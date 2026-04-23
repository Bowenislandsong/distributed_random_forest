"""Random Forest implementation with configurable splitting and voting rules."""

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from distributed_random_forest.parallelism import resolve_n_jobs
from distributed_random_forest.models.tree_utils import (
    compute_accuracy,
    compute_weighted_accuracy,
    _map_tree_predictions,
)


def _tree_predict_mapped(tree, X, classes):
    """For :class:`joblib.Parallel` across trees (module-level, picklable)."""
    preds = tree.predict(X)
    if classes is not None and hasattr(tree, 'classes_'):
        preds = _map_tree_predictions(preds, tree.classes_, classes)
    return preds


def _tree_proba(tree, X):
    """``tree.predict_proba`` for :class:`joblib.Parallel` workers."""
    return tree.predict_proba(X)


def _tree_wa_for_weight(tree, X_val, y_val, classes):
    """One tree’s weighted accuracy on validation (for weight vector)."""
    y_pred = tree.predict(X_val)
    if classes is not None and hasattr(tree, 'classes_'):
        y_pred = _map_tree_predictions(y_pred, tree.classes_, classes)
    return max(compute_weighted_accuracy(y_val, y_pred, classes), 1e-6)


class RandomForest:
    """Random Forest classifier with support for Gini/Entropy and SV/WV.

    Attributes:
        n_estimators: Number of trees in the forest.
        criterion: Splitting rule ('gini' or 'entropy').
        voting: Voting method ('simple' or 'weighted').
        max_depth: Maximum depth of trees.
        random_state: Random seed for reproducibility.
        n_jobs: Parallel workers (sklearn training, per-tree weighting, and
            merged-tree prediction; ``-1`` = all CPUs, ``1`` = sequential).
    """

    def __init__(
        self,
        n_estimators=100,
        criterion='gini',
        voting='simple',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=-1,
    ):
        """Initialize Random Forest.

        Args:
            n_estimators: Number of trees in the forest.
            criterion: Splitting rule ('gini' or 'entropy').
            voting: Voting method ('simple' or 'weighted').
            max_depth: Maximum depth of trees (None for unlimited).
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            random_state: Random seed for reproducibility.
            n_jobs: See :func:`distributed_random_forest.parallelism.resolve_n_jobs`.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.voting = voting
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._forest = None
        self._trees = []
        self._tree_weights = None
        self._classes = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the Random Forest to training data.

        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (n_samples,).
            X_val: Optional validation features for computing tree weights.
            y_val: Optional validation labels.

        Returns:
            self: Fitted RandomForest instance.
        """
        self._forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=resolve_n_jobs(self.n_jobs),
        )
        self._forest.fit(X, y)
        self._trees = list(self._forest.estimators_)
        self._classes = self._forest.classes_

        if self.voting == 'weighted' and X_val is not None and y_val is not None:
            self._compute_tree_weights(X_val, y_val)
        else:
            self._tree_weights = np.ones(len(self._trees))

        return self

    def _compute_tree_weights(self, X_val, y_val):
        """Compute weights for each tree based on class-specific accuracy.

        Args:
            X_val: Validation features.
            y_val: Validation labels.
        """
        n_workers = resolve_n_jobs(self.n_jobs)
        if n_workers <= 1 or len(self._trees) <= 1:
            weights = [
                _tree_wa_for_weight(t, X_val, y_val, self._classes)
                for t in self._trees
            ]
        else:
            weights = Parallel(n_jobs=n_workers)(
                delayed(_tree_wa_for_weight)(t, X_val, y_val, self._classes)
                for t in self._trees
            )

        self._tree_weights = np.array(weights)
        self._tree_weights /= self._tree_weights.sum()

    def predict(self, X):
        """Predict class labels for samples.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels.
        """
        if self.voting == 'simple':
            return self._simple_voting(X)
        else:
            return self._weighted_voting(X)

    def _stack_tree_predictions(self, X):
        """Shape (n_trees, n_samples) of class labels (mapped to ``self._classes``)."""
        n_workers = resolve_n_jobs(self.n_jobs)
        if n_workers <= 1 or len(self._trees) <= 1:
            return np.array(
                [_tree_predict_mapped(t, X, self._classes) for t in self._trees]
            )
        return np.array(
            Parallel(n_jobs=n_workers)(
                delayed(_tree_predict_mapped)(t, X, self._classes) for t in self._trees
            )
        )

    def _simple_voting(self, X):
        """Perform simple majority voting.

        Args:
            X: Features to predict.

        Returns:
            ndarray: Predicted labels via majority vote.
        """
        predictions = self._stack_tree_predictions(X)
        result = []
        for i in range(X.shape[0]):
            sample_preds = predictions[:, i]
            unique, counts = np.unique(sample_preds, return_counts=True)
            result.append(unique[np.argmax(counts)])
        return np.array(result)

    def _weighted_voting(self, X):
        """Perform weighted voting using tree weights.

        Args:
            X: Features to predict.

        Returns:
            ndarray: Predicted labels via weighted vote.
        """
        n_samples = X.shape[0]
        class_votes = np.zeros((n_samples, len(self._classes)))
        all_preds = self._stack_tree_predictions(X)

        for tree_idx, preds in enumerate(all_preds):
            weight = self._tree_weights[tree_idx]
            for sample_idx, pred in enumerate(preds):
                class_idx = np.where(self._classes == pred)[0][0]
                class_votes[sample_idx, class_idx] += weight

        return self._classes[np.argmax(class_votes, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities for samples.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            ndarray: Class probabilities (n_samples, n_classes).
        """
        if self._forest is not None:
            return self._forest.predict_proba(X)

        n_samples = X.shape[0]
        n_workers = resolve_n_jobs(self.n_jobs)
        if n_workers <= 1 or len(self._trees) <= 1:
            proba = np.zeros((n_samples, len(self._classes)))
            for tree_idx, tree in enumerate(self._trees):
                tree_proba = tree.predict_proba(X)
                proba += self._tree_weights[tree_idx] * tree_proba
        else:
            all_proba = Parallel(n_jobs=n_workers)(
                delayed(_tree_proba)(t, X) for t in self._trees
            )
            proba = np.zeros((n_samples, len(self._classes)))
            for tree_idx, tree_proba in enumerate(all_proba):
                proba += self._tree_weights[tree_idx] * tree_proba

        if self.voting == 'simple':
            proba /= len(self._trees)

        return proba

    def score(self, X, y):
        """Compute accuracy score.

        Args:
            X: Features.
            y: True labels.

        Returns:
            float: Accuracy score.
        """
        return compute_accuracy(y, self.predict(X))

    def get_trees(self):
        """Get individual decision trees from the forest.

        Returns:
            list: List of fitted DecisionTreeClassifier instances.
        """
        return self._trees

    def set_trees(self, trees, classes=None):
        """Set decision trees for the forest.

        Args:
            trees: List of fitted decision tree estimators.
            classes: Class labels (required if trees have different classes).
        """
        self._trees = list(trees)
        self.n_estimators = len(trees)
        self._tree_weights = np.ones(len(trees))

        if classes is not None:
            self._classes = np.array(classes)
        elif len(trees) > 0 and hasattr(trees[0], 'classes_'):
            self._classes = trees[0].classes_

        self._forest = None

    @property
    def classes_(self):
        """Get class labels."""
        return self._classes


class ClientRF:
    """Random Forest trainer for a single federated client.

    Attributes:
        client_id: Unique identifier for this client.
        rf: RandomForest instance.
    """

    def __init__(self, client_id, rf_params=None):
        """Initialize client trainer.

        Args:
            client_id: Unique client identifier.
            rf_params: Dictionary of RandomForest parameters.
        """
        self.client_id = client_id
        self.rf_params = rf_params or {}
        self.rf = None
        self.train_metrics = {}
        self.val_metrics = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest on client data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            self: Trained ClientRF instance.
        """
        self.rf = RandomForest(**self.rf_params)
        self.rf.fit(X_train, y_train, X_val, y_val)

        y_pred_train = self.rf.predict(X_train)
        self.train_metrics = {
            'accuracy': compute_accuracy(y_train, y_pred_train),
            'weighted_accuracy': compute_weighted_accuracy(
                y_train, y_pred_train, self.rf.classes_
            ),
        }

        if X_val is not None and y_val is not None:
            y_pred_val = self.rf.predict(X_val)
            self.val_metrics = {
                'accuracy': compute_accuracy(y_val, y_pred_val),
                'weighted_accuracy': compute_weighted_accuracy(
                    y_val, y_pred_val, self.rf.classes_
                ),
            }

        return self

    def get_trees(self):
        """Get trees from trained RF."""
        if self.rf is None:
            return []
        return self.rf.get_trees()

    def evaluate(self, X_test, y_test):
        """Evaluate on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        if self.rf is None:
            raise RuntimeError("RF not trained")

        y_pred = self.rf.predict(X_test)
        return {
            'accuracy': compute_accuracy(y_test, y_pred),
            'weighted_accuracy': compute_weighted_accuracy(
                y_test, y_pred, self.rf.classes_
            ),
        }
