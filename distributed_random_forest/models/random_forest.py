"""Random Forest implementation with configurable splitting and voting rules."""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from distributed_random_forest.federation.voting import simple_voting, weighted_voting
from distributed_random_forest.models.tree_utils import (
    _map_tree_predictions,
    _map_tree_probabilities,
    compute_class_distribution,
    compute_weighted_accuracy,
    evaluate_predictions,
)


class RandomForest:
    """Random Forest classifier with support for Gini/Entropy and SV/WV.

    Attributes:
        n_estimators: Number of trees in the forest.
        criterion: Splitting rule ('gini' or 'entropy').
        voting: Voting method ('simple' or 'weighted').
        max_depth: Maximum depth of trees.
        random_state: Random seed for reproducibility.
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
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.voting = voting
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self._forest = None
        self._trees = []
        self._tree_weights = None
        self._classes = None
        self.training_summary = {}

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
        X = np.asarray(X)
        y = np.asarray(y)

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.criterion not in {'gini', 'entropy', 'log_loss'}:
            raise ValueError("criterion must be one of: gini, entropy, log_loss")
        if self.voting not in {'simple', 'weighted'}:
            raise ValueError("voting must be either 'simple' or 'weighted'")

        self._forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._forest.fit(X, y)
        self._trees = list(self._forest.estimators_)
        self._classes = self._forest.classes_

        if self.voting == 'weighted' and X_val is not None and y_val is not None:
            self._compute_tree_weights(X_val, y_val)
        else:
            self._tree_weights = np.ones(len(self._trees))

        self.training_summary = {
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'n_classes': int(len(self._classes)),
            'class_distribution': compute_class_distribution(y, self._classes),
        }
        return self

    def _compute_tree_weights(self, X_val, y_val):
        """Compute weights for each tree based on class-specific accuracy.

        Args:
            X_val: Validation features.
            y_val: Validation labels.
        """
        weights = []
        for tree in self._trees:
            y_pred = tree.predict(X_val)
            # Map tree predictions to original class labels if needed
            if self._classes is not None and hasattr(tree, 'classes_'):
                y_pred = _map_tree_predictions(y_pred, tree.classes_, self._classes)
            wa = compute_weighted_accuracy(y_val, y_pred, self._classes)
            weights.append(max(wa, 1e-6))

        self._tree_weights = np.array(weights)
        self._tree_weights /= self._tree_weights.sum()

    def _ensure_ready(self):
        """Validate that the forest has fitted or attached trees."""
        if not self._trees:
            raise RuntimeError("RandomForest has no trained trees. Call fit() first.")

    def predict(self, X):
        """Predict class labels for samples.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels.
        """
        self._ensure_ready()
        if self.voting == 'simple':
            return self._simple_voting(X)
        return self._weighted_voting(X)

    def _simple_voting(self, X):
        """Perform simple majority voting.

        Args:
            X: Features to predict.

        Returns:
            ndarray: Predicted labels via majority vote.
        """
        all_preds = []
        for tree in self._trees:
            preds = tree.predict(X)
            # Map tree predictions to original class labels if needed
            if self._classes is not None and hasattr(tree, 'classes_'):
                preds = _map_tree_predictions(preds, tree.classes_, self._classes)
            all_preds.append(preds)

        predictions = np.array(all_preds)
        return simple_voting(predictions, self._classes)

    def _weighted_voting(self, X):
        """Perform weighted voting using tree weights.

        Args:
            X: Features to predict.

        Returns:
            ndarray: Predicted labels via weighted vote.
        """
        predictions = []
        for tree in self._trees:
            preds = tree.predict(X)
            if self._classes is not None and hasattr(tree, 'classes_'):
                preds = _map_tree_predictions(preds, tree.classes_, self._classes)
            predictions.append(preds)

        return weighted_voting(np.array(predictions), self._tree_weights, self._classes)

    def predict_proba(self, X):
        """Predict class probabilities for samples.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            ndarray: Class probabilities (n_samples, n_classes).
        """
        self._ensure_ready()

        if self._forest is not None and self.voting == 'simple':
            return self._forest.predict_proba(X)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self._classes)))

        for tree_idx, tree in enumerate(self._trees):
            tree_proba = tree.predict_proba(X)
            if self._classes is not None and hasattr(tree, 'classes_'):
                tree_proba = _map_tree_probabilities(
                    tree_proba,
                    tree.classes_,
                    self._classes,
                )
            weight = self._tree_weights[tree_idx]
            proba += weight * tree_proba

        total_weight = np.sum(self._tree_weights)
        if total_weight > 0:
            proba /= total_weight

        return proba

    def score(self, X, y):
        """Compute accuracy score.

        Args:
            X: Features.
            y: True labels.

        Returns:
            float: Accuracy score.
        """
        return evaluate_predictions(y, self.predict(X), self._classes)['accuracy']

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

    def evaluate(self, X, y):
        """Evaluate the forest using the default metric bundle."""
        return evaluate_predictions(y, self.predict(X), self._classes)

    def save(self, path):
        """Persist the forest with pickle for quick experimentation."""
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path):
        """Restore a previously persisted forest."""
        with open(path, 'rb') as handle:
            return pickle.load(handle)


class ClientRF:
    """Random Forest trainer for a single federated client.

    Attributes:
        client_id: Unique identifier for this client.
        rf: RandomForest instance.
    """

    def __init__(self, client_id, rf_params=None, client_name=None, metadata=None):
        """Initialize client trainer.

        Args:
            client_id: Unique client identifier.
            rf_params: Dictionary of RandomForest parameters.
        """
        self.client_id = client_id
        self.rf_params = rf_params or {}
        self.client_name = client_name or f"client-{client_id}"
        self.metadata = metadata or {}
        self.rf = None
        self.train_metrics = {}
        self.val_metrics = {}
        self.n_train_samples = 0
        self.n_validation_samples = 0
        self.class_distribution = {}

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

        self.n_train_samples = int(len(X_train))
        self.class_distribution = compute_class_distribution(y_train, self.rf.classes_)

        self.train_metrics = self.rf.evaluate(X_train, y_train)

        if X_val is not None and y_val is not None:
            self.n_validation_samples = int(len(X_val))
            self.val_metrics = self.rf.evaluate(X_val, y_val)

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

        return self.rf.evaluate(X_test, y_test)

    def summary(self):
        """Return a JSON-serializable snapshot of the client state."""
        return {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'n_train_samples': self.n_train_samples,
            'n_validation_samples': self.n_validation_samples,
            'class_distribution': self.class_distribution,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'metadata': self.metadata,
        }
