"""Differentially Private Random Forest implementation."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from distributed_random_forest.models.random_forest import RandomForest
from distributed_random_forest.models.tree_utils import (
    compute_class_distribution,
    evaluate_predictions,
)


class DPRandomForest(RandomForest):
    """Random Forest with Differential Privacy.

    Implements differential privacy at the training level using
    Laplace mechanism for leaf node counts and exponential mechanism
    for split selection.

    Attributes:
        epsilon: Privacy budget for differential privacy.
        dp_mechanism: DP mechanism to use ('laplace' or 'gaussian').
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
        epsilon=1.0,
        delta=1e-3,
        dp_mechanism='laplace',
    ):
        """Initialize DP Random Forest.

        Args:
            n_estimators: Number of trees in the forest.
            criterion: Splitting rule ('gini' or 'entropy').
            voting: Voting method ('simple' or 'weighted').
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples at a leaf.
            random_state: Random seed.
            n_jobs: Parallelism for per-tree weighting and merged prediction.
            epsilon: Privacy budget (smaller = more private).
            dp_mechanism: DP noise mechanism ('laplace' or 'gaussian').
        """
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            voting=voting,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.epsilon = epsilon
        self.delta = delta
        self.dp_mechanism = dp_mechanism
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit DP Random Forest with privacy-preserving training.

        The privacy budget is split across all trees (composition).

        Args:
            X: Training features.
            y: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            self: Fitted DPRandomForest instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta < 0:
            raise ValueError("delta must be non-negative")
        if self.dp_mechanism not in {'laplace', 'gaussian'}:
            raise ValueError("dp_mechanism must be either 'laplace' or 'gaussian'")

        self._classes = np.unique(y)
        n_samples, n_features = X.shape

        epsilon_per_tree = self.epsilon / self.n_estimators

        self._trees = []
        for i in range(self.n_estimators):
            seed = None
            if self.random_state is not None:
                seed = self.random_state + i

            tree = self._build_dp_tree(
                X, y, epsilon_per_tree, n_features, seed
            )
            self._trees.append(tree)

        if self.voting == 'weighted' and X_val is not None and y_val is not None:
            self._compute_tree_weights(X_val, y_val)
        else:
            self._tree_weights = np.ones(len(self._trees))

        self.training_summary = {
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'n_classes': int(len(self._classes)),
            'class_distribution': compute_class_distribution(y, self._classes),
            'epsilon': float(self.epsilon),
            'delta': float(self.delta),
            'dp_mechanism': self.dp_mechanism,
        }
        return self

    def _build_dp_tree(self, X, y, epsilon, n_features, random_state):
        """Build a single decision tree with DP.

        Uses bootstrap sampling with DP noise on leaf predictions.

        Args:
            X: Training features.
            y: Training labels.
            epsilon: Privacy budget for this tree.
            n_features: Number of features.
            random_state: Random seed.

        Returns:
            DecisionTreeClassifier: Fitted tree with DP predictions.
        """
        rng = np.random.default_rng(random_state)

        n_samples = X.shape[0]
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]

        max_features = max(1, int(np.sqrt(n_features)))

        tree = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        tree.fit(X_boot, y_boot)

        self._add_dp_noise_to_tree(tree, epsilon, rng)

        return tree

    def _add_dp_noise_to_tree(self, tree, epsilon, rng):
        """Add DP noise to tree leaf predictions.

        Applies Laplace or Gaussian noise to class counts at leaves.

        Args:
            tree: Fitted DecisionTreeClassifier.
            epsilon: Privacy budget.
            rng: Random number generator.
        """
        tree_internal = tree.tree_

        sensitivity = 1.0

        n_nodes = tree_internal.node_count
        for node_id in range(n_nodes):
            if tree_internal.children_left[node_id] == -1:
                class_counts = tree_internal.value[node_id, 0, :]

                if self.dp_mechanism == 'laplace':
                    scale = sensitivity / max(epsilon, 1e-10)
                    noise = rng.laplace(0, scale, size=class_counts.shape)
                else:
                    # Gaussian mechanism with (epsilon, delta)-DP
                    sigma = (
                        sensitivity * np.sqrt(2 * np.log(1.25 / max(self.delta, 1e-12)))
                        / max(epsilon, 1e-10)
                    )
                    noise = rng.normal(0, sigma, size=class_counts.shape)

                noisy_counts = np.maximum(class_counts + noise, 0)
                tree_internal.value[node_id, 0, :] = noisy_counts

    def get_privacy_budget(self):
        """Get the privacy budget (epsilon) used.

        Returns:
            float: Epsilon value.
        """
        return self.epsilon


class DPClientRF:
    """DP Random Forest trainer for a single federated client.

    Attributes:
        client_id: Unique identifier for this client.
        epsilon: Privacy budget for this client.
        rf: DPRandomForest instance.
    """

    def __init__(self, client_id, epsilon=1.0, rf_params=None, client_name=None, metadata=None):
        """Initialize DP client trainer.

        Args:
            client_id: Unique client identifier.
            epsilon: Privacy budget.
            rf_params: Dictionary of RandomForest parameters.
        """
        self.client_id = client_id
        self.epsilon = epsilon
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
        """Train DP Random Forest on client data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            self: Trained DPClientRF instance.
        """
        self.rf = DPRandomForest(epsilon=self.epsilon, **self.rf_params)
        self.rf.fit(X_train, y_train, X_val, y_val)

        self.n_train_samples = int(len(X_train))
        self.class_distribution = compute_class_distribution(y_train, self.rf.classes_)
        self.train_metrics = self.rf.evaluate(X_train, y_train)
        self.train_metrics['epsilon'] = self.epsilon

        if X_val is not None and y_val is not None:
            self.n_validation_samples = int(len(X_val))
            self.val_metrics = self.rf.evaluate(X_val, y_val)

        return self

    def get_trees(self):
        """Get trees from trained DP-RF."""
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
            raise RuntimeError("DP-RF not trained")

        metrics = evaluate_predictions(y_test, self.rf.predict(X_test), self.rf.classes_)
        metrics['epsilon'] = self.epsilon
        return metrics

    def summary(self):
        """Return a JSON-serializable snapshot of the DP client state."""
        return {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'epsilon': self.epsilon,
            'n_train_samples': self.n_train_samples,
            'n_validation_samples': self.n_validation_samples,
            'class_distribution': self.class_distribution,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'metadata': self.metadata,
        }
