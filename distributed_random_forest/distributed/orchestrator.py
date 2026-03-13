"""High-level orchestration for distributed and federated Random Forests."""

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split

from distributed_random_forest.distributed.partitioning import (
    create_partitions,
    summarize_partitions,
)
from distributed_random_forest.federation.aggregator import FederatedAggregator
from distributed_random_forest.models.dp_rf import DPClientRF
from distributed_random_forest.models.random_forest import ClientRF
from distributed_random_forest.models.tree_utils import evaluate_predictions

DEFAULT_AUTO_STRATEGIES = [
    'rf_s_dts_wa_all',
    'top_k_global_balanced_accuracy',
    'proportional_weighted_accuracy',
    'rf_s_dts_a_all',
]


@dataclass
class StrategyEvaluation:
    """Validation report for a candidate aggregation strategy."""

    strategy: str
    n_trees: int
    validation_metrics: dict
    aggregation_summary: dict

    def to_dict(self):
        """Convert the strategy evaluation to a dictionary."""
        return {
            'strategy': self.strategy,
            'n_trees': self.n_trees,
            'validation_metrics': self.validation_metrics,
            'aggregation_summary': self.aggregation_summary,
        }


@dataclass
class FederatedRunReport:
    """Structured report for a federated training run."""

    execution_backend: str
    partition_strategy: str
    selected_strategy: str
    partition_summary: list
    client_summaries: list
    strategy_reports: list
    validation_metrics: dict = field(default_factory=dict)
    global_metrics: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert the report to a JSON-serializable dictionary."""
        return {
            'execution_backend': self.execution_backend,
            'partition_strategy': self.partition_strategy,
            'selected_strategy': self.selected_strategy,
            'partition_summary': self.partition_summary,
            'client_summaries': self.client_summaries,
            'strategy_reports': self.strategy_reports,
            'validation_metrics': self.validation_metrics,
            'global_metrics': self.global_metrics,
        }

    def to_json(self, indent=2):
        """Serialize the report to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _safe_train_val_split(X, y, validation_split, random_state):
    """Split a local client dataset while handling small or skewed samples."""
    if validation_split is None or validation_split <= 0 or len(X) < 3:
        return X, None, y, None

    try:
        return train_test_split(
            X,
            y,
            test_size=validation_split,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    except ValueError:
        return X, None, y, None


def _train_client_task(task):
    """Train a single client model. This function is process-safe."""
    if task['use_dp']:
        rf_params = dict(task['rf_params'])
        rf_params.setdefault('dp_mechanism', task['dp_mechanism'])
        client = DPClientRF(
            client_id=task['client_id'],
            epsilon=task['epsilon'],
            rf_params=rf_params,
            metadata=task['metadata'],
        )
    else:
        client = ClientRF(
            client_id=task['client_id'],
            rf_params=task['rf_params'],
            metadata=task['metadata'],
        )

    X_train, X_val, y_train, y_val = _safe_train_val_split(
        task['X'],
        task['y'],
        task['client_validation_split'],
        task['random_state'],
    )
    client.train(X_train, y_train, X_val, y_val)
    return client


def _run_tasks(tasks, execution_backend='sequential', max_workers=None):
    """Run training tasks using the requested execution backend."""
    if execution_backend == 'sequential':
        return [_train_client_task(task) for task in tasks]

    executor_cls = {
        'thread': ThreadPoolExecutor,
        'process': ProcessPoolExecutor,
    }.get(execution_backend)

    if executor_cls is None:
        raise ValueError(
            "execution_backend must be one of: sequential, thread, process"
        )

    with executor_cls(max_workers=max_workers) as executor:
        return list(executor.map(_train_client_task, tasks))


class FederatedRandomForest:
    """Enterprise-grade orchestration wrapper for distributed Random Forests."""

    def __init__(
        self,
        n_clients=5,
        rf_params=None,
        partition_strategy='uniform',
        partition_kwargs=None,
        aggregation_strategy='auto',
        candidate_strategies=None,
        n_trees_per_client=10,
        n_total_trees=100,
        min_score=None,
        selection_metric='weighted_accuracy',
        execution_backend='sequential',
        max_workers=None,
        client_validation_split=0.2,
        global_validation_split=0.2,
        random_state=42,
        use_differential_privacy=False,
        epsilon=1.0,
        dp_mechanism='laplace',
    ):
        self.n_clients = n_clients
        self.rf_params = rf_params or {'n_estimators': 100, 'random_state': random_state}
        self.partition_strategy = partition_strategy
        self.partition_kwargs = partition_kwargs or {}
        self.aggregation_strategy = aggregation_strategy
        self.candidate_strategies = candidate_strategies or list(DEFAULT_AUTO_STRATEGIES)
        self.n_trees_per_client = n_trees_per_client
        self.n_total_trees = n_total_trees
        self.min_score = min_score
        self.selection_metric = selection_metric
        self.execution_backend = execution_backend
        self.max_workers = max_workers
        self.client_validation_split = client_validation_split
        self.global_validation_split = global_validation_split
        self.random_state = random_state
        self.use_differential_privacy = use_differential_privacy
        self.epsilon = epsilon
        self.dp_mechanism = dp_mechanism

        self.partitions = None
        self.partition_summary = None
        self.clients = []
        self.aggregator = None
        self.global_rf = None
        self.selected_strategy = None
        self.strategy_reports = []
        self.report = None

    def _resolve_candidate_strategies(self):
        """Resolve which strategies to evaluate for the current run."""
        if self.aggregation_strategy == 'auto':
            return list(self.candidate_strategies)
        return [self.aggregation_strategy]

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit a federated forest from a centralized dataset."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.global_validation_split,
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        else:
            X_train, y_train = X, y

        self.partitions = create_partitions(
            X_train,
            y_train,
            strategy=self.partition_strategy,
            n_clients=self.n_clients,
            random_state=self.random_state,
            **self.partition_kwargs,
        )
        self.partition_summary = summarize_partitions(self.partitions, classes=np.unique(y))
        return self.fit_from_partitions(self.partitions, X_val, y_val)

    def fit_from_partitions(self, partitions, X_val, y_val):
        """Fit a federated forest from precomputed client partitions."""
        tasks = []
        for client_id, (X_part, y_part) in enumerate(partitions):
            tasks.append(
                {
                    'client_id': client_id,
                    'X': np.asarray(X_part),
                    'y': np.asarray(y_part),
                    'rf_params': dict(self.rf_params),
                    'client_validation_split': self.client_validation_split,
                    'random_state': self.random_state + client_id,
                    'use_dp': self.use_differential_privacy,
                    'epsilon': self.epsilon,
                    'dp_mechanism': self.dp_mechanism,
                    'metadata': {'partition_strategy': self.partition_strategy},
                }
            )

        self.clients = _run_tasks(
            tasks,
            execution_backend=self.execution_backend,
            max_workers=self.max_workers,
        )

        classes = np.unique(
            np.concatenate(
                [np.asarray(y_val)] + [np.asarray(client.rf.classes_) for client in self.clients]
            )
        )

        best_score = -np.inf
        best_aggregator = None
        best_global_rf = None
        best_metrics = None
        self.strategy_reports = []

        for strategy in self._resolve_candidate_strategies():
            aggregator = FederatedAggregator(
                strategy=strategy,
                n_trees_per_client=self.n_trees_per_client,
                n_total_trees=self.n_total_trees,
                min_score=self.min_score,
            )
            aggregator.aggregate(self.clients, X_val, y_val, classes=classes)
            global_rf = aggregator.build_global_rf(
                classes=classes,
                voting=self.rf_params.get('voting', 'simple'),
            )
            validation_metrics = evaluate_predictions(y_val, global_rf.predict(X_val), classes)
            summary = aggregator.get_summary() or {}
            summary['validation_metrics'] = validation_metrics
            self.strategy_reports.append(
                StrategyEvaluation(
                    strategy=strategy,
                    n_trees=len(aggregator.global_trees),
                    validation_metrics=validation_metrics,
                    aggregation_summary=summary,
                ).to_dict()
            )

            score = validation_metrics[self.selection_metric]
            if score > best_score:
                best_score = score
                best_aggregator = aggregator
                best_global_rf = global_rf
                best_metrics = validation_metrics
                self.selected_strategy = strategy

        self.aggregator = best_aggregator
        self.global_rf = best_global_rf
        self.report = FederatedRunReport(
            execution_backend=self.execution_backend,
            partition_strategy=self.partition_strategy,
            selected_strategy=self.selected_strategy,
            partition_summary=(
                self.partition_summary
                or summarize_partitions(partitions, classes=classes)
            ),
            client_summaries=[client.summary() for client in self.clients],
            strategy_reports=self.strategy_reports,
            validation_metrics=best_metrics,
        )
        return self

    def predict(self, X):
        """Predict with the selected global forest."""
        if self.global_rf is None:
            raise RuntimeError("FederatedRandomForest has not been fitted yet.")
        return self.global_rf.predict(X)

    def predict_proba(self, X):
        """Predict probabilities with the selected global forest."""
        if self.global_rf is None:
            raise RuntimeError("FederatedRandomForest has not been fitted yet.")
        return self.global_rf.predict_proba(X)

    def score(self, X, y):
        """Return accuracy on a held-out dataset."""
        return self.evaluate(X, y)['accuracy']

    def evaluate(self, X, y):
        """Evaluate the trained global forest and persist results in the report."""
        if self.global_rf is None:
            raise RuntimeError("FederatedRandomForest has not been fitted yet.")

        metrics = evaluate_predictions(y, self.global_rf.predict(X), self.global_rf.classes_)
        if self.report is not None:
            self.report.global_metrics = metrics
        return metrics

    def export_report(self, path):
        """Write the run report to disk as JSON."""
        if self.report is None:
            raise RuntimeError("No report available. Fit the model first.")
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(self.report.to_json())

    @property
    def classes_(self):
        """Expose the classes of the selected global forest."""
        if self.global_rf is None:
            return None
        return self.global_rf.classes_
