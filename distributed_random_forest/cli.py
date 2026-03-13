"""Command-line entry points for quick experimentation."""

import argparse

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest.distributed.orchestrator import FederatedRandomForest


def build_parser():
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a quick distributed random forest benchmark on synthetic data.",
    )
    parser.add_argument('--samples', type=int, default=1200, help='Number of synthetic samples.')
    parser.add_argument('--features', type=int, default=20, help='Number of synthetic features.')
    parser.add_argument('--classes', type=int, default=3, help='Number of target classes.')
    parser.add_argument('--clients', type=int, default=4, help='Number of federated clients.')
    parser.add_argument('--trees', type=int, default=24, help='Trees per client forest.')
    parser.add_argument(
        '--partition-strategy',
        default='dirichlet',
        choices=['uniform', 'stratified', 'dirichlet', 'label_skew', 'feature', 'sized'],
        help='Data partitioning strategy.',
    )
    parser.add_argument(
        '--backend',
        default='thread',
        choices=['sequential', 'thread', 'process'],
        help='Execution backend for client training.',
    )
    parser.add_argument(
        '--aggregation-strategy',
        default='auto',
        help='Aggregation strategy or "auto" for strategy search.',
    )
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha when applicable.')
    parser.add_argument(
        '--dp-epsilon',
        type=float,
        default=None,
        help='Enable differential privacy with the given epsilon.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--json',
        action='store_true',
        help='Print the final report as JSON instead of a compact summary.',
    )
    return parser


def main(argv=None):
    """Run the quick synthetic benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_classes=args.classes,
        n_informative=max(args.classes * 2, args.features // 2),
        random_state=args.seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    partition_kwargs = {}
    if args.partition_strategy == 'dirichlet':
        partition_kwargs['alpha'] = args.alpha

    model = FederatedRandomForest(
        n_clients=args.clients,
        rf_params={
            'n_estimators': args.trees,
            'random_state': args.seed,
            'voting': 'weighted',
        },
        partition_strategy=args.partition_strategy,
        partition_kwargs=partition_kwargs,
        aggregation_strategy=args.aggregation_strategy,
        execution_backend=args.backend,
        max_workers=args.clients if args.backend != 'sequential' else None,
        random_state=args.seed,
        use_differential_privacy=args.dp_epsilon is not None,
        epsilon=1.0 if args.dp_epsilon is None else args.dp_epsilon,
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    if args.json:
        print(model.report.to_json())
        return 0

    print("Distributed Random Forest quickstart")
    print(f"  clients: {args.clients}")
    print(f"  partition strategy: {args.partition_strategy}")
    print(f"  selected aggregation strategy: {model.selected_strategy}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  weighted accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  f1 score: {metrics['f1_score']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
