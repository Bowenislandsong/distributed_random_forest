"""Non-IID federated training example with report export."""

from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import FederatedRandomForest


def main():
    X, y = make_classification(
        n_samples=1600,
        n_features=24,
        n_classes=4,
        n_informative=12,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=7,
        stratify=y,
    )

    model = FederatedRandomForest(
        n_clients=6,
        rf_params={'n_estimators': 32, 'random_state': 7, 'voting': 'weighted'},
        partition_strategy='dirichlet',
        partition_kwargs={'alpha': 0.35},
        aggregation_strategy='auto',
        execution_backend='thread',
        max_workers=6,
        random_state=7,
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    report_path = artifacts_dir / 'dirichlet-run.json'
    model.export_report(report_path)

    print("Selected strategy:", model.selected_strategy)
    print("Metrics:", metrics)
    print("Report:", report_path)


if __name__ == '__main__':
    main()
