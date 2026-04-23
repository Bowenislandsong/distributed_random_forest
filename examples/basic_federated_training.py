"""Basic federated training example."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import FederatedRandomForest


def main():
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = FederatedRandomForest(
        n_clients=4,
        rf_params={'n_estimators': 24, 'random_state': 42, 'voting': 'weighted'},
        partition_strategy='uniform',
        aggregation_strategy='auto',
        execution_backend='thread',
        max_workers=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("Selected strategy:", model.selected_strategy)
    print("Metrics:", metrics)


if __name__ == '__main__':
    main()
