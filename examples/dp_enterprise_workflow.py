"""Differentially private federated workflow example."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import FederatedRandomForest


def main():
    X, y = make_classification(
        n_samples=1400,
        n_features=18,
        n_classes=3,
        n_informative=9,
        random_state=13,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=13,
        stratify=y,
    )

    model = FederatedRandomForest(
        n_clients=5,
        rf_params={'n_estimators': 20, 'random_state': 13},
        partition_strategy='stratified',
        aggregation_strategy='top_k_global_balanced_accuracy',
        use_differential_privacy=True,
        epsilon=2.0,
        execution_backend='sequential',
        random_state=13,
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    print("DP selected strategy:", model.selected_strategy)
    print("Metrics:", metrics)


if __name__ == '__main__':
    main()
