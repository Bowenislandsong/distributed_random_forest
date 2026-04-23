"""Reproducible performance benchmark for the README and docs."""

from time import perf_counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import FederatedRandomForest, RandomForest


def run_case(name, builder, X_train, X_test, y_train, y_test):
    """Run a benchmark case and return a result dictionary."""
    start = perf_counter()
    model = builder()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    duration = perf_counter() - start

    return {
        'name': name,
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'weighted_accuracy': metrics['weighted_accuracy'],
        'f1_score': metrics['f1_score'],
        'seconds': duration,
        'strategy': getattr(model, 'selected_strategy', 'n/a'),
    }


def format_markdown_table(results):
    """Format results as a Markdown table."""
    header = (
        "| Scenario | Accuracy | Balanced Acc. | Weighted Acc. | F1 | Time (s) | Strategy |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    rows = []
    for result in results:
        rows.append(
            "| {name} | {accuracy:.4f} | {balanced_accuracy:.4f} | "
            "{weighted_accuracy:.4f} | {f1_score:.4f} | {seconds:.2f} | {strategy} |".format(
                **result
            )
        )
    return "\n".join([header] + rows)


def main():
    X, y = make_classification(
        n_samples=6000,
        n_features=40,
        n_classes=4,
        n_informative=18,
        n_redundant=8,
        class_sep=1.3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    results.append(
        run_case(
            "Centralized RF",
            lambda: RandomForest(
                n_estimators=96,
                voting='weighted',
                random_state=42,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    results.append(
        run_case(
            "Federated uniform",
            lambda: FederatedRandomForest(
                n_clients=6,
                rf_params={'n_estimators': 24, 'random_state': 42, 'voting': 'weighted'},
                partition_strategy='uniform',
                aggregation_strategy='auto',
                execution_backend='thread',
                max_workers=6,
                random_state=42,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    results.append(
        run_case(
            "Federated dirichlet",
            lambda: FederatedRandomForest(
                n_clients=6,
                rf_params={'n_estimators': 24, 'random_state': 42, 'voting': 'weighted'},
                partition_strategy='dirichlet',
                partition_kwargs={'alpha': 0.8},
                aggregation_strategy='auto',
                execution_backend='thread',
                max_workers=6,
                random_state=42,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    results.append(
        run_case(
            "Federated dirichlet + DP",
            lambda: FederatedRandomForest(
                n_clients=6,
                rf_params={'n_estimators': 24, 'random_state': 42, 'voting': 'weighted'},
                partition_strategy='dirichlet',
                partition_kwargs={'alpha': 0.8},
                aggregation_strategy='top_k_global_balanced_accuracy',
                execution_backend='sequential',
                use_differential_privacy=True,
                epsilon=10.0,
                random_state=42,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    print(format_markdown_table(results))


if __name__ == '__main__':
    main()
