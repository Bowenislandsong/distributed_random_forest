"""EXP 1 — RF Hyperparameter Selection.

Grid search over:
- Number of trees (odd numbers 1-100)
- Splitting rule (gini, entropy)
- Ensemble rule (SV, WV)

The best configuration is used for all remaining experiments.
"""

import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

from distributed_random_forest.models.random_forest import RandomForest
from distributed_random_forest.models.tree_utils import compute_accuracy, compute_weighted_accuracy, compute_f1_score


def run_exp1_hyperparameter_selection(
    X,
    y,
    n_estimators_range=None,
    criteria=None,
    voting_methods=None,
    validation_split=0.2,
    random_state=42,
    verbose=True,
):
    """Run hyperparameter selection experiment.

    Args:
        X: Features array.
        y: Labels array.
        n_estimators_range: List of n_estimators values to try.
            Default is odd numbers from 1 to 100.
        criteria: List of splitting criteria to try.
            Default is ['gini', 'entropy'].
        voting_methods: List of voting methods to try.
            Default is ['simple', 'weighted'].
        validation_split: Fraction of data for validation.
        random_state: Random seed.
        verbose: Whether to print progress.

    Returns:
        dict: Best parameters and results.
    """
    if n_estimators_range is None:
        n_estimators_range = list(range(1, 101, 2))

    if criteria is None:
        criteria = ['gini', 'entropy']

    if voting_methods is None:
        voting_methods = ['simple', 'weighted']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=random_state, stratify=y
    )

    results = []
    best_score = -1
    best_params = None
    best_rf = None

    total_configs = len(n_estimators_range) * len(criteria) * len(voting_methods)
    config_num = 0

    for n_est, criterion, voting in product(n_estimators_range, criteria, voting_methods):
        config_num += 1

        if verbose and config_num % 50 == 0:
            print(f"Testing configuration {config_num}/{total_configs}")

        rf = RandomForest(
            n_estimators=n_est,
            criterion=criterion,
            voting=voting,
            random_state=random_state,
        )
        rf.fit(X_train, y_train, X_val, y_val)

        y_pred = rf.predict(X_val)
        accuracy = compute_accuracy(y_val, y_pred)
        weighted_accuracy = compute_weighted_accuracy(y_val, y_pred)
        f1 = compute_f1_score(y_val, y_pred)

        result = {
            'n_estimators': n_est,
            'criterion': criterion,
            'voting': voting,
            'accuracy': accuracy,
            'weighted_accuracy': weighted_accuracy,
            'f1_score': f1,
        }
        results.append(result)

        if accuracy > best_score:
            best_score = accuracy
            best_params = {
                'n_estimators': n_est,
                'criterion': criterion,
                'voting': voting,
            }
            best_rf = rf

    if verbose:
        print(f"\nBest parameters found:")
        print(f"  n_estimators: {best_params['n_estimators']}")
        print(f"  criterion: {best_params['criterion']}")
        print(f"  voting: {best_params['voting']}")
        print(f"  accuracy: {best_score:.4f}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_rf': best_rf,
        'all_results': results,
    }


def quick_hyperparameter_selection(
    X,
    y,
    n_estimators_candidates=None,
    validation_split=0.2,
    random_state=42,
):
    """Quick hyperparameter selection with reduced search space.

    Args:
        X: Features array.
        y: Labels array.
        n_estimators_candidates: List of n_estimators to try.
        validation_split: Fraction of data for validation.
        random_state: Random seed.

    Returns:
        dict: Best parameters.
    """
    if n_estimators_candidates is None:
        n_estimators_candidates = [11, 21, 51, 101]

    return run_exp1_hyperparameter_selection(
        X,
        y,
        n_estimators_range=n_estimators_candidates,
        validation_split=validation_split,
        random_state=random_state,
        verbose=True,
    )


def get_default_best_params():
    """Get default best parameters.

    These can be used when hyperparameter selection hasn't been run.

    Returns:
        dict: Default RF parameters.
    """
    return {
        'n_estimators': 101,
        'criterion': 'gini',
        'voting': 'simple',
    }
