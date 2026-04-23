"""Load well-known *public* classification sets as NumPy arrays (optionally with pandas)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class BreastCancerBenchmark:
    """Stratified train / validation / test split of the UCI breast cancer set."""

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int_]
    X_val: NDArray[np.float64]
    y_val: NDArray[np.int_]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int_]
    feature_names: list[str]
    name: str
    description: str
    data_frame: Optional[pd.DataFrame] = None

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(np.concatenate([self.y_train, self.y_val, self.y_test]))))


def load_breast_cancer_bench(
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    as_frame: bool = False,
) -> BreastCancerBenchmark:
    """
    Wisconsin Breast Cancer (UCI) — 569 rows, 30 numeric features, binary target.

    Fetched with ``sklearn.datasets.load_breast_cancer`` and delivered as
    :class:`numpy.ndarray` for model code. If ``as_frame`` is true, a copy of
    the full *pandas* ``DataFrame`` (before splitting) is attached for
    inspection (``.data_frame``).

    Args:
        test_size: Fraction of data held out for the test set.
        val_size: Of the non-test rows, fraction held out for validation.
        random_state: Seed for :func:`sklearn.model_selection.train_test_split`.
        as_frame: If true, set ``data_frame`` on the returned object.

    Returns:
        BreastCancerBenchmark with aligned arrays and metadata.
    """
    bunch = load_breast_cancer(as_frame=True)
    frame: pd.DataFrame = bunch.frame
    x_cols = [c for c in frame.columns if c != "target"]
    X = frame[x_cols].to_numpy(dtype=np.float64, copy=True)
    y = frame["target"].to_numpy(copy=True).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    out = BreastCancerBenchmark(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=x_cols,
        name="wdbc_breast_cancer",
        description=(
            "Wisconsin Diagnostic Breast Cancer (UCI) via scikit-learn; "
            "binary classification, 30 real-valued features."
        ),
        data_frame=frame if as_frame else None,
    )
    return out


def summarize_split(split: Any) -> dict[str, Any]:
    """Lightweight key figures for logging or unit tests (works with a dataclass or dict)."""
    if isinstance(split, BreastCancerBenchmark):
        Xtr, ytr = split.X_train, split.y_train
        _, yv = split.X_val, split.y_val
        _, yte = split.X_test, split.y_test
    else:
        Xtr, ytr = split["X_train"], split["y_train"]
        _, yv = split["X_val"], split["y_val"]
        _, yte = split["X_test"], split["y_test"]
    return {
        "n_train": int(len(ytr)),
        "n_val": int(len(yv)),
        "n_test": int(len(yte)),
        "n_features": int(Xtr.shape[1]),
    }
