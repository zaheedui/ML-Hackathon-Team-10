"""Reusable training and cross-validation utilities."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


def apply_row_cap(
    data: pd.DataFrame,
    max_rows: int | None,
    group_column: str = "trip_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Apply a row cap without splitting or discarding trip groups unnecessarily.

    Whole trips are selected in a reproducible random order. A trip is included
    only when all its rows fit within the remaining capacity. Consequently,
    the returned row count may be below max_rows.
    """
    if max_rows is None or len(data) <= max_rows:
        return data.copy()

    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    if group_column not in data.columns:
        raise ValueError(f"Missing group column: {group_column}")

    if data[group_column].isna().any():
        raise ValueError(f"{group_column} contains missing values.")

    group_sizes = data.groupby(group_column, sort=False).size()
    shuffled_groups = group_sizes.sample(
        frac=1,
        random_state=random_state,
    )

    selected_groups = []
    used_rows = 0

    for trip_id, trip_rows in shuffled_groups.items():
        if used_rows + trip_rows <= max_rows:
            selected_groups.append(trip_id)
            used_rows += trip_rows

    if not selected_groups:
        raise ValueError(
            "No complete trip fits within max_rows. Increase the row cap."
        )

    return data[data[group_column].isin(selected_groups)].copy()


def evaluate_model(
    model,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    group_column: str = "trip_id",
    n_splits: int = 5,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Evaluate an estimator using GroupKFold and macro F1.

    All rows belonging to a trip stay in the same fold.
    Returns (fold_metrics, mean_macro_f1).
    """
    required = set(feature_columns) | {target_column, group_column}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if target_column in feature_columns or group_column in feature_columns:
        raise ValueError(
            "The target and trip ID must not be included as model features."
        )

    if data[list(required)].isna().any().any():
        raise ValueError("Required feature, target, or group columns contain missing values.")

    working_data = apply_row_cap(
        data,
        max_rows=max_rows,
        group_column=group_column,
        random_state=random_state,
    ).reset_index(drop=True)

    X = working_data[list(feature_columns)]
    y = working_data[target_column]
    groups = working_data[group_column]

    unique_trips = groups.nunique()
    if unique_trips < n_splits:
        raise ValueError(
            f"GroupKFold needs at least {n_splits} distinct trips; "
            f"received {unique_trips}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    results = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        train_groups = set(groups.iloc[train_idx])
        valid_groups = set(groups.iloc[valid_idx])

        if train_groups.intersection(valid_groups):
            raise RuntimeError("Trip leakage detected between train and validation.")

        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = fold_model.predict(X.iloc[valid_idx])

        score = f1_score(
            y.iloc[valid_idx],
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append({
            "fold": fold_number,
            "macro_f1": score,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx),
            "train_trips": len(train_groups),
            "validation_trips": len(valid_groups),
        })

    fold_metrics = pd.DataFrame(results)
    mean_macro_f1 = float(fold_metrics["macro_f1"].mean())

    return fold_metrics, mean_macro_f1
