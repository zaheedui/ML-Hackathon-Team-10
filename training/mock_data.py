"""Generate temporary synthetic data for testing the training harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_mock_data(
    n_trips: int = 20,
    rows_per_trip: int = 30,
    random_state: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Create synthetic trip-level classification data."""
    if n_trips < 5:
        raise ValueError("At least 5 trips are required for 5-fold GroupKFold.")

    if rows_per_trip < 1:
        raise ValueError("rows_per_trip must be positive.")

    rng = np.random.default_rng(random_state)
    records = []

    for trip_number in range(n_trips):
        trip_id = f"trip_{trip_number:03d}"

        # Give each trip a consistent underlying class.
        trip_class = trip_number % 2

        for point_number in range(rows_per_trip):
            feature_1 = rng.normal(loc=trip_class * 1.5, scale=1.0)
            feature_2 = rng.normal(loc=trip_class * 0.8, scale=1.0)
            feature_3 = rng.uniform(0.0, 1.0)

            records.append({
                "trip_id": trip_id,
                "feature_1": feature_1,
                "feature_2": feature_2,
                "feature_3": feature_3,
                "label": trip_class,
            })

    data = pd.DataFrame(records)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(destination, index=False)

    return data


if __name__ == "__main__":
    mock_data = generate_mock_data(
        output_path="data/mock_training_data.csv"
    )
    print(f"Generated {len(mock_data)} rows.")
    print(f"Generated {mock_data['trip_id'].nunique()} trips.")
    print(mock_data.head())
