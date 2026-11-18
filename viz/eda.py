
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium


# Project paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "results" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_numeric_file(path, expected_columns):
    """Read whitespace-separated numeric data."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        on_bad_lines="skip"
    )

    if df.shape[1] < expected_columns:
        raise ValueError(
            f"{path.name}: expected at least {expected_columns} columns, "
            f"found {df.shape[1]}"
        )

    return df.iloc[:, :expected_columns].apply(
        pd.to_numeric, errors="coerce"
    )


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance in metres between GPS coordinates."""
    earth_radius = 6371000

    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


def analyse_trip(trip_dir):
    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    track = read_numeric_file(track_file, 3)
    track.columns = ["longitude", "latitude", "obs_idx"]

    nodes = read_numeric_file(nodes_file, 2)
    nodes.columns = ["longitude", "latitude"]

    arcs = read_numeric_file(arcs_file, 2)
    arcs.columns = ["arc_col_1", "arc_col_2"]

    route = pd.read_csv(
        route_file, header=None, sep=r"\s+",
        names=["route_id"]
    )
    route["route_id"] = pd.to_numeric(
        route["route_id"], errors="coerce"
    )

    # Remove invalid GPS coordinates
    valid_track = track.dropna(
        subset=["longitude", "latitude", "obs_idx"]
    ).copy()

    valid_track = valid_track[
        valid_track["longitude"].between(-180, 180)
        & valid_track["latitude"].between(-90, 90)
    ]

    # Calculate distance travelled between consecutive observations
    if len(valid_track) > 1:
        distances = haversine_distance(
            valid_track["longitude"].iloc[:-1].to_numpy(),
            valid_track["latitude"].iloc[:-1].to_numpy(),
            valid_track["longitude"].iloc[1:].to_numpy(),
            valid_track["latitude"].iloc[1:].to_numpy()
        )
        total_distance_m = float(np.nansum(distances))
    else:
        total_distance_m = 0.0

    obs_gaps = valid_track["obs_idx"].diff().dropna()

    stats = {
        "trip_id": trip_id,
        "track_rows": len(track),
        "valid_gps_points": len(valid_track),
        "invalid_gps_rows": len(track) - len(valid_track),
        "missing_track_values": int(track.isna().sum().sum()),
        "node_rows": len(nodes),
        "missing_node_values": int(nodes.isna().sum().sum()),
        "arc_rows": len(arcs),
        "missing_arc_values": int(arcs.isna().sum().sum()),
        "route_rows": len(route),
        "missing_route_values": int(route.isna().sum().sum()),
        "total_gps_distance_m": round(total_distance_m, 2),
        "min_longitude": valid_track["longitude"].min(),
        "max_longitude": valid_track["longitude"].max(),
        "min_latitude": valid_track["latitude"].min(),
        "max_latitude": valid_track["latitude"].max(),
        "mean_observation_gap": (
            float(obs_gaps.mean()) if len(obs_gaps) else None
        ),
        "max_observation_gap": (
            float(obs_gaps.max()) if len(obs_gaps) else None
        )
    }

    return stats, valid_track, nodes, arcs, route


def create_map(track, nodes, trip_id):
    """Create an interactive GPS and road-node map."""

    center_lat = track["latitude"].mean()
    center_lon = track["longitude"].mean()

    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    # Approximate 300 metre bounding box around the GPS trip
    lat_margin = 300 / 111320
    lon_margin = 300 / (
        111320 * max(np.cos(np.radians(center_lat)), 0.01)
    )

    min_lat = track["latitude"].min() - lat_margin
    max_lat = track["latitude"].max() + lat_margin
    min_lon = track["longitude"].min() - lon_margin
    max_lon = track["longitude"].max() + lon_margin

    nearby_nodes = nodes.dropna().copy()

    nearby_nodes = nearby_nodes[
        nearby_nodes["latitude"].between(min_lat, max_lat)
        & nearby_nodes["longitude"].between(min_lon, max_lon)
    ]

    # Draw nearby road network nodes in grey
    node_group = folium.FeatureGroup(
        name="Road network nodes (grey)"
    )

    # Limit markers to keep the map responsive
    if len(nearby_nodes) > 10000:
        nearby_nodes = nearby_nodes.sample(
            10000, random_state=42
        )

    for _, node in nearby_nodes.iterrows():
        folium.CircleMarker(
            location=[node["latitude"], node["longitude"]],
            radius=1,
            color="gray",
            fill=True,
            fill_opacity=0.5,
            weight=1
        ).add_to(node_group)

    node_group.add_to(map_obj)

    # Draw GPS observations in blue
    gps_coordinates = list(zip(
        track["latitude"], track["longitude"]
    ))

    folium.PolyLine(
        gps_coordinates,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="GPS observations"
    ).add_to(map_obj)

    for _, point in track.iterrows():
        folium.CircleMarker(
            location=[point["latitude"], point["longitude"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Observation {int(point['obs_idx'])}"
        ).add_to(map_obj)

    folium.LayerControl().add_to(map_obj)

    map_path = OUTPUT_DIR / f"trip_{trip_id}_map.html"
    map_obj.save(str(map_path))

    return map_path, len(nearby_nodes)


def main():
    trip_dirs = sorted(
        [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    )

    if not trip_dirs:
        raise RuntimeError(f"No trip folders found in {TRAIN_DIR}")

    all_stats = []
    first_trip_data = None

    print(f"Found {len(trip_dirs)} training trips.")
    print("Starting EDA...\n")

    for index, trip_dir in enumerate(trip_dirs, start=1):
        try:
            stats, track, nodes, arcs, route = analyse_trip(trip_dir)
            all_stats.append(stats)

            if trip_dir.name == "00000001":
                first_trip_data = (track, nodes)

            print(
                f"[{index}/{len(trip_dirs)}] "
                f"Trip {trip_dir.name}: "
                f"{stats['valid_gps_points']} valid GPS points"
            )

        except Exception as error:
            print(f"ERROR in trip {trip_dir.name}: {error}")

    if not all_stats:
        raise RuntimeError("No trips could be analysed.")

    results = pd.DataFrame(all_stats)

    # Save per-trip EDA results
    results.to_csv(OUTPUT_DIR / "trip_statistics.csv", index=False)

    # Overall summary
    summary = {
        "trips_analysed": len(results),
        "total_gps_points": int(results["valid_gps_points"].sum()),
        "total_invalid_gps_rows": int(results["invalid_gps_rows"].sum()),
        "average_gps_points_per_trip": float(
            results["valid_gps_points"].mean()
        ),
        "average_trip_distance_m": float(
            results["total_gps_distance_m"].mean()
        ),
        "total_missing_track_values": int(
            results["missing_track_values"].sum()
        )
    }

    with open(OUTPUT_DIR / "summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    # Plot GPS points per trip
    plt.figure(figsize=(12, 5))
    plt.hist(results["valid_gps_points"], bins=15)
    plt.title("Distribution of GPS Points per Trip")
    plt.xlabel("Number of GPS points")
    plt.ylabel("Number of trips")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gps_points_distribution.png")
    plt.close()

    # Plot trip distances
    plt.figure(figsize=(12, 5))
    plt.hist(results["total_gps_distance_m"], bins=15)
    plt.title("Distribution of GPS Trip Distances")
    plt.xlabel("Distance (metres)")
    plt.ylabel("Number of trips")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trip_distance_distribution.png")
    plt.close()

    # Create map for first trip
    if first_trip_data is not None:
        track, nodes = first_trip_data
        map_path, node_count = create_map(
            track, nodes, "00000001"
        )
        print(f"\nMap saved: {map_path}")
        print(f"Nearby road nodes displayed: {node_count}")

    print("\nEDA COMPLETE")
    print(json.dumps(summary, indent=4))
    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
