import os
from src.pathsnap.graph import build_trip_graph

def test_build_trip_graph():
    trip_dir = "data/train/00000001"
    if os.path.exists(trip_dir):
        successors, reverse_of = build_trip_graph(trip_dir)
        assert isinstance(successors, dict)
        assert isinstance(reverse_of, dict)