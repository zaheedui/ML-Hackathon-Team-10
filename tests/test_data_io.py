import os
from src.pathsnap.data_io import load_trip

def test_load_trip():
    # Provide the path to a known training trip folder you unzipped
    trip_dir = "data/train/00000001" 
    
    if os.path.exists(trip_dir):
        track, nodes, arcs, route = load_trip(trip_dir)
        
        # Verify projections exist
        assert 'x' in track.columns and 'y' in track.columns
        assert 'start_x' in arcs.columns and 'length' in arcs.columns
        
        # Verify route loaded (since this is a train trip)
        assert route is not None