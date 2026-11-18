import os
import pandas as pd
import numpy as np

def load_trip(trip_dir):
    """
    Loads raw trip data and applies a local metric projection.
    Returns: track, nodes, arcs, route
    """
    # 1. Load whitespace-separated files without headers
    track_path = [f for f in os.listdir(trip_dir) if f.endswith('.track')][0]
    nodes_path = [f for f in os.listdir(trip_dir) if f.endswith('.nodes')][0]
    arcs_path = [f for f in os.listdir(trip_dir) if f.endswith('.arcs')][0]
    route_file = [f for f in os.listdir(trip_dir) if f.endswith('.route')]

    track = pd.read_csv(os.path.join(trip_dir, track_path), sep='\s+', header=None, names=['lon', 'lat', 'timestamp'])
    nodes = pd.read_csv(os.path.join(trip_dir, nodes_path), sep='\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(os.path.join(trip_dir, arcs_path), sep='\s+', header=None, names=['from_node_id', 'to_node_id'])
    
    route = None
    if route_file:
        route = pd.read_csv(os.path.join(trip_dir, route_file[0]), sep='\s+', header=None, names=['arc_id'])

    # 2. Local Metric Projection
    lon0 = track['lon'].mean()
    lat0 = track['lat'].mean()
    
    def project(lon, lat):
        x = (lon - lon0) * np.cos(np.radians(lat0)) * 111320
        y = (lat - lat0) * 110574
        return x, y

    track['x'], track['y'] = project(track['lon'], track['lat'])
    nodes['x'], nodes['y'] = project(nodes['lon'], nodes['lat'])

    # 3. Arc Geometry Calculations
    # 0-based row index serves as arc_id and node_id
    start_nodes = nodes.iloc[arcs['from_node_id']].reset_index(drop=True)
    end_nodes = nodes.iloc[arcs['to_node_id']].reset_index(drop=True)

    arcs['start_x'] = start_nodes['x']
    arcs['start_y'] = start_nodes['y']
    arcs['end_x'] = end_nodes['x']
    arcs['end_y'] = end_nodes['y']

    dx = arcs['end_x'] - arcs['start_x']
    dy = arcs['end_y'] - arcs['start_y']
    
    arcs['length'] = np.sqrt(dx**2 + dy**2)
    arcs['bearing'] = np.degrees(np.arctan2(dx, dy)) % 360

    return track, nodes, arcs, route