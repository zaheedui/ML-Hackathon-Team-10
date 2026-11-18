import os
from collections import defaultdict
from src.pathsnap.data_io import load_trip

def build_trip_graph(trip_dir):
    """Builds a successor dictionary for the road network arcs."""
    track, nodes, arcs, route = load_trip(trip_dir)
    
    successors = defaultdict(list)
    reverse_of = {}
    
    # Identify successors based on from_node and to_node matching
    for arc_id, row in arcs.iterrows():
        # Find arcs where their from_node == this arc's to_node
        succ_arcs = arcs[arcs['from_node_id'] == row['to_node_id']].index.tolist()
        successors[arc_id] = succ_arcs
        
        # Identify reverse arcs (if a two-way street exists)
        rev = arcs[(arcs['from_node_id'] == row['to_node_id']) & (arcs['to_node_id'] == row['from_node_id'])].index.tolist()
        if rev:
            reverse_of[arc_id] = rev[0]
            
    return successors, reverse_of

def calculate_hop_distance(start_arcs, target_arcs, successors, max_hops=3):
    """Calculates BFS hop distance up to max_hops. Returns dictionary of {(start, end): hops}."""
    hop_table = {}
    for start in start_arcs:
        visited = {start: 0}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            current_hops = visited[current]
            
            if current_hops < max_hops:
                for nxt in successors.get(current, []):
                    if nxt not in visited:
                        visited[nxt] = current_hops + 1
                        queue.append(nxt)
                        
        for target in target_arcs:
            if target in visited:
                hop_table[(start, target)] = visited[target]
                
    return hop_table