import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
from src.pathsnap.data_io import load_trip

def densify_segment(row, step_m=10):
    length = row['length']
    num_steps = max(2, int(length // step_m) + 1)
    t = np.linspace(0, 1, num_steps)
    xs = row['start_x'] + t * (row['end_x'] - row['start_x'])
    ys = row['start_y'] + t * (row['end_y'] - row['start_y'])
    return pd.DataFrame({'arc_id': row.name, 'x': xs, 'y': ys})

def point_to_segment_dist(px, py, sx, sy, ex, ey, length):
    if length == 0:
        return np.sqrt((px - sx)**2 + (py - sy)**2), 0.0
    t = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / (length**2)
    t = np.clip(t, 0, 1) 
    proj_x = sx + t * (ex - sx)
    proj_y = sy + t * (ey - sy)
    dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    return dist, t

def process_trip_candidates(trip_dir):
    trip_id = os.path.basename(trip_dir)
    track, nodes, arcs, route = load_trip(trip_dir)
    
    densified_points = pd.concat([densify_segment(row) for _, row in arcs.iterrows()], ignore_index=True)
    tree = cKDTree(densified_points[['x', 'y']].values)
    
    candidates_list = []
    for obs_idx, row in track.iterrows():
        px, py = row['x'], row['y']
        
        idx_within_radius = tree.query_ball_point([px, py], r=75)
        if len(set(densified_points.iloc[idx_within_radius]['arc_id'])) < 3:
            idx_within_radius = tree.query_ball_point([px, py], r=150)
            
        found_arcs = densified_points.iloc[idx_within_radius]['arc_id'].unique()
        
        obs_cands = []
        for arc_id in found_arcs:
            arc = arcs.iloc[arc_id]
            dist, frac = point_to_segment_dist(px, py, arc['start_x'], arc['start_y'], arc['end_x'], arc['end_y'], arc['length'])
            obs_cands.append({
                'trip_id': trip_id,
                'obs_idx': obs_idx,
                'arc_id': arc_id,
                'dist_m': dist,
                'along_frac': frac
            })
            
        obs_cands = sorted(obs_cands, key=lambda x: x['dist_m'])[:10]
        for rank, cand in enumerate(obs_cands):
            cand['cand_rank'] = rank
            
        candidates_list.extend(obs_cands)
    return pd.DataFrame(candidates_list)

def generate_all_candidates(base_dir, output_parquet):
    trip_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    with Pool() as pool:
        results = pool.map(process_trip_candidates, trip_dirs)
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_parquet(output_parquet)
    print(f"Saved {len(final_df)} candidates to {output_parquet}")

if __name__ == "__main__":
    generate_all_candidates("data/train", "data_cache/cands_train.parquet")
    generate_all_candidates("data/student_test", "data_cache/cands_student_test.parquet")