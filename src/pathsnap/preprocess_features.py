"""
PathSnap preprocessing + feature engineering (stand-in for DATA-04 and FEAT-01).

Per trip:
  1. Load .track / .nodes / .arcs / .route
  2. Project lon/lat to a local metric frame per trip
     (x = (lon-lon0)*cos(lat0)*111320, y = (lat-lat0)*110574)
  3. Candidate road segments: cKDTree over arcs densified every ~10 m,
     75 m radius (150 m fallback if <3 found), up to K=10 unique arcs
  4. Label (train only): the point's true arc = nearest arc by
     point-to-segment distance inside the trip's ground-truth route set
  5. Features per (point, candidate): distance, snapped coordinates,
     heading difference, speed, acceleration, dt, rank, distance ratio,
     arc length

Outputs one row per (GPS point, candidate arc):
  trip_id, point_idx, t, lon, lat, cand_arc_id, dist_m, snap_lon, snap_lat,
  heading_diff, speed_mps, accel_mps2, dt_s, rank, dist_ratio, arc_len_m, label

Usage:
  python preprocess_features.py --data-dir "Student folder" --out-dir data_cache
Writes: data_cache/features_train.parquet, data_cache/features_student_test.parquet
"""
import argparse, os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

K_CAND, RADIUS, RADIUS_FALLBACK, DENSIFY_M = 10, 75.0, 150.0, 10.0

def load_trip(d, tid):
    f = lambda ext: os.path.join(d, tid, tid + ext)
    track = np.loadtxt(f(".track"))
    nodes = np.loadtxt(f(".nodes"))
    arcs  = np.loadtxt(f(".arcs"), dtype=int)
    route = np.loadtxt(f(".route"), dtype=int).ravel()
    return track, nodes, arcs, route

def project(lon, lat, lon0, lat0):
    return ((lon - lon0) * np.cos(np.radians(lat0)) * 111320.0,
            (lat - lat0) * 110574.0)

def unproject(x, y, lon0, lat0):
    return (x / (np.cos(np.radians(lat0)) * 111320.0) + lon0,
            y / 110574.0 + lat0)

def seg_snap(px, py, ax, ay, bx, by):
    """distance and snapped point of (px,py) to each segment (ax,ay)-(bx,by)"""
    dx, dy = bx - ax, by - ay
    L2 = dx*dx + dy*dy
    L2[L2 == 0] = 1e-12
    t = np.clip(((px[...,None]-ax)*dx + (py[...,None]-ay)*dy) / L2, 0, 1)
    cx, cy = ax + t*dx, ay + t*dy
    return np.hypot(px[...,None]-cx, py[...,None]-cy), cx, cy

def ang_diff(a, b):
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)

def trip_table(tid, track, nodes, arcs, route, with_labels):
    lon0, lat0 = nodes[:,0].mean(), nodes[:,1].mean()
    nx, ny = project(nodes[:,0], nodes[:,1], lon0, lat0)
    tx, ty = project(track[:,0], track[:,1], lon0, lat0)
    ax, ay, bx, by = nx[arcs[:,0]], ny[arcs[:,0]], nx[arcs[:,1]], ny[arcs[:,1]]
    arc_len = np.hypot(bx-ax, by-ay)
    arc_brg = np.arctan2(by-ay, bx-ax)

    # densified candidate tree (vectorized)
    ns = np.maximum(1, (arc_len // DENSIFY_M).astype(int))
    counts = ns + 1
    total = counts.sum()
    starts = np.zeros(len(ns)+1, dtype=int); np.cumsum(counts, out=starts[1:])
    k_idx = np.arange(total) - np.repeat(starts[:-1], counts)
    ts_ = k_idx / np.repeat(ns, counts)
    arc_of = np.repeat(np.arange(len(ns)), counts)
    dens = np.column_stack([ax[arc_of] + ts_*(bx[arc_of]-ax[arc_of]),
                            ay[arc_of] + ts_*(by[arc_of]-ay[arc_of])])
    tree = cKDTree(dens)

    # point motion features
    dt = np.diff(track[:,2]); dt[dt <= 0] = 1.0
    spd = np.hypot(np.diff(tx), np.diff(ty)) / dt
    spd = np.clip(np.concatenate([[spd[0]], spd]), 0, 60)
    acc = np.concatenate([[0.0], np.diff(spd)])
    hdg = np.arctan2(np.gradient(ty), np.gradient(tx))
    dt_full = np.concatenate([[1.0], dt])

    # labels: nearest arc within ground-truth route
    true_arc = np.full(len(tx), -1)
    if with_labels:
        vr = route[(route >= 0) & (route < len(arcs))]
        d_route, _, _ = seg_snap(tx, ty, ax[vr], ay[vr], bx[vr], by[vr])
        true_arc = vr[np.argmin(d_route, axis=1)]

    # candidates per point (batch tree query)
    dd, ii = tree.query(np.column_stack([tx, ty]), k=40,
                        distance_upper_bound=RADIUS_FALLBACK, workers=-1)
    rows = []
    for pi in range(len(tx)):
        ua, seen = [], set()
        radius_ok = dd[pi] <= RADIUS
        for j, dj in zip(ii[pi], dd[pi]):
            if j >= len(dens): continue
            a = int(arc_of[j])
            if a not in seen and (dj <= RADIUS or len(ua) < 3):
                seen.add(a); ua.append(a)
            if len(ua) == K_CAND: break
        if not ua: continue
        ua = np.array(ua)
        d_c, cx, cy = seg_snap(tx[pi:pi+1], ty[pi:pi+1], ax[ua], ay[ua], bx[ua], by[ua])
        d_c = d_c[0]; cx = cx[0]; cy = cy[0]
        d0 = max(d_c.min(), 1e-6)
        s_lon, s_lat = unproject(cx, cy, lon0, lat0)
        for rank in range(len(ua)):
            a = int(ua[rank])
            rows.append((tid, pi, track[pi,2], track[pi,0], track[pi,1], a,
                         d_c[rank], s_lon[rank], s_lat[rank],
                         ang_diff(hdg[pi], arc_brg[a]), spd[pi], acc[pi],
                         dt_full[pi], rank, d_c[rank]/d0, arc_len[a],
                         1 if a == true_arc[pi] else 0))
    return rows

COLS = ["trip_id","point_idx","t","lon","lat","cand_arc_id","dist_m",
        "snap_lon","snap_lat","heading_diff","speed_mps","accel_mps2",
        "dt_s","rank","dist_ratio","arc_len_m","label"]

def run_split(data_dir, split, out_path, with_labels):
    base = os.path.join(data_dir, split)
    trips = sorted(os.listdir(base))
    writer, n = None, 0
    for tid in trips:
        rows = trip_table(tid, *load_trip(base, tid), with_labels)
        arr = np.array([[float(c) if not isinstance(c, str) else np.nan for c in r[1:]] for r in rows])
        d = {c: arr[:, i] for i, c in enumerate(COLS[1:], 0)}
        d = {COLS[i+1]: arr[:, i] for i in range(len(COLS)-1)}
        d["trip_id"] = np.array([r[0] for r in rows])
        table = pa.table({c: d[c] for c in COLS})
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        n += len(rows)
        print(f"{split}/{tid}: {len(rows)} rows", flush=True)
    if writer: writer.close()
    print(f"{split}: {n} rows -> {out_path}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="data_cache")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_split(args.data_dir, "train",
              os.path.join(args.out_dir, "features_train.parquet"), True)
    run_split(args.data_dir, "student_test",
              os.path.join(args.out_dir, "features_student_test.parquet"), False)

if __name__ == "__main__":
    main()
