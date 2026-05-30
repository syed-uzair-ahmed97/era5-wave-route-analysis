"""
map1_optimized.py
=================
Optimized ERA5 wave route joint-probability analysis.

Improvements over map1.py
--------------------------
* Fixed hardcoded path (Zone -> Zone_IDE configurable via CLI)
* Chunked Dask loading  → parallel file reads
* Single .load() call   → one network/disk round-trip per variable
* Vectorized 2-D histogram over all points simultaneously
* Vectorized trapezoidal weights (no Python loop)
* argparse for all configurable parameters
* Real-time progress output

Usage
-----
    python map1_optimized.py                          # uses defaults
    python map1_optimized.py --folder D:/ERA5/wavedata/Zone_IDE \
                              --track track.txt       \
                              --years 2016 2025       \
                              --output output
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# =====================================================
# CLI ARGUMENTS
# =====================================================

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="ERA5 wave route joint-probability analysis (optimized)"
    )
    parser.add_argument(
        "--folder",
        default=os.path.join(script_dir, "data"),
        help="Folder containing Hs_YYYY.nc and Tp_YYYY.nc files",
    )
    parser.add_argument(
        "--track",
        default=os.path.join(script_dir, "data", "track.txt"),
        help="Path to track.txt (lon,lat CSV)",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        default=[2016, 2025],
        metavar=("START", "END"),
        help="Year range inclusive (default: 2016 2025)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "results"),
        help="Output folder for PNG, SVG, and JSON results",
    )
    parser.add_argument(
        "--hs-max",
        type=float,
        default=9.0,
        help="Upper bound for Hs bins in metres (default: 9.0)",
    )
    parser.add_argument(
        "--hs-step",
        type=float,
        default=1.0,
        help="Hs bin width in metres (default: 1.0)",
    )
    parser.add_argument(
        "--tp-max",
        type=float,
        default=21.0,
        help="Upper bound for Tp bins in seconds (default: 21.0)",
    )
    parser.add_argument(
        "--tp-step",
        type=float,
        default=1.0,
        help="Tp bin width in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write results.json (used by the web frontend)",
    )
    return parser.parse_args()


# =====================================================
# HELPER: STATUS MESSAGES
# =====================================================

def log(msg, level="INFO"):
    prefix = {"INFO": "[OK]", "WARN": "[WARN]", "ERR": "[ERR]", "STEP": "[STEP]"}.get(level, "*")
    print(f"  {prefix}  {msg}", flush=True)


# =====================================================
# 1. LOAD DATA  (chunked + parallel via Dask)
# =====================================================

def load_data(folder, years, chunk_size=500):
    # Auto-resolve fallback path if raw files are in data/ subfolder
    test_file = os.path.join(folder, f"Hs_{years[0]}.nc")
    if not os.path.isfile(test_file):
        alt_folder = os.path.join(folder, "data")
        if os.path.isfile(os.path.join(alt_folder, f"Hs_{years[0]}.nc")):
            folder = alt_folder

    log(f"Loading ERA5 data for years {years[0]}-{years[1]} from:\n       {folder}", "STEP")

    Hs_files = [
        os.path.join(folder, f"Hs_{y}.nc")
        for y in range(years[0], years[1] + 1)
    ]
    Tp_files = [
        os.path.join(folder, f"Tp_{y}.nc")
        for y in range(years[0], years[1] + 1)
    ]

    missing = [f for f in Hs_files + Tp_files if not os.path.isfile(f)]
    if missing:
        log(f"Missing files:\n  " + "\n  ".join(missing), "ERR")
        sys.exit(1)

    t0 = time.time()

    # Open lazily with Dask chunks for parallel reads
    ds_Hs = xr.open_mfdataset(
        Hs_files,
        combine="by_coords",
        chunks={"time": chunk_size},
        engine="netcdf4",
    )
    ds_Tp = xr.open_mfdataset(
        Tp_files,
        combine="by_coords",
        chunks={"time": chunk_size},
        engine="netcdf4",
    )

    Hs = ds_Hs["swh"]    # significant wave height
    Tp = ds_Tp["pp1d"]   # peak wave period

    log(f"Dataset opened ({len(Hs_files)} Hs + {len(Tp_files)} Tp files) in {time.time()-t0:.1f}s")
    return Hs, Tp


# =====================================================
# 2. LOAD TRACK
# =====================================================

def load_track(track_file):
    # Auto-resolve fallback paths in case an old path was passed by the frontend
    if not os.path.isfile(track_file):
        parent_dir = os.path.dirname(track_file)
        alt1 = os.path.join(parent_dir, "data", "track.txt")
        alt2 = os.path.join(parent_dir, "track.txt")
        if os.path.isfile(alt1):
            track_file = alt1
        elif os.path.isfile(alt2):
            track_file = alt2

    log(f"Loading track: {track_file}", "STEP")
    if not os.path.isfile(track_file):
        log(f"Track file not found: {track_file}. Please check the path.", "ERR")
        sys.exit(1)

    track = np.loadtxt(track_file, delimiter=",")
    track_lon = track[:, 0]
    track_lat = track[:, 1]
    log(f"Track loaded - {len(track_lon)} waypoints")
    return track_lat, track_lon


# =====================================================
# 3. SELECT NEAREST ERA5 GRID POINTS
# =====================================================

def select_grid_points(Hs, track_lat, track_lon):
    log("Selecting nearest ERA5 grid points along route", "STEP")

    era_lat = Hs.latitude.values
    era_lon = Hs.longitude.values

    lat_min, lat_max = float(era_lat.min()), float(era_lat.max())
    lon_min, lon_max = float(era_lon.min()), float(era_lon.max())

    inside = (
        (track_lat >= lat_min) & (track_lat <= lat_max) &
        (track_lon >= lon_min) & (track_lon <= lon_max)
    )
    valid_lat = track_lat[inside]
    valid_lon = track_lon[inside]

    if len(valid_lat) == 0:
        log("No track points inside the ERA5 domain — check coordinates!", "ERR")
        sys.exit(1)

    # Vectorised nearest-neighbour search (no Python loop)
    # shape: (n_valid, n_era_lat)
    lat_idx = np.abs(era_lat[None, :] - valid_lat[:, None]).argmin(axis=1)
    lon_idx = np.abs(era_lon[None, :] - valid_lon[:, None]).argmin(axis=1)

    nearest = list(dict.fromkeys(zip(era_lat[lat_idx], era_lon[lon_idx])))

    selected_lat = np.array([p[0] for p in nearest])
    selected_lon = np.array([p[1] for p in nearest])

    log(f"Track points inside domain : {len(valid_lat)}")
    log(f"Unique ERA5 grid points    : {len(nearest)}")
    for la, lo in nearest:
        print(f"        ({la:.4f}, {lo:.4f})")

    return selected_lat, selected_lon


# =====================================================
# 4. HAVERSINE DISTANCE  (vectorised)
# =====================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised Haversine — accepts scalars or arrays, returns km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))


# =====================================================
# 5. TRAPEZOIDAL ROUTE WEIGHTS  (fully vectorised)
# =====================================================

def compute_weights(selected_lat, selected_lon):
    log("Computing trapezoidal route weights", "STEP")

    n = len(selected_lat)
    seg = haversine_distance(
        selected_lat[:-1], selected_lon[:-1],
        selected_lat[1:],  selected_lon[1:],
    )

    weights = np.empty(n)
    weights[0]    = 0.5 * seg[0]
    weights[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    weights[-1]   = 0.5 * seg[-1]
    weights /= weights.sum()

    log(f"Total route distance : {seg.sum():.1f} km")
    log(f"Sum of weights       : {weights.sum():.6f}")
    return weights, seg


# =====================================================
# 6. EXTRACT DATA AT SELECTED POINTS  (one .load() call)
# =====================================================

def extract_data(Hs, Tp, selected_lat, selected_lon):
    log("Extracting Hs and Tp at selected grid points (loading into RAM)...", "STEP")
    t0 = time.time()

    Hs_sel = Hs.sel(
        latitude  = xr.DataArray(selected_lat, dims="points"),
        longitude = xr.DataArray(selected_lon, dims="points"),
    )
    Tp_sel = Tp.sel(
        latitude  = xr.DataArray(selected_lat, dims="points"),
        longitude = xr.DataArray(selected_lon, dims="points"),
    )

    # Compute both lazily-loaded arrays at once (Dask parallel read)
    import dask
    Hs_arr, Tp_arr = dask.compute(Hs_sel, Tp_sel)

    log(f"Data loaded in {time.time()-t0:.1f}s — shape: {np.array(Hs_arr).shape}")
    return np.array(Hs_arr), np.array(Tp_arr)


# =====================================================
# 7. VECTORISED 2-D JOINT PROBABILITY MATRICES
# =====================================================

def compute_joint_prob_matrices(Hs_arr, Tp_arr, Hs_bins, Tp_bins):
    """
    Hs_arr : (time, points)
    Tp_arr : (time, points)

    Returns D_all : (points, n_Hs_bins, n_Tp_bins)  — probability (0..1)
    """
    log("Computing joint probability matrices (vectorised)...", "STEP")
    t0 = time.time()

    n_time, n_pts = Hs_arr.shape
    n_hs = len(Hs_bins) - 1
    n_tp = len(Tp_bins) - 1

    # Valid means physically available data. Values outside the selected
    # Hs/Tp bins must remain in the denominator so coverage can drop below 100%.
    valid = ~np.isnan(Hs_arr) & ~np.isnan(Tp_arr)

    D_all = np.zeros((n_pts, n_hs, n_tp), dtype=np.float64)

    for p in range(n_pts):
        v = valid[:, p]
        if v.sum() == 0:
            continue
        D_all[p] = np.histogram2d(
            Hs_arr[v, p], Tp_arr[v, p],
            bins=[Hs_bins, Tp_bins]
        )[0] / v.sum()

    log(f"Matrices computed in {time.time()-t0:.1f}s")
    return D_all


# =====================================================
# 8. ROUTE-WEIGHTED COMBINED MATRIX
# =====================================================

def compute_weighted_matrix(D_all, weights):
    log("Combining matrices with route weights...", "STEP")
    # Einstein summation: w_p * D[p, i, j]  →  sum over p
    D_weighted = np.einsum("p,pij->ij", weights, D_all)
    log(f"Probability captured inside selected bins: {D_weighted.sum():.6f}")
    log(f"Probability outside selected bins        : {max(0.0, 1.0 - D_weighted.sum()):.6f}")
    return D_weighted


# =====================================================
# 9. PLOT
# =====================================================

def save_plot(output_folder, D_weighted, Hs_bins, Tp_bins):
    log("Rendering joint probability table figure...", "STEP")
    os.makedirs(output_folder, exist_ok=True)

    D_percent = D_weighted * 100.0
    hs_centers = (Hs_bins[:-1] + Hs_bins[1:]) / 2.0
    tp_centers = (Tp_bins[:-1] + Tp_bins[1:]) / 2.0
    captured_pct = float(D_percent.sum())
    outside_pct = max(0.0, 100.0 - captured_pct)

    hs_mask = np.ones_like(hs_centers, dtype=bool)
    tp_mask = np.ones_like(tp_centers, dtype=bool)

    table_data = D_percent[hs_mask, :][:, tp_mask]
    row_total  = table_data.sum(axis=1).reshape(-1, 1)
    col_total  = table_data.sum(axis=0).reshape(1, -1)
    grand_total = np.array([[table_data.sum()]])

    table_with_totals = np.block([
        [table_data, row_total],
        [col_total,  grand_total],
    ])

    row_labels = [f"{h:.1f}" for h in hs_centers[hs_mask]] + ["Total"]
    col_labels = [f"{t:.1f}" for t in tp_centers[tp_mask]] + ["Total"]

    cell_text = [
        [f"{v:.2f}%" for v in row]
        for row in table_with_totals
    ]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText  = cell_text,
        rowLabels = row_labels,
        colLabels = col_labels,
        cellLoc   = "center",
        rowLoc    = "center",
        loc       = "center",
        bbox      = [0.05, 0.08, 0.92, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(0.80, 0.90)

    cmap  = plt.cm.YlOrRd
    vmax  = np.max(table_data) if np.max(table_data) > 0 else 1
    norm  = mcolors.Normalize(vmin=0, vmax=vmax)
    n_rows, n_cols = table_with_totals.shape

    for r in range(n_rows):
        for c in range(n_cols):
            cell = table[(r + 1, c)]
            if r < n_rows - 1 and c < n_cols - 1:
                cell.set_facecolor(cmap(norm(table_data[r, c])))
            else:
                cell.set_facecolor("#E6E6E6")
                cell.set_text_props(fontweight="bold")

    for c in range(n_cols):
        table[(0, c)].set_text_props(fontweight="bold")
    for r in range(1, n_rows + 1):
        table[(r, -1)].set_text_props(fontweight="bold")

    ax.text(0.5, 0.82,
            "Route-Specific Joint Probability Distribution of Hs and Tp",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(0.5, 0.765,
            f"Coverage inside selected bins: {captured_pct:.2f}%   |   Outside selected bins: {outside_pct:.2f}%",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(0.515, 0.665, "Peak Period (Tp) [s]",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.text(0.008, 0.355, "Significant Wave Height (Hs) [m]",
            transform=ax.transAxes, rotation=90, ha="center", va="center",
            fontsize=11, fontweight="bold")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.03)

    png_path = os.path.join(output_folder, "Route_Weighted_Joint_Probability.png")
    svg_path = os.path.join(output_folder, "Route_Weighted_Joint_Probability.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    log(f"PNG saved -> {png_path}")
    log(f"SVG saved -> {svg_path}")
    return png_path, svg_path


# =====================================================
# 11. SAVE JSON (for frontend)
# =====================================================

def save_json(
    output_folder,
    D_weighted, D_all,
    selected_lat, selected_lon, weights,
    Hs_bins, Tp_bins,
    segment_km,
):
    log("Writing results.json for web frontend...", "STEP")
    os.makedirs(output_folder, exist_ok=True)

    hs_centers = ((Hs_bins[:-1] + Hs_bins[1:]) / 2.0).tolist()
    tp_centers = ((Tp_bins[:-1] + Tp_bins[1:]) / 2.0).tolist()
    captured_pct = float(D_weighted.sum() * 100.0)
    outside_pct = max(0.0, 100.0 - captured_pct)

    per_point = []
    for p in range(len(selected_lat)):
        per_point.append({
            "lat"   : float(selected_lat[p]),
            "lon"   : float(selected_lon[p]),
            "weight": float(weights[p]),
            "matrix": (D_all[p] * 100).tolist(),
        })

    payload = {
        "hs_centers"    : hs_centers,
        "tp_centers"    : tp_centers,
        "D_weighted_pct": (D_weighted * 100).tolist(),
        "captured_pct"   : captured_pct,
        "outside_pct"    : outside_pct,
        "per_point"     : per_point,
        "total_km"      : float(segment_km.sum()),
    }

    json_path = os.path.join(output_folder, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f)

    log(f"JSON saved -> {json_path}")
    return json_path


# =====================================================
# MAIN
# =====================================================

def main():
    print()
    print("=" * 60)
    print("  ERA5 Wave Route Joint-Probability Analysis (Optimized)")
    print("=" * 60)

    args = parse_args()

    # Include the requested upper limit so, for example, --hs-max 7 covers 0 <= Hs <= 7.
    Hs_bins = np.arange(0.0, args.hs_max + args.hs_step * 0.5, args.hs_step)
    Tp_bins = np.arange(0.0, args.tp_max + args.tp_step * 0.5, args.tp_step)

    t_total = time.time()

    # --- Load ---
    Hs, Tp = load_data(args.folder, args.years)

    # --- Track ---
    track_lat, track_lon = load_track(args.track)

    # --- Grid points ---
    selected_lat, selected_lon = select_grid_points(Hs, track_lat, track_lon)

    # --- Weights ---
    weights, seg_km = compute_weights(selected_lat, selected_lon)

    # --- Extract ---
    Hs_arr, Tp_arr = extract_data(Hs, Tp, selected_lat, selected_lon)
    # extract_data always returns plain numpy (time, points) or (points, time)
    # Ensure shape is (time, points)
    if Hs_arr.shape[0] == len(selected_lat):
        Hs_arr = Hs_arr.T
        Tp_arr = Tp_arr.T

    # --- Joint probability ---
    D_all      = compute_joint_prob_matrices(Hs_arr, Tp_arr, Hs_bins, Tp_bins)
    D_weighted = compute_weighted_matrix(D_all, weights)

    # --- Save outputs ---
    save_plot(args.output, D_weighted, Hs_bins, Tp_bins)

    if args.json:
        save_json(args.output, D_weighted, D_all,
                  selected_lat, selected_lon, weights,
                  Hs_bins, Tp_bins, seg_km)

    print()
    print(f"  [OK]  All done in {time.time()-t_total:.1f}s")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
