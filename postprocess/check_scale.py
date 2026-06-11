#!/usr/bin/env python3
"""
check_scale.py
==============
Measure the metric scale accuracy of a raw COLMAP reconstruction using
ArUco markers of known physical size.

For each detected marker the script:
  1. Triangulates the 4 corner points from multi-view observations.
  2. Filters markers whose triangulated side lengths are inconsistent
     (fraction > --marker-tol from the post-filter median).
  3. Derives mpu from the filtered markers' median side length.
  4. Reports per-marker and overall scale accuracy.

This gives two independent quality signals:
  • Per-marker squareness error  — reprojection / triangulation quality
  • Cross-marker scale spread    — scale uniformity across the scene
    (NOTE: only meaningful when ≥2 markers survive filtering)

Usage:
    python check_scale.py \\
        --colmap  data/go_pro/scene_1/sparse/0 \\
        --images  data/go_pro/scene_1/images \\
        --label   "GoPro scene_1" \\
        [--marker-size 0.10] \\
        [--aruco-dict DICT_4X4_250] \\
        [--min-views 2]

Notes:
  --min-views 2  enables triangulation from just 2 views (no overdetermination).
                 Increase to 3 for higher confidence at the cost of fewer markers.
  --sample-every controls how many images are probed for auto dict-selection.
                 Defaults to max(1, n_images // 100) so sparse datasets are
                 fully probed rather than seeing only 1–2 images.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse all helpers from align.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from align import (
    read_cameras_binary, read_images_binary,
    pick_best_aruco_dict, detect_in_dataset,
    triangulate_marker_set, filter_markers_by_size,
)


def marker_sides(corners):
    """Return 4 side lengths (COLMAP units) for a (4,3) corner array."""
    return [float(np.linalg.norm(corners[(i+1) % 4] - corners[i])) for i in range(4)]


def marker_diags(corners):
    """Return 2 diagonal lengths (COLMAP units) for a (4,3) corner array."""
    return [float(np.linalg.norm(corners[2] - corners[0])),
            float(np.linalg.norm(corners[3] - corners[1]))]


def median_side_from_markers(markers_3d):
    """Compute median side length across all sides of all markers."""
    sides = []
    for c in markers_3d.values():
        sides.extend(marker_sides(c))
    return float(np.median(sides)) if sides else None


def analyse_markers(markers_3d, mpu, marker_size, label):
    """
    Print scale accuracy report for a set of triangulated markers.

    markers_3d : dict { marker_id → (4, 3) float64 corner array }
    mpu        : metres-per-COLMAP-unit (derived from filtered median side)
    marker_size: physical side length in metres (e.g. 0.10)
    """
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    n = len(markers_3d)
    if n == 1:
        print(f"  NOTE: only 1 marker — cross-marker spread is not meaningful.")

    all_sides_m = []
    all_diags_m = []
    rows = []

    for mid in sorted(markers_3d):
        c = markers_3d[mid]
        sides_m = [s * mpu for s in marker_sides(c)]
        diags_m = [d * mpu for d in marker_diags(c)]
        mean_s  = float(np.mean(sides_m))
        std_s   = float(np.std(sides_m))
        pct_err = (mean_s - marker_size) / marker_size * 100
        sq_err  = std_s / mean_s * 100
        rows.append((mid, sides_m, diags_m, mean_s, std_s, pct_err, sq_err))
        all_sides_m.extend(sides_m)
        all_diags_m.extend(diags_m)

    # Table
    print(f"\n  {'ID':>4}  {'mean side (m)':>14}  {'scale err':>10}  "
          f"{'squareness':>10}  sides [m]")
    print(f"  {'-'*4}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*35}")
    for (mid, sides_m, diags_m, mean_s, std_s, pct_err, sq_err) in rows:
        sides_str = "  ".join(f"{s:.4f}" for s in sides_m)
        print(f"  {mid:>4}  {mean_s:>14.4f}  {pct_err:>+9.2f}%  "
              f"{sq_err:>9.2f}%    {sides_str}")

    # Summary
    all_sides_m = np.array(all_sides_m)
    all_diags_m = np.array(all_diags_m)
    diag_target = marker_size * np.sqrt(2)
    diag_err    = (all_diags_m.mean() - diag_target) / diag_target * 100

    print(f"\n  Marker size target  : {marker_size:.4f} m")
    print(f"  mpu                 : {mpu:.6f} m/unit  "
          f"(derived from {n} filtered marker{'s' if n>1 else ''})")
    print(f"  Side length (mean)  : {all_sides_m.mean():.4f} m")
    print(f"  Scale error (mean)  : {(all_sides_m.mean()-marker_size)/marker_size*100:+.2f}%")
    spread_label = ("cross-marker + squareness" if n > 1
                    else "squareness only — single marker, no cross-marker info")
    print(f"  Scale spread (±std) : {all_sides_m.std()/marker_size*100:.2f}%  ({spread_label})")
    print(f"  Diagonal (mean)     : {all_diags_m.mean():.4f} m  "
          f"(target √2×side = {diag_target:.4f} m,  err={diag_err:+.2f}%)")

    return {
        "label":            label,
        "n_markers":        n,
        "mpu":              float(mpu),
        "marker_size_m":    float(marker_size),
        "side_mean_m":      float(all_sides_m.mean()),
        "side_std_m":       float(all_sides_m.std()),
        "scale_error_pct":  float((all_sides_m.mean() - marker_size) / marker_size * 100),
        "scale_spread_pct": float(all_sides_m.std() / marker_size * 100),
        "diag_error_pct":   float(diag_err),
        "single_marker":    n == 1,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap",       required=True,
                    help="COLMAP sparse/0 dir (cameras.bin + images.bin)")
    ap.add_argument("--images",       required=True,
                    help="Corresponding images directory")
    ap.add_argument("--label",        default=None,
                    help="Human-readable label for this dataset")
    ap.add_argument("--marker-size",  type=float, default=0.10,
                    help="Physical marker side length in metres (default 0.10)")
    ap.add_argument("--aruco-dict",   default=None,
                    help="Force ArUco dict (e.g. DICT_4X4_250); default: auto-detect")
    ap.add_argument("--min-views",    type=int, default=2,
                    help="Min views to triangulate a corner (default 2; use 3 for "
                         "overdetermined triangulation)")
    ap.add_argument("--max-reproj",   type=float, default=2.0,
                    help="Max reprojection error in px (default 2.0)")
    ap.add_argument("--marker-tol",   type=float, default=0.35,
                    help="Max fractional side-length deviation for filtering (default 0.35)")
    ap.add_argument("--sample-every", type=int, default=None,
                    help="Stride for auto dict-detection probe "
                         "(default: max(1, n_images // 100))")
    ap.add_argument("--out",          default=None,
                    help="Optional JSON output path for the summary")
    args = ap.parse_args()

    label = args.label or Path(args.colmap).parent.parent.name

    # ── Load COLMAP ────────────────────────────────────────────────────────────
    print(f"Loading COLMAP from {args.colmap} …")
    cams = read_cameras_binary(f"{args.colmap}/cameras.bin")
    imgs = read_images_binary(f"{args.colmap}/images.bin")
    n_imgs = len(imgs)
    print(f"  {len(cams)} camera(s), {n_imgs} images")

    # Adaptive sampling stride: probe ~100 images regardless of dataset size
    sample_every = args.sample_every or max(1, n_imgs // 100)

    # ── Detect ArUco markers ───────────────────────────────────────────────────
    print(f"\nDetecting ArUco markers (probing every {sample_every}th image) …")
    dname, adict, _ = pick_best_aruco_dict(
        args.images, imgs, cams,
        force_dict=args.aruco_dict,
        sample_every=sample_every)

    print(f"\nFull detection with {dname} …")
    obs = detect_in_dataset(args.images, imgs, cams, adict,
                            max_reproj=args.max_reproj)

    # ── Triangulate ────────────────────────────────────────────────────────────
    print(f"\nTriangulating (min {args.min_views} views, max_reproj {args.max_reproj}px) …")
    markers_raw = triangulate_marker_set(obs, args.min_views, args.max_reproj)
    print(f"  Triangulated: {sorted(markers_raw.keys())}")

    if not markers_raw:
        print("ERROR: no markers triangulated.")
        sys.exit(1)

    # ── Filter by size consistency ─────────────────────────────────────────────
    # Use raw median as the filter reference (standard approach), then recompute
    # mpu from the *filtered* set so that outlier markers don't bias the scale.
    raw_median = median_side_from_markers(markers_raw)
    print(f"\nFiltering markers (raw median side = {raw_median:.6f} COLMAP units, "
          f"tol={args.marker_tol}) …")
    markers = filter_markers_by_size(markers_raw, raw_median, tol=args.marker_tol)
    print(f"  Kept {len(markers)}/{len(markers_raw)} markers: {sorted(markers.keys())}")

    if not markers:
        print("ERROR: no markers survived filtering.")
        sys.exit(1)

    # Recompute mpu from the filtered set (fix: avoid outlier bias from raw set)
    filtered_median = median_side_from_markers(markers)
    mpu = args.marker_size / filtered_median
    print(f"  Filtered median side: {filtered_median:.6f} COLMAP units")
    print(f"  → mpu = {args.marker_size:.4f} / {filtered_median:.6f} = {mpu:.6f} m/unit")

    if len(markers) == 1:
        print("  NOTE: only 1 marker survived — mpu is from 4 sides. "
              "Consider lowering --min-views or --max-reproj if more markers exist.")

    # ── Analyse ────────────────────────────────────────────────────────────────
    result = analyse_markers(markers, mpu, args.marker_size, label)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSummary → {args.out}")


if __name__ == "__main__":
    main()
