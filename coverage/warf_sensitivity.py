#!/usr/bin/env python3
"""
warf_sensitivity.py — Sensitivity analysis of the WARF threshold angle.

Sweeps theta* from 5° to 60° across pairs of COLMAP datasets and plots
per-dataset WARF curves alongside the two group band envelopes.  Clean
separation (GoPro min > Rover max) is highlighted.

Usage:
  python warf_sensitivity.py --data-root /path/to/data --out-dir ./results

Expected layout under --data-root:
  go_pro/scene_1/sparse/0
  go_pro/scene_2/sparse/0
  go_pro/scene_3/sparse/0
  mastcam/scene_1/scene1_full/sparse/0
  mastcam/scan_2/sparse/0
  mastcam/scan3/scan3_full/sparse/0

Output: <out-dir>/warf_sensitivity.pdf + .png
"""

import argparse
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

THRESHOLDS_DEG = np.arange(5, 65, 5)   # 5, 10, ..., 60
SAMPLE         = 20000
SEED           = 42
C_GOPRO        = "#1A6EB5"
C_ROVER        = "#C0392B"
THETA_REF      = 30.0


# ── COLMAP readers ─────────────────────────────────────────────────────────────

def _read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id       = struct.unpack("<i", f.read(4))[0]
            qw,qx,qy,qz = struct.unpack("<4d", f.read(32))
            tx,ty,tz     = struct.unpack("<3d", f.read(24))
            _            = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n2d * 24)
            images[img_id] = {"qvec": np.array([qw,qx,qy,qz]),
                               "tvec": np.array([tx,ty,tz])}
    return images


def _read_points3d_bin(path: Path) -> dict:
    pts = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid  = struct.unpack("<Q", f.read(8))[0]
            xyz  = np.array(struct.unpack("<3d", f.read(24)))
            f.read(3); f.read(8)
            tlen = struct.unpack("<Q", f.read(8))[0]
            track = struct.unpack(f"<{2*tlen}i", f.read(8 * tlen))
            iids = [track[2*k] for k in range(tlen)]
            pts[pid] = {"xyz": xyz, "image_ids": iids}
    return pts


def _read_images_txt(path: Path) -> dict:
    images = {}
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 9: i += 1; continue
        images[int(p[0])] = {
            "qvec": np.array(list(map(float, p[1:5]))),
            "tvec": np.array(list(map(float, p[5:8]))),
        }
        i += 2
    return images


def _read_points3d_txt(path: Path) -> dict:
    pts = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        p    = line.split()
        iids = [int(p[8 + k]) for k in range(0, len(p) - 8, 2)]
        pts[int(p[0])] = {
            "xyz":       np.array([float(p[1]), float(p[2]), float(p[3])]),
            "image_ids": iids,
        }
    return pts


def load_reconstruction(sparse_dir: Path):
    if (sparse_dir / "images.bin").exists():
        return (_read_images_bin(sparse_dir / "images.bin"),
                _read_points3d_bin(sparse_dir / "points3D.bin"))
    return (_read_images_txt(sparse_dir / "images.txt"),
            _read_points3d_txt(sparse_dir / "points3D.txt"))


# ── geometry ───────────────────────────────────────────────────────────────────

def _qvec2R(q):
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [1-2*qy**2-2*qz**2,   2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
        [2*qx*qy+2*qw*qz,     1-2*qx**2-2*qz**2, 2*qy*qz-2*qw*qx],
        [2*qx*qz-2*qw*qy,     2*qy*qz+2*qw*qx,   1-2*qx**2-2*qy**2],
    ])


def _cam_center(img):
    return -_qvec2R(img["qvec"]).T @ img["tvec"]


# ── core computation ───────────────────────────────────────────────────────────

def precompute_max_angles(images, points3d, sample=SAMPLE, seed=SEED):
    rng         = np.random.default_rng(seed)
    centers_map = {iid: _cam_center(img) for iid, img in images.items()}

    all_pts = list(points3d.values())
    n_total = len(all_pts)
    if n_total > sample:
        idx = rng.choice(n_total, sample, replace=False)
        pts = [all_pts[i] for i in idx]
    else:
        pts = all_pts

    max_angles = []
    for pt in pts:
        obs = [iid for iid in pt["image_ids"] if iid in centers_map]
        if len(obs) < 2:
            continue
        obs_c  = np.array([centers_map[iid] for iid in obs])
        dirs   = obs_c - pt["xyz"][None, :]
        norms  = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms  = np.where(norms < 1e-9, 1e-9, norms)
        dirs_u = dirs / norms
        dots   = np.clip(dirs_u @ dirs_u.T, -1.0, 1.0)
        max_angles.append(float(np.arccos(dots).max()))

    return np.array(max_angles)


def warf_at_threshold(max_angles, thresh_deg):
    thresh_rad = np.radians(thresh_deg)
    return float((max_angles > thresh_rad).mean()) if len(max_angles) > 0 else 0.0


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root of the data tree (contains go_pro/ and mastcam/)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"),
                        help="Output directory for figures (default: ./results)")
    parser.add_argument("--sample", type=int, default=SAMPLE,
                        help=f"Max points to sample per dataset (default: {SAMPLE})")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    datasets = [
        {"label": "GoPro S1",   "group": "GoPro",   "path": data_root / "go_pro/scene_1/sparse/0"},
        {"label": "GoPro S2",   "group": "GoPro",   "path": data_root / "go_pro/scene_2/sparse/0"},
        {"label": "GoPro S3",   "group": "GoPro",   "path": data_root / "go_pro/scene_3/sparse/0"},
        {"label": "Rover S1",   "group": "Rover",   "path": data_root / "mastcam/scene_1/scene1_full/sparse/0"},
        {"label": "Rover S2",   "group": "Rover",   "path": data_root / "mastcam/scan_2/sparse/0"},
        {"label": "Rover S3",   "group": "Rover",   "path": data_root / "mastcam/scan3/scan3_full/sparse/0"},
    ]

    print("Loading datasets and precomputing triangulation angles …")
    results = []
    for ds in datasets:
        print(f"  {ds['label']:14s}  {ds['path']}")
        images, points3d = load_reconstruction(ds["path"])
        max_angles = precompute_max_angles(images, points3d, sample=args.sample)
        warfs = np.array([warf_at_threshold(max_angles, t) * 100
                          for t in THRESHOLDS_DEG])
        results.append({"label": ds["label"], "group": ds["group"], "warfs": warfs})
        print(f"    {len(images):,} images  {len(points3d):,} pts  "
              f"valid angles: {len(max_angles):,}")

    # ── separation gap at each threshold ──────────────────────────────────────
    gp_warfs = np.array([r["warfs"] for r in results if r["group"] == "GoPro"])
    mc_warfs = np.array([r["warfs"] for r in results if r["group"] == "Rover"])
    gp_min, gp_max = gp_warfs.min(0), gp_warfs.max(0)
    mc_min, mc_max = mc_warfs.min(0), mc_warfs.max(0)
    gap   = gp_min - mc_max
    ratio = gp_min / np.maximum(mc_max, 1e-6)

    print("\nThreshold | GoPro range       | MastCam range    | Gap   | Ratio")
    print("-" * 72)
    for i, t in enumerate(THRESHOLDS_DEG):
        sep_flag = "✓" if gap[i] > 0 else "✗"
        print(f"  {t:4.0f}°  | {gp_min[i]:5.1f}% – {gp_max[i]:5.1f}%  "
              f"| {mc_min[i]:4.1f}% – {mc_max[i]:4.1f}%   "
              f"| {gap[i]:+5.1f}% {sep_flag}  | {ratio[i]:.1f}×")

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(wspace=0.12)

    ax = axes[0]
    ax.fill_between(THRESHOLDS_DEG, gp_min, gp_max,
                    color=C_GOPRO, alpha=0.18, label="GoPro range")
    ax.fill_between(THRESHOLDS_DEG, mc_min, mc_max,
                    color=C_ROVER, alpha=0.18, label="Rover range")

    ls_cycle = ["-", "--", ":"]
    gi = mi = 0
    for r in results:
        color = C_GOPRO if r["group"] == "GoPro" else C_ROVER
        ls    = ls_cycle[gi] if r["group"] == "GoPro" else ls_cycle[mi]
        ax.plot(THRESHOLDS_DEG, r["warfs"], color=color, lw=1.6,
                ls=ls, label=r["label"], zorder=4)
        if r["group"] == "GoPro": gi += 1
        else:                      mi += 1

    ax.axvline(THETA_REF, color="0.3", lw=1.2, ls="--", zorder=5,
               label=f"θ* = {THETA_REF:.0f}° (chosen)")
    ax.set_xlabel("Triangulation angle threshold θ* (°)", fontsize=12)
    ax.set_ylabel("WARF  (%)", fontsize=12)
    ax.set_xlim(THRESHOLDS_DEG[0], THRESHOLDS_DEG[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=10)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)

    ax2          = axes[1]
    color_gap    = "#2CA02C"
    color_ratio  = "#FF7F0E"
    ax2.bar(THRESHOLDS_DEG, np.clip(gap, 0, None),
            width=3.5, color=color_gap, alpha=0.7,
            label="Gap (GoPro min − Rover max)")
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.axvline(THETA_REF, color="0.3", lw=1.2, ls="--")

    ax2b = ax2.twinx()
    ax2b.plot(THRESHOLDS_DEG, ratio, "o-", color=color_ratio, lw=1.6,
              ms=5, label="Ratio (GoPro min / Rover max)")
    ax2b.set_ylabel("GoPro-min / MastCam-max  (×)", fontsize=10, color=color_ratio)
    ax2b.tick_params(axis="y", colors=color_ratio, labelsize=9)

    ax2.set_xlabel("θ* (°)", fontsize=12)
    ax2.set_ylabel("Separation gap  (pp)", fontsize=10, color=color_gap)
    ax2.tick_params(axis="y", colors=color_gap, labelsize=9)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.set_xlim(THRESHOLDS_DEG[0], THRESHOLDS_DEG[-1])
    ax2.grid(True, lw=0.4, alpha=0.4)

    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
               loc="upper right", framealpha=0.9)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = args.out_dir / f"warf_sensitivity.{fmt}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"\nSaved → {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
