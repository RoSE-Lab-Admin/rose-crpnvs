#!/usr/bin/env python3
"""
compute_cs.py — Coverage Score from a COLMAP reconstruction.

Usage:
  python compute_cs.py <colmap_path>
  python compute_cs.py <colmap_path> --sample 10000

<colmap_path> can be any of:
  - sparse/0/          directory containing images.bin / images.txt
  - dataset_root/      directory containing sparse/0/ and/or database.db
  - path/to/database.db

Examples:
  python compute_cs.py data/go_pro/scene_1/sparse/0
  python compute_cs.py data/mastcam/scan_2
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# ── reference values: best observed across GoPro Scene1/Scene2/Scene3 ─────────
WARF_REF           = 0.143   # GoPro Scene1 (14.3%)
WAPD_REF           = 13.70   # GoPro Scene2
WARF_THRESHOLD_DEG = 30.0


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
            iids  = [track[2*k] for k in range(tlen)]
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


# ── path resolver ──────────────────────────────────────────────────────────────

def resolve_sparse_dir(input_path: Path) -> Path:
    p = input_path.resolve()
    if (p / "images.bin").exists() or (p / "images.txt").exists():
        return p
    for candidate in [p / "sparse" / "0", p / "distorted" / "sparse" / "0"]:
        if candidate.exists() and (
            (candidate / "images.bin").exists() or
            (candidate / "images.txt").exists()
        ):
            return candidate
    if p.suffix == ".db":
        for candidate in [p.parent / "sparse" / "0",
                          p.parent.parent / "sparse" / "0"]:
            if candidate.exists() and (
                (candidate / "images.bin").exists() or
                (candidate / "images.txt").exists()
            ):
                return candidate
    raise FileNotFoundError(
        f"Could not find a COLMAP sparse/0 directory from: {input_path}\n"
        f"Expected images.bin or images.txt inside sparse/0/."
    )


def load_reconstruction(sparse_dir: Path):
    if (sparse_dir / "images.bin").exists():
        return (_read_images_bin(sparse_dir / "images.bin"),
                _read_points3d_bin(sparse_dir / "points3D.bin"))
    return (_read_images_txt(sparse_dir / "images.txt"),
            _read_points3d_txt(sparse_dir / "points3D.txt"))


# ── geometry ───────────────────────────────────────────────────────────────────

def _qvec2R(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [1-2*qy**2-2*qz**2,   2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
        [2*qx*qy+2*qw*qz,     1-2*qx**2-2*qz**2, 2*qy*qz-2*qw*qx],
        [2*qx*qz-2*qw*qy,     2*qy*qz+2*qw*qx,   1-2*qx**2-2*qy**2],
    ])


def _cam_center(img: dict) -> np.ndarray:
    return -_qvec2R(img["qvec"]).T @ img["tvec"]


# ── metrics ────────────────────────────────────────────────────────────────────

def cam_diagonal(images: dict) -> float:
    centers = np.array([_cam_center(img) for img in images.values()])
    return float(np.linalg.norm(centers.max(0) - centers.min(0)))


def compute_warf(images: dict, points3d: dict,
                 sample: int, seed: int = 42) -> tuple[float, float]:
    """Returns (WARF, n_wide_total_est)."""
    rng         = np.random.default_rng(seed)
    centers_map = {iid: _cam_center(img) for iid, img in images.items()}

    all_pts = list(points3d.values())
    n_total = len(all_pts)
    if n_total > sample:
        idx = rng.choice(n_total, sample, replace=False)
        pts = [all_pts[i] for i in idx]
    else:
        pts = all_pts

    thresh_rad = np.radians(WARF_THRESHOLD_DEG)
    n_wide = n_valid = 0

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
        if np.arccos(dots).max() > thresh_rad:
            n_wide += 1
        n_valid += 1

    warf          = n_wide / n_valid if n_valid > 0 else 0.0
    scale         = n_total / len(pts) if len(pts) > 0 else 1.0
    n_wide_total  = n_wide * scale
    return warf, n_wide_total


def compute_wapd(n_wide_total: float, cam_diag: float) -> float:
    if cam_diag < 1e-6:
        return 0.0
    return float(n_wide_total / cam_diag ** 3)


def coverage_score(warf: float, wapd: float) -> float:
    w = min(warf / WARF_REF, 1.0)
    d = min(wapd / WAPD_REF, 1.0)
    return float((w * d) ** 0.5)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("colmap_path", type=Path,
                        help="sparse/0 dir, dataset root, or database.db")
    parser.add_argument("--sample", type=int, default=20000,
                        help="Max 3D points to sample for WARF (default: 20000)")
    args = parser.parse_args()

    try:
        sparse_dir = resolve_sparse_dir(args.colmap_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading reconstruction from: {sparse_dir}")
    images, points3d = load_reconstruction(sparse_dir)
    print(f"  {len(images):,} registered images   {len(points3d):,} 3D points")

    print(f"\nComputing coverage metrics (WARF sample = {args.sample:,}) …")
    diag             = cam_diagonal(images)
    warf, n_wide_est = compute_warf(images, points3d, sample=args.sample)
    wapd             = compute_wapd(n_wide_est, diag)
    cs               = coverage_score(warf, wapd)

    W = 58
    print(f"\n{'═'*W}")
    print(f"  Coverage Score")
    print(f"{'═'*W}")
    print(f"  Path          : {sparse_dir}")
    print(f"  Camera cluster: {diag:.2f} units diagonal   {len(images)} images")
    print(f"{'─'*W}")
    print(f"  WARF (>{WARF_THRESHOLD_DEG:.0f}°)   : {warf*100:.1f}%"
          f"   (ref {WARF_REF*100:.1f}% → {min(warf/WARF_REF,1)*100:.0f}% of ref)")
    print(f"  WAPD          : {wapd:.2f}"
          f"   (ref {WAPD_REF:.2f} → {min(wapd/WAPD_REF,1)*100:.0f}% of ref)")
    print(f"{'─'*W}")
    print(f"  CS            : {cs:.3f}")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
