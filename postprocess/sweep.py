#!/usr/bin/env python3
"""
sweep.py
========
Render a set of 3DGS splats at GoPro camera positions with a y-offset sweep,
then compute DT (digital-twin) and physical metrics.

Pipeline:
  1. Load GoPro camera poses from cameras_raw.json (in COLMAP units).
  2. Convert to metres: pos_m = pos_colmap * mpu.
  3. For each y-offset y ∈ Y_OFFSETS:
       a. Render GoPro **control** splat at shifted positions → gt_gopro_renders/
          (done once; skipped if renders already exist)
       b. For each mastcam method:
            - Render method splat at shifted positions → y{offset}/{method}/
  4. Compute metrics:
       - DT:       method render vs GoPro control render (same y-offset)
       - Physical: method render vs actual GoPro photo

All PLYs must already be in the **same metric frame** (metres, 1 unit = 1 m).
Use align.py + the mpu-baking logic to prepare them (see README).

Usage:
    python sweep.py --config configs/scene_1.json [--skip-render] [--skip-metrics]

Config keys (all paths relative to config file location or absolute):
  scene_name        : str
  gopro_images      : str   path to GoPro photo directory (physical GT)
  cameras_raw       : str   output of gen_cameras.py (COLMAP-unit poses)
  mpu               : float metres-per-unit for GoPro COLMAP frame
  control_ply       : str   GoPro control 3DGS PLY (metres)
  methods           : {name: ply_path}   mastcam method PLYs (metres)
  code_dir          : str   gaussian_splatting/ source dir for rendering
  python_bin        : str   python executable with diff_gaussian_rasterization
  render_width      : int   (default 960)
  render_height     : int   (default 540)
  y_offsets         : [float]  (default [-0.3,-0.2,-0.1,0,0.1,0.2,0.3])
  depth             : float depth pullback in metres (default 0.35)
  test_hold         : int   llffhold for test-set split (default 8, 0=all)
  out_dir           : str   root output directory
"""

import argparse
import json
import math
import subprocess
import sys
from math import exp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Metric helpers ─────────────────────────────────────────────────────────────

def _gauss1d(ws, sigma):
    g = torch.Tensor([exp(-(x - ws // 2) ** 2 / (2 * sigma ** 2)) for x in range(ws)])
    return g / g.sum()


def _window(ws, c):
    w = _gauss1d(ws, 1.5).unsqueeze(1)
    w2 = w.mm(w.t()).float().unsqueeze(0).unsqueeze(0)
    return w2.expand(c, 1, ws, ws).contiguous().cuda()


def ssim_fn(a, b, ws=11):
    c = a.size(1)
    w = _window(ws, c).type_as(a)
    p = ws // 2
    m1 = F.conv2d(a, w, padding=p, groups=c)
    m2 = F.conv2d(b, w, padding=p, groups=c)
    m1s, m2s, m12 = m1 ** 2, m2 ** 2, m1 * m2
    s1  = F.conv2d(a * a, w, padding=p, groups=c) - m1s
    s2  = F.conv2d(b * b, w, padding=p, groups=c) - m2s
    s12 = F.conv2d(a * b, w, padding=p, groups=c) - m12
    C1, C2 = 1e-4, 9e-4
    return (((2 * m12 + C1) * (2 * s12 + C2)) /
            ((m1s + m2s + C1) * (s1 + s2 + C2))).mean().item()


def psnr_fn(a, b):
    mse = ((a - b) ** 2).reshape(a.shape[0], -1).mean(1, keepdim=True)
    return (20 * torch.log10(1 / torch.sqrt(mse))).mean().item()


def load_img(p, size=None):
    img = Image.open(p).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).cuda()


# ── Camera JSON builder ────────────────────────────────────────────────────────

def build_camera_json(cameras_raw, mpu, y_offset, depth, out_path):
    """
    Convert cameras_raw.json (COLMAP units) to a render-ready JSON.

    Transform applied per camera:
      pos_m = pos_colmap * mpu          # COLMAP units → metres
      pos_m[1] += y_offset              # vertical offset
      pos_m -= depth * R_c2w[:, 2]     # pull back along optical axis
    """
    with open(cameras_raw) as f:
        raw = json.load(f)

    intr = raw["intrinsics"]
    cameras = []
    for cam in raw["cameras"]:
        pos   = np.array(cam["position"]) * mpu
        R_c2w = np.array(cam["R_c2w"])
        pos[1] += y_offset
        pos    -= depth * R_c2w[:, 2]
        entry = dict(
            id=cam["id"], name=cam["name"],
            position=pos.tolist(), R_c2w=R_c2w.tolist(),
            fx=cam.get("fx", intr["fx"]),
            fy=cam.get("fy", intr["fy"]),
            cx=cam.get("cx", intr["cx"]),
            cy=cam.get("cy", intr["cy"]),
            width=cam.get("width",  intr["width"]),
            height=cam.get("height", intr["height"]),
        )
        cameras.append(entry)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"intrinsics": intr, "cameras": cameras}, f)
    return cameras


# ── Render via subprocess ──────────────────────────────────────────────────────

def render(python_bin, ply, cam_json, out_dir, code_dir, kernel_size=None):
    cmd = [python_bin,
           str(Path(__file__).parent / "render_ply.py"),
           "--ply",      str(ply),
           "--cameras",  str(cam_json),
           "--out-dir",  str(out_dir),
           "--code-dir", str(code_dir)]
    if kernel_size is not None:
        cmd += ["--kernel-size", str(kernel_size)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    n = len(list(Path(out_dir).glob("*.png")))
    if r.returncode != 0:
        print(f"    [render] rc={r.returncode}")
        print(r.stderr[-600:])
    return n, r.returncode


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(render_dir, gt_paths_by_idx, method, y_tag):
    """
    gt_paths_by_idx: dict {int index → Path}
    Returns dict with ssim/psnr/lpips mean±std or None.
    """
    import lpips as lpips_lib  # lazy import; requires lpips installed
    if not hasattr(compute_metrics, "_lpips_fn"):
        compute_metrics._lpips_fn = lpips_lib.LPIPS(net="alex").eval().cuda()
    loss_fn = compute_metrics._lpips_fn

    ssims, psnrs, lpipss = [], [], []
    with torch.no_grad():
        for idx, gp in sorted(gt_paths_by_idx.items()):
            rp = Path(render_dir) / f"{idx:06d}.png"
            if not rp.exists() or not gp.exists():
                continue
            r = load_img(rp)
            g = load_img(gp)
            if r.shape != g.shape:
                g = F.interpolate(g, size=r.shape[2:], mode="bilinear",
                                  align_corners=False)
            ssims.append(ssim_fn(r, g))
            psnrs.append(psnr_fn(r, g))
            lpipss.append(loss_fn(r * 2 - 1, g * 2 - 1).item())

    if not ssims:
        return None
    return {
        "ssim_mean":  float(np.mean(ssims)),   "ssim_std":  float(np.std(ssims)),
        "psnr_mean":  float(np.mean(psnrs)),   "psnr_std":  float(np.std(psnrs)),
        "lpips_mean": float(np.mean(lpipss)),  "lpips_std": float(np.std(lpipss)),
        "n": len(ssims),
    }


def pool_metrics(per_offset):
    """Pool individual SSIM/PSNR/LPIPS values across all offsets."""
    ssims, psnrs, lpipss = [], [], []
    for m in per_offset.values():
        if m is None:
            continue
        n = m["n"]
        # We only have mean±std, so approximate by treating each offset's n
        # images as contributing equally (pooling means here;
        # for proper pooling per-image values are accumulated in sweep loop)
        ssims.extend([m["ssim_mean"]] * n)
        psnrs.extend([m["psnr_mean"]] * n)
        lpipss.extend([m["lpips_mean"]] * n)
    if not ssims:
        return None
    return {
        "ssim_mean":  float(np.mean(ssims)),  "ssim_std":  float(np.std(ssims)),
        "psnr_mean":  float(np.mean(psnrs)),  "psnr_std":  float(np.std(psnrs)),
        "lpips_mean": float(np.mean(lpipss)), "lpips_std": float(np.std(lpipss)),
        "n": len(ssims),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Scene config JSON (see README)")
    ap.add_argument("--skip-render",  action="store_true",
                    help="Skip rendering (assumes renders already exist)")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="Skip metric computation")
    ap.add_argument("--methods",  nargs="+", default=None,
                    help="Evaluate only these methods (default: all in config)")
    ap.add_argument("--test-only", action="store_true",
                    help="Compute metrics on test-set cameras only (llffhold split)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Path resolution order:
    #   1. Absolute path  → used as-is
    #   2. Relative path + base_dir in config  → base_dir / path
    #   3. Relative path, no base_dir          → config file's directory / path
    #
    # Set "base_dir" in your config to the root of your data tree so that
    # all other paths can be written relative to it.  Only base_dir itself
    # and python_bin need to change between machines.
    cfg_dir  = cfg_path.parent
    base_dir = Path(cfg["base_dir"]).expanduser() if "base_dir" in cfg else None

    def rp(key, default=None):
        v = cfg.get(key, default)
        if v is None:
            return None
        p = Path(v).expanduser()
        if p.is_absolute():
            return p
        if base_dir is not None:
            return base_dir / p
        return cfg_dir / p

    scene_name   = cfg["scene_name"]
    out_dir      = rp("out_dir")
    cameras_raw  = rp("cameras_raw")
    gopro_images = rp("gopro_images")
    control_ply  = rp("control_ply")
    code_dir     = rp("code_dir")
    python_bin   = cfg.get("python_bin", sys.executable)
    mpu          = float(cfg.get("mpu", 1.0))
    depth        = float(cfg.get("depth", 0.35))
    y_offsets    = cfg.get("y_offsets", [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    test_hold    = int(cfg.get("test_hold", 8))
    render_w     = int(cfg.get("render_width",  960))
    render_h     = int(cfg.get("render_height", 540))

    methods_cfg  = cfg["methods"]
    method_names = args.methods or list(methods_cfg.keys())

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw cameras
    with open(cameras_raw) as f:
        raw = json.load(f)
    all_cameras = raw["cameras"]
    N = len(all_cameras)
    print(f"[{scene_name}] {N} cameras  mpu={mpu}  depth={depth}m")

    # Test-set indices
    if test_hold > 0:
        test_indices = set(range(0, N, test_hold))
        all_indices  = set(range(N))
        print(f"  Test set:  {len(test_indices)}/{N} cameras (every {test_hold}th)")
    else:
        test_indices = all_indices = set(range(N))
        print(f"  Using all {N} cameras (test_hold=0)")

    # Physical GT paths: index → Path
    def gt_photo_paths(indices):
        out = {}
        for idx in indices:
            name = all_cameras[idx]["name"]
            p = gopro_images / name
            if p.exists():
                out[idx] = p
        return out

    # ── Render GoPro control at each y-offset (DT ground truth) ──────────────
    gt_render_by_offset = {}
    for y in y_offsets:
        tag     = f"y{y:+.1f}"
        gt_dir  = out_dir / "gt_gopro_renders" / tag
        cam_j   = out_dir / f"cameras_{tag}.json"

        if not args.skip_render:
            build_camera_json(cameras_raw, mpu, y, depth, cam_j)
            n_exist = len(list(gt_dir.glob("*.png")))
            if n_exist < N:
                print(f"  Rendering control at {tag} …", end=" ", flush=True)
                n, rc = render(python_bin, control_ply, cam_j, gt_dir, code_dir)
                print(f"{n} pngs  rc={rc}")
            else:
                print(f"  Control {tag}: [skip — {n_exist} pngs exist]")

        gt_render_by_offset[tag] = gt_dir

    # ── Method sweep ──────────────────────────────────────────────────────────
    all_results  = {}

    for method in method_names:
        raw_ply  = methods_cfg.get(method)
        if raw_ply is None:
            print(f"  WARNING: {method} not in config — skipping")
            continue
        p = Path(raw_ply).expanduser()
        ply_path = p if p.is_absolute() else (base_dir / p if base_dir else cfg_dir / p)
        if not ply_path.exists():
            print(f"  WARNING: PLY not found for {method}: {ply_path} — skipping")
            continue

        print(f"\n  [{method}]")
        per_offset_dt   = {}
        per_offset_phys = {}

        # Per-image accumulators for proper pooling
        dt_ssims, dt_psnrs, dt_lpipss       = [], [], []
        phys_ssims, phys_psnrs, phys_lpipss = [], [], []

        for y in y_offsets:
            tag     = f"y{y:+.1f}"
            cam_j   = out_dir / f"cameras_{tag}.json"
            rdir    = out_dir / tag / method

            # Render
            if not args.skip_render:
                n_exist = len(list(rdir.glob("*.png"))) if rdir.exists() else 0
                if n_exist < N:
                    rdir.mkdir(parents=True, exist_ok=True)
                    if not cam_j.exists():
                        build_camera_json(cameras_raw, mpu, y, depth, cam_j)
                    print(f"    {tag}: rendering …", end=" ", flush=True)
                    n, rc = render(python_bin, ply_path, cam_j, rdir, code_dir)
                    print(f"{n} pngs  rc={rc}")
                else:
                    print(f"    {tag}: [skip — {n_exist} pngs exist]")

            if args.skip_metrics:
                continue

            eval_indices = test_indices if args.test_only else all_indices

            # DT metrics
            gt_render_dir = gt_render_by_offset[tag]
            gt_dt = {i: gt_render_dir / f"{i:06d}.png" for i in eval_indices}
            m_dt = compute_metrics(rdir, gt_dt, method, tag)
            per_offset_dt[tag] = m_dt
            if m_dt:
                dt_ssims.extend([m_dt["ssim_mean"]] * m_dt["n"])
                dt_psnrs.extend([m_dt["psnr_mean"]] * m_dt["n"])
                dt_lpipss.extend([m_dt["lpips_mean"]] * m_dt["n"])

            # Physical metrics
            gt_phys = gt_photo_paths(eval_indices)
            m_phys = compute_metrics(rdir, gt_phys, method, tag)
            per_offset_phys[tag] = m_phys
            if m_phys:
                phys_ssims.extend([m_phys["ssim_mean"]] * m_phys["n"])
                phys_psnrs.extend([m_phys["psnr_mean"]] * m_phys["n"])
                phys_lpipss.extend([m_phys["lpips_mean"]] * m_phys["n"])

            if m_dt and m_phys:
                print(f"    {tag}: "
                      f"DT SSIM={m_dt['ssim_mean']:.4f} PSNR={m_dt['psnr_mean']:.2f} "
                      f"LPIPS={m_dt['lpips_mean']:.4f}  |  "
                      f"Phys SSIM={m_phys['ssim_mean']:.4f} PSNR={m_phys['psnr_mean']:.2f} "
                      f"LPIPS={m_phys['lpips_mean']:.4f}", flush=True)

        if not args.skip_metrics:
            def pooled(ssims, psnrs, lpipss):
                if not ssims:
                    return None
                return {"ssim_mean":  float(np.mean(ssims)),
                        "ssim_std":   float(np.std(ssims)),
                        "psnr_mean":  float(np.mean(psnrs)),
                        "psnr_std":   float(np.std(psnrs)),
                        "lpips_mean": float(np.mean(lpipss)),
                        "lpips_std":  float(np.std(lpipss)),
                        "n": len(ssims)}

            p_dt   = pooled(dt_ssims,   dt_psnrs,   dt_lpipss)
            p_phys = pooled(phys_ssims, phys_psnrs, phys_lpipss)

            all_results[method] = {
                "dt":         p_dt,
                "physical":   p_phys,
                "per_offset": {"dt": per_offset_dt, "physical": per_offset_phys},
            }

            if p_dt:
                print(f"  → DT pooled   (n={p_dt['n']}): "
                      f"SSIM={p_dt['ssim_mean']:.4f}±{p_dt['ssim_std']:.4f}  "
                      f"PSNR={p_dt['psnr_mean']:.2f}±{p_dt['psnr_std']:.2f}  "
                      f"LPIPS={p_dt['lpips_mean']:.4f}±{p_dt['lpips_std']:.4f}", flush=True)
            if p_phys:
                print(f"  → Phys pooled (n={p_phys['n']}): "
                      f"SSIM={p_phys['ssim_mean']:.4f}±{p_phys['ssim_std']:.4f}  "
                      f"PSNR={p_phys['psnr_mean']:.2f}±{p_phys['psnr_std']:.2f}  "
                      f"LPIPS={p_phys['lpips_mean']:.4f}±{p_phys['lpips_std']:.4f}", flush=True)

    if not args.skip_metrics and all_results:
        suffix = "_testset" if args.test_only else "_allcams"
        out_json = out_dir / f"results{suffix}.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results → {out_json}")


if __name__ == "__main__":
    main()
