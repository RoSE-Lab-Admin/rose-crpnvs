#!/usr/bin/env python3
"""
eval_dome/render_ply.py
=======================
Render a 3DGS PLY from dome camera poses defined in a JSON file.
Works in any env with diff_gaussian_rasterization installed.

Usage (called by run_eval.py via subprocess):
    python render_ply.py \
        --ply       path/to/splat.ply \
        --cameras   path/to/dome_cameras.json \
        --out-dir   path/to/output_dir \
        --code-dir  path/to/method_code_dir \
        [--y-offset 0.0] \
        [--kernel-size 0.1]   # omit for methods without kernel_size

The script emits one PNG per camera: 000000.png, 000001.png, …
"""

import sys, os, math, json, argparse
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply",         required=True)
ap.add_argument("--cameras",     required=True, help="dome_cameras.json")
ap.add_argument("--out-dir",     required=True)
ap.add_argument("--code-dir",    required=True, help="Method source dir (e.g. mip-splatting/)")
ap.add_argument("--y-offset",    type=float, default=0.0, help="Vertical shift in GoPro units")
ap.add_argument("--kernel-size", type=float, default=None,
                help="Pass kernel_size to render() if the method supports it")
ap.add_argument("--sh-degree",   type=int, default=3)
args = ap.parse_args()

# ── Add method code dir to path ───────────────────────────────────────────────
sys.path.insert(0, args.code_dir)

from gaussian_renderer import GaussianModel, render as gs_render
from scene.cameras import Camera
import inspect as _inspect

# ── Helpers ───────────────────────────────────────────────────────────────────
class _Pipe:
    convert_SHs_python  = False
    compute_cov3D_python = False
    debug               = False
    beta                = 5.0    # SparseGS rasterizer expects this

# Detect Camera variant once at import time
_cam_sig = set(_inspect.signature(Camera.__init__).parameters.keys())
_CAM_HAS_K        = "K"         in _cam_sig   # GaussianPro, SparseGS
_CAM_HAS_WARP     = "warp_mask" in _cam_sig   # SparseGS only
_CAM_HAS_DEPTH_POS= ("depth"    in _cam_sig and
                     list(_cam_sig).index("depth") < 8)  # SparseGS: positional

def lookat_c2w(pos, target, world_up=np.array([0., 1., 0.])):
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:          # degenerate (looking straight up/down)
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # C2W: columns are [right, up, -forward]  (OpenGL / COLMAP convention)
    return np.stack([right, up, -fwd], axis=1).astype(np.float32)

def make_cam(pos, R_c2w, fx, fy, cx, cy, W, H, uid):
    T    = -(R_c2w.T @ pos).astype(np.float32)
    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))
    img  = torch.zeros((3, H, W), dtype=torch.float32)

    if _CAM_HAS_WARP:
        # SparseGS: Camera(colmap_id, R, T, FoVx, FoVy, image, depth,
        #                  gt_alpha_mask, image_name, uid, warp_mask,
        #                  K, src_R, src_T, src_uid)
        K_mat = torch.tensor([[fx,  0, cx],
                              [ 0, fy, cy],
                              [ 0,  0,  1]], dtype=torch.float32).cuda()
        return Camera(
            colmap_id=uid, R=R_c2w, T=T,
            FoVx=fovx, FoVy=fovy,
            image=img, depth=None, gt_alpha_mask=None,
            image_name=f"{uid:06d}", uid=uid,
            warp_mask=None, K=K_mat,
            src_R=R_c2w, src_T=T, src_uid=uid,
        )
    elif _CAM_HAS_K:
        # GaussianPro: Camera(..., K=[fx, fy, cx, cy])
        return Camera(
            colmap_id=uid, R=R_c2w, T=T,
            FoVx=fovx, FoVy=fovy,
            image=img, gt_alpha_mask=None,
            image_name=f"{uid:06d}", uid=uid,
            K=[fx, fy, cx, cy],
        )
    else:
        # Standard (mip-splatting, BAGS, SuGaR/3DGS)
        return Camera(
            colmap_id=uid, R=R_c2w, T=T,
            FoVx=fovx, FoVy=fovy,
            image=img, gt_alpha_mask=None,
            image_name=f"{uid:06d}", uid=uid,
        )

# ── Load ──────────────────────────────────────────────────────────────────────
with open(args.cameras) as f:
    cam_data = json.load(f)

intrinsics = cam_data["intrinsics"]
fx, fy = intrinsics["fx"], intrinsics["fy"]
cx, cy = intrinsics["cx"], intrinsics["cy"]
W,  H  = intrinsics["width"], intrinsics["height"]

cameras_cfg = cam_data["cameras"]   # list of {id, position, target}

gaussians = GaussianModel(args.sh_degree)
gaussians.load_ply(args.ply)

bg    = torch.zeros(3, dtype=torch.float32, device="cuda")
pipe  = _Pipe()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

import torchvision

# ── Render ────────────────────────────────────────────────────────────────────
for cfg in cameras_cfg:
    uid = cfg["id"]
    pos = np.array(cfg["position"], dtype=np.float32)
    pos[1] += args.y_offset                    # apply vertical jitter (0 for real cameras)

    # Per-camera intrinsics override global if present (GoPro eval mode)
    _fx = cfg.get("fx", fx)
    _fy = cfg.get("fy", fy)
    _cx = cfg.get("cx", cx)
    _cy = cfg.get("cy", cy)
    _W  = cfg.get("width",  W)
    _H  = cfg.get("height", H)

    # Use exact COLMAP rotation if provided, else lookat approximation (dome mode)
    if "R_c2w" in cfg:
        R_c2w = np.array(cfg["R_c2w"], dtype=np.float32)
    else:
        tgt   = np.array(cfg["target"], dtype=np.float32)
        R_c2w = lookat_c2w(pos, tgt)

    cam = make_cam(pos, R_c2w, _fx, _fy, _cx, _cy, _W, _H, uid)

    with torch.no_grad():
        if args.kernel_size is not None:
            out_img = gs_render(cam, gaussians, pipe, bg,
                                kernel_size=args.kernel_size)["render"]
        else:
            out_img = gs_render(cam, gaussians, pipe, bg)["render"]

    torchvision.utils.save_image(out_img, out / f"{uid:06d}.png")

print(f"Rendered {len(cameras_cfg)} views → {out}")
