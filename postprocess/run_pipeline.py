#!/usr/bin/env python3
"""
run_pipeline.py
===============
Single entry point for the cross-domain 3DGS evaluation pipeline.

Steps:
  1. gen_cameras.py  — Extract GoPro camera poses from COLMAP → cameras_raw.json
  2. sweep.py        — Render all methods at y-offset sweep, compute DT + physical metrics

Pre-requisites (must be done manually before running this):
  • Run align.py to compute the mastcam→GoPro transform and bake mpu into the PLYs.
    See README.md → "Step 0: Align splats".

Usage:
    python run_pipeline.py --config configs/scene_1.json

    # Skip rendering (if renders already exist) and just compute metrics:
    python run_pipeline.py --config configs/scene_1.json --skip-render

    # Only evaluate specific methods:
    python run_pipeline.py --config configs/scene_1.json --methods 3DGS MipSplatting

    # Test-set only (llffhold cameras):
    python run_pipeline.py --config configs/scene_1.json --test-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"\nERROR: command exited with code {r.returncode}")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Scene config JSON")
    ap.add_argument("--skip-gen-cameras", action="store_true",
                    help="Skip cameras_raw.json generation (if already done)")
    ap.add_argument("--skip-render",  action="store_true",
                    help="Skip rendering (assumes renders already exist)")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="Skip metric computation")
    ap.add_argument("--methods",  nargs="+", default=None,
                    help="Evaluate only these methods")
    ap.add_argument("--test-only", action="store_true",
                    help="Compute metrics on test-set cameras only")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path) as f:
        cfg = json.load(f)

    cfg_dir  = cfg_path.parent
    base_dir = Path(cfg["base_dir"]).expanduser() if "base_dir" in cfg else None

    def rp(key, default=None):
        v = cfg.get(key, default)
        if v is None:
            return None
        p = Path(v).expanduser()
        if p.is_absolute():
            return str(p)
        if base_dir is not None:
            return str(base_dir / p)
        return str(cfg_dir / p)

    scene_name   = cfg["scene_name"]
    gopro_colmap = rp("gopro_colmap")
    out_dir      = rp("out_dir")
    cameras_raw  = rp("cameras_raw")
    render_w     = cfg.get("render_width", 960)
    render_h     = cfg.get("render_height", 540)

    scripts = Path(__file__).parent

    print(f"\n{'#'*60}")
    print(f"#  Cross-Domain 3DGS Eval  —  {scene_name}")
    print(f"{'#'*60}")

    # ── Step 1: Generate cameras_raw.json ─────────────────────────────────────
    if not args.skip_gen_cameras:
        cmd = [sys.executable, str(scripts / "gen_cameras.py"),
               "--colmap", gopro_colmap,
               "--out",    cameras_raw,
               "--render-width",  str(render_w),
               "--render-height", str(render_h)]
        run(cmd, "Step 1: gen_cameras — extract GoPro COLMAP poses")
    else:
        print(f"\nStep 1 skipped (--skip-gen-cameras).  Using: {cameras_raw}")

    # ── Step 2: Sweep + metrics ────────────────────────────────────────────────
    cmd = [sys.executable, "-u", str(scripts / "sweep.py"),
           "--config", str(cfg_path)]
    if args.skip_render:
        cmd.append("--skip-render")
    if args.skip_metrics:
        cmd.append("--skip-metrics")
    if args.test_only:
        cmd.append("--test-only")
    if args.methods:
        cmd += ["--methods"] + args.methods

    run(cmd, "Step 2: sweep — render + compute DT and physical metrics")

    print(f"\n{'='*60}")
    print(f"  Done.  Results in: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
