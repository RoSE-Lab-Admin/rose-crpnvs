#!/usr/bin/env python3
"""
gen_cameras.py
==============
Read a COLMAP sparse reconstruction (binary format) and write a cameras JSON
file suitable for sweep.py / render_ply.py.

The output JSON stores camera positions in **COLMAP units** (not metres).
sweep.py converts to metres using the `mpu` (metres-per-unit) value from the
scene config.

Usage:
    python gen_cameras.py \
        --colmap  data/go_pro/scene_1/sparse/0 \
        --out     results/scene_1/cameras_raw.json \
        [--render-width 960 --render-height 540]

Output format:
    {
      "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...,
                     "width": ..., "height": ...},
      "cameras": [
        {"id": 0, "name": "img_001.jpg",
         "position": [x, y, z],     ← COLMAP units
         "R_c2w": [[...], [...], [...]],
         "fx": ..., "fy": ..., "cx": ..., "cy": ...,
         "width": ..., "height": ...},
        ...
      ]
    }
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# ── COLMAP binary readers ─────────────────────────────────────────────────────

COLMAP_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5,
                  8: 4, 9: 5, 10: 12}


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid   = struct.unpack("<I", f.read(4))[0]
            model = struct.unpack("<I", f.read(4))[0]
            w     = struct.unpack("<Q", f.read(8))[0]
            h     = struct.unpack("<Q", f.read(8))[0]
            npar  = COLMAP_NPARAMS.get(model, 4)
            params = struct.unpack(f"<{npar}d", f.read(8 * npar))
            cams[cid] = {"model": model, "w": w, "h": h, "params": list(params)}
    return cams


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid          = struct.unpack("<I",  f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz   = struct.unpack("<3d", f.read(24))
            cam_id       = struct.unpack("<I",  f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)
            R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            R_c2w = R_w2c.T
            pos   = -R_w2c.T @ np.array([tx, ty, tz])
            images[iid] = {
                "name":   name.decode(),
                "R_c2w":  R_c2w.tolist(),
                "position": pos.tolist(),
                "cam_id": cam_id,
            }
    return images


def intrinsics_from_colmap_cam(cam, render_w=None, render_h=None):
    """Extract fx, fy, cx, cy from a COLMAP camera dict.

    Scales to render_w×render_h if provided (COLMAP stores native resolution).
    Supports SIMPLE_PINHOLE(0), PINHOLE(1), SIMPLE_RADIAL(2), RADIAL(3),
    OPENCV(4), and OPENCV_FISHEYE(5).
    """
    p     = cam["params"]
    model = cam["model"]
    w0, h0 = cam["w"], cam["h"]

    if model == 0:                        # SIMPLE_PINHOLE
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    elif model == 1:                      # PINHOLE
        fx = p[0]; fy = p[1]; cx = p[2]; cy = p[3]
    elif model in (2, 3):                 # SIMPLE_RADIAL / RADIAL
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    elif model in (4, 5, 6):              # OPENCV / OPENCV_FISHEYE / FULL_OPENCV
        fx = p[0]; fy = p[1]; cx = p[2]; cy = p[3]
    else:
        raise ValueError(f"Unsupported COLMAP camera model {model}")

    if render_w is not None and render_h is not None:
        sx = render_w / w0
        sy = render_h / h0
        fx *= sx; cx *= sx
        fy *= sy; cy *= sy
        w0, h0 = render_w, render_h

    return dict(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w0), height=int(h0))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap", required=True,
                    help="Path to COLMAP sparse/0 directory (contains cameras.bin, images.bin)")
    ap.add_argument("--out", required=True,
                    help="Output cameras_raw.json path")
    ap.add_argument("--render-width",  type=int, default=None,
                    help="Target render width in pixels (downscales intrinsics if needed)")
    ap.add_argument("--render-height", type=int, default=None,
                    help="Target render height in pixels")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap)
    cams  = read_cameras_bin(str(colmap_dir / "cameras.bin"))
    imgs  = read_images_bin(str(colmap_dir  / "images.bin"))

    # Use the first (or only) camera for global intrinsics
    cam0     = list(cams.values())[0]
    intr     = intrinsics_from_colmap_cam(cam0, args.render_width, args.render_height)

    cameras = []
    for i, img in enumerate(sorted(imgs.values(), key=lambda x: x["name"])):
        cam_i = cams[img["cam_id"]]
        intr_i = intrinsics_from_colmap_cam(cam_i, args.render_width, args.render_height)
        entry = dict(id=i, name=img["name"],
                     position=img["position"], R_c2w=img["R_c2w"])
        entry.update(intr_i)
        cameras.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"intrinsics": intr, "cameras": cameras}, f)

    print(f"Wrote {len(cameras)} cameras → {out_path}")
    print(f"  Intrinsics: {intr}")


if __name__ == "__main__":
    main()
