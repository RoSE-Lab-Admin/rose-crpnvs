#!/usr/bin/env python3
"""
Align two Gaussian Splatting PLY files using ArUco marker correspondences,
followed by ICP refinement on COLMAP sparse point clouds.

Pipeline:
  1. Auto-detect ArUco dictionary (or use --aruco-dict).
  2. Detect markers with subpixel corner refinement.
  3. Multi-view DLT triangulation with per-observation reprojection filtering.
  4. Filter triangulated markers by side-length consistency
     (real markers should form ~square with --marker-size sides).
  5. RANSAC Umeyama similarity transform (src → dst).
  6. ICP rigid refinement on COLMAP sparse 3D points (scale fixed from step 5).
  7. Apply combined transform to src PLY:
       positions:    x' = s * R_total * x + t_total
       3DGS scales:  scale_i += log(s)
       3DGS quats:   q' = q_R_total ⊗ q_gaussian

Usage:
    python align_by_aruco.py \\
        --src-colmap /path/gopro/sparse/0 \\
        --src-images /path/gopro/images \\
        --src-ply    /path/gopro.ply \\
        --dst-colmap /path/mastcam/sparse/0 \\
        --dst-images /path/mastcam/images \\
        --output     aligned_gopro.ply \\
        [--marker-size 0.10] [--min-views 3] [--no-scale] [--skip-icp]
"""

import argparse
import struct
import math
import json
import numpy as np
import cv2
import cv2.aruco as aruco
from pathlib import Path
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

# ─── COLMAP binary readers ────────────────────────────────────────────────────

COLMAP_CAMERA_MODELS = {
    0:  ("SIMPLE_PINHOLE",        3),
    1:  ("PINHOLE",               4),
    2:  ("SIMPLE_RADIAL",         4),
    3:  ("RADIAL",                5),
    4:  ("OPENCV",                8),
    5:  ("OPENCV_FISHEYE",        8),
    6:  ("FULL_OPENCV",          12),
    7:  ("FOV",                   5),
    8:  ("SIMPLE_RADIAL_FISHEYE", 4),
    9:  ("RADIAL_FISHEYE",        5),
    10: ("THIN_PRISM_FISHEYE",   12),
}


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id  = struct.unpack("<i", f.read(4))[0]
            model   = struct.unpack("<i", f.read(4))[0]
            width   = struct.unpack("<Q", f.read(8))[0]
            height  = struct.unpack("<Q", f.read(8))[0]
            _, npar = COLMAP_CAMERA_MODELS[model]
            params  = list(struct.unpack(f"<{npar}d", f.read(8 * npar)))
            cameras[cam_id] = {"model": model, "width": width,
                               "height": height, "params": params}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec   = struct.unpack("<4d", f.read(32))   # qw qx qy qz
            tvec   = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name   = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)
            images[img_id] = {"qvec": qvec, "tvec": tvec,
                              "camera_id": cam_id, "name": name.decode()}
    return images


def read_points3d_binary(path):
    """Return (N, 3) float64 array of well-observed 3D points."""
    pts = []
    errs = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            f.read(8)                             # point_id (uint64)
            xyz   = struct.unpack("<3d", f.read(24))
            f.read(3)                             # rgb (3×uint8)
            err   = struct.unpack("<d", f.read(8))[0]
            tlen  = struct.unpack("<Q", f.read(8))[0]
            f.read(tlen * 8)                      # track (image_id + pt2D_idx)
            pts.append(xyz)
            errs.append(err)
    pts  = np.array(pts,  dtype=np.float64)
    errs = np.array(errs, dtype=np.float64)
    return pts, errs


def camera_matrix(cam):
    p, m = cam["params"], cam["model"]
    if m == 0:      return np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]], np.float64)
    if m == 1:      return np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1]], np.float64)
    if m in (2,3):  return np.array([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]], np.float64)
    if m in (4,5,6):return np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1]], np.float64)
    raise ValueError(f"Unsupported camera model {m}")


def distortion_coeffs(cam):
    p, m = cam["params"], cam["model"]
    if m == 0:   return np.zeros(4)
    if m == 1:   return np.zeros(4)
    if m == 2:   return np.array([p[3], 0, 0, 0])
    if m == 3:   return np.array([p[3], p[4], 0, 0])
    if m == 4:   return np.array([p[4], p[5], p[6], p[7]])
    if m == 5:   return np.array([p[4], p[5], p[6], p[7]])
    return np.zeros(4)


def image_projection_matrix(cam, img):
    """Return K, R, t, P=K@[R|t]  (world→image)."""
    qw, qx, qy, qz = img["qvec"]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    t = np.array(img["tvec"])
    K = camera_matrix(cam)
    return K, R, t, K @ np.hstack([R, t[:, None]])


def undistort_pts(pts2d, cam):
    K    = camera_matrix(cam)
    dist = distortion_coeffs(cam)
    arr  = pts2d.astype(np.float64).reshape(-1, 1, 2)
    if cam["model"] == 5:
        out = cv2.fisheye.undistortPoints(arr, K, dist, P=K)
    else:
        out = cv2.undistortPoints(arr, K, dist, P=K)
    return out.reshape(-1, 2)

# ─── Multi-view DLT triangulation ────────────────────────────────────────────

def triangulate_dlt(Ps, pts2d):
    A = []
    for P, (x, y) in zip(Ps, pts2d):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reproj_errors(X, Ps, pts2d):
    Xh = np.append(X, 1.0)
    errs = []
    for P, pt in zip(Ps, pts2d):
        proj = P @ Xh
        if proj[2] <= 0:
            errs.append(np.inf)
        else:
            errs.append(np.linalg.norm(proj[:2] / proj[2] - pt))
    return np.array(errs)


def triangulate_robust(Ps, pts2d, max_reproj=1.5, min_views=2):
    """
    DLT with iterative reprojection-error outlier removal.
    Returns 3D point or None if too few inliers remain.
    """
    active = list(range(len(Ps)))
    for _ in range(3):   # up to 3 rounds of filtering
        if len(active) < min_views:
            return None
        Ps_a  = [Ps[i]    for i in active]
        pts_a = [pts2d[i] for i in active]
        X     = triangulate_dlt(Ps_a, pts_a)
        errs  = reproj_errors(X, Ps_a, pts_a)
        mask  = errs < max_reproj
        if mask.all():
            return X
        keep  = [active[i] for i, ok in enumerate(mask) if ok]
        if len(keep) == len(active):
            break
        active = keep
    if len(active) < min_views:
        return None
    return triangulate_dlt([Ps[i] for i in active],
                           [pts2d[i] for i in active])

# ─── ArUco detection with subpixel refinement ─────────────────────────────────

CANDIDATE_DICTS = [
    ("DICT_APRILTAG_36h11", aruco.DICT_APRILTAG_36h11),
    ("DICT_APRILTAG_36h10", aruco.DICT_APRILTAG_36h10),
    ("DICT_APRILTAG_25h9",  aruco.DICT_APRILTAG_25h9),
    ("DICT_APRILTAG_16h5",  aruco.DICT_APRILTAG_16h5),
    ("DICT_4X4_50",         aruco.DICT_4X4_50),
    ("DICT_4X4_100",        aruco.DICT_4X4_100),
    ("DICT_4X4_250",        aruco.DICT_4X4_250),
    ("DICT_5X5_50",         aruco.DICT_5X5_50),
    ("DICT_5X5_100",        aruco.DICT_5X5_100),
    ("DICT_6X6_50",         aruco.DICT_6X6_50),
    ("DICT_6X6_100",        aruco.DICT_6X6_100),
    ("DICT_ARUCO_ORIGINAL", aruco.DICT_ARUCO_ORIGINAL),
]

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_in_dataset(images_dir, colmap_images, colmap_cameras, aruco_dict,
                      sample_every=1, max_reproj=1.5, verbose=True,
                      id_offset=0):
    """
    Returns dict[marker_id] → list of (P_3x4, corners_4x2_undistorted).
    Applies subpixel corner refinement before undistortion.
    id_offset: added to every detected marker ID (used for multi-dict namespacing).
    """
    params   = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    by_name = {}
    for img in colmap_images.values():
        by_name[img["name"]]            = img
        by_name[Path(img["name"]).name] = img

    obs = {}
    img_files = sorted(
        set(Path(images_dir).glob("*.jpg")) |
        set(Path(images_dir).glob("*.JPG")) |
        set(Path(images_dir).glob("*.png")) |
        set(Path(images_dir).glob("*.PNG"))
    )

    processed = 0
    for i, ip in enumerate(img_files):
        if i % sample_every != 0:
            continue
        meta = by_name.get(ip.name) or by_name.get(str(ip))
        if meta is None:
            continue

        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        processed += 1

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue

        # subpixel refinement on raw (distorted) image before undistortion
        corners_refined = []
        for mc in corners:
            mc_r = cv2.cornerSubPix(gray, mc, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
            corners_refined.append(mc_r)

        cam = colmap_cameras[meta["camera_id"]]
        _, _, _, P = image_projection_matrix(cam, meta)

        for mc, mid in zip(corners_refined, ids.flatten()):
            mc_u = undistort_pts(mc[0], cam)   # (4, 2)
            obs.setdefault(int(mid) + id_offset, []).append((P, mc_u))

    if verbose:
        total = sum(len(v) for v in obs.values())
        print(f"    {processed} images → {total} obs, "
              f"{len(obs)} IDs: {sorted(obs)}")
    return obs


def detect_multi_dict(images_dir, colmap_images, colmap_cameras,
                      dict_names, sample_every=1, max_reproj=1.5):
    """
    Detect markers with multiple ArUco/AprilTag dictionaries simultaneously.
    Each dictionary's marker IDs are offset by dict_index × 10000 to avoid
    collisions (e.g. ArUco ID 21 and AprilTag ID 21 become 21 and 10021).

    Returns: combined obs dict, legend dict {namespaced_id → (dict_name, raw_id)}
    """
    ID_STRIDE = 10000
    combined = {}
    legend   = {}   # namespaced_id → (dict_name, raw_id)

    for di, dname in enumerate(dict_names):
        d_code = getattr(aruco, dname)
        d_obj  = aruco.getPredefinedDictionary(d_code)
        offset = di * ID_STRIDE
        print(f"  [{di}] {dname}  (id_offset={offset})")
        obs = detect_in_dataset(images_dir, colmap_images, colmap_cameras,
                                d_obj, sample_every=sample_every,
                                max_reproj=max_reproj, verbose=True,
                                id_offset=offset)
        for nsid, views in obs.items():
            combined[nsid] = combined.get(nsid, []) + views
            legend[nsid]   = (dname, nsid - offset)

    total = sum(len(v) for v in combined.values())
    print(f"  Combined: {total} obs across {len(combined)} namespaced marker IDs")
    return combined, legend


def pick_best_aruco_dict(images_dir, colmap_images, colmap_cameras,
                         force_dict=None, sample_every=10):
    if force_dict is not None:
        d   = aruco.getPredefinedDictionary(getattr(aruco, force_dict))
        obs = detect_in_dataset(images_dir, colmap_images, colmap_cameras,
                                d, sample_every=sample_every, verbose=False)
        return force_dict, d, obs

    best = (None, None, {}, 0)
    for name, code in CANDIDATE_DICTS:
        d   = aruco.getPredefinedDictionary(code)
        obs = detect_in_dataset(images_dir, colmap_images, colmap_cameras,
                                d, sample_every=sample_every, verbose=False)
        n   = sum(len(v) for v in obs.values())
        print(f"    {name:30s} → {n:5d} obs  {len(obs)} IDs")
        if n > best[3]:
            best = (name, d, obs, n)
    print(f"  → Best: {best[0]}  ({best[3]} obs)")
    return best[0], best[1], best[2]


def triangulate_marker_set(obs, min_views=2, max_reproj=1.5):
    """Returns dict[marker_id] → (4,3) 3D corners."""
    out = {}
    for mid, views in obs.items():
        if len(views) < min_views:
            continue
        Ps   = [v[0] for v in views]
        c4x3 = []
        ok   = True
        for ci in range(4):
            pts2d = [v[1][ci] for v in views]
            X = triangulate_robust(Ps, pts2d, max_reproj=max_reproj,
                                   min_views=min_views)
            if X is None:
                ok = False
                break
            c4x3.append(X)
        if ok:
            out[mid] = np.array(c4x3)
    return out


def filter_markers_by_size(markers_3d, expected_size, tol=0.35):
    """
    Remove any marker whose triangulated corner side lengths deviate
    more than `tol` (fraction) from expected_size.
    """
    kept = {}
    for mid, corners in markers_3d.items():
        sides = [np.linalg.norm(corners[(i+1)%4] - corners[i]) for i in range(4)]
        mean_side = np.mean(sides)
        if abs(mean_side - expected_size) / expected_size <= tol:
            kept[mid] = corners
        else:
            print(f"    [filter] marker {mid}: mean side {mean_side:.4f} "
                  f"vs expected {expected_size:.4f} → rejected")
    return kept

# ─── Umeyama + RANSAC ────────────────────────────────────────────────────────

def umeyama(src, dst, with_scale=True):
    """dst ≈ s * R @ src + t.  Returns s, R (3×3), t (3,)."""
    n = len(src)
    mu_s = src.mean(0);  mu_d = dst.mean(0)
    sc   = src - mu_s;   dc   = dst - mu_d
    var_s = (sc**2).sum() / n
    H     = (dc.T @ sc) / n
    U, S, Vt = np.linalg.svd(H)
    det   = int(round(np.linalg.det(U @ Vt)))
    D     = np.diag([1.0] * 2 + [float(det)])
    R     = U @ D @ Vt
    s     = float((S * np.diag(D)).sum() / var_s) if with_scale else 1.0
    t     = mu_d - s * R @ mu_s
    return s, R, t


def umeyama_ransac(src, dst, threshold, with_scale=True,
                   n_iter=2000, rng_seed=42):
    """
    RANSAC wrapper around Umeyama.
    src, dst: (N, 3) matched points (N ≥ 3).
    threshold: inlier distance threshold in dst units.
    Returns s, R, t, inlier_mask (bool N).
    """
    n = len(src)
    if n <= 3:
        s, R, t = umeyama(src, dst, with_scale)
        return s, R, t, np.ones(n, bool)

    rng = np.random.default_rng(rng_seed)
    best_inliers = None
    best_count   = 0

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx], with_scale)
        except Exception:
            continue
        xfm   = (s * (R @ src.T)).T + t
        errs  = np.linalg.norm(xfm - dst, axis=1)
        inliers = errs < threshold
        cnt   = inliers.sum()
        if cnt > best_count or (cnt == best_count and
                                errs[inliers].mean() < errs[best_inliers].mean()):
            best_count   = cnt
            best_inliers = inliers

    # Final refit on all inliers
    s, R, t = umeyama(src[best_inliers], dst[best_inliers], with_scale)
    return s, R, t, best_inliers

# ─── ICP refinement ──────────────────────────────────────────────────────────

def icp_rigid(src_pts, dst_pts, max_iter=100, tol=1e-7,
              inlier_pct=50, verbose=True):
    """
    Point-to-point ICP, rigid only (no scale).
    src_pts already in dst frame (after ArUco similarity transform).
    Returns R_acc (3×3), t_acc (3,) — the additional rigid correction.
    """
    tree  = KDTree(dst_pts)
    src   = src_pts.copy()
    R_acc = np.eye(3)
    t_acc = np.zeros(3)
    prev_err = np.inf

    for it in range(max_iter):
        dists, idx = tree.query(src, workers=-1)
        thresh = np.percentile(dists, inlier_pct)
        mask   = dists < thresh
        n_in   = mask.sum()

        if n_in < 6:
            print(f"  ICP: only {n_in} inliers, stopping.")
            break

        matched_src = src[mask]
        matched_dst = dst_pts[idx[mask]]

        _, R_step, t_step = umeyama(matched_src, matched_dst, with_scale=False)
        src   = (R_step @ src.T).T + t_step
        R_acc = R_step @ R_acc
        t_acc = R_step @ t_acc + t_step

        err = np.mean(dists[mask])
        if verbose:
            print(f"  ICP {it+1:3d}: {n_in:6d} inliers  "
                  f"mean dist={err:.5f}  max={dists[mask].max():.5f}")
        if abs(prev_err - err) < tol:
            if verbose:
                print("  ICP converged.")
            break
        prev_err = err

    return R_acc, t_acc

# ─── SH rotation (Wigner D-matrices, numerical) ──────────────────────────────

def _sh1_basis(dirs):
    """l=1 SH basis per 3DGS convention: (-y, z, -x). dirs: (N,3)→(N,3)"""
    C = 0.4886025119029199
    return np.stack([-C*dirs[:,1], C*dirs[:,2], -C*dirs[:,0]], axis=1)


def _sh2_basis(dirs):
    """l=2 SH basis per 3DGS convention. dirs: (N,3)→(N,5)"""
    x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
    C = [1.0925484305920792, -1.0925484305920792,
         0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    return np.stack([C[0]*x*y, C[1]*y*z, C[2]*(2*z**2-x**2-y**2),
                     C[3]*x*z, C[4]*(x**2-y**2)], axis=1)


def _sh3_basis(dirs):
    """l=3 SH basis per 3DGS convention. dirs: (N,3)→(N,7)"""
    x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
    xx, yy, zz = x*x, y*y, z*z
    C = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.4453057213202770, -0.5900435899266435]
    return np.stack([C[0]*y*(3*xx-yy), C[1]*x*y*z, C[2]*y*(4*zz-xx-yy),
                     C[3]*z*(2*zz-3*xx-3*yy), C[4]*x*(4*zz-xx-yy),
                     C[5]*z*(xx-yy), C[6]*x*(xx-3*yy)], axis=1)


def compute_sh_rotation_matrices(R):
    """
    Numerically compute SH Wigner D-matrices D1,D2,D3 for rotation R.
    Returns D[l] such that  new_coeff = D[l] @ old_coeff  for l=1,2,3.

    Derivation: when the world rotates by R, a camera at direction d_new in
    the new frame was at R^T d_new in the old frame. So SH value at d_new
    equals old SH evaluated at R^T d_new:
        Y(R^T d) = D^T Y(d)  →  D = lstsq(Y(d), Y(R^T d))^T
    """
    rng = np.random.default_rng(42)
    u = rng.standard_normal((3000, 3))
    dirs = u / np.linalg.norm(u, axis=1, keepdims=True)
    dirs_Rt = (R.T @ dirs.T).T   # R^T applied to all directions

    D = {}
    for l, fn in [(1, _sh1_basis), (2, _sh2_basis), (3, _sh3_basis)]:
        B     = fn(dirs)     # (3000, 2l+1)
        B_Rt  = fn(dirs_Rt)  # (3000, 2l+1)
        # B_Rt = B @ D^T  →  D^T = lstsq(B, B_Rt)
        DT, _, _, _ = np.linalg.lstsq(B, B_Rt, rcond=None)
        D[l] = DT.T
    return D


def rotate_sh_coefficients(data, R):
    """
    Rotate all SH coefficients (l=1,2,3) in a 3DGS vertex array in-place.

    PLY storage (degree-3 SH, 45 f_rest values per Gaussian):
      Color R: f_rest_0..14  (sh idx 0-2=l1, 3-7=l2, 8-14=l3)
      Color G: f_rest_15..29
      Color B: f_rest_30..44
    """
    names = data.dtype.names
    if 'f_rest_0' not in names or 'f_rest_14' not in names:
        return data   # no SH or too few coefficients

    D = compute_sh_rotation_matrices(R)

    for c_off in [0, 15, 30]:          # R, G, B color offsets
        for l, slc in [(1, slice(0,3)), (2, slice(3,8)), (3, slice(8,15))]:
            keys = [f'f_rest_{c_off + i}' for i in range(*slc.indices(15))]
            if not all(k in names for k in keys):
                continue
            old = np.stack([data[k].astype(np.float64) for k in keys], axis=1)
            new = (D[l] @ old.T).T
            for i, k in enumerate(keys):
                data[k] = new[:, i].astype(data[k].dtype)

    print("  Rotated SH coefficients (l=1,2,3).")
    return data


# ─── Scale error analysis ────────────────────────────────────────────────────

def report_scale_errors(src_3d, dst_3d, common_ids, s, R,
                        src_median, dst_median, marker_size,
                        gravity_axis=1):
    """
    Compare inter-corner distances between GoPro (ground truth) and mastcam.

    IMPORTANT: the two COLMAP frames have arbitrary orientations.
    We rotate GoPro vectors into the mastcam frame (using R from Umeyama) before
    decomposing along the vertical axis — otherwise per-axis comparison is invalid.

    gravity_axis: axis in the DST (mastcam) COLMAP frame pointing physically up.
                  Default 1 = Y (most common COLMAP convention).
    """
    src_m  = marker_size / src_median   # src units → metres
    dst_m  = marker_size / dst_median   # dst units → metres
    horiz  = [a for a in range(3) if a != gravity_axis]

    all_src = np.vstack([src_3d[m] for m in common_ids]) * src_m   # GoPro metric
    all_dst = np.vstack([dst_3d[m] for m in common_ids]) * dst_m   # mastcam metric

    n = len(all_src)
    all_ratios, h_ratios, v_ratios = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            d_src     = all_src[i] - all_src[j]
            d_dst     = all_dst[i] - all_dst[j]
            d_src_rot = R @ d_src          # GoPro vector expressed in mastcam frame

            dist_s = np.linalg.norm(d_src_rot)
            dist_d = np.linalg.norm(d_dst)
            if dist_s < 0.01:
                continue
            all_ratios.append(dist_d / dist_s)

            # horizontal = XZ plane (or whichever two axes are not gravity_axis)
            h_s = np.linalg.norm(d_src_rot[horiz])
            h_d = np.linalg.norm(d_dst[horiz])
            v_s = abs(d_src_rot[gravity_axis])
            v_d = abs(d_dst[gravity_axis])

            if h_s > 0.02:
                h_ratios.append(h_d / h_s)
            if v_s > 0.02:
                v_ratios.append(v_d / v_s)

    grav_name = "XYZ"[gravity_axis]
    horiz_name = "".join("XYZ"[a] for a in horiz)
    print(f"\n=== Scale error analysis  "
          f"(GoPro=ground truth, up={grav_name} in mastcam frame) ===")
    print(f"  1 src (GoPro) unit   = {src_m:.4f} m")
    print(f"  1 dst (mastcam) unit = {dst_m:.4f} m")
    print(f"  Similarity scale s   = {s:.6f}  "
          f"(geometry-derived ratio = {dst_m/src_m:.6f})")
    if all_ratios:
        a = np.array(all_ratios)
        print(f"  Overall  distance ratio:     {a.mean():.4f} ± {a.std():.4f}  "
              f"({(a.mean()-1)*100:+.1f}%)  "
              f"[ideal = 1.00 in real-world metric]")
    if h_ratios:
        h = np.array(h_ratios)
        print(f"  Horizontal ({horiz_name}) scale:  "
              f"{h.mean():.4f} ± {h.std():.4f}  "
              f"({(h.mean()-1)*100:+.1f}%)  [{len(h)} pairs]")
    else:
        print("  Horizontal: too few separated pairs")
    if v_ratios:
        v = np.array(v_ratios)
        print(f"  Vertical   ({grav_name})  scale:  "
              f"{v.mean():.4f} ± {v.std():.4f}  "
              f"({(v.mean()-1)*100:+.1f}%)  [{len(v)} pairs]")
    else:
        print(f"  Vertical ({grav_name}): too few vertically-separated pairs")


# ─── 3DGS-aware PLY I/O ───────────────────────────────────────────────────────

_PLY_DTYPE = {
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
    "int": np.int32,  "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "uchar": np.uint8, "uint8": np.uint8,
    "short": np.int16, "ushort": np.uint16, "char": np.int8,
}


def read_ply_positions(path, max_pts=150_000, seed=0):
    """
    Read only xyz positions from a binary PLY file.
    Randomly subsample to max_pts if larger (deterministic, seed=0).
    Returns (N, 3) float64.
    """
    path = str(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").rstrip()
            header_lines.append(line)
            if line == "end_header":
                break
        raw = f.read()

    num_verts = 0
    props, in_vertex = [], False
    for line in header_lines:
        toks = line.split()
        if not toks:
            continue
        if toks[:2] == ["element", "vertex"]:
            num_verts = int(toks[2]);  in_vertex = True
        elif len(toks) >= 2 and toks[0] == "element" and toks[1] != "vertex":
            in_vertex = False
        elif in_vertex and len(toks) >= 3 and toks[0] == "property" and toks[1] != "list":
            props.append((toks[2], toks[1]))

    dtype = np.dtype([(n, _PLY_DTYPE[t]) for n, t in props])
    data  = np.frombuffer(raw[:num_verts * dtype.itemsize], dtype=dtype)

    if max_pts and num_verts > max_pts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(num_verts, max_pts, replace=False)
        data = data[idx]

    return np.stack([data["x"].astype(np.float64),
                     data["y"].astype(np.float64),
                     data["z"].astype(np.float64)], axis=1)


def read_ply_vertices(path):
    path = str(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").rstrip()
            header_lines.append(line)
            if line == "end_header":
                break
        header_bytes = ("\n".join(header_lines) + "\n").encode()
        raw = f.read()

    num_verts = 0
    props     = []
    in_vertex = False
    is_binary_le = False
    for line in header_lines:
        toks = line.split()
        if not toks:
            continue
        if toks[:2] == ["format", "binary_little_endian"]:
            is_binary_le = True
        elif toks[:2] == ["element", "vertex"]:
            num_verts = int(toks[2])
            in_vertex = True
        elif len(toks) >= 2 and toks[0] == "element" and toks[1] != "vertex":
            in_vertex = False
        elif in_vertex and len(toks) >= 3 and toks[0] == "property" and toks[1] != "list":
            props.append((toks[2], toks[1]))

    if not is_binary_le:
        raise ValueError("Only binary_little_endian PLY supported.")
    if not props:
        raise ValueError("No vertex properties parsed from PLY header.")

    dtype = np.dtype([(name, _PLY_DTYPE[typ]) for name, typ in props])
    data  = np.frombuffer(raw[:num_verts * dtype.itemsize], dtype=dtype).copy()
    return data, header_bytes, num_verts, [p[0] for p in props]


def write_ply_vertices(path, data, header_bytes):
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(data.tobytes())


def transform_gaussians(data, s, R, t):
    """Apply similarity transform to 3DGS vertex array in-place."""
    names = data.dtype.names

    if all(k in names for k in ("x", "y", "z")):
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        xyz = (s * (R @ xyz.T)).T + t
        data["x"] = xyz[:, 0].astype(data["x"].dtype)
        data["y"] = xyz[:, 1].astype(data["y"].dtype)
        data["z"] = xyz[:, 2].astype(data["z"].dtype)
        print(f"  Transformed {len(data):,} positions.")

    if all(k in names for k in ("scale_0", "scale_1", "scale_2")):
        log_s = math.log(s)
        for k in ("scale_0", "scale_1", "scale_2"):
            data[k] = (data[k].astype(np.float64) + log_s).astype(data[k].dtype)
        print(f"  Shifted log-scales by log(s)={log_s:.6f}.")

    if all(k in names for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        q_align = Rotation.from_matrix(R)
        qw = data["rot_0"].astype(np.float64)
        qx = data["rot_1"].astype(np.float64)
        qy = data["rot_2"].astype(np.float64)
        qz = data["rot_3"].astype(np.float64)
        norms = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2) + 1e-12
        qw /= norms; qx /= norms; qy /= norms; qz /= norms
        q_gauss = Rotation.from_quat(np.stack([qx, qy, qz, qw], axis=1))
        q_new   = q_align * q_gauss
        q_xyzw  = q_new.as_quat()
        data["rot_0"] = q_xyzw[:, 3].astype(data["rot_0"].dtype)
        data["rot_1"] = q_xyzw[:, 0].astype(data["rot_1"].dtype)
        data["rot_2"] = q_xyzw[:, 1].astype(data["rot_2"].dtype)
        data["rot_3"] = q_xyzw[:, 2].astype(data["rot_3"].dtype)
        print("  Rotated Gaussian orientation quaternions.")

    data = rotate_sh_coefficients(data, R)
    return data

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-colmap",    default=None,
                    help="Src COLMAP sparse dir (required unless --load-transform is used)")
    ap.add_argument("--src-images",    default=None,
                    help="Src images dir (required unless --load-transform is used)")
    ap.add_argument("--src-ply",       required=True)
    ap.add_argument("--dst-colmap",    default=None,
                    help="Dst COLMAP sparse dir (required unless --load-transform is used)")
    ap.add_argument("--dst-images",    default=None,
                    help="Dst images dir (required unless --load-transform is used)")
    ap.add_argument("--dst-ply",       default=None,
                    help="Mastcam 3DGS PLY — used for ICP if provided "
                         "(preferred over COLMAP sparse); omit to use COLMAP sparse")
    ap.add_argument("--output",        required=True)
    ap.add_argument("--aruco-dict",    default=None,
                    help="Force a single ArUco dict (e.g. DICT_4X4_250); default: auto-detect best")
    ap.add_argument("--aruco-dicts",   nargs="+", default=None,
                    help="Combine multiple dicts (e.g. DICT_4X4_250 DICT_APRILTAG_36h11). "
                         "IDs are namespaced by dict-index × 10000 to avoid collision. "
                         "Overrides --aruco-dict and auto-detect.")
    ap.add_argument("--marker-size",   type=float, default=0.10,
                    help="Physical marker side length in metres (default 0.10)")
    ap.add_argument("--min-views",     type=int, default=3,
                    help="Min camera views for triangulation (default 3)")
    ap.add_argument("--max-reproj",    type=float, default=1.5,
                    help="Max reprojection error in px for inlier observations (default 1.5)")
    ap.add_argument("--marker-tol",    type=float, default=0.35,
                    help="Max fractional deviation from --marker-size (default 0.35)")
    ap.add_argument("--ransac-thresh", type=float, default=None,
                    help="RANSAC inlier threshold in dst coords; default: 0.5*marker_size_dst")
    ap.add_argument("--gravity-axis",  type=int, default=1, choices=[0, 1, 2],
                    help="Vertical axis in DST (mastcam) COLMAP frame: 0=X 1=Y 2=Z (default 1=Y)")
    ap.add_argument("--no-scale",      action="store_true")
    ap.add_argument("--skip-icp",      action="store_true",
                    help="Skip ICP refinement step")
    ap.add_argument("--force-icp",     action="store_true",
                    help="Accept ICP result even if it increases marker residual")
    ap.add_argument("--icp-max-iter",  type=int, default=100)
    ap.add_argument("--icp-inlier-pct",type=float, default=50,
                    help="Percentile of NN distances to keep as ICP inliers (default 50)")
    ap.add_argument("--icp-max-pts",   type=int, default=150_000,
                    help="Max points to subsample from each cloud for ICP (default 150000)")
    ap.add_argument("--save-transform", default=None,
                    help="Path to save the computed (s, R, t) as JSON for reuse. "
                         "All mastcam PLYs share the same COLMAP frame so the same "
                         "transform applies to all of them.")
    ap.add_argument("--load-transform", default=None,
                    help="Path to a previously saved transform JSON. "
                         "Skips ArUco/Umeyama entirely and goes straight to "
                         "optional ICP + PLY transformation.")
    args = ap.parse_args()

    # ── Load saved transform (skip ArUco/Umeyama entirely) ───────────────────
    if args.load_transform:
        with open(args.load_transform) as f:
            tf = json.load(f)
        s       = float(tf["s"])
        R_total = np.array(tf["R"], dtype=np.float64)
        t_total = np.array(tf["t"], dtype=np.float64)
        print(f"=== Loaded transform from {args.load_transform} ===")
        print(f"  Scale:       {s:.6f}")
        print(f"  Translation: {t_total}")
        euler = Rotation.from_matrix(R_total).as_euler("xyz", degrees=True)
        print(f"  Rotation xyz°: {euler}")

        # Optionally re-run ICP for this specific PLY (useful if the dense
        # point cloud geometry differs from the reference PLY used to compute
        # the transform, but can be skipped for speed).
        if not args.skip_icp and args.src_ply:
            dst_ply_path = args.dst_ply or None
            if dst_ply_path and Path(dst_ply_path).exists():
                print(f"\n=== ICP refinement using saved-transform initialisation ===")
                src_pts = read_ply_positions(args.src_ply, max_pts=args.icp_max_pts)
                dst_pts = read_ply_positions(dst_ply_path, max_pts=args.icp_max_pts)
                src_in_dst = (s * (R_total @ src_pts.T)).T + t_total
                hard_cap = dst_pts.std() * 0.3
                tree_dst = KDTree(dst_pts)
                d_init, _ = tree_dst.query(src_in_dst, workers=-1)
                src_in_dst = src_in_dst[d_init < hard_cap]
                print(f"  Hard cap {hard_cap:.4f} → {len(src_in_dst):,} src pts within range")
                if len(src_in_dst) >= 50:
                    R_icp, t_icp = icp_rigid(src_in_dst, dst_pts,
                                              max_iter=args.icp_max_iter,
                                              inlier_pct=args.icp_inlier_pct)
                    R_total = R_icp @ R_total
                    t_total = R_icp @ t_total + t_icp
                    print("  ICP applied on top of loaded transform.")
            else:
                print("  (skipping ICP — no --dst-ply provided or file missing)")

        # Apply and write
        print(f"\n=== Transforming src PLY ===")
        data, hdr, nverts, pnames = read_ply_vertices(args.src_ply)
        print(f"  {nverts:,} vertices  |  "
              f"attrs: {pnames[:6]}{'...' if len(pnames) > 6 else ''}")
        data = transform_gaussians(data, s, R_total, t_total)
        write_ply_vertices(args.output, data, hdr)
        print(f"\nDone → {args.output}")
        return   # ← skip the full ArUco pipeline below

    # ── Validate that COLMAP paths are given when needed ─────────────────────
    for flag, val in [("--src-colmap", args.src_colmap),
                      ("--src-images", args.src_images),
                      ("--dst-colmap", args.dst_colmap),
                      ("--dst-images", args.dst_images)]:
        if val is None:
            print(f"ERROR: {flag} is required when not using --load-transform")
            return

    # ── Load COLMAP ───────────────────────────────────────────────────────────
    print("=== Loading src COLMAP ===")
    src_cams = read_cameras_binary(f"{args.src_colmap}/cameras.bin")
    src_imgs = read_images_binary(f"{args.src_colmap}/images.bin")
    print(f"  {len(src_cams)} cameras, {len(src_imgs)} images")

    print("=== Loading dst COLMAP ===")
    dst_cams = read_cameras_binary(f"{args.dst_colmap}/cameras.bin")
    dst_imgs = read_images_binary(f"{args.dst_colmap}/images.bin")
    print(f"  {len(dst_cams)} cameras, {len(dst_imgs)} images")

    # ── ArUco / AprilTag detection ────────────────────────────────────────────
    if args.aruco_dicts:
        # ── Multi-dictionary mode: combine ArUco + AprilTag (or any mix) ──────
        print(f"\n=== Multi-dict detection: {args.aruco_dicts} ===")
        print(f"  Detecting in src …")
        src_obs, src_legend = detect_multi_dict(
            args.src_images, src_imgs, src_cams,
            args.aruco_dicts, max_reproj=args.max_reproj)
        print(f"  Detecting in dst …")
        dst_obs, dst_legend = detect_multi_dict(
            args.dst_images, dst_imgs, dst_cams,
            args.aruco_dicts, max_reproj=args.max_reproj)
        # Print legend (namespaced ID → dict name + raw ID)
        all_ids = sorted(set(src_obs) | set(dst_obs))
        legend  = {k: src_legend.get(k) or dst_legend.get(k) for k in all_ids}
        print("  Namespace legend:")
        for nsid, info in sorted(legend.items()):
            if info:
                print(f"    ns_id {nsid:6d} → {info[0]}  raw_id={info[1]}")
        dname = "+".join(args.aruco_dicts)   # label for later prints
    else:
        # ── Single-dict mode (original behaviour) ─────────────────────────────
        print(f"\n=== Selecting ArUco dict (probing every 10th src image) ===")
        dname, adict, _ = pick_best_aruco_dict(
            args.src_images, src_imgs, src_cams,
            force_dict=args.aruco_dict, sample_every=10)

        print(f"\n=== Detecting in src ({dname}) ===")
        src_obs = detect_in_dataset(args.src_images, src_imgs, src_cams, adict,
                                    max_reproj=args.max_reproj)

        print(f"\n=== Detecting in dst ({dname}) ===")
        dst_obs = detect_in_dataset(args.dst_images, dst_imgs, dst_cams, adict,
                                    max_reproj=args.max_reproj)

    # ── Triangulate + filter ──────────────────────────────────────────────────
    print(f"\n=== Triangulating (min {args.min_views} views, "
          f"max_reproj {args.max_reproj}px) ===")
    src_3d_raw = triangulate_marker_set(src_obs, args.min_views, args.max_reproj)
    dst_3d_raw = triangulate_marker_set(dst_obs, args.min_views, args.max_reproj)
    print(f"  src triangulated: {sorted(src_3d_raw)}")
    print(f"  dst triangulated: {sorted(dst_3d_raw)}")

    # Estimate expected marker size in each coordinate system from good markers
    def median_side(markers):
        sides = []
        for corners in markers.values():
            sides += [np.linalg.norm(corners[(i+1)%4] - corners[i]) for i in range(4)]
        return np.median(sides) if sides else None

    # Filter each set independently using its own expected size
    # (we know physical size, but COLMAP units are arbitrary → use cross-check)
    # If scale is unknown, use the median side length from the set itself as guide.
    src_median = median_side(src_3d_raw)
    dst_median = median_side(dst_3d_raw)
    print(f"  src median side: {src_median:.5f}  dst median side: {dst_median:.5f}")

    print("  Filtering src:")
    src_3d = filter_markers_by_size(src_3d_raw, src_median, tol=args.marker_tol)
    print("  Filtering dst:")
    dst_3d = filter_markers_by_size(dst_3d_raw, dst_median, tol=args.marker_tol)

    common = sorted(set(src_3d) & set(dst_3d))
    print(f"\n  Common good markers: {common}")
    if len(common) < 2:
        print("ERROR: Need ≥2 common markers after filtering.")
        return

    src_corr = np.vstack([src_3d[m] for m in common])
    dst_corr = np.vstack([dst_3d[m] for m in common])

    # ── Umeyama + RANSAC ──────────────────────────────────────────────────────
    if args.ransac_thresh is None:
        args.ransac_thresh = dst_median * 0.5
    print(f"\n=== RANSAC Umeyama  "
          f"({len(common)} markers → {len(src_corr)} corners, "
          f"thresh={args.ransac_thresh:.4f}) ===")
    s, R, t, inlier_mask = umeyama_ransac(
        src_corr, dst_corr,
        threshold=args.ransac_thresh,
        with_scale=not args.no_scale)

    euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    print(f"  Scale:           {s:.6f}")
    print(f"  Translation:     {t}")
    print(f"  Rotation xyz°:   {euler}")
    print(f"  RANSAC inliers:  {inlier_mask.sum()}/{len(inlier_mask)}")

    src_xfm  = (s * (R @ src_corr.T)).T + t
    residuals = np.linalg.norm(src_xfm - dst_corr, axis=1)
    print(f"  Residual (dst units): "
          f"mean={residuals.mean():.5f}  max={residuals.max():.5f}  "
          f"std={residuals.std():.5f}")
    # estimate real-world residual
    m_per_dst = args.marker_size / dst_median
    print(f"  ≈ real world: mean={residuals.mean()*m_per_dst*100:.1f}cm  "
          f"max={residuals.max()*m_per_dst*100:.1f}cm  "
          f"(1 dst unit = {m_per_dst:.4f} m)")
    for i, mid in enumerate(common):
        r = residuals[i*4:(i+1)*4].mean()
        rw = r * m_per_dst * 100
        status = "INLIER" if inlier_mask[i*4:(i+1)*4].all() else "outlier"
        print(f"    marker {mid:3d}: mean={r:.5f} ({rw:.1f}cm)  [{status}]")

    aruco_mean_res = residuals.mean()   # save for ICP hard-cap

    # ── Scale error analysis ──────────────────────────────────────────────────
    report_scale_errors(src_3d, dst_3d, common,
                        s, R,
                        src_median, dst_median, args.marker_size,
                        gravity_axis=args.gravity_axis)

    # current best transform
    R_total = R.copy()
    t_total = t.copy()

    # ── ICP refinement ────────────────────────────────────────────────────────
    if not args.skip_icp:
        # Prefer dense 3DGS cloud if --dst-ply is given; fall back to COLMAP sparse.
        if args.dst_ply is not None:
            print(f"\n=== ICP refinement on dense 3DGS point clouds "
                  f"(scale fixed at {s:.6f}) ===")
            print(f"  Loading src PLY positions (max {args.icp_max_pts:,}) ...")
            src_pts = read_ply_positions(args.src_ply,
                                         max_pts=args.icp_max_pts)
            print(f"  Loading dst PLY positions (max {args.icp_max_pts:,}) ...")
            dst_pts = read_ply_positions(args.dst_ply,
                                         max_pts=args.icp_max_pts)
        else:
            print(f"\n=== ICP refinement on COLMAP sparse clouds "
                  f"(scale fixed at {s:.6f}) ===")
            src_pts, src_errs = read_points3d_binary(
                f"{args.src_colmap}/points3D.bin")
            dst_pts, dst_errs = read_points3d_binary(
                f"{args.dst_colmap}/points3D.bin")
            # keep low-error reconstructed points
            src_pts = src_pts[src_errs < np.percentile(src_errs, 80)]
            dst_pts = dst_pts[dst_errs < np.percentile(dst_errs, 80)]

        print(f"  src: {len(src_pts):,} pts   dst: {len(dst_pts):,} pts")

        # Apply ArUco transform to src cloud
        src_in_dst = (s * (R_total @ src_pts.T)).T + t_total

        # Hard cap: only ICP-match points within 3× the ArUco marker residual.
        # Prevents wrong cross-matching from dominating ICP.
        hard_cap = max(aruco_mean_res * 3.0, dst_median * 0.2)
        tree_dst  = KDTree(dst_pts)
        d_init, _ = tree_dst.query(src_in_dst, workers=-1)
        src_in_dst = src_in_dst[d_init < hard_cap]
        print(f"  Hard cap {hard_cap:.4f} dst units → "
              f"{len(src_in_dst):,} src pts within range")

        if len(src_in_dst) < 50:
            print("  WARNING: too few src points within hard cap — "
                  "skipping ICP (ArUco result kept).")
        else:
            R_icp, t_icp = icp_rigid(src_in_dst, dst_pts,
                                      max_iter=args.icp_max_iter,
                                      inlier_pct=args.icp_inlier_pct,
                                      verbose=True)

            # Check whether ICP actually improved the marker alignment
            R_cand = R_icp @ R_total
            t_cand = R_icp @ t_total + t_icp
            src_xfm2  = (s * (R_cand @ src_corr.T)).T + t_cand
            res2      = np.linalg.norm(src_xfm2 - dst_corr, axis=1)
            accept = res2.mean() < residuals.mean() * 1.05 or args.force_icp
            if accept:
                R_total = R_cand
                t_total = t_cand
                label = "(forced)" if args.force_icp and res2.mean() >= residuals.mean() * 1.05 else ""
                print(f"\n  Post-ICP marker residual {label}: "
                      f"mean={res2.mean():.5f} ({res2.mean()*m_per_dst*100:.1f}cm)  "
                      f"max={res2.max():.5f} ({res2.max()*m_per_dst*100:.1f}cm)")
            else:
                print(f"\n  ICP degraded marker residual "
                      f"({res2.mean():.5f} > {residuals.mean():.5f}) — "
                      f"reverting to ArUco-only transform.  Use --force-icp to override.")

    # ── Save transform for reuse across PLYs sharing the same COLMAP frame ───
    if args.save_transform:
        tf_data = {
            "s": float(s),
            "R": R_total.tolist(),
            "t": t_total.tolist(),
            "src_colmap": str(args.src_colmap),
            "dst_colmap": str(args.dst_colmap),
        }
        Path(args.save_transform).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_transform, "w") as f:
            json.dump(tf_data, f, indent=2)
        print(f"\n  Transform saved → {args.save_transform}")
        print(f"  (apply to other PLYs with --load-transform {args.save_transform})")

    # ── Transform PLY ─────────────────────────────────────────────────────────
    print(f"\n=== Transforming src PLY ===")
    data, hdr, nverts, pnames = read_ply_vertices(args.src_ply)
    print(f"  {nverts:,} vertices  |  "
          f"attrs: {pnames[:6]}{'...' if len(pnames) > 6 else ''}")

    data = transform_gaussians(data, s, R_total, t_total)
    write_ply_vertices(args.output, data, hdr)
    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
