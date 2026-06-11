# rose-crpnvs — Cross-Reconstruction Photorealistic Novel View Synthesis

A pipeline to quantify cross-reconstruction photorealistic novel view synthesis in reconstructed Lunar analog digital twins.

Evaluate 3D Gaussian Splatting models trained on one camera system (MastCam)
against a reference from a different camera system (GoPro), across two stages:

| Stage | What it does |
|---|---|
| **Pre-processing** (`coverage/`) | Assess dataset capture quality before training via the Coverage Score (CS) |
| **Post-processing** (root scripts) | Align splats across camera systems, render, and compute cross-domain metrics |

Post-processing metrics:

| Comparison | Abbreviation | Description |
|---|---|---|
| Method render vs GoPro control render | **DT** (Digital-Twin) | Both rendered from the same pose; measures appearance accuracy |
| Method render vs actual GoPro photo | **Physical** | Render vs real photograph; measures realism at cross-domain views |

The sweep applies a y-offset (±0.3 m, 7 values) to account for physical
height differences between the two camera rigs, and a fixed depth pull-back
(0.35 m) to prevent near-clipping artefacts.

---

## Directory layout

```
rose-crpnvs/
├── README.md
├── environment.yml              # conda env (Python + lpips + opencv + scipy)
├── coverage/                    # ── PRE-PROCESSING ────────────────────────
│   ├── compute_cs.py            # Coverage Score for a single COLMAP dataset
│   └── warf_sensitivity.py     # Sensitivity analysis for the θ*=30° threshold
├── postprocess/                 # ── POST-PROCESSING ───────────────────────
│   ├── run_pipeline.py          # Single entry point
│   ├── align.py                 # ArUco + Umeyama + ICP splat alignment
│   ├── check_scale.py           # Scale accuracy audit via ArUco/AprilTag markers
│   ├── gen_cameras.py           # COLMAP sparse → cameras_raw.json
│   ├── render_ply.py            # 3DGS renderer (wraps diff_gaussian_rasterization)
│   └── sweep.py                 # y-offset sweep + DT/physical metrics
├── docs/
│   └── warf_sensitivity.md      # WARF threshold sensitivity table + rationale
└── configs/
    ├── example_scene.json       # Template config
    └── scene_1.json             # Verified scene 1 config
```

---

## Quick start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate crpnvs
```

---

## Pre-processing: Coverage Score

Before training any splat, assess whether the dataset has sufficient geometric
coverage to support reliable novel-view synthesis evaluation.

### Coverage Score (`coverage/compute_cs.py`)

Computes WARF and WAPD from the COLMAP sparse reconstruction and combines them
into a Coverage Score CS ∈ [0, 1].

**Reference values** (best observed across GoPro handheld captures):
- WARF_ref = 14.3%  (GoPro Scene 1)
- WAPD_ref = 13.70  (GoPro Scene 2)

```bash
# Single dataset
python coverage/compute_cs.py data/go_pro/scene_1/sparse/0

# Dataset root — auto-resolves sparse/0
python coverage/compute_cs.py data/mastcam/scan_2

# Reduce sample size for faster runs
python coverage/compute_cs.py data/go_pro/scene_3/sparse/0 --sample 10000
```

CS scores across canonical datasets:

| Dataset | WARF (>30°) | WAPD | CS |
|---|---|---|---|
| GoPro Scene 1   | 14.3% | 12.98 | **0.973** |
| GoPro Scene 2   | 11.7% | 13.70 | **0.905** |
| GoPro Scene 3   | 6.3%  | 7.25  | **0.483** |
| MastCam Scene 1 | 3.5%  | 0.21  | **0.061** |
| MastCam Scan 2  | 1.7%  | 0.13  | **0.033** |
| MastCam Scan 3  | 4.7%  | 0.36  | **0.094** |

GoPro min CS (0.483) is 5× higher than MastCam max CS (0.094) — clean separation.

### Threshold sensitivity (`coverage/warf_sensitivity.py`)

Sweeps θ* from 5° to 60° to justify the choice of 30° as the WARF threshold.
See [docs/warf_sensitivity.md](docs/warf_sensitivity.md) for the full separation
table and rationale.

```bash
python coverage/warf_sensitivity.py \
    --data-root /path/to/data \
    --out-dir   ./results
```

`--data-root` must contain `go_pro/scene_{1,2,3}/` and `mastcam/` subdirectories.
Outputs `warf_sensitivity.pdf` and `.png`.

---

## Post-processing

### 2. (Optional) Audit scale accuracy

```bash
# ArUco 4×4 markers, 100 mm (scenes 1 & 2)
python postprocess/check_scale.py \
    --colmap  data/go_pro/scene_1/sparse/0 \
    --images  data/go_pro/scene_1/images \
    --label   "GoPro scene_1"

# AprilTag 36h11, 94 mm (scene 3)
python postprocess/check_scale.py \
    --colmap  data/go_pro/scene_3/sparse/0 \
    --images  data/go_pro/scene_3/images \
    --label   "GoPro scene_3" \
    --aruco-dict DICT_APRILTAG_36h11 \
    --marker-size 0.094
```

The script reports:
- **mpu** (metres-per-COLMAP-unit) — used as the `mpu` value in `configs/*.json`
- **Scale error** — systematic over/under-estimation vs physical marker size
- **Scale spread** — standard deviation of all measured side lengths (triangulation quality)
- **Squareness error** — intra-marker side consistency (per-marker triangulation noise)

Verified results:

| Dataset | Tag type | Markers | mpu | Scale error | Spread |
|---|---|---|---|---|---|
| GoPro scene_1   | ArUco 4×4      | 5 | 0.902 | +0.58% | ±2.85%  |
| Mastcam scene_1 | ArUco 4×4      | 3 | 0.361 | −7.68% | ±21.5%  |
| GoPro scene_2   | ArUco 4×4      | 5 | 1.074 | −3.22% | ±16.6%  |
| Mastcam scene_2 | ArUco 4×4      | 3 | 0.559 | +7.57% | ±22.2%  |
| GoPro scene_3   | AprilTag 36h11 | 3 | 1.317 | −0.04% | ±2.75%  |
| Mastcam scene_3 | AprilTag 36h11 | 1 | 0.593 | −0.18% | ±14.4%  |

### 3. Install the 3DGS renderer

`diff_gaussian_rasterization` must be compiled against your CUDA version:

```bash
cd path/to/gaussian_splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### 4. Align mastcam splats to the GoPro frame

```bash
# Compute transform (mastcam → GoPro COLMAP frame)
python postprocess/align.py \
    --src-colmap  data/go_pro/scene_1/sparse/0 \
    --src-images  data/go_pro/scene_1/images \
    --src-ply     splats/gopro_control.ply \
    --dst-colmap  data/mastcam/scene_1/sparse/0 \
    --dst-images  data/mastcam/scene_1/images \
    --output      splats/gopro_control_aligned.ply \
    --marker-size 0.10 \
    --save-transform results/scene_1_transform.json

# Apply the same transform to all mastcam PLYs
python postprocess/align.py \
    --src-ply     splats/mastcam_3dgs.ply \
    --output      splats/mastcam_3dgs_aligned.ply \
    --load-transform results/scene_1_transform.json \
    --skip-icp
```

Known mpu values:

| Scene | mpu |
|---|---|
| scene_1 | 0.9018 |
| scene_2 | 1.090  |
| scene_3 | 1.066  |

### 5. Run the evaluation pipeline

```bash
# Full pipeline: gen cameras + render + metrics
python postprocess/run_pipeline.py --config configs/scene_1.json

# Skip rendering (renders already exist)
python postprocess/run_pipeline.py --config configs/scene_1.json --skip-render

# Test-set only (llffhold=8)
python postprocess/run_pipeline.py --config configs/scene_1.json --test-only

# Selected methods only
python postprocess/run_pipeline.py --config configs/scene_1.json --methods BAGS MipSplatting
```

Results are written to `out_dir/results_allcams.json` (or `results_testset.json`).

---

## Config reference

All paths resolve in this order: absolute → relative + `base_dir` → relative to config file.
To port to a new machine, change only **`base_dir`** and **`python_bin`**.

| Key | Type | Description |
|---|---|---|
| `scene_name` | str | Human label |
| `base_dir` | path | Root of data tree |
| `gopro_colmap` | path | GoPro COLMAP sparse dir |
| `gopro_images` | path | GoPro photo directory (physical GT) |
| `cameras_raw` | path | Output path for gen_cameras.py |
| `mpu` | float | Metres-per-unit for GoPro COLMAP frame |
| `control_ply` | path | GoPro control 3DGS PLY in metres |
| `methods` | dict | `{method_name: ply_path}` — mastcam PLYs in metres |
| `code_dir` | path | `gaussian_splatting/` source (added to Python path) |
| `python_bin` | str | Python executable with `diff_gaussian_rasterization` |
| `render_width` | int | Render width in pixels (default 960) |
| `render_height` | int | Render height in pixels (default 540) |
| `y_offsets` | list | Y-offset sweep values in metres |
| `depth` | float | Depth pull-back in metres (default 0.35) |
| `test_hold` | int | llffhold for test set (0 = all cameras) |
| `out_dir` | path | Root output directory |

---

## Output structure

```
out_dir/
├── cameras_raw.json
├── cameras_y+0.0.json
├── cameras_y-0.1.json  …
├── gt_gopro_renders/
│   ├── y+0.0/  000000.png …
│   └── …
├── y+0.0/
│   ├── 3DGS/
│   ├── MipSplatting/  …
└── results_allcams.json
    results_testset.json
```

---

## Scene-specific notes

### Scene 1
- GoPro COLMAP: `data/go_pro/scene_1/sparse/0`  (900 cameras)
- Mastcam COLMAP: `data/mastcam/scene_1/scene1_full/sparse/0`
- mpu = 0.9018
- Render resolution: 960×540
- Known good results (BAGS, all cameras, y-offset pooled): DT SSIM ≈ 0.594 / Physical SSIM ≈ 0.542

### Scene 2
- GoPro COLMAP: `data/go_pro/scene_2/sparse/0`  (729 cameras)
- Mastcam COLMAP: `data/mastcam/scan_2/sparse/0`
- mpu = 1.090

### Scene 3
- GoPro COLMAP: `data/go_pro/scene_3/sparse/0`  (961 cameras)
- Mastcam COLMAP: `data/mastcam/scan3/scan3_full/sparse/0`
- mpu = 1.066

---

## Authors & Credits

**Concept, Reconstruction & Quantification:**
Antar Mazumder — PhD Student, Robotics, Dept. of ME, Colorado School of Mines · [antar_mazumder@mines.edu](mailto:antar_mazumder@mines.edu)

**Supervision:**
Dr. Frances (Frankie) Zhu — Asst. Prof., Dept. of ME, Colorado School of Mines · [frankie.zhu@mines.edu](mailto:frankie.zhu@mines.edu)

**Perception Dataset:**
Ryan Hartzell — PhD Student, Robotics, Dept. of ME, Colorado School of Mines · [ryan_hartzell@mines.edu](mailto:ryan_hartzell@mines.edu)

**Team**
| Name | Contribution | Contact |
|------|-------------|---------|
| Alex Robinson | Isaac Sim 5 USD Generation | [alex_robinson@mines.edu](mailto:alex_robinson@mines.edu) |
| Luke Boyd | Fiducial Marker Detection & Triangulation | [lboyd@mines.edu](mailto:lboyd@mines.edu) |
| Neha Sharma | Testbed Scans | [neha_sharma@mines.edu](mailto:neha_sharma@mines.edu) |
| Chance Carnival | Scans | [chance_carnival@mines.edu](mailto:chance_carnival@mines.edu) |

---

## Acknowledgements

1. Dr. Christopher Dreyer and Mines NASA Aspect Research Team
2. Hans Mertens — PhD Candidate, University of Hawaii

Claude (Anthropic) was used as a development assistant to help debug and refine
code in this repository. All design decisions, experimental results, and
intellectual contributions are the work of the authors.
