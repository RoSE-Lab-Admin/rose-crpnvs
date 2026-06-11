# WARF Threshold Sensitivity Analysis

`coverage/warf_sensitivity.py` sweeps θ* from 5° to 60° across all 6 canonical
datasets (GoPro and MastCam Scenes 1–3) to justify the choice of **θ* = 30°**
as the WARF threshold.

## Separation table

| θ* | GoPro range | MastCam range | Gap | Ratio | Separated? |
|---|---|---|---|---|---|
| 5°  | 69.1% – 80.9% | 66.9% – 75.7% | −6.6pp | 0.9× | ✗ |
| 10° | 40.1% – 57.9% | 36.5% – 47.1% | −7.0pp | 0.9× | ✗ |
| 15° | 24.3% – 40.5% | 19.3% – 27.6% | −3.3pp | 0.9× | ✗ |
| 20° | 14.8% – 28.5% |  9.1% – 15.2% | −0.4pp | 1.0× | ✗ |
| 25° |  9.4% – 20.1% |  4.0% –  7.9% | +1.4pp | 1.2× | ✓ |
| **30°** | **6.3% – 14.3%** | **1.7% – 4.7%** | **+1.6pp** | **1.3×** | **✓** |
| 35° |  4.2% – 10.3% |  0.6% –  3.2% | +1.0pp | 1.3× | ✓ |
| 40° |  3.0% –  7.2% |  0.2% –  2.2% | +0.8pp | 1.4× | ✓ |
| 45° |  2.2% –  5.3% |  0.1% –  1.4% | +0.8pp | 1.6× | ✓ |
| 50° |  1.7% –  3.9% |  0.0% –  0.8% | +1.0pp | 2.3× | ✓ |
| 55° |  1.2% –  3.2% |  0.0% –  0.5% | +0.7pp | 2.4× | ✓ |
| 60° |  0.8% –  2.5% |  0.0% –  0.2% | +0.6pp | 3.4× | ✓ |

Gap = GoPro min − MastCam max (positive → no overlap between groups).

## Why 30°

- **Below 25°**: GoPro and MastCam ranges overlap — no reliable discrimination.
- **25°–30°**: clean separation first emerges; 30° gives the **largest absolute gap (+1.6pp)** while keeping WARF values numerically meaningful (6–14% vs 2–5%).
- **Above 30°**: separation holds but absolute WARF values collapse toward zero, making the metric noisy and the reference values poorly grounded.
- The standard photogrammetric criterion (baseline/depth ≥ 0.5 ↔ apex angle ≈ 28°) independently lands in the same region, providing principled justification beyond the empirical result.

## Reproducing

```bash
python coverage/warf_sensitivity.py \
    --data-root /path/to/data \
    --out-dir   ./results
```

Outputs `results/warf_sensitivity.pdf` and `.png`.
