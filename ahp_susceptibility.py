import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio import windows as rio_windows
import warnings
warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WITH_LS_DIR = os.path.join(BASE_DIR, "landslide_dataset", "with_landslide")
OUT_DIR     = os.path.join(BASE_DIR, "ahp_output")
os.makedirs(OUT_DIR, exist_ok=True)

BAND_NAMES = ["B2","B3","B4","B8","B11","B12","NDVI","NDWI","BSI","NBR","slope","aspect"]
BAND_IDX   = {name: i+1 for i, name in enumerate(BAND_NAMES)}  # rasterio is 1-indexed

FACTORS = [
    ("Slope",  "slope",  False),
    ("Aspect", "aspect", False),
    ("NDVI",   "NDVI",   True),
    ("NDWI",   "NDWI",   False),
    ("BSI",    "BSI",    False),
    ("NBR",    "NBR",    True),
]
FACTOR_NAMES = [f[0] for f in FACTORS]
N = len(FACTOR_NAMES)

PC_MATRIX = np.array([
    [1,    3,    2,    3,    4,    5   ],
    [1/3,  1,    1/2,  1,    2,    3   ],
    [1/2,  2,    1,    2,    3,    4   ],
    [1/3,  1,    1/2,  1,    2,    3   ],
    [1/4,  1/2,  1/3,  1/2,  1,    2   ],
    [1/5,  1/3,  1/4,  1/3,  1/2,  1   ],
], dtype=np.float64)

RI_TABLE = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

SUSCEPTIBILITY_CLASSES = {
    1: ("Very Low",  0.00, 0.20),
    2: ("Low",       0.20, 0.40),
    3: ("Moderate",  0.40, 0.60),
    4: ("High",      0.60, 0.80),
    5: ("Very High", 0.80, 1.00),
}

def compute_ahp_weights(matrix):
    n = matrix.shape[0]
    geo_mean   = np.power(np.prod(matrix, axis=1), 1.0 / n)
    weights    = geo_mean / geo_mean.sum()
    col_sums   = matrix.sum(axis=0)
    lambda_max = np.dot(col_sums, weights)
    CI = (lambda_max - n) / (n - 1)
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return weights, lambda_max, CI, CR, RI

def process_single_tif(tif_path, factors, weights, percentiles):
    """
    Processes one TIF patch:
      - reads required bands
      - normalizes using global percentiles
      - applies weighted sum
      - returns (susceptibility_array, profile)
    """
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        nodata  = src.nodata
        score   = None
        valid_mask = None

        for (factor_name, band_name, invert), w in zip(factors, weights):
            band_idx = BAND_IDX[band_name]
            band = src.read(band_idx).astype(np.float32)

            # mask nodata
            if nodata is not None:
                invalid = np.isclose(band, nodata, atol=1.0)
                band[invalid] = np.nan

            # Clip and normalize using pre-computed global percentiles
            p2, p98, vmin, vmax = percentiles[factor_name]
            band = np.clip(band, p2, p98)
            if vmax - vmin > 1e-10:
                norm = (band - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(band)
            if invert:
                norm = 1.0 - norm

            nan_m = np.isnan(norm)
            if score is None:
                score      = np.where(nan_m, 0.0, norm) * w
                valid_mask = ~nan_m
            else:
                score     += np.where(nan_m, 0.0, norm) * w
                valid_mask = valid_mask & (~nan_m)

        score[~valid_mask] = np.nan
    return score, profile

def gather_global_percentiles(tif_files, factors):
    """
    Two-pass: gather p2/p98 across ALL files for each factor band.
    We sample each band to compute percentiles without loading all data.
    """
    print("  Computing global percentiles across all patches...")
    samples = {f[0]: [] for f in factors}
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            nodata = src.nodata
            for factor_name, band_name, _ in factors:
                band_idx = BAND_IDX[band_name]
                band = src.read(band_idx).astype(np.float32)
                if nodata is not None:
                    band[np.isclose(band, nodata, atol=1.0)] = np.nan
                valid = band[~np.isnan(band)]
                # subsample for memory efficiency (every 10th pixel)
                samples[factor_name].append(valid[::10])

    percentiles = {}
    for factor_name, band_name, invert in factors:
        all_vals = np.concatenate(samples[factor_name])
        p2  = float(np.percentile(all_vals, 2))
        p98 = float(np.percentile(all_vals, 98))
        clipped = np.clip(all_vals, p2, p98)
        vmin = float(np.min(clipped))
        vmax = float(np.max(clipped))
        percentiles[factor_name] = (p2, p98, vmin, vmax)
        print(f"    {factor_name:8s} | p2={p2:.4f}  p98={p98:.4f}  invert={invert}")
    return percentiles

def classify_susceptibility(score):
    classes = np.zeros(score.shape, dtype=np.uint8)
    for cls, (label, lo, hi) in SUSCEPTIBILITY_CLASSES.items():
        mask = (score >= lo) & (score <= hi if cls == 5 else score < hi)
        classes[mask] = cls
    classes[np.isnan(score)] = 0
    return classes

def main():
    print("\n" + "="*70)
    print("  AHP SUSCEPTIBILITY MODEL - LANDSLIDE RISK MAPPING, HIMACHAL PRADESH")
    print("="*70)

    # Step 1+2: AHP Weights
    print("\n[1] Computing AHP weights...")
    weights, lambda_max, CI, CR, RI = compute_ahp_weights(PC_MATRIX)

    df_pc = pd.DataFrame(PC_MATRIX, index=FACTOR_NAMES, columns=FACTOR_NAMES)
    print(f"\n  Pairwise Comparison Matrix ({N}x{N}):")
    print(df_pc.round(3).to_string())
    print(f"\n  lambda_max = {lambda_max:.4f}")
    print(f"  CI         = {CI:.4f}")
    print(f"  RI         = {RI:.4f}  (n={N})")
    print(f"  CR         = {CR:.4f}")

    print("\n[2] Consistency Check...")
    consistent = bool(CR < 0.1)
    if consistent:
        print(f"  OK  CR = {CR:.4f}  (< 0.10) -- Matrix is CONSISTENT")
    else:
        print(f"  WARN CR = {CR:.4f}  (>= 0.10) -- Revise pairwise judgements!")

    weight_df = pd.DataFrame({
        "Factor":   FACTOR_NAMES,
        "Band":     [f[1] for f in FACTORS],
        "Weight":   np.round(weights, 4),
        "Weight_%": np.round(weights * 100, 2),
        "Invert":   [f[2] for f in FACTORS],
    })
    print(f"\n  Weight Table:\n{weight_df.to_string(index=False)}")

    csv_path  = os.path.join(OUT_DIR, "ahp_weights.csv")
    json_path = os.path.join(OUT_DIR, "ahp_weights.json")
    weight_df.to_csv(csv_path, index=False)

    ahp_meta = {
        "factors": FACTOR_NAMES,
        "weights": weights.tolist(),
        "weights_pct": (weights * 100).round(2).tolist(),
        "lambda_max": round(float(lambda_max), 6),
        "CI": round(float(CI), 6),
        "RI": round(float(RI), 4),
        "CR": round(float(CR), 6),
        "consistent": consistent,
        "pairwise_matrix": PC_MATRIX.tolist(),
        "susceptibility_classes": {
            str(k): {"label": v[0], "lo": v[1], "hi": v[2]}
            for k, v in SUSCEPTIBILITY_CLASSES.items()
        }
    }
    with open(json_path, "w") as f:
        json.dump(ahp_meta, f, indent=2)
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {json_path}")

    # Step 4: List TIF files
    tif_files = sorted([
        os.path.join(WITH_LS_DIR, f)
        for f in os.listdir(WITH_LS_DIR) if f.endswith(".tif")
    ])
    print(f"\n[3] Found {len(tif_files)} TIF patches")
    for tf in tif_files:
        print(f"  - {os.path.basename(tf)}")

    # Step 5: Global percentiles (two-pass, memory safe)
    print("\n[4] Building normalized factor layers (global stats)...")
    percentiles = gather_global_percentiles(tif_files, FACTORS)

    # Step 6+7: Process each TIF and write individual susceptibility rasters
    print("\n[5] Computing weighted susceptibility overlay per patch...")
    patches_dir = os.path.join(OUT_DIR, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    all_scores   = []
    all_class_dist = {cls: 0 for cls in SUSCEPTIBILITY_CLASSES}
    total_valid  = 0

    for tif_path in tif_files:
        site = os.path.basename(tif_path).replace(".tif", "")
        score, profile = process_single_tif(tif_path, FACTORS, weights, percentiles)

        s_valid = score[~np.isnan(score)]
        if len(s_valid) > 0:
            all_scores.extend(s_valid[::5])  # subsample for summary stats
        print(f"  {site}: range=[{np.nanmin(score):.3f}, {np.nanmax(score):.3f}]  mean={np.nanmean(score):.3f}")

        # Classify
        cls_arr = classify_susceptibility(score)
        for cls_id in SUSCEPTIBILITY_CLASSES:
            all_class_dist[cls_id] += int((cls_arr == cls_id).sum())
        total_valid += int((cls_arr > 0).sum())

        # Save susceptibility patch
        base_profile = profile.copy()
        base_profile.update({"count": 1, "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256})

        susc_path = os.path.join(patches_dir, f"{site}_susceptibility.tif")
        p_susc = base_profile.copy()
        p_susc.update({"dtype": "float32", "nodata": -9999.0})
        susc_out = np.where(np.isnan(score), -9999.0, score).astype(np.float32)
        with rasterio.open(susc_path, "w", **p_susc) as dst:
            dst.write(susc_out, 1)
            dst.update_tags(
                DESCRIPTION="AHP Weighted Susceptibility [0-1]",
                CR=f"{CR:.4f}",
                WEIGHTS=",".join(f"{w:.4f}" for w in weights),
            )

        cls_path = os.path.join(patches_dir, f"{site}_susceptibility_class.tif")
        p_cls = base_profile.copy()
        p_cls.update({"dtype": "uint8", "nodata": 0})
        with rasterio.open(cls_path, "w", **p_cls) as dst:
            dst.write(cls_arr, 1)
            dst.update_tags(DESCRIPTION="AHP Class 1=VeryLow..5=VeryHigh")

    # Summary stats
    all_scores_arr = np.array(all_scores, dtype=np.float32)
    print(f"\n[6] Overall Susceptibility Statistics (all {len(tif_files)} patches):")
    print(f"  Min   : {all_scores_arr.min():.4f}")
    print(f"  Max   : {all_scores_arr.max():.4f}")
    print(f"  Mean  : {all_scores_arr.mean():.4f}")
    print(f"  Std   : {all_scores_arr.std():.4f}")

    print(f"\n  Class Distribution ({total_valid:,} valid pixels total):")
    for cls, (label, lo, hi) in SUSCEPTIBILITY_CLASSES.items():
        count = all_class_dist[cls]
        pct   = 100.0 * count / total_valid if total_valid > 0 else 0
        bar   = "#" * int(pct / 2)
        print(f"    {cls}. {label:10s} [{lo:.2f}-{hi:.2f}]: {count:>8,} px  ({pct:5.1f}%)  {bar}")

    # Save summary stats to JSON
    summary_path = os.path.join(OUT_DIR, "susceptibility_summary.json")
    summary = {
        "n_patches": len(tif_files),
        "total_valid_pixels": int(total_valid),
        "susceptibility_stats": {
            "min":  float(all_scores_arr.min()),
            "max":  float(all_scores_arr.max()),
            "mean": float(all_scores_arr.mean()),
            "std":  float(all_scores_arr.std()),
        },
        "class_distribution": {
            SUSCEPTIBILITY_CLASSES[cls][0]: {
                "class_id": cls,
                "pixel_count": all_class_dist[cls],
                "pct": round(100.0 * all_class_dist[cls] / total_valid, 2) if total_valid > 0 else 0.0,
                "range": [SUSCEPTIBILITY_CLASSES[cls][1], SUSCEPTIBILITY_CLASSES[cls][2]],
            }
            for cls in SUSCEPTIBILITY_CLASSES
        },
        "ahp_weights": {
            name: float(w) for name, w in zip(FACTOR_NAMES, weights)
        },
        "CR": round(float(CR), 6),
        "consistent": consistent,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")

    print("\n" + "="*70)
    print("  PHASE 1 COMPLETE")
    print("="*70)
    print(f"\n  ahp_output/")
    print(f"  +-- ahp_weights.csv")
    print(f"  +-- ahp_weights.json")
    print(f"  +-- susceptibility_summary.json")
    print(f"  +-- patches/  ({len(tif_files)*2} rasters: *_susceptibility.tif + *_class.tif)")
    print(f"\n  AHP Weights:")
    for name, w in zip(FACTOR_NAMES, weights):
        bar = "#" * int(w * 40)
        print(f"    {name:8s} {w:.4f} ({w*100:5.2f}%)  {bar}")
    print(f"\n  CR = {CR:.4f}  ->  {'CONSISTENT' if consistent else 'INCONSISTENT'}")
    print()

if __name__ == "__main__":
    main()
