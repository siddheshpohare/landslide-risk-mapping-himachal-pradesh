"""
phase3_validation.py
====================
Phase 3 -- AHP Susceptibility Validation
Himachal Pradesh Landslide Risk Mapping Project

Workflow:
  Step 1 -- Alignment Verification
  Step 2 -- Ground-Truth Binary Mask Generation (dNBR/dNDVI change detection)
  Step 3 -- Pixel Extraction & Cleaning
  Step 4 -- Global Metrics (ROC-AUC, Confusion Matrix)
  Step 5 -- Per-Site Breakdown
  Step 6 -- Visualizations
  Step 7 -- Optional Bhusanket/Inventory Point Cross-Check

Outputs (validation_output/):
  alignment_report.json
  roc_auc.json
  confusion_matrix.csv
  per_site_auc.csv
  bhusanket_crosscheck.csv
  roc_curve.png
  per_site_auc_bar.png
  weakest_site_comparison.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PATCHES_DIR    = os.path.join(BASE_DIR, "ahp_output", "patches")
WITH_LS_DIR    = os.path.join(BASE_DIR, "landslide_dataset", "with_landslide")
WITHOUT_LS_DIR = os.path.join(BASE_DIR, "landslide_dataset", "without_landslide")
OUT_DIR        = os.path.join(BASE_DIR, "validation_output")
os.makedirs(OUT_DIR, exist_ok=True)

BAND_NDVI = 7   # 1-indexed rasterio band
BAND_NBR  = 10  # 1-indexed rasterio band

CHANGE_SIGMA = 1.5

HIGH_PLUS_THRESHOLD     = 4
MODERATE_PLUS_THRESHOLD = 3

SITE_MAP = {
    "Chamba_Ravi":     "Chamba",
    "Kangra_Banganga": "Kangra",
    "Kinnaur_Sutlej":  "Kinnaur",
    "Kullu_Beas":      "Kullu",
    "Lahaul_Spiti":    "Lahaul & Spiti",
    "Mandi_NH3":       "Mandi",
    "Rampur_Sutlej":   "Shimla",
    "Shimla_NH5":      "Shimla",
    "Sirmaur_Giri":    "Sirmaur",
    "Solan_Shivalik":  "Solan",
}

WEAK_FIT_AUC_THRESHOLD = 0.65

# Fix-specific constants
TARGET_SITES = {"Lahaul_Spiti", "Sirmaur_Giri", "Rampur_Sutlej"}
LS_FIXED_THRESHOLDS = [0.05, 0.04, 0.03]
N_BASELINE_SAMPLES  = 1000
RNG_SEED = 42
GLOBAL_AUC_V1 = 0.6239120774838096

def _site_key(filename):
    """Extract site key from a patch filename."""
    basename = os.path.basename(filename)
    for key in SITE_MAP:
        if basename.startswith(key):
            return key
    return None
def discover_sites():
    """Return list of (site_key, susc_path, susc_class_path, post_path, pre_path)."""
    sites = []
    for f in sorted(os.listdir(PATCHES_DIR)):
        if f.endswith("_susceptibility.tif") and "_class" not in f:
            key = _site_key(f)
            if key is None:
                continue
            susc_path       = os.path.join(PATCHES_DIR, f)
            susc_class_path = os.path.join(PATCHES_DIR,
                                           f.replace("_susceptibility.tif",
                                                     "_susceptibility_class.tif"))
            post_tif = os.path.join(WITH_LS_DIR,    f"{key}_POST_with_landslide.tif")
            pre_tif  = os.path.join(WITHOUT_LS_DIR, f"{key}_PRE_without_landslide.tif")

            if not os.path.exists(susc_class_path):
                print(f"  [WARN] Missing class TIF for {key}, skipping.")
                continue
            if not os.path.exists(post_tif) or not os.path.exists(pre_tif):
                print(f"  [WARN] Missing PRE/POST TIF for {key}, skipping.")
                continue
            sites.append((key, susc_path, susc_class_path, post_tif, pre_tif))
    return sites


# ---------------------------------------------------------------------------
# Step 1 -- Alignment Verification
# ---------------------------------------------------------------------------
def verify_alignment(susc_path, mask_arr, mask_profile, site_key):
    """
    Compare spatial properties of susceptibility vs GT mask.
    Reproject mask to susceptibility grid if needed.
    Returns aligned_mask_array only (status is discarded for simplicity in fix script).
    """
    with rasterio.open(susc_path) as src:
        susc_crs       = src.crs
        susc_transform = src.transform
        susc_width     = src.width
        susc_height    = src.height

    mask_crs       = mask_profile["crs"]
    mask_transform = mask_profile["transform"]
    mask_width     = mask_profile["width"]
    mask_height    = mask_profile["height"]

    crs_match  = (str(susc_crs) == str(mask_crs))
    size_match = (susc_width == mask_width) and (susc_height == mask_height)
    tol = 1e-9
    res_match = (
        abs(susc_transform.a - mask_transform.a) < tol and
        abs(susc_transform.e - mask_transform.e) < tol and
        abs(susc_transform.c - mask_transform.c) < tol and
        abs(susc_transform.f - mask_transform.f) < tol
    )

    if crs_match and size_match and res_match:
        return mask_arr

    # Reproject/resample mask to susceptibility grid (nearest-neighbor, categorical)
    print(f"    [{site_key}] Grid mismatch -- resampling mask to susceptibility grid ...")
    resampled = np.full((susc_height, susc_width), 255, dtype=np.uint8)
    reproject(
        source=mask_arr,
        destination=resampled,
        src_transform=mask_transform,
        src_crs=mask_crs,
        dst_transform=susc_transform,
        dst_crs=susc_crs,
        resampling=Resampling.nearest,
    )
    return resampled


# ---------------------------------------------------------------------------
# Step 2 -- Ground-Truth Binary Mask Generation (Change Detection)
# ---------------------------------------------------------------------------
def extract_pixels(site_key, susc_path, susc_class_path, mask_arr):
    """
    Load susceptibility score + class rasters, intersect with GT mask.
    Returns DataFrame with columns: site, score, cls, gt.
    Valid pixels: score != NoData AND class != NoData AND mask != 255 (nodata).
    """
    with rasterio.open(susc_path) as src:
        score      = src.read(1).astype(np.float32)
        susc_nodata = src.nodata if src.nodata is not None else -9999.0

    with rasterio.open(susc_class_path) as src:
        cls       = src.read(1).astype(np.uint8)
        cls_nodata = int(src.nodata) if src.nodata is not None else 0

    valid = (
        (score != susc_nodata) &
        np.isfinite(score) &
        (cls != cls_nodata) &
        (mask_arr != 255)
    )

    df = pd.DataFrame({
        "site":  site_key,
        "score": score[valid].ravel(),
        "cls":   cls  [valid].ravel(),
        "gt":    mask_arr[valid].ravel().astype(np.uint8),
    })
    return df


# ---------------------------------------------------------------------------
# Step 4 -- Metrics Computation
# ---------------------------------------------------------------------------
def compute_metrics(scores, labels, cls_arr, threshold_cls=HIGH_PLUS_THRESHOLD):
    """Compute AUC + confusion matrix metrics. Returns dict."""
    results = {}
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())

    # ROC-AUC
    if n_pos == 0 or n_neg == 0:
        results["auc"]  = None
        results["note"] = "Only one class present -- AUC undefined"
    else:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        results["auc"]            = float(auc)
        results["roc_fpr"]        = fpr.tolist()
        results["roc_tpr"]        = tpr.tolist()
        results["roc_thresholds"] = thresholds.tolist()

    # Confusion matrix at class threshold
    pred = (cls_arr >= threshold_cls).astype(np.uint8)
    gt   = labels.astype(np.uint8)

    cm = confusion_matrix(gt, pred, labels=[0, 1])
    results["confusion_matrix"] = cm.tolist()
    results["threshold_cls"]    = threshold_cls
    results["accuracy"]         = float(accuracy_score (gt, pred))
    results["precision"]        = float(precision_score(gt, pred, zero_division=0))
    results["recall"]           = float(recall_score   (gt, pred, zero_division=0))
    results["f1"]               = float(f1_score       (gt, pred, zero_division=0))
    results["n_pixels"]         = int(len(labels))
    results["n_positive"]       = n_pos
    results["n_negative"]       = n_neg
    return results


# ---------------------------------------------------------------------------
# Step 5 -- Per-Site Breakdown
# ---------------------------------------------------------------------------
def per_site_breakdown(df, threshold_cls=HIGH_PLUS_THRESHOLD):
    """Compute per-site AUC and confusion matrix stats. Returns DataFrame."""
    records = []
    for site_key, grp in df.groupby("site"):
        m = compute_metrics(
            grp["score"].values,
            grp["gt"].values,
            grp["cls"].values,
            threshold_cls=threshold_cls,
        )
        records.append({
            "site":       site_key,
            "district":   SITE_MAP.get(site_key, "Unknown"),
            "AUC":        m.get("auc"),
            "precision":  m.get("precision"),
            "recall":     m.get("recall"),
            "f1":         m.get("f1"),
            "accuracy":   m.get("accuracy"),
            "n_pixels":   m.get("n_pixels"),
            "n_positive": m.get("n_positive"),
            "n_negative": m.get("n_negative"),
            "weak_fit":   (m.get("auc") is not None and
                           m.get("auc") < WEAK_FIT_AUC_THRESHOLD),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 6 -- Visualizations
# ---------------------------------------------------------------------------
def plot_roc_curve_v2(global_scores, global_labels, per_site_df, out_path):
    """Combined figure: ROC curve (left) + per-site AUC bars (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Phase 3 (v2 Masks) -- Susceptibility Validation  |  Himachal Pradesh",
        fontsize=14, fontweight="bold",
    )

    # -- Panel A: ROC Curve --------------------------------------------------
    ax = axes[0]
    fpr, tpr, _ = roc_curve(global_labels, global_scores)
    auc = roc_auc_score(global_labels, global_scores)

    ax.plot(fpr, tpr, color="#d7191c", lw=2.5,
            label=f"Global ROC (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#d7191c")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6,
            label="Random classifier (AUC = 0.50)")
    ax.axhline(0.80, color="#4393c3", lw=1.0, linestyle=":", alpha=0.8,
               label="TPR = 0.80 reference")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("A  Combined ROC Curve (All Sites)", fontsize=12,
                 loc="left", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_aspect("equal")

    quality = (
        "Excellent (>=0.90)" if auc >= 0.90 else
        "Good (>=0.80)"      if auc >= 0.80 else
        "Acceptable (>=0.70)" if auc >= 0.70 else
        "Marginal (>=0.60)"  if auc >= 0.60 else
        "Poor (<0.60)"
    )
    ax.text(0.97, 0.04, f"AUC = {auc:.4f}\n{quality}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff7bc",
                      edgecolor="#d7191c", alpha=0.92))

    # -- Panel B: Per-site AUC bars -----------------------------------------
    ax2 = axes[1]
    df_s = per_site_df.dropna(subset=["AUC"]).sort_values("AUC", ascending=True)
    bar_colors = ["#d7191c" if w else "#2166ac" for w in df_s["weak_fit"]]
    bars = ax2.barh(df_s["site"], df_s["AUC"], color=bar_colors,
                    edgecolor="#333", linewidth=0.6, height=0.65)

    ax2.axvline(WEAK_FIT_AUC_THRESHOLD, color="#d7191c", lw=1.5, linestyle="--",
                label=f"Weak-fit threshold ({WEAK_FIT_AUC_THRESHOLD})")
    ax2.axvline(0.70, color="#f4a582", lw=1.2, linestyle=":",
                label="Industry minimum (0.70)")
    ax2.axvline(0.80, color="#92c5de", lw=1.2, linestyle=":",
                label="Good (0.80)")

    for bar, auc_val in zip(bars, df_s["AUC"]):
        ax2.text(auc_val + 0.007, bar.get_y() + bar.get_height() / 2,
                 f"{auc_val:.3f}", va="center", fontsize=9, fontweight="bold")

    ax2.set_xlim([0, 1.14])
    ax2.set_xlabel("AUC Score", fontsize=11)
    ax2.set_title("B  Per-Site AUC (sorted ascending)", fontsize=12,
                  loc="left", fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.35)

    weak_p = mpatches.Patch(facecolor="#d7191c", edgecolor="#333",
                             label=f"Weak fit (AUC<{WEAK_FIT_AUC_THRESHOLD})")
    good_p = mpatches.Patch(facecolor="#2166ac", edgecolor="#333",
                             label="Acceptable")
    ax2.legend(handles=[weak_p, good_p], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

def plot_per_site_auc_bar_v2(per_site_df, out_path):
    """Standalone horizontal bar chart -- per-site AUC."""
    df_s = per_site_df.dropna(subset=["AUC"]).sort_values("AUC", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#d7191c" if w else "#2166ac" for w in df_s["weak_fit"]]
    bars   = ax.barh(df_s["site"], df_s["AUC"], color=colors,
                     edgecolor="#333", linewidth=0.7, height=0.6)

    ax.axvline(WEAK_FIT_AUC_THRESHOLD, color="#d7191c", lw=1.8, linestyle="--",
               label=f"Weak-fit ({WEAK_FIT_AUC_THRESHOLD})")
    ax.axvline(0.70, color="#f4a582", lw=1.4, linestyle=":",
               label="Industry min (0.70)")
    ax.axvline(0.80, color="#92c5de", lw=1.4, linestyle=":",
               label="Good (0.80)")

    for bar, (_, row) in zip(bars, df_s.iterrows()):
        ls_pct = 100 * row["n_positive"] / row["n_pixels"] if row["n_pixels"] else 0
        lbl = f"{row['AUC']:.3f}  |  n={row['n_pixels']:,}  |  LS={ls_pct:.1f}%"
        ax.text(row["AUC"] + 0.009, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=8.5)

    ax.set_xlim([0, 1.20])
    ax.set_xlabel("AUC Score", fontsize=12)
    ax.set_title(
        "Per-Site ROC-AUC -- AHP Susceptibility Validation\n"
        "Himachal Pradesh Landslide Risk Mapping",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    weak_p = mpatches.Patch(facecolor="#d7191c", label=f"Weak fit (AUC<{WEAK_FIT_AUC_THRESHOLD})")
    good_p = mpatches.Patch(facecolor="#2166ac", label="Acceptable")
    ax.legend(handles=[weak_p, good_p, *ax.get_legend_handles_labels()[0]],
              fontsize=8.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

def build_original_mask(post_tif, pre_tif):
    with rasterio.open(post_tif) as src:
        ndvi_post = src.read(BAND_NDVI).astype(np.float32)
        nbr_post  = src.read(BAND_NBR ).astype(np.float32)
        profile   = src.profile.copy()
    with rasterio.open(pre_tif) as src:
        ndvi_pre = src.read(BAND_NDVI).astype(np.float32)
        nbr_pre  = src.read(BAND_NBR ).astype(np.float32)
    dnbr  = nbr_pre  - nbr_post
    dndvi = ndvi_pre - ndvi_post
    t_nbr  = float(dnbr .mean() + CHANGE_SIGMA * dnbr .std())
    t_ndvi = float(dndvi.mean() + CHANGE_SIGMA * dndvi.std())
    mask   = ((dnbr > t_nbr) | (dndvi > t_ndvi)).astype(np.uint8)
    profile.update(dtype="uint8", count=1, nodata=255)
    return mask, profile


def build_lahaul_mask_v2(post_tif, pre_tif):
    with rasterio.open(post_tif) as src:
        ndvi_post = src.read(BAND_NDVI).astype(np.float32)
        profile   = src.profile.copy()
    with rasterio.open(pre_tif) as src:
        ndvi_pre = src.read(BAND_NDVI).astype(np.float32)
    dndvi = ndvi_pre - ndvi_post
    used_t = LS_FIXED_THRESHOLDS[-1]
    for t in LS_FIXED_THRESHOLDS:
        mask   = (dndvi > t).astype(np.uint8)
        ls_pct = float(100.0 * mask.mean())
        print("    [LS_v2] dNDVI threshold = " + str(round(t,2)) + "  ->  " + str(round(ls_pct,2)) + "% LS pixels")
        if ls_pct >= 2.0:
            used_t = t
            profile.update(dtype="uint8", count=1, nodata=255)
            return mask, profile, ls_pct, used_t
    mask   = (dndvi > used_t).astype(np.uint8)
    ls_pct = float(100.0 * mask.mean())
    print("    [LS_v2] WARN: best threshold " + str(used_t) + " gives " + str(round(ls_pct,2)) + "% -- using it.")
    profile.update(dtype="uint8", count=1, nodata=255)
    return mask, profile, ls_pct, used_t


def build_seasonal_norm_mask_v2(post_tif, pre_tif, site_key):
    with rasterio.open(post_tif) as src:
        ndvi_post = src.read(BAND_NDVI).astype(np.float32)
        nbr_post  = src.read(BAND_NBR ).astype(np.float32)
        profile   = src.profile.copy()
    with rasterio.open(pre_tif) as src:
        ndvi_pre = src.read(BAND_NDVI).astype(np.float32)
        nbr_pre  = src.read(BAND_NBR ).astype(np.float32)
    raw_dnbr  = nbr_pre  - nbr_post
    raw_dndvi = ndvi_pre - ndvi_post
    h, w  = raw_dnbr.shape
    rng   = np.random.default_rng(RNG_SEED)
    idx   = rng.choice(h * w, size=min(N_BASELINE_SAMPLES, h * w), replace=False)
    rs, cs = np.unravel_index(idx, (h, w))
    samp_dnbr  = raw_dnbr [rs, cs]
    samp_dndvi = raw_dndvi[rs, cs]
    fin = np.isfinite(samp_dnbr) & np.isfinite(samp_dndvi)
    offset_nbr  = float(np.mean(samp_dnbr [fin]))
    offset_ndvi = float(np.mean(samp_dndvi[fin]))
    print("    [" + site_key + "] Seasonal offsets: dNBR=" + str(round(offset_nbr,4)) + ", dNDVI=" + str(round(offset_ndvi,4)))
    corr_dnbr  = raw_dnbr  - offset_nbr
    corr_dndvi = raw_dndvi - offset_ndvi
    t_nbr_c  = float(corr_dnbr .mean() + CHANGE_SIGMA * corr_dnbr .std())
    t_ndvi_c = float(corr_dndvi.mean() + CHANGE_SIGMA * corr_dndvi.std())
    print("    [" + site_key + "] Corrected thresholds: dNBR=" + str(round(t_nbr_c,4)) + ", dNDVI=" + str(round(t_ndvi_c,4)))
    mask   = ((corr_dnbr > t_nbr_c) | (corr_dndvi > t_ndvi_c)).astype(np.uint8)
    profile.update(dtype="uint8", count=1, nodata=255)
    ls_pct = float(100.0 * mask.mean())
    diag = {
        "raw_dnbr_mean":     float(raw_dnbr.mean()),
        "raw_dndvi_mean":    float(raw_dndvi.mean()),
        "offset_nbr":        offset_nbr,
        "offset_ndvi":       offset_ndvi,
        "corr_dnbr_thresh":  t_nbr_c,
        "corr_dndvi_thresh": t_ndvi_c,
        "ls_pct":            ls_pct,
    }
    return mask, profile, ls_pct, diag


def save_mask_tif(mask_arr, reference_tif, out_path):
    with rasterio.open(reference_tif) as src:
        prof = src.profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(mask_arr[np.newaxis, :, :])
    print("    Saved -> " + out_path)


def main():
    print("\n" + "="*70)
    print("  PHASE 3 MASK FIXES (v2)")
    print("  Fix 1: Lahaul_Spiti -> NDVI-only fixed threshold (no dNBR)")
    print("  Fix 2: Sirmaur_Giri, Rampur_Sutlej -> Seasonal normalisation")
    print("="*70)
    sites_data = discover_sites()
    if not sites_data:
        raise RuntimeError("No valid sites found.")
    site_dict = {k: (sp, scp, pt, pret) for k, sp, scp, pt, pret in sites_data}
    print("\n[0] Found " + str(len(sites_data)) + " sites.")
    existing_csv = os.path.join(OUT_DIR, "per_site_auc.csv")
    if not os.path.exists(existing_csv):
        raise FileNotFoundError("Missing " + existing_csv + ". Run phase3_validation.py first.")
    df_existing = pd.read_csv(existing_csv)
    auc_old_map = dict(zip(df_existing["site"], df_existing["AUC"]))
    ls_old_map  = {r["site"]: 100*r["n_positive"]/r["n_pixels"]
                   for _, r in df_existing.iterrows() if r["n_pixels"]}

    # Step 1: Pixel extraction for unchanged 7 sites
    print("\n[1] Extracting pixels from unchanged sites (original masks) ...")
    all_pxl = []
    for key, susc_path, susc_class_path, post_tif, pre_tif in sites_data:
        if key in TARGET_SITES:
            continue
        mask, profile = build_original_mask(post_tif, pre_tif)
        mask = verify_alignment(susc_path, mask, profile, key)
        df_px = extract_pixels(key, susc_path, susc_class_path, mask)
        lspct = 100*df_px["gt"].mean()
        print("    " + key + "  LS=" + str(round(lspct,1)) + "%  pixels=" + str(len(df_px)))
        all_pxl.append(df_px)

    # Step 2: Fix 1 - Lahaul_Spiti
    print("\n[2] Fix 1 -- Lahaul_Spiti: NDVI-only fixed threshold ...")
    key_ls = "Lahaul_Spiti"
    sp_ls, scp_ls, post_ls, pre_ls = site_dict[key_ls]
    row_v1_ls = df_existing[df_existing["site"] == key_ls].iloc[0]
    ls_pct_v1_ls = 100*row_v1_ls["n_positive"]/row_v1_ls["n_pixels"]
    mask_ls, prof_ls, ls_pct_ls, used_t = build_lahaul_mask_v2(post_ls, pre_ls)
    save_mask_tif(mask_ls, post_ls, os.path.join(OUT_DIR, "lahaul_spiti_mask_v2.tif"))
    mask_ls_aln = verify_alignment(sp_ls, mask_ls, prof_ls, key_ls)
    df_ls = extract_pixels(key_ls, sp_ls, scp_ls, mask_ls_aln)
    n_ls_pos = int(df_ls["gt"].sum())
    print("    Before: " + str(round(ls_pct_v1_ls,2)) + "%  |  After: " + str(round(ls_pct_ls,2)) + "%  (threshold=" + str(used_t) + ")")
    print("    Valid pixels: " + str(len(df_ls)) + "  LS: " + str(n_ls_pos) + " (" + str(round(100*df_ls["gt"].mean(),1)) + "%)")
    all_pxl.append(df_ls)

    # Step 3: Fix 2 - Sirmaur_Giri
    print("\n[3] Fix 2 -- Sirmaur_Giri: seasonal normalisation ...")
    key_sg = "Sirmaur_Giri"
    sp_sg, scp_sg, post_sg, pre_sg = site_dict[key_sg]
    row_v1_sg = df_existing[df_existing["site"] == key_sg].iloc[0]
    ls_pct_v1_sg = 100*row_v1_sg["n_positive"]/row_v1_sg["n_pixels"]
    mask_sg, prof_sg, ls_pct_sg, diag_sg = build_seasonal_norm_mask_v2(post_sg, pre_sg, key_sg)
    save_mask_tif(mask_sg, post_sg, os.path.join(OUT_DIR, "sirmaur_giri_mask_v2.tif"))
    mask_sg_aln = verify_alignment(sp_sg, mask_sg, prof_sg, key_sg)
    df_sg = extract_pixels(key_sg, sp_sg, scp_sg, mask_sg_aln)
    print("    Before: " + str(round(ls_pct_v1_sg,2)) + "%  |  After: " + str(round(ls_pct_sg,2)) + "%")
    print("    Valid pixels: " + str(len(df_sg)) + "  LS: " + str(int(df_sg["gt"].sum())) + " (" + str(round(100*df_sg["gt"].mean(),1)) + "%)")
    all_pxl.append(df_sg)

    # Step 4: Fix 2 - Rampur_Sutlej
    print("\n[4] Fix 2 -- Rampur_Sutlej: seasonal normalisation ...")
    key_rs = "Rampur_Sutlej"
    sp_rs, scp_rs, post_rs, pre_rs = site_dict[key_rs]
    row_v1_rs = df_existing[df_existing["site"] == key_rs].iloc[0]
    ls_pct_v1_rs = 100*row_v1_rs["n_positive"]/row_v1_rs["n_pixels"]
    mask_rs, prof_rs, ls_pct_rs, diag_rs = build_seasonal_norm_mask_v2(post_rs, pre_rs, key_rs)
    save_mask_tif(mask_rs, post_rs, os.path.join(OUT_DIR, "rampur_sutlej_mask_v2.tif"))
    mask_rs_aln = verify_alignment(sp_rs, mask_rs, prof_rs, key_rs)
    df_rs = extract_pixels(key_rs, sp_rs, scp_rs, mask_rs_aln)
    print("    Before: " + str(round(ls_pct_v1_rs,2)) + "%  |  After: " + str(round(ls_pct_rs,2)) + "%")
    print("    Valid pixels: " + str(len(df_rs)) + "  LS: " + str(int(df_rs["gt"].sum())) + " (" + str(round(100*df_rs["gt"].mean(),1)) + "%)")
    all_pxl.append(df_rs)

    # Step 5: Merge and compute metrics
    print("\n[5] Merging and computing v2 per-site AUC ...")
    df_all_v2     = pd.concat(all_pxl, ignore_index=True)
    per_site_v2   = per_site_breakdown(df_all_v2)
    global_scores = df_all_v2["score"].values
    global_labels = df_all_v2["gt"].values
    global_cls    = df_all_v2["cls"].values
    m_high_v2 = compute_metrics(global_scores, global_labels, global_cls, HIGH_PLUS_THRESHOLD)
    m_mod_v2  = compute_metrics(global_scores, global_labels, global_cls, MODERATE_PLUS_THRESHOLD)
    global_auc_v2 = m_high_v2["auc"]
    delta_g = global_auc_v2 - GLOBAL_AUC_V1
    quality = ("EXCELLENT(>=0.90)" if global_auc_v2>=0.90 else "GOOD(>=0.80)" if global_auc_v2>=0.80
               else "ACCEPTABLE(>=0.70)" if global_auc_v2>=0.70 else "MARGINAL(>=0.60)" if global_auc_v2>=0.60
               else "POOR(<0.60)")

    print("\n  Site                         AUC_v1    AUC_v2   LS%_v1   LS%_v2    Delta")
    print("  " + "-"*76)
    summary_rows = []
    for _, row in per_site_v2.sort_values("AUC", ascending=False, na_position="last").iterrows():
        s = row["site"]; an = row["AUC"]; ao = auc_old_map.get(s, 0)
        ln = 100*row["n_positive"]/row["n_pixels"] if row["n_pixels"] else 0
        lo = ls_old_map.get(s, 0)
        d  = (an - ao) if (an is not None and ao is not None) else None
        ds = ("+" if d > 0 else "") + str(round(d,4)) if d is not None else "N/A"
        tag = " [v2]" if s in TARGET_SITES else ""
        print("  " + (s+tag)[:27].ljust(27) + " " + str(round(ao,4)).rjust(8) + "  " + str(round(an,4) if an else 0).rjust(8) + "  " + (str(round(lo,2))+"%").rjust(8) + "  " + (str(round(ln,2))+"%").rjust(8) + "  " + ds)
        summary_rows.append({"site":s,"auc_v1":ao,"auc_v2":an,"ls_pct_v1":lo,"ls_pct_v2":ln,"delta_auc":d,"target":s in TARGET_SITES})

    print("\n  Global AUC: v1=" + str(round(GLOBAL_AUC_V1,4)) + "  ->  v2=" + str(round(global_auc_v2,4)) + "  (" + ("+" if delta_g>0 else "") + str(round(delta_g,4)) + ")  [" + quality + "]")
    print("  High+ Prec=" + str(round(m_high_v2["precision"],3)) + "  Rec=" + str(round(m_high_v2["recall"],3)) + "  F1=" + str(round(m_high_v2["f1"],3)))
    print("  Mod+  Prec=" + str(round(m_mod_v2["precision"],3))  + "  Rec=" + str(round(m_mod_v2["recall"],3))  + "  F1=" + str(round(m_mod_v2["f1"],3)))

    per_site_v2.to_csv(os.path.join(OUT_DIR,"per_site_auc_v2.csv"), index=False)
    print("\n  Saved -> per_site_auc_v2.csv")

    roc_v2 = {
        "global_auc_v1": GLOBAL_AUC_V1, "global_auc_v2": global_auc_v2,
        "delta_global_auc": delta_g,
        "n_total_pixels": int(len(global_labels)),
        "n_landslide_pixels": int(global_labels.sum()),
        "n_no_landslide_pixels": int((global_labels==0).sum()),
        "mask_fixes": {
            "Lahaul_Spiti":  {"method":"NDVI-only fixed threshold","threshold":float(used_t),"ls_pct_v1":ls_pct_v1_ls,"ls_pct_v2":ls_pct_ls},
            "Sirmaur_Giri":  {"method":"seasonal normalisation (Option B)",**diag_sg},
            "Rampur_Sutlej": {"method":"seasonal normalisation (Option B)",**diag_rs},
        },
        "thresholds": {
            "class_gte_4_High+":      {"precision":m_high_v2["precision"],"recall":m_high_v2["recall"],"f1":m_high_v2["f1"],"accuracy":m_high_v2["accuracy"],"confusion_matrix":m_high_v2["confusion_matrix"]},
            "class_gte_3_Moderate+":  {"precision":m_mod_v2["precision"],"recall":m_mod_v2["recall"],"f1":m_mod_v2["f1"],"accuracy":m_mod_v2["accuracy"],"confusion_matrix":m_mod_v2["confusion_matrix"]},
        }
    }
    with open(os.path.join(OUT_DIR,"roc_auc_v2.json"),"w") as fh: json.dump(roc_v2,fh,indent=2)
    print("  Saved -> roc_auc_v2.json")

    cm_rows = []
    for lbl, m in [("class_gte_4_High+",m_high_v2),("class_gte_3_Moderate+",m_mod_v2)]:
        cm = np.array(m["confusion_matrix"])
        tn,fp,fn,tp = (cm.ravel() if cm.size==4 else (0,0,0,0))
        cm_rows.append({"threshold":lbl,"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"precision":m["precision"],"recall":m["recall"],"f1":m["f1"],"accuracy":m["accuracy"]})
    pd.DataFrame(cm_rows).to_csv(os.path.join(OUT_DIR,"confusion_matrix_v2.csv"), index=False)
    print("  Saved -> confusion_matrix_v2.csv")

    target_met = {}
    for sk, tgt in [("Lahaul_Spiti",0.55),("Sirmaur_Giri",0.65),("Rampur_Sutlej",0.70)]:
        rv = float(per_site_v2[per_site_v2["site"]==sk]["AUC"].values[0])
        target_met[sk] = bool(rv > tgt)
    fix_sum = {"run_timestamp":str(pd.Timestamp.now()),"global_auc_v1":GLOBAL_AUC_V1,
               "global_auc_v2":global_auc_v2,"global_delta":delta_g,
               "site_summaries":summary_rows,"target_auc_met":target_met}
    with open(os.path.join(OUT_DIR,"mask_fix_summary.json"),"w") as fh: json.dump(fix_sum,fh,indent=2,default=str)
    print("  Saved -> mask_fix_summary.json")

    if global_auc_v2 is not None:
        print("\n[6] Generating v2 visualizations ...")
        plot_roc_curve_v2(global_scores, global_labels, per_site_v2,
                          os.path.join(OUT_DIR,"roc_curve_v2.png"))
        plot_per_site_auc_bar_v2(per_site_v2, os.path.join(OUT_DIR,"per_site_auc_bar_v2.png"))

    print("\n" + "="*70)
    print("  MASK FIXES COMPLETE")
    print("="*70)
    print("\n  Global AUC: v1=" + str(round(GLOBAL_AUC_V1,4)) + "  ->  v2=" + str(round(global_auc_v2,4)) + "  (" + ("+" if delta_g>0 else "") + str(round(delta_g,4)) + ")")
    print()
    for sk, tgt in [("Lahaul_Spiti",0.55),("Sirmaur_Giri",0.65),("Rampur_Sutlej",0.70)]:
        rv = float(per_site_v2[per_site_v2["site"]==sk]["AUC"].values[0])
        ao = auc_old_map.get(sk,0)
        met = "YES" if rv > tgt else "NO"
        print("  " + sk.ljust(22) + "  " + str(round(ao,4)) + " -> " + str(round(rv,4)) + "  (target>" + str(tgt) + ")  [" + met + "]")
    print()


if __name__ == "__main__":
    main()
