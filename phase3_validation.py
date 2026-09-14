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
INVENTORY_CSV  = os.path.join(BASE_DIR, "landslide_inventory_preprocessed.csv")
OUT_DIR        = os.path.join(BASE_DIR, "validation_output")
os.makedirs(OUT_DIR, exist_ok=True)

# Band layout in the 12-band Sentinel-2 TIFs (from Phase 1 ahp_susceptibility.py)
# BAND_NAMES = ["B2","B3","B4","B8","B11","B12","NDVI","NDWI","BSI","NBR","slope","aspect"]
BAND_NDVI = 7   # 1-indexed rasterio band
BAND_NBR  = 10  # 1-indexed rasterio band

# Change-detection threshold: pixels with dNBR or dNDVI > mean+sigma*std flagged as landslide
CHANGE_SIGMA = 1.5

# Classification thresholds
HIGH_PLUS_THRESHOLD     = 4   # class >= 4  -> predicted positive
MODERATE_PLUS_THRESHOLD = 3   # class >= 3  -> sensitivity check

# Site metadata: patch prefix -> district name
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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
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
    Returns (status_dict, aligned_mask_array).
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

    status = {
        "site":        site_key,
        "susc_crs":    str(susc_crs),
        "mask_crs":    str(mask_crs),
        "susc_shape":  [susc_height, susc_width],
        "mask_shape":  [mask_height, mask_width],
        "crs_match":   crs_match,
        "size_match":  size_match,
        "res_match":   res_match,
    }

    if crs_match and size_match and res_match:
        status["result"] = "matched"
        return status, mask_arr

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
    status["result"] = "resampled"
    return status, resampled


# ---------------------------------------------------------------------------
# Step 2 -- Ground-Truth Binary Mask Generation (Change Detection)
# ---------------------------------------------------------------------------
def build_gt_mask(post_tif, pre_tif, sigma=CHANGE_SIGMA):
    """
    Create binary landslide mask from PRE/POST Sentinel-2 imagery using
    normalized burn ratio (NBR) and NDVI change detection.

    dNBR  = NBR_pre  - NBR_post   (positive  ->  vegetation/NBR loss  ->  likely LS)
    dNDVI = NDVI_pre - NDVI_post  (positive  ->  vegetation loss       ->  likely LS)

    Pixel labelled 1 (landslide) if:
        dNBR  > mean(dNBR)  + sigma * std(dNBR)   OR
        dNDVI > mean(dNDVI) + sigma * std(dNDVI)

    Returns: (mask_array uint8, profile dict, ls_pct float, t_nbr, t_ndvi)
    """
    with rasterio.open(post_tif) as src:
        ndvi_post = src.read(BAND_NDVI).astype(np.float32)
        nbr_post  = src.read(BAND_NBR ).astype(np.float32)
        profile   = src.profile.copy()

    with rasterio.open(pre_tif) as src:
        ndvi_pre = src.read(BAND_NDVI).astype(np.float32)
        nbr_pre  = src.read(BAND_NBR ).astype(np.float32)

    dnbr  = nbr_pre  - nbr_post
    dndvi = ndvi_pre - ndvi_post

    t_nbr  = float(dnbr .mean() + sigma * dnbr .std())
    t_ndvi = float(dndvi.mean() + sigma * dndvi.std())

    mask   = ((dnbr > t_nbr) | (dndvi > t_ndvi)).astype(np.uint8)
    profile.update(dtype="uint8", count=1, nodata=255)

    ls_pct = float(100.0 * mask.mean())
    return mask, profile, ls_pct, t_nbr, t_ndvi


# ---------------------------------------------------------------------------
# Step 3 -- Pixel Extraction
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
def plot_roc_curve(global_scores, global_labels, per_site_df, out_path):
    """Combined figure: ROC curve (left) + per-site AUC bars (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Phase 3 -- AHP Susceptibility Validation  |  Himachal Pradesh",
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


def plot_per_site_auc_bar(per_site_df, out_path):
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


def plot_weakest_site_comparison(weakest_site, susc_class_path, mask_arr, out_path):
    """Side-by-side raster comparison for the weakest-fit site."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    with rasterio.open(susc_class_path) as src:
        cls = src.read(1)

    CLASS_COLORS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
    cls_cmap = ListedColormap(CLASS_COLORS)
    cls_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cls_cmap.N)

    mask_cmap = ListedColormap(["#4393c3", "#d7191c"])
    mask_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Weakest-Fit Site: {weakest_site}\n"
        "AHP Susceptibility Class  vs  Change-Detection Ground-Truth Mask",
        fontsize=13, fontweight="bold"
    )

    cls_disp = cls.astype(float)
    cls_disp[cls == 0] = np.nan
    axes[0].imshow(cls_disp, cmap=cls_cmap, norm=cls_norm, interpolation="nearest")
    axes[0].set_title("AHP Susceptibility Class\n(1=Very Low  ...  5=Very High)", fontsize=11)
    axes[0].axis("off")
    class_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    patches = [mpatches.Patch(facecolor=c, label=f"Class {i+1}: {l}")
               for i, (c, l) in enumerate(zip(CLASS_COLORS, class_labels))]
    axes[0].legend(handles=patches, fontsize=8, loc="lower left", framealpha=0.87)

    mask_disp = mask_arr.astype(float)
    mask_disp[mask_arr == 255] = np.nan
    axes[1].imshow(mask_disp, cmap=mask_cmap, norm=mask_norm, interpolation="nearest")
    axes[1].set_title("Change-Detection Ground-Truth Mask\n(Blue=No Landslide  |  Red=Landslide)", fontsize=11)
    axes[1].axis("off")
    gt_patches = [
        mpatches.Patch(facecolor="#4393c3", label="0 = No Landslide"),
        mpatches.Patch(facecolor="#d7191c", label="1 = Landslide (dNBR/dNDVI)"),
    ]
    axes[1].legend(handles=gt_patches, fontsize=8, loc="lower left", framealpha=0.87)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Step 7 -- Inventory Point Cross-Check (Bhusanket substitute)
# ---------------------------------------------------------------------------
def bhusanket_crosscheck(sites_data, inventory_csv, out_path):
    """
    Cross-check HP landslide inventory point events against AHP susceptibility class.
    Also samples matched random background points per site.
    Outputs bhusanket_crosscheck.csv with columns:
      site, district, type, lon, lat, susc_class, high_plus
    """
    print("\n[7] Inventory/Bhusanket Point Cross-Check ...")
    if not os.path.exists(inventory_csv):
        print("    [WARN] Inventory CSV not found, skipping.")
        return None

    df_inv = pd.read_csv(inventory_csv)
    df_inv["district_norm"] = df_inv["district"].str.strip().str.lower()
    hp_dist_lower = [d.lower() for d in SITE_MAP.values()]

    df_hp = df_inv[df_inv["district_norm"].apply(
        lambda d: any(h.split("&")[0].strip() in d or d in h for h in hp_dist_lower)
    )].copy()

    if df_hp.empty:
        print("    [WARN] No HP inventory events found.")
        return None
    print(f"    HP inventory events: {len(df_hp)}")

    records = []
    # Sample landslide inventory points per site
    for site_key, susc_path, susc_class_path, post_tif, pre_tif in sites_data:
        district = SITE_MAP.get(site_key, "")
        dist_l   = district.lower().split("&")[0].strip()

        site_events = df_hp[df_hp["district_norm"].apply(
            lambda d: dist_l in d or d in dist_l
        )]

        with rasterio.open(susc_class_path) as src:
            cls_arr   = src.read(1)
            transform = src.transform
            height    = src.height
            width     = src.width

        n_ls = 0
        for _, row in site_events.iterrows():
            lon, lat = row.get("longitude"), row.get("latitude")
            if pd.isna(lon) or pd.isna(lat):
                continue
            try:
                r, c = rowcol(transform, lon, lat)
                if 0 <= r < height and 0 <= c < width:
                    cv = int(cls_arr[r, c])
                    if cv == 0:
                        continue  # NoData pixel
                    records.append({
                        "site":       site_key,
                        "district":   district,
                        "type":       "landslide_inventory",
                        "lon":        lon,
                        "lat":        lat,
                        "susc_class": cv,
                        "high_plus":  int(cv >= HIGH_PLUS_THRESHOLD),
                    })
                    n_ls += 1
            except Exception:
                pass

        # Generate matched background points
        rng = np.random.default_rng(42)
        n_bg = max(n_ls, 100)
        valid_rows, valid_cols = np.where(cls_arr > 0)
        if len(valid_rows) >= n_bg:
            idx = rng.choice(len(valid_rows), size=n_bg, replace=False)
            for ri, ci in zip(valid_rows[idx], valid_cols[idx]):
                lon_pt = transform.c + (ci + 0.5) * transform.a
                lat_pt = transform.f + (ri + 0.5) * transform.e
                cv = int(cls_arr[ri, ci])
                records.append({
                    "site":       site_key,
                    "district":   district,
                    "type":       "background",
                    "lon":        lon_pt,
                    "lat":        lat_pt,
                    "susc_class": cv,
                    "high_plus":  int(cv >= HIGH_PLUS_THRESHOLD),
                })

    if not records:
        print("    [WARN] No valid cross-check records generated.")
        return None

    df_cross = pd.DataFrame(records)
    df_cross.to_csv(out_path, index=False)
    print(f"    Saved -> {out_path}  ({len(df_cross)} rows)")

    # Print summary
    for pt_type in ["landslide_inventory", "background"]:
        sub = df_cross[df_cross["type"] == pt_type].dropna(subset=["susc_class"])
        if sub.empty:
            continue
        h_pct = 100 * sub["high_plus"].mean()
        l_pct = 100 * (sub["susc_class"] <= 2).mean()
        dist  = sub["susc_class"].value_counts().sort_index().to_dict()
        print(f"\n    [{pt_type}]  n={len(sub)}")
        print(f"      % in class >= 4 (High+):  {h_pct:.1f}%")
        print(f"      % in class <= 2 (Low-):   {l_pct:.1f}%")
        print(f"      Class dist: {dist}")

    return df_cross


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  PHASE 3 -- AHP SUSCEPTIBILITY VALIDATION")
    print("  Landslide Risk Mapping, Himachal Pradesh")
    print("=" * 70)

    # Discover sites
    print("\n[0] Discovering sites ...")
    sites_data = discover_sites()
    if not sites_data:
        raise RuntimeError("No valid sites found. Check PATCHES_DIR and dataset paths.")
    print(f"    Found {len(sites_data)} sites.")
    for key, *_ in sites_data:
        print(f"      {key:<22} -> District: {SITE_MAP.get(key,'?')}")

    # Steps 1-3: Per-site alignment + mask generation + pixel extraction
    print(f"\n[1-3] Alignment verification, mask generation, pixel extraction ...")
    alignment_report = {}
    all_dfs   = []
    site_masks = {}   # site_key -> (susc_class_path, mask_arr)

    for site_key, susc_path, susc_class_path, post_tif, pre_tif in sites_data:
        print(f"\n  -- {site_key} --")

        # Step 2
        mask_arr, mask_profile, ls_pct, t_nbr, t_ndvi = build_gt_mask(post_tif, pre_tif)
        print(f"    GT mask: {ls_pct:.1f}% landslide pixels  "
              f"(dNBR_thr={t_nbr:.3f}, dNDVI_thr={t_ndvi:.3f})")

        # Step 1
        al_status, mask_arr = verify_alignment(susc_path, mask_arr, mask_profile, site_key)
        al_status.update({"ls_pct": ls_pct, "t_nbr": t_nbr, "t_ndvi": t_ndvi})
        alignment_report[site_key] = al_status
        print(f"    Alignment: {al_status['result']}")

        # Step 3
        df_site = extract_pixels(site_key, susc_path, susc_class_path, mask_arr)
        n_ls = int(df_site["gt"].sum())
        print(f"    Valid pixels: {len(df_site):,}  |  "
              f"Landslide: {n_ls:,} ({100*df_site['gt'].mean():.1f}%)")

        all_dfs.append(df_site)
        site_masks[site_key] = (susc_class_path, mask_arr)

    # Save alignment report
    al_path = os.path.join(OUT_DIR, "alignment_report.json")
    with open(al_path, "w") as f:
        json.dump(alignment_report, f, indent=2)
    print(f"\n  Alignment report -> {al_path}")

    # Merge all site pixels
    df_all        = pd.concat(all_dfs, ignore_index=True)
    global_scores = df_all["score"].values
    global_labels = df_all["gt"].values
    global_cls    = df_all["cls"].values
    print(f"\n  Total valid pixels: {len(df_all):,}")
    print(f"  Global LS prevalence: {100*global_labels.mean():.2f}%")

    # Step 4: Global Metrics
    print("\n[4] Computing global metrics ...")
    m_high = compute_metrics(global_scores, global_labels, global_cls,
                             threshold_cls=HIGH_PLUS_THRESHOLD)
    m_mod  = compute_metrics(global_scores, global_labels, global_cls,
                             threshold_cls=MODERATE_PLUS_THRESHOLD)

    if m_high["auc"] is not None:
        quality = (
            "EXCELLENT (>=0.90)" if m_high["auc"] >= 0.90 else
            "GOOD (>=0.80)"      if m_high["auc"] >= 0.80 else
            "ACCEPTABLE (>=0.70)" if m_high["auc"] >= 0.70 else
            "MARGINAL (>=0.60)"  if m_high["auc"] >= 0.60 else
            "POOR (<0.60)"
        )
        print(f"    Global AUC: {m_high['auc']:.4f}  ->  {quality}")
    else:
        print("    AUC: UNDEFINED (single-class data)")

    print(f"    High+  (cls>=4): Prec={m_high['precision']:.3f}  "
          f"Rec={m_high['recall']:.3f}  F1={m_high['f1']:.3f}  "
          f"Acc={m_high['accuracy']:.3f}")
    print(f"    Mod+   (cls>=3): Prec={m_mod['precision']:.3f}  "
          f"Rec={m_mod['recall']:.3f}  F1={m_mod['f1']:.3f}  "
          f"Acc={m_mod['accuracy']:.3f}")

    # Save roc_auc.json
    roc_json = {
        "global_auc":           m_high["auc"],
        "n_total_pixels":       int(len(global_labels)),
        "n_landslide_pixels":   int(global_labels.sum()),
        "n_no_landslide_pixels": int((global_labels == 0).sum()),
        "change_detection_sigma": CHANGE_SIGMA,
        "thresholds": {
            f"class_gte_{HIGH_PLUS_THRESHOLD}_High+": {
                "precision":       m_high["precision"],
                "recall":          m_high["recall"],
                "f1":              m_high["f1"],
                "accuracy":        m_high["accuracy"],
                "confusion_matrix": m_high["confusion_matrix"],
            },
            f"class_gte_{MODERATE_PLUS_THRESHOLD}_Moderate+": {
                "precision":       m_mod["precision"],
                "recall":          m_mod["recall"],
                "f1":              m_mod["f1"],
                "accuracy":        m_mod["accuracy"],
                "confusion_matrix": m_mod["confusion_matrix"],
            },
        },
    }
    roc_path = os.path.join(OUT_DIR, "roc_auc.json")
    with open(roc_path, "w") as f:
        json.dump(roc_json, f, indent=2)
    print(f"    Saved -> {roc_path}")

    # Save confusion_matrix.csv
    cm_rows = []
    for label, m in [(f"class_gte_{HIGH_PLUS_THRESHOLD}_High+", m_high),
                     (f"class_gte_{MODERATE_PLUS_THRESHOLD}_Moderate+", m_mod)]:
        cm = np.array(m["confusion_matrix"])
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        cm_rows.append({
            "threshold": label,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "accuracy": m["accuracy"],
        })
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.csv")
    pd.DataFrame(cm_rows).to_csv(cm_path, index=False)
    print(f"    Saved -> {cm_path}")

    # Step 5: Per-site breakdown
    print("\n[5] Per-site AUC breakdown ...")
    per_site_df = per_site_breakdown(df_all, threshold_cls=HIGH_PLUS_THRESHOLD)
    print(f"\n    {'Site':<22} {'AUC':>6}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  "
          f"{'n_px':>8}  {'LS%':>6}  Flag")
    print("    " + "-" * 75)
    for _, row in per_site_df.sort_values("AUC", ascending=False, na_position="last").iterrows():
        auc_s  = f"{row['AUC']:.3f}" if row["AUC"] is not None else "  N/A"
        ls_s   = f"{100*row['n_positive']/row['n_pixels']:.1f}%" if row["n_pixels"] else "N/A"
        flag   = " *** WEAK FIT ***" if row["weak_fit"] else ""
        print(f"    {row['site']:<22} {auc_s:>6}  "
              f"{row['precision']:>5.3f}  {row['recall']:>5.3f}  {row['f1']:>5.3f}  "
              f"{int(row['n_pixels']):>8,}  {ls_s:>6}{flag}")

    ps_path = os.path.join(OUT_DIR, "per_site_auc.csv")
    per_site_df.to_csv(ps_path, index=False)
    print(f"\n    Saved -> {ps_path}")

    weak_sites = per_site_df[per_site_df["weak_fit"] == True]["site"].tolist()
    if weak_sites:
        print(f"\n  [!!]  Weak-fit sites (AUC < {WEAK_FIT_AUC_THRESHOLD}): {', '.join(weak_sites)}")
    else:
        print(f"\n  [OK]  All sites meet AUC >= {WEAK_FIT_AUC_THRESHOLD}")

    # Step 6: Visualizations
    if m_high["auc"] is not None:
        print("\n[6] Generating visualizations ...")

        roc_png = os.path.join(OUT_DIR, "roc_curve.png")
        plot_roc_curve(global_scores, global_labels, per_site_df, roc_png)

        bar_png = os.path.join(OUT_DIR, "per_site_auc_bar.png")
        plot_per_site_auc_bar(per_site_df, bar_png)

        weakest = per_site_df.dropna(subset=["AUC"]).sort_values("AUC").iloc[0]
        wk = weakest["site"]
        ws_class_path, ws_mask = site_masks[wk]
        comp_png = os.path.join(OUT_DIR, "weakest_site_comparison.png")
        plot_weakest_site_comparison(wk, ws_class_path, ws_mask, comp_png)
    else:
        print("\n[6] Skipping visualizations (AUC undefined).")

    # Step 7: Inventory cross-check
    crosscheck_path = os.path.join(OUT_DIR, "bhusanket_crosscheck.csv")
    bhusanket_crosscheck(sites_data, INVENTORY_CSV, crosscheck_path)

    # Final summary
    print("\n" + "=" * 70)
    print("  PHASE 3 COMPLETE")
    print("=" * 70)
    if m_high["auc"]:
        print(f"\n  Global AUC: {m_high['auc']:.4f}  ({quality})")
    print(f"\n  validation_output/")
    print(f"  +--- alignment_report.json")
    print(f"  +--- roc_auc.json")
    print(f"  +--- confusion_matrix.csv")
    print(f"  +--- per_site_auc.csv")
    print(f"  +--- bhusanket_crosscheck.csv")
    print(f"  +--- roc_curve.png")
    print(f"  +--- per_site_auc_bar.png")
    print(f"  +--- weakest_site_comparison.png")
    print()


if __name__ == "__main__":
    main()
