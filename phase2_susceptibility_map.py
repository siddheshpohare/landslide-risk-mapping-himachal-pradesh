"""
phase2_susceptibility_map.py
============================
Phase 2 — Static Susceptibility Map
Himachal Pradesh Landslide Risk Mapping Project

Workflow:
  1. Load all per-patch susceptibility rasters from Phase 1
  2. Mosaic them into a single continuous raster
  3. Reclassify using Jenks Natural Breaks (5 classes: Very Low -> Very High)
  4. Save mosaicked continuous raster  -> static_susceptibility_map.tif
  5. Save classified raster            -> static_susceptibility_classified.tif
  6. Generate publication-quality map  -> static_susceptibility_map.png
  7. Save classification metadata      -> jenks_classification.json
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm, ListedColormap
import rasterio
from rasterio.merge import merge
import jenkspy

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PATCHES_DIR = os.path.join(BASE_DIR, "ahp_output", "patches")
AHP_JSON    = os.path.join(BASE_DIR, "ahp_output", "ahp_weights.json")
OUT_DIR     = os.path.join(BASE_DIR, "phase2_output")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MOSAIC  = os.path.join(OUT_DIR, "static_susceptibility_map.tif")
OUT_CLASS   = os.path.join(OUT_DIR, "static_susceptibility_classified.tif")
OUT_MAP_PNG = os.path.join(OUT_DIR, "static_susceptibility_map.png")
OUT_JENKS   = os.path.join(OUT_DIR, "jenks_classification.json")

# ---------------------------------------------------------------------------
# Colour palette & class definitions
# ---------------------------------------------------------------------------
CLASS_LABELS  = ["Very Low", "Low", "Moderate", "High", "Very High"]
CLASS_COLORS  = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
N_CLASSES     = 5


def load_susceptibility_patches(patches_dir):
    files = sorted([
        os.path.join(patches_dir, f)
        for f in os.listdir(patches_dir)
        if f.endswith("_susceptibility.tif") and "_class" not in f
    ])
    if not files:
        raise FileNotFoundError(
            f"No *_susceptibility.tif found in {patches_dir}. Run Phase 1 first."
        )
    return files


def mosaic_patches(patch_paths):
    print(f"  Opening {len(patch_paths)} patches ...")
    src_list = [rasterio.open(p) for p in patch_paths]
    mosaic, transform = merge(src_list, nodata=-9999.0, method="first")
    profile = src_list[0].profile.copy()
    profile.update({
        "height": mosaic.shape[1],
        "width":  mosaic.shape[2],
        "transform": transform,
        "count": 1,
        "dtype": "float32",
        "nodata": -9999.0,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })
    for s in src_list:
        s.close()
    mosaic_2d = mosaic[0]
    return mosaic_2d, transform, profile


def jenks_classify(arr_2d, nodata=-9999.0, n_classes=5, sample_frac=0.05):
    valid_mask = (arr_2d != nodata) & np.isfinite(arr_2d)
    valid_vals = arr_2d[valid_mask].astype(np.float64)

    n_sample = max(50000, int(len(valid_vals) * sample_frac))
    if len(valid_vals) > n_sample:
        rng = np.random.default_rng(42)
        sample = rng.choice(valid_vals, size=n_sample, replace=False)
    else:
        sample = valid_vals

    print(f"  Running Jenks on {len(sample):,} samples (total valid: {len(valid_vals):,}) ...")
    breaks = jenkspy.jenks_breaks(sample.tolist(), n_classes=n_classes)

    class_arr = np.zeros(arr_2d.shape, dtype=np.uint8)
    for i in range(n_classes):
        lo = breaks[i]
        hi = breaks[i + 1]
        if i < n_classes - 1:
            mask = valid_mask & (arr_2d >= lo) & (arr_2d < hi)
        else:
            mask = valid_mask & (arr_2d >= lo) & (arr_2d <= hi)
        class_arr[mask] = i + 1

    return breaks, class_arr, valid_mask


def save_mosaic(arr_2d, transform, profile, out_path):
    p = profile.copy()
    p.update({"dtype": "float32", "nodata": -9999.0})
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr_2d.astype(np.float32), 1)
        dst.update_tags(
            DESCRIPTION="AHP Weighted Susceptibility Mosaic [0-1]",
            PHASE="Phase 2 — Static Susceptibility Map"
        )
    print(f"  Saved mosaic  -> {out_path}")


def save_classified(class_arr, transform, profile, out_path):
    p = profile.copy()
    p.update({"dtype": "uint8", "nodata": 0})
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(class_arr.astype(np.uint8), 1)
        dst.update_tags(
            DESCRIPTION="Jenks-classified susceptibility 1=VeryLow..5=VeryHigh",
            PHASE="Phase 2 — Static Susceptibility Map"
        )
    print(f"  Saved classes -> {out_path}")


def _add_gridlines(ax):
    ax.grid(True, linestyle="--", linewidth=0.4, color="#aaaaaa", alpha=0.6)
    ax.set_aspect("equal")


def _downsample(arr, max_dim=4000):
    """Subsample a 2-D array so its longest dimension <= max_dim."""
    H, W  = arr.shape
    step  = max(1, max(H, W) // max_dim)
    return arr[::step, ::step]


def build_map(mosaic_2d, class_arr, transform, breaks, nodata=-9999.0, out_path=OUT_MAP_PNG):
    # ── Downsample for display (mosaic can be 22k×26k → OOM in matplotlib) ──
    MAX_DIM = 3000
    H_full, W_full = mosaic_2d.shape
    step = max(1, max(H_full, W_full) // MAX_DIM)
    print(f"  Display downsample: step={step}  ({H_full}×{W_full} -> {H_full//step}×{W_full//step})")

    disp_cont  = _downsample(mosaic_2d, MAX_DIM)
    disp_cls   = _downsample(class_arr, MAX_DIM)

    valid_mask = (disp_cont != nodata) & np.isfinite(disp_cont)
    cont_ma    = np.ma.array(disp_cont, mask=~valid_mask)
    cls_ma     = np.ma.array(disp_cls.astype(float), mask=(disp_cls == 0))

    H, W   = disp_cont.shape
    left   = transform.c
    top    = transform.f
    res_x  = transform.a
    res_y  = transform.e
    right  = left + W * res_x
    bottom = top  + H * res_y
    extent = [left, right, bottom, top]

    cont_cmap = plt.get_cmap("RdYlGn_r")
    cls_cmap  = ListedColormap(CLASS_COLORS)
    bounds    = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    cls_norm  = BoundaryNorm(bounds, cls_cmap.N)

    fig = plt.figure(figsize=(20, 14), dpi=150, facecolor="#f0f0f0")
    fig.suptitle(
        "Landslide Susceptibility Map — Himachal Pradesh\n"
        "AHP Weighted Overlay | Phase 2 Analysis",
        fontsize=16, fontweight="bold", y=0.97, color="#1a1a2e"
    )

    gs = GridSpec(
        2, 3, figure=fig,
        left=0.04, right=0.96, top=0.90, bottom=0.08,
        wspace=0.12, hspace=0.30,
        width_ratios=[1, 1, 0.35],
        height_ratios=[3, 1],
    )

    ax_cont = fig.add_subplot(gs[0, 0])
    ax_cls  = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_bar  = fig.add_subplot(gs[1, 1])

    # Panel A: continuous
    im_cont = ax_cont.imshow(
        cont_ma, extent=extent, origin="upper",
        cmap=cont_cmap, vmin=0, vmax=1,
        interpolation="nearest", aspect="auto"
    )
    ax_cont.set_title("A  Continuous Susceptibility Score", fontsize=11, pad=6, loc="left", fontweight="bold")
    ax_cont.set_xlabel("Longitude (degrees E)", fontsize=9)
    ax_cont.set_ylabel("Latitude (degrees N)", fontsize=9)
    ax_cont.tick_params(labelsize=7)
    _add_gridlines(ax_cont)
    cb = plt.colorbar(im_cont, ax=ax_cont, orientation="vertical",
                      fraction=0.035, pad=0.03, shrink=0.85)
    cb.set_label("Susceptibility Score [0-1]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for b in breaks[1:-1]:
        cb.ax.axhline(b, color="white", linewidth=1.2, linestyle="--")

    # Panel B: classified
    ax_cls.imshow(
        cls_ma, extent=extent, origin="upper",
        cmap=cls_cmap, norm=cls_norm,
        interpolation="nearest", aspect="auto"
    )
    ax_cls.set_title("B  Classified Susceptibility (Jenks Natural Breaks)", fontsize=11, pad=6, loc="left", fontweight="bold")
    ax_cls.set_xlabel("Longitude (degrees E)", fontsize=9)
    ax_cls.set_ylabel("Latitude (degrees N)", fontsize=9)
    ax_cls.tick_params(labelsize=7)
    _add_gridlines(ax_cls)
    legend_handles = [
        mpatches.Patch(
            facecolor=CLASS_COLORS[i], edgecolor="#333",
            label=f"Class {i+1}: {CLASS_LABELS[i]}  [{breaks[i]:.3f}-{breaks[i+1]:.3f}]"
        )
        for i in range(N_CLASSES)
    ]
    ax_cls.legend(
        handles=legend_handles, loc="lower left", fontsize=7,
        framealpha=0.85, title="Risk Class  (Jenks)", title_fontsize=8,
        handlelength=1.2, handleheight=1.2, edgecolor="#aaa", fancybox=True,
    )

    # Panel C: metadata
    ax_info.axis("off")
    with open(AHP_JSON) as f:
        ahp = json.load(f)
    weights      = ahp["weights"]
    factor_names = ahp["factors"]
    cr           = ahp["CR"]
    info_lines = [
        "AHP PARAMETERS",
        "-" * 22,
        f"  CR = {cr:.4f}  (consistent)",
        "",
        "Factor Weights:",
    ]
    for fn, w in zip(factor_names, weights):
        bar_len = int(w * 14)
        info_lines.append(f"  {fn:<7s} {w*100:5.1f}%  {'|'*bar_len}")
    info_lines += [
        "",
        "CLASSIFICATION",
        "-" * 22,
        "  Method: Jenks Natural Breaks",
        f"  n_classes = {N_CLASSES}",
        "",
        "Break Values:",
    ]
    for i, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
        info_lines.append(f"  {CLASS_LABELS[i]:<10s} {lo:.3f}-{hi:.3f}")
    ax_info.text(
        0.04, 0.96, "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=8.5, family="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )


    # Panel D: histogram — use downsampled display array (valid_mask already matches disp_cont)
    valid_vals = disp_cont[valid_mask].ravel()
    ax_hist.hist(valid_vals, bins=100, color="#4c72b0", edgecolor="none", alpha=0.85)
    for b in breaks[1:-1]:
        ax_hist.axvline(b, color="#d7191c", linewidth=1.2, linestyle="--", label=f"{b:.3f}")
    ax_hist.set_xlabel("Susceptibility Score", fontsize=9)
    ax_hist.set_ylabel("Pixel Count (display subsample)", fontsize=9)
    ax_hist.set_title("D  Score Distribution + Jenks Breaks", fontsize=10, loc="left", fontweight="bold")
    ax_hist.tick_params(labelsize=7)
    ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_hist.legend(title="Breaks", fontsize=7, title_fontsize=7, loc="upper right")
    for i in range(N_CLASSES):
        ax_hist.axvspan(breaks[i], breaks[i+1], alpha=0.12, color=CLASS_COLORS[i])

    # Panel E: class area bar — use downsampled class array
    counts = np.array([(disp_cls == c).sum() for c in range(1, N_CLASSES + 1)])
    total  = counts.sum()
    pcts   = 100.0 * counts / total if total > 0 else counts
    bars   = ax_bar.barh(range(N_CLASSES), pcts, color=CLASS_COLORS, edgecolor="#333", linewidth=0.6)

    ax_bar.set_yticks(range(N_CLASSES))
    ax_bar.set_yticklabels(CLASS_LABELS, fontsize=9)
    ax_bar.set_xlabel("Area (%)", fontsize=9)
    ax_bar.set_title("E  Class Area Distribution", fontsize=10, loc="left", fontweight="bold")
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_xlim(0, max(pcts) * 1.28)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax_bar.text(
            pct + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%  ({cnt:,} px)",
            va="center", fontsize=8, color="#222"
        )

    fig.text(
        0.5, 0.02,
        "Data: Sentinel-2 L2A  |  AHP factors: Slope, Aspect, NDVI, NDWI, BSI, NBR  |  "
        "Classification: Jenks Natural Breaks  |  CRS: EPSG:4326",
        ha="center", fontsize=7.5, color="#555", style="italic"
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved map     -> {out_path}")


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2 - STATIC SUSCEPTIBILITY MAP")
    print("  Landslide Risk Mapping, Himachal Pradesh")
    print("=" * 70)

    print("\n[1] Discovering Phase-1 susceptibility patches ...")
    patch_paths = load_susceptibility_patches(PATCHES_DIR)
    for p in patch_paths:
        print(f"    {os.path.basename(p)}")

    print("\n[2] Mosaicking patches into single continuous raster ...")
    mosaic_2d, transform, profile = mosaic_patches(patch_paths)
    valid_mask = (mosaic_2d != -9999.0) & np.isfinite(mosaic_2d)
    print(f"    Mosaic shape  : {mosaic_2d.shape}")
    print(f"    Valid pixels  : {valid_mask.sum():,}")
    print(f"    Score range   : [{mosaic_2d[valid_mask].min():.4f}, {mosaic_2d[valid_mask].max():.4f}]")
    print(f"    Score mean    : {mosaic_2d[valid_mask].mean():.4f} +/- {mosaic_2d[valid_mask].std():.4f}")

    print("\n[3] Computing Jenks Natural Breaks classification ...")
    breaks, class_arr, _ = jenks_classify(mosaic_2d, nodata=-9999.0, n_classes=N_CLASSES)
    print(f"\n    Jenks Break Values:")
    for i, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
        cnt = int((class_arr == i + 1).sum())
        pct = 100.0 * cnt / valid_mask.sum()
        print(f"    [{i+1}] {CLASS_LABELS[i]:<10s}: [{lo:.4f}, {hi:.4f}]  ->  {cnt:>8,} px  ({pct:.1f}%)")

    print("\n[4] Saving continuous susceptibility mosaic ...")
    save_mosaic(mosaic_2d, transform, profile, OUT_MOSAIC)

    print("\n[5] Saving Jenks-classified raster ...")
    save_classified(class_arr, transform, profile, OUT_CLASS)

    jenks_meta = {
        "method": "Jenks Natural Breaks",
        "n_classes": N_CLASSES,
        "breaks": breaks,
        "classes": [
            {
                "id":    i + 1,
                "label": CLASS_LABELS[i],
                "color": CLASS_COLORS[i],
                "lo":    breaks[i],
                "hi":    breaks[i + 1],
                "pixel_count": int((class_arr == i + 1).sum()),
                "pct":   round(100.0 * (class_arr == i + 1).sum() / valid_mask.sum(), 2)
            }
            for i in range(N_CLASSES)
        ],
        "mosaic_stats": {
            "min":  float(mosaic_2d[valid_mask].min()),
            "max":  float(mosaic_2d[valid_mask].max()),
            "mean": float(mosaic_2d[valid_mask].mean()),
            "std":  float(mosaic_2d[valid_mask].std()),
            "valid_pixels": int(valid_mask.sum()),
        }
    }
    with open(OUT_JENKS, "w") as f:
        json.dump(jenks_meta, f, indent=2)
    print(f"\n[6] Saved Jenks metadata -> {OUT_JENKS}")

    print("\n[7] Rendering publication-quality map (this may take ~30s) ...")
    build_map(mosaic_2d, class_arr, transform, breaks, out_path=OUT_MAP_PNG)

    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"\n  phase2_output/")
    print(f"  +-- static_susceptibility_map.tif        (continuous mosaic)")
    print(f"  +-- static_susceptibility_classified.tif (Jenks 5-class)")
    print(f"  +-- jenks_classification.json            (break metadata)")
    print(f"  +-- static_susceptibility_map.png        (publication map)")
    print()


if __name__ == "__main__":
    main()
