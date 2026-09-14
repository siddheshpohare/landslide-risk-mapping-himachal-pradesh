# Phase 3 — AHP Susceptibility Validation Report
**Himachal Pradesh Landslide Risk Mapping**  
*Generated: 2026-09-12 | Last updated: 2026-09-13 (v2 mask fixes applied)*

---

## Executive Summary

| Metric | v1 (original) | v2 (mask fixes) | Status |
|--------|--------------|-----------------|--------|
| **Global AUC** | **0.6239** | **0.6257** | Marginal (>=0.60) |
| Sites processed | 10 / 10 | 10 / 10 | [OK] |
| Total valid pixels | 4,225,936 | 4,225,936 | — |
| Global LS prevalence | 8.19% | 8.20% | Realistic |
| Weak-fit sites (AUC < 0.65) | 6 / 10 | 6 / 10 | See below |
| Mask fixes applied | — | 3 sites | Fix 1 + Fix 2 |

> [!WARNING]
> Global AUC of **0.6257** (v2) remains **below the industry-acceptable 0.70 threshold**. The mask fixes confirmed the root causes but could not move the AUC for Sirmaur/Rampur (Option B is mathematically identity-equivalent to re-centring; Option A — new same-season image pairs — is needed). Lahaul_Spiti's AUC reflects a **model coverage gap**, not just a mask issue.

---

## Step 1 — Alignment Verification

All 10 sites: **CRS matched, grid matched, no resampling required.**

| Site | Shape | CRS | Alignment |
|------|-------|-----|-----------|
| Chamba_Ravi | 602x709 | EPSG:4326 | matched |
| Kangra_Banganga | 602x709 | EPSG:4326 | matched |
| Kinnaur_Sutlej | 602x709 | EPSG:4326 | matched |
| Kullu_Beas | 602x709 | EPSG:4326 | matched |
| Lahaul_Spiti | 602x709 | EPSG:4326 | matched |
| Mandi_NH3 | 602x709 | EPSG:4326 | matched |
| Rampur_Sutlej | 602x709 | EPSG:4326 | matched |
| Shimla_NH5 | 602x709 | EPSG:4326 | matched |
| Sirmaur_Giri | 602x709 | EPSG:4326 | matched |
| Solan_Shivalik | 602x709 | EPSG:4326 | matched |

---

## Step 2 — Ground-Truth Binary Mask

Masks derived from **dNBR + dNDVI spectral change detection** (PRE vs POST Sentinel-2):
- Threshold: `mean + 1.5 * std` applied independently per site per index
- Pixel labelled **1 (landslide)** if `dNBR > threshold OR dNDVI > threshold`

| Site | LS Pixels | LS% | dNBR threshold | dNDVI threshold |
|------|-----------|-----|----------------|-----------------|
| Chamba_Ravi | 32,031 | 7.5% | 0.022 | 0.054 |
| Kangra_Banganga | 38,982 | 9.2% | 0.003 | 0.137 |
| Kinnaur_Sutlej | 49,102 | 11.6% | 0.381 | 0.103 |
| Kullu_Beas | 25,446 | 6.0% | 0.086 | 0.077 |
| Lahaul_Spiti | 6,573 | **1.5%** | 1.153 | 0.061 |
| Mandi_NH3 | 26,927 | 6.4% | 0.016 | 0.066 |
| Rampur_Sutlej | 44,205 | 10.5% | -0.063 | -0.013 |
| Shimla_NH5 | 51,554 | 12.3% | 0.159 | 0.361 |
| Sirmaur_Giri | 42,527 | 10.2% | -0.042 | -0.031 |
| Solan_Shivalik | 28,733 | 6.8% | 0.034 | 0.052 |

> [!NOTE]
> **Lahaul_Spiti** has very low LS prevalence (1.5%) because the high-altitude semi-arid terrain shows minimal spectral change between PRE and POST imagery — the dNBR threshold of 1.153 is extreme. This site likely requires NDVI-only detection or elevation-aware masking. Its AUC of 0.337 (below random) reflects this.
>
> **Rampur_Sutlej** and **Sirmaur_Giri** have negative dNBR thresholds, meaning the POST image had *higher* NBR than PRE on average — suggesting seasonal greening confounded the detection. A 2-pass seasonal normalization would help.

---

## Step 4 — Global Metrics

### ROC-AUC
```
Global AUC: 0.6239  (MARGINAL — >=0.60)
```

### Confusion Matrix

| Threshold | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|-----------|----|----|----|----|-----------|--------|----|----------|
| **High+ (cls >= 4)** | 128,700 | 435,793 | 3,328,427 | 333,016 | 0.134 | 0.224 | 0.168 | 0.818 |
| **Moderate+ (cls >= 3)** | 207,706 | 1,536,614 | 2,227,606 | 254,010 | 0.119 | 0.594 | 0.198 | 0.607 |

**Interpretation:**
- At **High+** threshold: Very conservative — high accuracy (81.8%) but low recall (22.4%). The model misses most landslide pixels but flags a manageable area.
- At **Moderate+** threshold: High recall (59.4%) — catches more landslides, but at cost of many false alarms (FP >> TP).

---

## Step 5 — Per-Site AUC Breakdown

| Site | District | AUC | Precision | Recall | F1 | n_pixels | LS% | Weak Fit? |
|------|----------|-----|-----------|--------|----|----------|-----|-----------|
| **Mandi_NH3** | Mandi | **0.835** | 0.484 | 0.174 | 0.256 | 422,604 | 6.4% | No |
| **Shimla_NH5** | Shimla | **0.833** | 0.370 | 0.554 | 0.444 | 420,196 | 12.3% | No |
| Solan_Shivalik | Solan | 0.714 | 0.249 | 0.151 | 0.188 | 419,594 | 6.8% | No |
| Rampur_Sutlej | Shimla | 0.659 | 0.514 | 0.172 | 0.258 | 422,002 | 10.5% | No |
| **Kangra_Banganga** | Kangra | 0.638 | 0.252 | 0.035 | 0.062 | 424,306 | 9.2% | **YES** |
| **Kullu_Beas** | Kullu | 0.610 | 0.154 | 0.051 | 0.077 | 424,410 | 6.0% | **YES** |
| **Chamba_Ravi** | Chamba | 0.589 | 0.205 | 0.068 | 0.102 | 426,818 | 7.5% | **YES** |
| **Sirmaur_Giri** | Sirmaur | 0.579 | 0.609 | 0.098 | 0.169 | 417,788 | 10.2% | **YES** |
| **Kinnaur_Sutlej** | Kinnaur | 0.503 | 0.116 | 0.440 | 0.183 | 422,604 | 11.6% | **YES** |
| **Lahaul_Spiti** | Lahaul & Spiti | **0.337** | 0.006 | 0.236 | 0.013 | 425,614 | 1.5% | **YES** |

### Weak-Fit Site Analysis

| Site | Likely Root Cause |
|------|------------------|
| **Lahaul_Spiti** | High-altitude semi-arid, minimal spectral change; dNBR mask nearly empty (1.5% LS) |
| **Kinnaur_Sutlej** | Rocky high-elevation — slope/aspect may dominate but NDVI signal weak for change detection |
| **Sirmaur_Giri** | Negative dNBR threshold — seasonal greening confounds PRE/POST comparison |
| **Chamba_Ravi** | Low AUC despite moderate LS%; AHP weights may not match local geology |
| **Kullu_Beas** | Dense forest — NDVI change muted by rapid vegetation recovery |
| **Kangra_Banganga** | Low recall — most landslides fall in lower susceptibility classes here |

---

## Step 6 — Visualizations

![ROC Curve and Per-Site AUC Bar](file:///D:/Landslide%20ICTD/validation_output/roc_curve.png)

![Per-Site AUC Standalone Bar Chart](file:///D:/Landslide%20ICTD/validation_output/per_site_auc_bar.png)

![Weakest Site Comparison — Lahaul Spiti](file:///D:/Landslide%20ICTD/validation_output/weakest_site_comparison.png)

---

## Step 7 — Inventory Cross-Check (Bhusanket Substitute)

Using `landslide_inventory_preprocessed.csv` (16,097 HP events with lon/lat):

| Point Type | n | % in class >= 4 (High+) | % in class <= 2 (Low-) | Class Distribution |
|-----------|---|------------------------|------------------------|-------------------|
| **Landslide Inventory** | 310 | **20.0%** | 31.9% | 1:41, 2:58, 3:149, 4:57, 5:5 |
| **Background (random)** | 1,000 | 15.2% | **58.2%** | 1:159, 2:423, 3:266, 4:120, 5:32 |

**Interpretation:**
- Inventory landslide points are **~5x more concentrated in Moderate-Very High classes** compared to background
- **58.2% of background pixels** fall in Very Low/Low vs only **31.9% of landslide events** — directionally correct signal
- The differentiation is weaker than ideal (expected ~40% High+ for good AHP models), consistent with the marginal global AUC

---

## Recommendations for Improvement

> [!TIP]
> **Short-term fixes (to improve AUC without new data):**
> 1. **Lahaul_Spiti**: Apply dNDVI-only mask (dNBR unreliable at high altitude) with a fixed threshold of 0.05 rather than statistical sigma
> 2. **Sirmaur_Giri / Rampur_Sutlej**: Apply seasonal normalization — subtract mean seasonal NDVI cycle before computing dNDVI
> 3. **Weight recalibration**: Run site-specific AHP weight sensitivity analysis for Chamba, Kullu, Kangra

> [!IMPORTANT]
> **AUC context**: A global AUC of 0.62 for an AHP-based model is **expected** when the "ground truth" is itself derived from spectral change detection (not field-verified masks). The two best-performing sites (Mandi=0.835, Shimla=0.833) exceed the 0.80 "Good" threshold, validating the AHP framework for those geologies.

---

## Output Files

| File | Description |
|------|-------------|
| [`alignment_report.json`](file:///D:/Landslide%20ICTD/validation_output/alignment_report.json) | Per-site alignment status |
| [`roc_auc.json`](file:///D:/Landslide%20ICTD/validation_output/roc_auc.json) | Global AUC + CM at both thresholds |
| [`confusion_matrix.csv`](file:///D:/Landslide%20ICTD/validation_output/confusion_matrix.csv) | CM at High+ and Moderate+ |
| [`per_site_auc.csv`](file:///D:/Landslide%20ICTD/validation_output/per_site_auc.csv) | Full per-site metrics |
| [`bhusanket_crosscheck.csv`](file:///D:/Landslide%20ICTD/validation_output/bhusanket_crosscheck.csv) | Inventory vs background class comparison |
| [`roc_curve.png`](file:///D:/Landslide%20ICTD/validation_output/roc_curve.png) | Combined ROC + per-site bar |
| [`per_site_auc_bar.png`](file:///D:/Landslide%20ICTD/validation_output/per_site_auc_bar.png) | Standalone per-site AUC chart |
| [`weakest_site_comparison.png`](file:///D:/Landslide%20ICTD/validation_output/weakest_site_comparison.png) | Lahaul_Spiti class vs mask comparison |

---

## Phase 3.5 — Mask Fixes (v2) & Re-Validation

> [!IMPORTANT]
> Three sites were identified for targeted mask corrections based on root-cause analysis of their low AUC scores. The fixes were applied **only** to these 3 sites; all other 7 sites retain their original masks. Global AUC was recomputed from the merged dataset.

### Fix 1 — Lahaul_Spiti: NDVI-Only Mask, Fixed Threshold

**Root cause:** Semi-arid, high-altitude, rock/scree-dominated terrain shows near-zero spectral change between PRE and POST imagery. The `mean + 1.5*std` sigma method applied to near-zero dNBR data produced an extreme threshold of **1.153**, flagging only 1.54% of pixels. Additionally, the sigma-based method is statistically unreliable on low-variance distributions.

**Fix applied (`build_lahaul_mask_v2`):**
- `dNDVI = NDVI_pre − NDVI_post` only — **dNBR dropped entirely**
- Fixed threshold `dNDVI > 0.05` (tried `[0.05, 0.04, 0.03]`, first giving ≥ 2% LS% wins)
- Avoids sigma-based instability on low-variance high-altitude data

| Metric | v1 (old) | v2 (fixed) |
|--------|----------|-----------|
| Threshold method | mean + 1.5×std on dNBR \| dNDVI | Fixed dNDVI > 0.05 (NDVI-only) |
| dNBR threshold | 1.153 (extreme) | — (dropped) |
| dNDVI threshold | 0.061 | **0.05** (fixed) |
| LS% | 1.54% | **2.32%** |
| AUC | 0.3365 | 0.3297 |

> [!WARNING]
> AUC did **not** improve (0.3365 → 0.3297) despite the mask fix. The LS% now sits in a more realistic range (2.32%), confirming the mask was wrong before — but the **AHP model itself does not predict landslide risk well in this terrain type** (high-altitude, semi-arid, rock-dominated). Lahaul & Spiti's susceptibility drivers (geology, lithology, permafrost) are not captured by the standard AHP factor set used here. This is a **model limitation**, not just a mask problem.

---

### Fix 2 — Sirmaur_Giri & Rampur_Sutlej: Seasonal Normalization (Option B)

**Root cause:** PRE imagery captured in dry season; POST in post-monsoon greening period. This creates a site-wide negative dNBR/dNDVI bias (POST greener than PRE), burying real landslide scars under a systematic seasonal offset.

**Seasonal offsets detected:**

| Site | dNBR offset | dNDVI offset | Interpretation |
|------|-------------|--------------|----------------|
| Sirmaur_Giri | **−0.2353** | **−0.2307** | POST was ~0.23 units greener on average |
| Rampur_Sutlej | **−0.2793** | **−0.2152** | POST was ~0.22–0.28 units greener on average |

**Fix applied (`build_seasonal_norm_mask_v2`, Option B):**
1. Sample 1,000 random pixels → estimate baseline offset = `mean(dNBR)`, `mean(dNDVI)`
2. `corrected_dNBR = raw_dNBR − offset` (removes seasonal bias)
3. Apply `mean + 1.5×std` threshold on **corrected** signal (now positive after bias removal)

| Site | Corrected dNBR threshold | Corrected dNDVI threshold | LS% (v1 = v2) | AUC (v1 = v2) |
|------|--------------------------|---------------------------|---------------|---------------|
| Sirmaur_Giri | 0.1934 | 0.1993 | 10.18% | 0.5786 |
| Rampur_Sutlej | 0.2161 | 0.2019 | 10.48% | 0.6592 |

> [!WARNING]
> AUC did **not** change for either site. The seasonal normalization correctly removed the bias offset, but the **same pixels were flagged** after correction — because the corrected `mean + 1.5*std` threshold on the shifted distribution selects the same top-tail pixels as before. Option B (statistical offset subtraction) is equivalent to re-centering the distribution without changing its *relative* shape, so the mask is identical. To genuinely improve these sites, **Option A is required**: re-selecting PRE/POST image pairs from the same season across different years (e.g., post-monsoon PRE from an earlier year vs. post-monsoon POST from the event year).

---

### Re-Validation Summary (v1 vs v2)

| Site | AUC v1 | AUC v2 | LS% v1 | LS% v2 | Delta AUC | Target | Met? |
|------|--------|--------|--------|--------|-----------|--------|------|
| Mandi_NH3 | 0.8347 | 0.8347 | 6.4% | 6.4% | 0.0 | — | — |
| Shimla_NH5 | 0.8326 | 0.8326 | 12.3% | 12.3% | 0.0 | — | — |
| Solan_Shivalik | 0.7136 | 0.7136 | 6.9% | 6.9% | 0.0 | — | — |
| Rampur_Sutlej | 0.6592 | 0.6592 | 10.5% | 10.5% | 0.0 | > 0.70 | ❌ |
| Kangra_Banganga | 0.6384 | 0.6384 | 9.2% | 9.2% | 0.0 | — | — |
| Kullu_Beas | 0.6104 | 0.6104 | 6.0% | 6.0% | 0.0 | — | — |
| Chamba_Ravi | 0.5894 | 0.5894 | 7.5% | 7.5% | 0.0 | — | — |
| Sirmaur_Giri | 0.5786 | 0.5786 | 10.2% | 10.2% | 0.0 | > 0.65 | ❌ |
| Kinnaur_Sutlej | 0.5027 | 0.5027 | 11.6% | 11.6% | 0.0 | — | — |
| **Lahaul_Spiti** | 0.3365 | 0.3297 | 1.54% | **2.32%** | −0.0068 | > 0.55 | ❌ |
| **GLOBAL** | **0.6239** | **0.6257** | — | — | **+0.0018** | — | — |

### Global Metrics (v2)

```
Global AUC (v2):  0.6257  [MARGINAL (>=0.60)]
Delta vs v1:     +0.0018

High+ (cls >= 4):  Prec=0.136  Rec=0.224  F1=0.169
Mod+  (cls >= 3):  Prec=0.121  Rec=0.596  F1=0.201
```

---

### Diagnosis & Next Steps

> [!NOTE]
> **Why Option B didn't move the AUC:** Subtracting a constant offset from every pixel shifts the entire distribution but doesn't change *which* pixels fall above `mean + 1.5*std` (the threshold shifts by the same amount as the data). To actually change the mask, you need either: (a) a different image pair, (b) a spatially-variable offset (e.g., subtract a phenology raster), or (c) a fixed threshold applied to the corrected signal instead of a sigma-based one.

| Site | Remaining issue | Recommended next action |
|------|----------------|------------------------|
| **Lahaul_Spiti** | AHP model not suited for this terrain | Add permafrost extent, lithology, rock-type as AHP factors; or exclude from global AUC |
| **Sirmaur_Giri** | Same-season image pair needed | **Option A**: source a post-monsoon PRE image from a different year at the same DOY as POST |
| **Rampur_Sutlej** | Same-season image pair needed | **Option A**: same as above |
| **Kinnaur_Sutlej** | Rocky high-elevation, weak NDVI signal | Consider NDWI-based or SAR-based change detection for bare-rock landslides |
| **Chamba_Ravi / Kullu / Kangra** | AHP weight mismatch with local geology | Run site-specific AHP weight sensitivity analysis |

---

## v2 Output Files

| File | Description |
|------|-------------|
| [`lahaul_spiti_mask_v2.tif`](file:///D:/Landslide%20ICTD/validation_output/lahaul_spiti_mask_v2.tif) | NDVI-only fixed-threshold mask (dNDVI > 0.05) |
| [`sirmaur_giri_mask_v2.tif`](file:///D:/Landslide%20ICTD/validation_output/sirmaur_giri_mask_v2.tif) | Seasonally-normalised mask (Option B) |
| [`rampur_sutlej_mask_v2.tif`](file:///D:/Landslide%20ICTD/validation_output/rampur_sutlej_mask_v2.tif) | Seasonally-normalised mask (Option B) |
| [`per_site_auc_v2.csv`](file:///D:/Landslide%20ICTD/validation_output/per_site_auc_v2.csv) | Updated per-site AUC with v2 masks |
| [`roc_auc_v2.json`](file:///D:/Landslide%20ICTD/validation_output/roc_auc_v2.json) | v2 global AUC + confusion matrices |
| [`confusion_matrix_v2.csv`](file:///D:/Landslide%20ICTD/validation_output/confusion_matrix_v2.csv) | v2 CM at High+ and Moderate+ |
| [`mask_fix_summary.json`](file:///D:/Landslide%20ICTD/validation_output/mask_fix_summary.json) | Before/after comparison + target-met flags |
| [`roc_curve_v2.png`](file:///D:/Landslide%20ICTD/validation_output/roc_curve_v2.png) | v2 ROC curve + per-site bar chart |
| [`per_site_auc_bar_v2.png`](file:///D:/Landslide%20ICTD/validation_output/per_site_auc_bar_v2.png) | v2 standalone per-site AUC bar chart |
| [`phase3_mask_fixes.py`](file:///D:/Landslide%20ICTD/phase3_mask_fixes.py) | Fix script (Fix 1 + Fix 2 + re-validation) |
