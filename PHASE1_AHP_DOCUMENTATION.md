# Phase 1 — AHP Susceptibility Model
## Landslide Risk Mapping, Himachal Pradesh

---

## Table of Contents
1. [What is AHP?](#1-what-is-ahp)
2. [Project Context](#2-project-context)
3. [Factors Used](#3-factors-used)
4. [Step-by-Step Pipeline](#4-step-by-step-pipeline)
5. [Pairwise Comparison Matrix](#5-pairwise-comparison-matrix)
6. [AHP Weight Computation](#6-ahp-weight-computation)
7. [Consistency Check](#7-consistency-check)
8. [Normalization Logic](#8-normalization-logic)
9. [Weighted Overlay Formula](#9-weighted-overlay-formula)
10. [5-Class Susceptibility Map](#10-5-class-susceptibility-map)
11. [Results Summary](#11-results-summary)
12. [Output Files](#12-output-files)
13. [Design Decisions & Tradeoffs](#13-design-decisions--tradeoffs)
14. [How to Re-run / Modify](#14-how-to-re-run--modify)

---

## 1. What is AHP?

**AHP (Analytic Hierarchy Process)** is a structured multi-criteria decision-making method developed by Thomas Saaty (1980).

It works in 3 steps:

```
Step 1: Expert compares every factor pair → Pairwise Matrix
        e.g., "Slope is 3x more important than Aspect"

Step 2: Compute eigenvector → priority weights for each factor
        All weights sum to 1.0

Step 3: Validate consistency → Consistency Ratio (CR) must be < 0.10
        If CR ≥ 0.10, re-examine your judgements
```

In landslide susceptibility mapping, AHP is used to **combine multiple terrain/environmental
factors into a single continuous risk score** without needing labelled training data.

---

## 2. Project Context

### Data Source
- **Satellite:** Sentinel-2 SR Harmonized (10 m/pixel) via Google Earth Engine
- **Site type:** POST-monsoon composites (July–September 2023) at 10 known landslide sites in HP
- **Each TIF:** 12-band stack over a ~6×6 km patch

### 12-Band Stack (per TIF file)
| Band Index | Band Name | Description |
|------------|-----------|-------------|
| 1 | B2 | Blue (490 nm) |
| 2 | B3 | Green (560 nm) |
| 3 | B4 | Red (665 nm) |
| 4 | B8 | NIR (842 nm) |
| 5 | B11 | SWIR-1 (1610 nm) |
| 6 | B12 | SWIR-2 (2190 nm) |
| 7 | NDVI | Normalised Difference Vegetation Index |
| 8 | NDWI | Normalised Difference Water Index |
| 9 | BSI | Bare Soil Index |
| 10 | NBR | Normalised Burn Ratio |
| 11 | slope | Terrain slope in degrees (from ALOS DEM) |
| 12 | aspect | Terrain aspect in degrees (from ALOS DEM) |

We do **not** use all 12 bands for AHP — only the 6 that represent physically meaningful
susceptibility factors.

---

## 3. Factors Used

Six factors were selected for the AHP model:

| # | Factor | Source Band | Physical Meaning | Direction |
|---|--------|-------------|-----------------|-----------|
| F1 | Slope | `slope` | Steeper = more likely to fail | Higher → More Risk |
| F2 | Aspect | `aspect` | Directional solar/rain exposure | Higher → More Risk |
| F3 | NDVI | `NDVI` | Vegetation cover (protective) | **Inverted** — Lower NDVI → More Risk |
| F4 | NDWI | `NDWI` | Moisture / water saturation | Higher → More Risk |
| F5 | BSI | `BSI` | Bare/exposed soil (geology proxy) | Higher → More Risk |
| F6 | NBR | `NBR` | Disturbance / soil health | **Inverted** — Lower NBR → More Risk |

> **Why invert NDVI and NBR?**
> Dense healthy vegetation stabilizes slopes (root cohesion, less surface runoff).
> A pixel with high NDVI is LESS susceptible. So we flip it: `1 - normalized_NDVI`
> so that high raw value becomes low susceptibility score.

---

## 4. Step-by-Step Pipeline

```
INPUT: 10 × GeoTIFF patches (12 bands each, ~11 MB each)
         └─ landslide_dataset/with_landslide/*.tif

    │
    ▼
[Step 1] Build 6×6 Pairwise Comparison Matrix
         Expert judgement: How much more important is factor A vs B?
         Uses Saaty's 1–9 scale
    │
    ▼
[Step 2] Compute AHP Weights
         Geometric mean of each row → normalized priority vector
         Result: 6 weights that sum to 1.0
    │
    ▼
[Step 3] Consistency Check
         λ_max → CI → CR = CI / RI
         Must be CR < 0.10
    │
    ▼
[Step 4] Gather Global Percentiles (2-pass, memory-safe)
         Read every TIF, subsample every 10th pixel
         Compute p2, p98 for each factor across ALL patches
    │
    ▼
[Step 5] Per-file Normalization + Weighted Sum
         For each of 10 TIF files:
           - Extract factor band
           - Clip to [p2, p98]
           - Normalize to [0, 1]
           - Invert if needed
           - Multiply by AHP weight
           - Accumulate weighted sum
    │
    ▼
[Step 6] Classify into 5 Classes
         [0.0–0.2) = Very Low
         [0.2–0.4) = Low
         [0.4–0.6) = Moderate
         [0.6–0.8) = High
         [0.8–1.0] = Very High
    │
    ▼
OUTPUT:
  ahp_output/
    ahp_weights.csv
    ahp_weights.json
    susceptibility_summary.json
    patches/*_susceptibility.tif        (continuous float32)
    patches/*_susceptibility_class.tif  (uint8, 1–5)
```

---

## 5. Pairwise Comparison Matrix

The **6×6 matrix** below captures how much more important one factor is compared to another,
using Saaty's scale:

| Scale | Meaning |
|-------|---------|
| 1 | Equal importance |
| 3 | Moderate importance of one over another |
| 5 | Strong importance |
| 7 | Very strong importance |
| 9 | Extreme importance |
| 2,4,6,8 | Intermediate values |

```
           Slope  Aspect  NDVI   NDWI   BSI    NBR
Slope    [  1      3      2      3      4      5   ]
Aspect   [  1/3    1      1/2    1      2      3   ]
NDVI     [  1/2    2      1      2      3      4   ]
NDWI     [  1/3    1      1/2    1      2      3   ]
BSI      [  1/4    1/2    1/3    1/2    1      2   ]
NBR      [  1/5    1/3    1/4    1/3    1/2    1   ]
```

**Reading example:**
- Slope vs Aspect = 3 → Slope is 3× more important than Aspect
- Slope vs NBR = 5 → Slope is 5× more important than NBR
- NDVI vs Aspect = 2 → NDVI is 2× more important than Aspect

**Rule:** If A vs B = k, then B vs A = 1/k  (matrix is reciprocal)

---

## 6. AHP Weight Computation

### Method: Geometric Mean (Approximate Eigenvector)

For each factor row i:

```
geo_mean[i] = (product of all elements in row i) ^ (1/n)

weight[i] = geo_mean[i] / sum(geo_mean)
```

### Results

| Factor | Geometric Mean | Weight | Weight % |
|--------|---------------|--------|----------|
| Slope  | 2.6052 | **0.3639** | **36.39%** |
| NDVI   | 1.6619 | **0.2317** | **23.17%** |
| Aspect | 0.9779 | 0.1364 | 13.64% |
| NDWI   | 0.9779 | 0.1364 | 13.64% |
| BSI    | 0.5756 | 0.0803 | 8.03% |
| NBR    | 0.3674 | 0.0512 | 5.12% |

**Weights sum = 1.0000** ✅

```
Weight distribution:
Slope    ████████████████████████████████████  36.4%
NDVI     ███████████████████████               23.2%
Aspect   █████████████                         13.6%
NDWI     █████████████                         13.6%
BSI      ████████                               8.0%
NBR      █████                                  5.1%
```

---

## 7. Consistency Check

Saaty defined a Consistency Ratio (CR) to detect contradictory judgements.

### Formula

```
1. Compute λ_max (principal eigenvalue):
   λ_max = Σ (column_sum[j] × weight[j])

2. Consistency Index:
   CI = (λ_max - n) / (n - 1)

3. Random Index (Saaty's table for n=6):
   RI = 1.24

4. Consistency Ratio:
   CR = CI / RI
```

### Our Results

| Metric | Value |
|--------|-------|
| n (factors) | 6 |
| λ_max | **6.0768** |
| CI | **0.0154** |
| RI | **1.2400** |
| **CR** | **0.0124** |
| Threshold | 0.10 |
| Decision | ✅ **CONSISTENT** |

> CR = 0.0124 means only **1.2%** inconsistency relative to a fully random matrix.
> This is excellent — our expert judgements are internally consistent.

---

## 8. Normalization Logic

Each factor band must be scaled to [0, 1] before combining.

### Why normalize?
- Different units: slope is in degrees (0–90), NDVI is unitless (−1 to +1)
- Without normalization, large-range factors would dominate regardless of weight

### Method: Percentile Clip + Min-Max

```python
# Step 1: Mask nodata pixels (GEE nodata ~ -9999)
band[band == nodata] = NaN

# Step 2: Clip outliers (cloud residuals, water bodies)
p2  = 2nd  percentile of all valid pixels across all patches
p98 = 98th percentile of all valid pixels across all patches
band = clip(band, p2, p98)

# Step 3: Min-max normalize to [0, 1]
normalized = (band - vmin) / (vmax - vmin)

# Step 4: Invert if needed (for NDVI and NBR)
if invert:
    normalized = 1.0 - normalized
```

### Global percentiles computed (across all 10 patches)

| Factor | p2 | p98 | Inverted? |
|--------|-----|------|-----------|
| Slope | 0.00° | 53.00° | No |
| Aspect | 0.00° | 348.00° | No |
| NDVI | -0.029 | 0.874 | **Yes** |
| NDWI | -0.773 | 0.013 | No |
| BSI | -0.320 | 0.194 | No |
| NBR | -0.122 | 0.663 | **Yes** |

> **Why global percentiles?**
> If each patch was normalized independently, a flat valley patch and a steep gorge
> patch would both get 0–1 ranges, making them incomparable.
> Using global stats ensures the same physical slope value maps to the same score
> everywhere.

---

## 9. Weighted Overlay Formula

The final susceptibility score is a **linear weighted combination**:

```
S(x,y) = w1 × Slope_norm(x,y)
        + w2 × Aspect_norm(x,y)
        + w3 × (1 - NDVI_norm(x,y))     ← inverted
        + w4 × NDWI_norm(x,y)
        + w5 × BSI_norm(x,y)
        + w6 × (1 - NBR_norm(x,y))      ← inverted

Where:
  w1=0.3639, w2=0.1364, w3=0.2317,
  w4=0.1364, w5=0.0803, w6=0.0512

  S(x,y) ∈ [0, 1]
  0 = Very Low susceptibility
  1 = Very High susceptibility
```

**If ANY factor is NoData at pixel (x,y), the output is marked NoData.**

---

## 10. 5-Class Susceptibility Map

The continuous score is reclassified into 5 zones for interpretability:

| Class | Label | Score Range | Colour (suggested) |
|-------|-------|-------------|-------------------|
| 1 | Very Low | 0.00 – 0.20 | Green |
| 2 | Low | 0.20 – 0.40 | Yellow-Green |
| 3 | Moderate | 0.40 – 0.60 | Yellow |
| 4 | High | 0.60 – 0.80 | Orange |
| 5 | Very High | 0.80 – 1.00 | Red |

This classification follows NDMA (National Disaster Management Authority) and
NRSC standard 5-class landslide hazard zonation.

---

## 11. Results Summary

### Overall Statistics (4,225,936 pixels across 10 sites)

| Statistic | Value |
|-----------|-------|
| Min susceptibility | 0.0000 |
| Max susceptibility | 0.9925 |
| Mean susceptibility | 0.3704 |
| Std deviation | 0.2009 |

### Class Distribution

| Class | Label | Pixels | % |
|-------|-------|--------|---|
| 1 | Very Low | 844,104 | 20.0% |
| 2 | Low | 1,656,011 | 39.2% |
| 3 | Moderate | 1,148,469 | 27.2% |
| 4 | High | 456,385 | 10.8% |
| 5 | Very High | 120,967 | 2.9% |

### Per-Site Susceptibility (sorted by mean score)

| Site | District | Mean Score | Dominant Class |
|------|----------|-----------|----------------|
| Lahaul_Spiti | Lahaul & Spiti | **0.633** | High |
| Kinnaur_Sutlej | Kinnaur | **0.552** | Moderate |
| Shimla_NH5 | Shimla | 0.412 | Moderate |
| Rampur_Sutlej | Shimla | 0.331 | Low |
| Chamba_Ravi | Chamba | 0.324 | Low |
| Kullu_Beas | Kullu | 0.304 | Low |
| Solan_Shivalik | Solan | 0.298 | Low |
| Sirmaur_Giri | Sirmaur | 0.289 | Low |
| Mandi_NH3 | Mandi | 0.281 | Low |
| Kangra_Banganga | Kangra | 0.278 | Low |

**Lahaul-Spiti and Kinnaur show highest susceptibility** — consistent with their
high-altitude, sparsely vegetated, steep-gorge terrain along the Sutlej and
Chandrabhaga valleys.

---

## 12. Output Files

```
D:\Landslide ICTD\
│
├── ahp_susceptibility.py           ← Main pipeline script
│
└── ahp_output\
    ├── ahp_weights.csv             ← Weight table (Factor, Band, Weight, %)
    ├── ahp_weights.json            ← Full AHP metadata:
    │                                    weights, lambda_max, CI, RI, CR,
    │                                    pairwise matrix, class definitions
    ├── susceptibility_summary.json ← Overall stats + per-class pixel counts
    │
    └── patches\                    ← 20 GeoTIFF rasters (2 per site × 10 sites)
        ├── *_susceptibility.tif        float32  [0.0–1.0]  NoData=-9999
        └── *_susceptibility_class.tif  uint8    [1–5]      NoData=0
```

### File Format Details

**`*_susceptibility.tif`** (continuous)
- Dtype: `float32`
- NoData: `-9999.0`
- Range: `[0.0, 1.0]`
- Projection: WGS84 (EPSG:4326)
- Resolution: 10 m/pixel
- Tags: `CR`, `WEIGHTS`, `DESCRIPTION`

**`*_susceptibility_class.tif`** (classified)
- Dtype: `uint8`
- NoData: `0`
- Values: `1=Very Low, 2=Low, 3=Moderate, 4=High, 5=Very High`
- Use this for visualization in QGIS / ArcGIS

---

## 13. Design Decisions & Tradeoffs

### Why not use all 12 bands?
Raw spectral bands (B2–B12) represent reflectance, not physical susceptibility factors.
We use derived indices (NDVI, NDWI, etc.) that have established physical interpretations
in landslide literature.

### Why percentile clipping at p2/p98?
Water bodies (NDWI ~ +1), shadow pixels, and cloud residuals create extreme outliers.
Clipping at p2/p98 prevents these from stretching the normalization and masking real variation.

### Why global (not per-patch) percentiles?
Per-patch normalization makes each site independently normalized — a flat area and a
steep gorge both span [0,1]. Global percentiles preserve relative differences across sites.

### Memory-safe processing
The full mosaic would require **~27 GB RAM** to load at once.
Instead we use a **2-pass approach**:
- Pass 1: Subsample every 10th pixel to compute global percentiles
- Pass 2: Process each TIF file one at a time, apply global stats, write output

### Why these 6 factors and not DEM/TWI directly?
The GEE export already computed `slope` and `aspect` from ALOS AW3D30 DEM.
TWI (Topographic Wetness Index) was not exported — it can be added in a future iteration.
NDWI serves as a moisture/hydrology proxy in this phase.

---

## 14. How to Re-run / Modify

### Re-run the pipeline
```bash
cd "D:\Landslide ICTD"
python ahp_susceptibility.py
```

### Change the AHP weights
Edit the `PC_MATRIX` in `ahp_susceptibility.py`:

```python
PC_MATRIX = np.array([
    #Slope  Aspect  NDVI   NDWI   BSI    NBR
    [1,     3,      2,     3,     4,     5   ],   # Slope   <- change these
    ...
], dtype=np.float64)
```

After changing, the script automatically recomputes weights and checks CR.
If CR ≥ 0.10, revise your judgements until consistency is achieved.

### Add a new factor
1. Add the factor to the `FACTORS` list:
   ```python
   FACTORS = [
       ...
       ("TWI", "twi_band", False),   # new factor
   ]
   ```
2. Add it to `BAND_NAMES` with the correct band index in your TIF
3. Expand `PC_MATRIX` from 6×6 to 7×7
4. Re-run

### Visualize in QGIS
1. Open QGIS → Layer → Add Raster Layer
2. Load `*_susceptibility_class.tif`
3. Layer Properties → Symbology → Categorized
4. Use colour ramp: Green (1) → Red (5)

---

## References

- Saaty, T.L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Yalcin, A. (2008). GIS-based landslide susceptibility mapping using AHP method.
  *Environmental Geology*, 54, 1147–1156.
- NRSC (2023). *Landslide Atlas of India*. National Remote Sensing Centre, ISRO.
- Sentinel-2 data: ESA Copernicus Programme via Google Earth Engine.
- DEM: JAXA ALOS AW3D30 v4.1 (30 m → 10 m resampled).
