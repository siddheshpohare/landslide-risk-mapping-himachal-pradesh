# Landslide Image Dataset — Himachal Pradesh
### Using Sentinel-2 via Google Earth Engine

---

## 📁 Output Folder Structure

```
landslide_dataset/
├── with_landslide/          ← Sentinel-2 patches at known landslide sites
│   ├── Mandi_NH3_POST.tif   (post-monsoon 2023 — landslide visible)
│   ├── Mandi_NH3_PRE.tif    (pre-monsoon 2023 — baseline)
│   ├── Kullu_Beas_POST.tif
│   └── ...
└── without_landslide/       ← Stable, unaffected patches
    ├── Bilaspur_stable_POST.tif
    ├── Bilaspur_stable_PRE.tif
    └── ...
```

---

## 🛰️ Data Source

| Property | Value |
|---|---|
| **Satellite** | Sentinel-2 MSI (Copernicus) |
| **Collection** | `COPERNICUS/S2_SR_HARMONIZED` |
| **Resolution** | 10 m/pixel (RGB + NIR), 20 m resampled for SWIR |
| **PRE-event window** | March 1 – May 31, 2023 (dry season) |
| **POST-event window** | July 1 – Sep 30, 2023 (peak monsoon) |
| **Cloud filter** | < 30% cloud cover per scene |
| **Composite** | Median composite of filtered scenes |

### Bands Exported
| Band | Description |
|---|---|
| B2 | Blue (490 nm) |
| B3 | Green (560 nm) |
| B4 | Red (665 nm) |
| B8 | NIR (842 nm) |
| B11 | SWIR-1 (1610 nm) |
| B12 | SWIR-2 (2190 nm) |
| NDVI | Normalised Difference Vegetation Index |
| NDWI | Normalised Difference Water Index |
| BSI | Bare Soil Index |
| NBR | Normalised Burn Ratio |

---

## 📍 Sites Covered

### With Landslide (High-Risk Zones — 2023 Monsoon)
| Site Name | Lat | Lon | District |
|---|---|---|---|
| Mandi_NH3 | 31.7074 | 76.9279 | Mandi |
| Kullu_Beas | 31.9579 | 77.1059 | Kullu |
| Kinnaur_Sutlej | 31.5900 | 78.4680 | Kinnaur |
| Shimla_NH5 | 31.1048 | 77.1734 | Shimla |
| Chamba_Ravi | 32.5590 | 76.1260 | Chamba |
| Lahaul_Spiti | 32.2700 | 77.5500 | Lahaul & Spiti |
| Solan_Shivalik | 30.9045 | 77.1020 | Solan |
| Sirmaur_Giri | 30.5600 | 77.6700 | Sirmaur |

### Without Landslide (Stable Zones)
| Site Name | Lat | Lon | District |
|---|---|---|---|
| Bilaspur_stable | 31.3200 | 76.5200 | Bilaspur |
| Una_stable | 32.1020 | 76.3890 | Una |
| Hamirpur_stable | 31.5230 | 76.7890 | Hamirpur |
| Nalagarh_stable | 31.1040 | 76.2540 | Solan |
| Rampur_stable | 31.6700 | 77.4500 | Shimla |
| Kangra_valley | 32.0940 | 75.8600 | Kangra |

Each site is a **3 km radius circular patch** (~6×6 km area).

---

## 🚀 Option A — GEE Code Editor (Recommended)

1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Copy the contents of **`gee_landslide_hp.js`** into the editor
3. Click **Run** — the map will show all sites
4. Open the **Tasks** tab (top-right panel)
5. Click **RUN** for each export task
6. Files will appear in **Google Drive → `HP_Landslide/`**

> **Note:** GEE export runs on Google's servers. No local compute needed.

---

## 🐍 Option B — Python Local Download

### Prerequisites
```bash
pip install earthengine-api geemap
earthengine authenticate   # one-time browser login
```

### Run
```bash
# Edit download_landslide_images.py first:
# Change 'your-gee-project-id' to your actual GEE project ID
# (Create one at https://console.cloud.google.com/)

python download_landslide_images.py
```

Files download directly to `landslide_dataset/` next to the script.

---

## 🔍 Change Detection Logic

The script automatically computes:

```
dNDVI = NDVI_post − NDVI_pre

Landslide Pixel = (dNDVI < −0.15)      # significant vegetation loss
               AND (dBSI > 0.05)        # bare soil increased
               AND (slope > 10°)        # on steep terrain
```

A **landslide_mask** band is also included in the `with_landslide` exports.

---

## ⚠️ Important Notes

1. **Cloud cover** during the 2023 monsoon is heavy. The 30% filter + median compositing minimises cloud artifacts but some patches may have gaps.
2. **Validation**: Cross-reference detected sites with [NRSC Landslide Atlas of India](https://www.nrsc.gov.in) or state SDMA reports.
3. **Expand the dataset**: Add more point coordinates in the `LANDSLIDE_SITES` list to generate more training samples.
4. For **deep learning / ML training**, use the PRE+POST pairs to create difference images as model inputs.
