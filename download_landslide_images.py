"""
=============================================================
  LANDSLIDE IMAGE DOWNLOADER — HIMACHAL PRADESH
  Google Earth Engine Python API + geemap
  Satellite: Sentinel-2 SR Harmonized (10 m)

  SAME LOCATIONS for BOTH folders:

  landslide_dataset/
  ├── with_landslide/          ← POST-event images (Jul–Sep 2023)
  │   ├── Mandi_NH3_POST_with_landslide.tif
  │   └── ...                   Same places AFTER the landslide
  └── without_landslide/       ← PRE-event images (Mar–May 2023)
      ├── Mandi_NH3_PRE_without_landslide.tif
      └── ...                   Same places BEFORE the landslide

  SETUP:
    pip install earthengine-api geemap
    earthengine authenticate        (one-time browser login)
    Then set your GEE_PROJECT below.
=============================================================
"""

import sys
import io
# Force UTF-8 output so emoji don't crash on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import ee
import os
import time
import geemap

# ─── CONFIG — EDIT THIS ──────────────────────────────────────
GEE_PROJECT  = 'landslide-507011'   # ← replace with your GEE Cloud Project ID
BUFFER_M     = 3000    # 3 km radius patch around each site
EXPORT_SCALE = 10      # 10 m/pixel (Sentinel-2 native)
PRE_START    = '2023-03-01'
PRE_END      = '2023-05-31'
POST_START   = '2023-07-01'
POST_END     = '2023-09-30'
BANDS        = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'BSI', 'NBR', 'slope', 'aspect']

# ─── 0. AUTHENTICATE & INITIALISE ────────────────────────────
try:
    ee.Initialize(project=GEE_PROJECT)
    print("[OK] GEE initialised.")
except Exception:
    print("[AUTH] Authenticating with Google Earth Engine ...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("[OK] GEE initialised.")

# ─── 1. OUTPUT FOLDERS ───────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landslide_dataset')
LS_DIR    = os.path.join(BASE_DIR, 'with_landslide')
NO_LS_DIR = os.path.join(BASE_DIR, 'without_landslide')

os.makedirs(LS_DIR,    exist_ok=True)
os.makedirs(NO_LS_DIR, exist_ok=True)
print(f"\n📁 Output folders:")
print(f"   {LS_DIR}")
print(f"   {NO_LS_DIR}\n")

# ─── 2. LANDSLIDE SITES ──────────────────────────────────────
# Same coordinates used for BOTH folders.
# PRE composite  → without_landslide/  (before the event)
# POST composite → with_landslide/     (after the event)
SITES = [
    (76.9279, 31.7074, 'Mandi_NH3'),
    (77.1059, 31.9579, 'Kullu_Beas'),
    (78.4680, 31.5900, 'Kinnaur_Sutlej'),
    (77.1734, 31.1048, 'Shimla_NH5'),
    (76.1260, 32.5590, 'Chamba_Ravi'),
    (77.5500, 32.2700, 'Lahaul_Spiti'),
    (77.1020, 30.9045, 'Solan_Shivalik'),
    (77.6700, 30.5600, 'Sirmaur_Giri'),
    (76.2680, 32.0990, 'Kangra_Banganga'),
    (77.6270, 31.4470, 'Rampur_Sutlej'),
]

# ─── 3. CLOUD MASKING ────────────────────────────────────────
def mask_s2_clouds(image):
    qa   = image.select('QA60')
    mask = (qa.bitwiseAnd(1 << 10).eq(0)
              .And(qa.bitwiseAnd(1 << 11).eq(0)))
    return (image.updateMask(mask)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start', 'system:index']))

# ─── 4. SPECTRAL INDICES ─────────────────────────────────────
def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    bsi  = image.expression(
        '((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))',
        {'SWIR': image.select('B11'), 'RED':  image.select('B4'),
         'NIR':  image.select('B8'),  'BLUE': image.select('B2')}
    ).rename('BSI')
    nbr  = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    return image.addBands([ndvi, ndwi, bsi, nbr])

# ─── 5. COMPOSITE BUILDER ────────────────────────────────────
def get_s2_composite(geom: ee.Geometry, start: str, end: str) -> ee.Image:
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(geom)
              .filterDate(start, end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .map(mask_s2_clouds)
              .map(add_indices)
              .median()
              .clip(geom))

# ─── 6. TERRAIN (ALOS DEM) ───────────────────────────────────
# ALOS AW3D30 has better Himalayan coverage than SRTM.
# Slope + aspect are merged as bands before download.
_dem     = ee.Image('JAXA/ALOS/AW3D30/V4_1').select('DSM')
_terrain = ee.Terrain.products(_dem)
SLOPE    = _terrain.select('slope')   # degrees (0–90)
ASPECT   = _terrain.select('aspect')  # degrees (0–360)

# ─── 7. DOWNLOAD HELPER ──────────────────────────────────────
def download_patch(image: ee.Image, geom: ee.Geometry, filename: str, out_dir: str):
    filepath = os.path.join(out_dir, filename + '.tif')
    if os.path.exists(filepath):
        print(f"  [SKIP] {filename}.tif already exists.")
        return
    print(f"  [⬇]   Downloading {filename}.tif …", end=' ', flush=True)
    try:
        # Merge slope + aspect into the composite before selecting bands
        combined = image.addBands(SLOPE.clip(geom)).addBands(ASPECT.clip(geom))
        geemap.ee_export_image(
            combined.select(BANDS),
            filename=filepath,
            scale=EXPORT_SCALE,
            region=geom,
            file_per_band=False,
        )
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌  ERROR: {e}")
    time.sleep(2)

# ─── 8. MAIN LOOP ────────────────────────────────────────────
print("=" * 60)
print(f"Processing {len(SITES)} sites — same locations, two time windows")
print("=" * 60)

for lon, lat, name in SITES:
    print(f"\n📍 {name}  ({lat:.4f}°N, {lon:.4f}°E)")
    geom = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)

    pre_img  = get_s2_composite(geom, PRE_START,  PRE_END)   # no landslide visible
    post_img = get_s2_composite(geom, POST_START, POST_END)  # landslide visible

    # ── with_landslide/    → POST composite ───────────────────
    download_patch(post_img, geom, f"{name}_POST_with_landslide",   LS_DIR)

    # ── without_landslide/ → PRE  composite ───────────────────
    download_patch(pre_img,  geom, f"{name}_PRE_without_landslide", NO_LS_DIR)

# ─── 8. SUMMARY ──────────────────────────────────────────────
ls_files    = [f for f in os.listdir(LS_DIR)    if f.endswith('.tif')]
no_ls_files = [f for f in os.listdir(NO_LS_DIR) if f.endswith('.tif')]

print("\n" + "=" * 60)
print("🎉  DOWNLOAD COMPLETE")
print("=" * 60)
print(f"  with_landslide/    → {len(ls_files)} files")
print(f"  without_landslide/ → {len(no_ls_files)} files")
print(f"\n  Saved under: {BASE_DIR}")
