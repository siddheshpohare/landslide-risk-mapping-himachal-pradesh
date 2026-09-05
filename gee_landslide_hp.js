// ============================================================
//  LANDSLIDE IMAGE COLLECTOR — HIMACHAL PRADESH
//  Google Earth Engine Code Editor Script
//  Satellite: Sentinel-2 SR Harmonized (10 m)
//
//  SAME LOCATIONS used for BOTH folders:
//
//  with_landslide/    → POST-event images (Jul–Sep 2023)
//                       Same places AFTER the landslide hit
//
//  without_landslide/ → PRE-event images (Mar–May 2023)
//                       Same places BEFORE the landslide
//                       (no damage visible — normal terrain)
//
//  This is the correct approach for ML training:
//  identical geography, different time window.
// ============================================================
//
//  HOW TO USE:
//  1. Open https://code.earthengine.google.com/
//  2. Paste this entire script and click "Run".
//  3. Open the Tasks tab → click RUN for every export task.
//  4. Images saved to Google Drive:
//       HP_Landslide/with_landslide/     ← POST images
//       HP_Landslide/without_landslide/  ← PRE  images
// ============================================================

// ─── 1. STUDY AREA ───────────────────────────────────────────
var states = ee.FeatureCollection('FAO/GAUL/2015/level1');
var hp     = states.filter(ee.Filter.eq('ADM1_NAME', 'Himachal Pradesh'));
Map.addLayer(hp, {color: '4682B4'}, 'Himachal Pradesh');
Map.centerObject(hp.geometry(), 8);

// ─── 2. LANDSLIDE SITES (SAME COORDS FOR BOTH FOLDERS) ───────
// Known high-risk corridors that were struck during 2023 monsoon.
// PRE images  → without_landslide/
// POST images → with_landslide/
var landslideSites = ee.FeatureCollection([
  // Mandi — NH-3 / NH-154 corridor (frequent landslide zone)
  ee.Feature(ee.Geometry.Point([76.9279, 31.7074]), {name: 'Mandi_NH3',       district: 'Mandi'}),
  // Kullu — Beas River valley
  ee.Feature(ee.Geometry.Point([77.1059, 31.9579]), {name: 'Kullu_Beas',      district: 'Kullu'}),
  // Kinnaur — Sutlej river gorge
  ee.Feature(ee.Geometry.Point([78.4680, 31.5900]), {name: 'Kinnaur_Sutlej',  district: 'Kinnaur'}),
  // Shimla — NH-5 (Hindustan–Tibet Road)
  ee.Feature(ee.Geometry.Point([77.1734, 31.1048]), {name: 'Shimla_NH5',      district: 'Shimla'}),
  // Chamba — Ravi River gorge
  ee.Feature(ee.Geometry.Point([76.1260, 32.5590]), {name: 'Chamba_Ravi',     district: 'Chamba'}),
  // Lahaul & Spiti — Chandrabhaga valley
  ee.Feature(ee.Geometry.Point([77.5500, 32.2700]), {name: 'Lahaul_Spiti',    district: 'Lahaul'}),
  // Solan — Shivalik foothills
  ee.Feature(ee.Geometry.Point([77.1020, 30.9045]), {name: 'Solan_Shivalik',  district: 'Solan'}),
  // Sirmaur — Giri River
  ee.Feature(ee.Geometry.Point([77.6700, 30.5600]), {name: 'Sirmaur_Giri',    district: 'Sirmaur'}),
  // Kangra — Banganga River area
  ee.Feature(ee.Geometry.Point([76.2680, 32.0990]), {name: 'Kangra_Banganga', district: 'Kangra'}),
  // Rampur — Sutlej valley (NH-5 Rampur area)
  ee.Feature(ee.Geometry.Point([77.6270, 31.4470]), {name: 'Rampur_Sutlej',   district: 'Shimla'}),
]);

// 3 km radius patch around each site
var sitesBuffered = landslideSites.map(function(f) {
  return f.buffer(3000);
});

// Visualise the sites
Map.addLayer(sitesBuffered, {color: 'FF4500'}, 'Landslide Sites (both folders)');

// ─── 3. CLOUD MASKING ────────────────────────────────────────
function maskS2clouds(image) {
  var qa            = image.select('QA60');
  var cloudBitMask  = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image
    .updateMask(mask)
    .divide(10000)   // scale to 0–1 reflectance
    .copyProperties(image, ['system:time_start', 'system:index']);
}

// ─── 4. SPECTRAL INDICES ─────────────────────────────────────
function addIndices(image) {
  // NDVI — vegetation health (drops steeply after a landslide)
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  // NDWI — moisture / water content
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  // BSI  — bare soil index (rises after soil exposure)
  var bsi  = image.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    {SWIR: image.select('B11'), RED: image.select('B4'),
     NIR:  image.select('B8'),  BLUE: image.select('B2')}
  ).rename('BSI');
  // NBR  — Normalised Burn Ratio (sensitive to soil disturbance)
  var nbr  = image.normalizedDifference(['B8', 'B12']).rename('NBR');
  return image.addBands([ndvi, ndwi, bsi, nbr]);
}

// ─── 5. DATE WINDOWS ─────────────────────────────────────────
//  PRE  = dry season before 2023 monsoon  → NO landslide visible
//  POST = peak 2023 monsoon               → landslide visible
var preStart  = '2023-03-01';
var preEnd    = '2023-05-31';
var postStart = '2023-07-01';
var postEnd   = '2023-09-30';

// ─── 6. COMPOSITE BUILDER ────────────────────────────────────
function getS2Composite(geom, startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(maskS2clouds)
    .map(addIndices)
    .median()
    .clip(geom);
}

// ─── 7. TERRAIN (SLOPE + ASPECT) ─────────────────────────────
// ALOS AW3D30 has better coverage over steep Himalayan terrain than SRTM.
// Slope is exported as a band (not used as a hard mask) so the ML model
// can learn its own relevance from the data.
var dem     = ee.Image('JAXA/ALOS/AW3D30/V4_1').select('DSM');
var terrain = ee.Terrain.products(dem);   // computes slope, aspect, hillshade
var slope   = terrain.select('slope');    // degrees (0–90)
var aspect  = terrain.select('aspect');   // degrees (0–360)

// ─── 8. CHANGE DETECTION ─────────────────────────────────────
// Used only for map visualisation — not stored in exports
function changeDetection(preComp, postComp, slopeBand) {
  var dNDVI = postComp.select('NDVI').subtract(preComp.select('NDVI')).rename('dNDVI');
  var dBSI  = postComp.select('BSI').subtract(preComp.select('BSI')).rename('dBSI');
  var dNBR  = postComp.select('NBR').subtract(preComp.select('NBR')).rename('dNBR');

  // Slope > 10° as a soft spectral condition (keeps all pixels, just flags steep ones)
  var steepCondition = slopeBand.gt(10);

  // Landslide pixels: vegetation dropped + bare soil rose + steep slope
  var lsMask = dNDVI.lt(-0.15)
      .and(dBSI.gt(0.05))
      .and(steepCondition)
      .rename('landslide_mask');

  return ee.Image.cat([dNDVI, dBSI, dNBR, lsMask]);
}

// ─── 9. VISUALISATION ────────────────────────────────────────
var rgbVis    = {min: 0,    max: 0.35, bands: ['B4','B3','B2']};
var falseVis  = {min: 0,    max: 0.35, bands: ['B8','B4','B3']};  // NIR-RGB
var changVis  = {min: -0.5, max: 0.1,  palette: ['FF0000','FFFFFF']};
var maskVis   = {min: 0,    max: 1,    palette: ['white','red']};

// Build and display all 10 sites on the map
var siteList = sitesBuffered.toList(20);
var nSites   = landslideSites.size().getInfo();

print('Rendering', nSites, 'sites on map …');

for (var i = 0; i < nSites; i++) {
  var feat     = ee.Feature(siteList.get(i));
  var geom     = feat.geometry();

  var preComp  = getS2Composite(geom, preStart,  preEnd);   // without_landslide
  var postComp = getS2Composite(geom, postStart, postEnd);  // with_landslide
  var change   = changeDetection(preComp, postComp, slope.clip(geom));

  // PRE layer (without landslide)
  Map.addLayer(preComp,  rgbVis,   'PRE_RGB_'  + i, false);
  // POST layer (with landslide)
  Map.addLayer(postComp, rgbVis,   'POST_RGB_' + i, true);
  // False colour POST (NIR helps see exposed soil)
  Map.addLayer(postComp, falseVis, 'POST_NIR_' + i, false);
  // Change map
  Map.addLayer(change.select('dNDVI'),         changVis, 'dNDVI_'  + i, false);
  Map.addLayer(change.select('landslide_mask'), maskVis,  'LSmask_' + i, false);
}

// ─── 10. EXPORT SETTINGS ─────────────────────────────────────
var EXPORT_SCALE  = 10;              // 10 m/pixel (S2 native)
var EXPORT_FOLDER = 'HP_Landslide';
// Bands: True colour + NIR + SWIR + 4 indices + slope + aspect = 12 bands per file
var BANDS = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','BSI','NBR','slope','aspect'];

function exportPatch(image, fileName, subfolder, geom) {
  // Merge slope + aspect into the S2 composite before band selection
  var combined = image.addBands(slope.clip(geom)).addBands(aspect.clip(geom));
  Export.image.toDrive({
    image:          combined.select(BANDS),
    description:    fileName,
    folder:         EXPORT_FOLDER + '/' + subfolder,
    fileNamePrefix: fileName,
    region:         geom,
    scale:          EXPORT_SCALE,
    maxPixels:      1e9,
    fileFormat:     'GeoTIFF',
    formatOptions:  {cloudOptimized: true}
  });
}

// ─── 11. EXPORT — SAME SITES, TWO TIME WINDOWS ───────────────
//
//  with_landslide/    ← POST composites (Jul–Sep 2023)
//                        Same places where landslides occurred
//
//  without_landslide/ ← PRE  composites (Mar–May 2023)
//                        Same places BEFORE any landslide —
//                        terrain looks normal/vegetated

var siteList2 = sitesBuffered.toList(20);

landslideSites.evaluate(function(fc) {
  fc.features.forEach(function(feat, idx) {
    var geom     = ee.Feature(siteList2.get(idx)).geometry();
    var siteName = feat.properties.name;

    // Build PRE and POST composites for this exact location
    var preImg  = getS2Composite(geom, preStart,  preEnd);
    var postImg = getS2Composite(geom, postStart, postEnd);

    // ── with_landslide → POST image (landslide has occurred) ──
    exportPatch(postImg, siteName + '_POST_with_landslide',    'with_landslide',    geom);

    // ── without_landslide → PRE image (same place, no damage) ─
    exportPatch(preImg,  siteName + '_PRE_without_landslide',  'without_landslide', geom);
  });
});

// ─── 12. CONSOLE OUTPUT ──────────────────────────────────────
print('');
print('✅ Script loaded successfully!');
print('────────────────────────────────────────────');
print('SAME LOCATIONS — TWO TIME WINDOWS:');
print('  with_landslide/    → POST-monsoon 2023 (Jul–Sep)');
print('                        Landslide scars visible');
print('  without_landslide/ → PRE-monsoon  2023 (Mar–May)');
print('                        Same places, no damage yet');
print('────────────────────────────────────────────');
print('Open the Tasks tab → click RUN for each task.');
print('Files → Google Drive: HP_Landslide/');
