"""
convert_landslide_inventory.py
================================
Converts the HP Landslide Inventory PDF (204 pages) into a clean CSV.

Source columns (PDF):
    Sr.No. | District | Longitude (East) | Latitude (North)
    | Landuse/Landcover | Geomorphology | Length in Meter
    | Area in Sq Meter | Toposheet | Object ID

Outputs
-------
landslide_inventory_raw.csv            – verbatim table extraction
landslide_inventory_preprocessed.csv   – cleaned & feature-engineered

Usage
-----
    pip install pdfplumber pandas numpy tqdm
    python convert_landslide_inventory.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent

# Auto-detect the inventory PDF by filename prefix
candidates = sorted(BASE_DIR.glob("Landslide Inventory*.pdf"))
if not candidates:
    sys.exit("[ERROR] No 'Landslide Inventory*.pdf' found in the script directory.")
PDF_PATH  = candidates[0]

RAW_CSV   = BASE_DIR / "landslide_inventory_raw.csv"
CLEAN_CSV = BASE_DIR / "landslide_inventory_preprocessed.csv"

COLUMNS = [
    "sr_no", "district", "longitude", "latitude",
    "landuse_landcover", "geomorphology",
    "length_m", "area_sqm", "toposheet", "object_id",
]

# Himachal Pradesh approximate bounding box
HP_LAT = (30.0, 33.5)
HP_LON = (75.5, 79.0)

HEADER_PATTERNS = [
    re.compile(r"inventory of probable", re.I),
    re.compile(r"spatial information", re.I),
    re.compile(r"sr\.?\s*no\.?", re.I),
    re.compile(r"details of landslide", re.I),
    re.compile(r"^landuse", re.I),
    re.compile(r"^geomorphology", re.I),
    re.compile(r"himachal pradesh", re.I),
    re.compile(r"lenth in meter", re.I),
]

# Vegetation density mapping (ordinal)
VEG_MAP = {
    "Sparsely Vegetated":  1,
    "Moderate Vegetation": 2,
    "Moderately Vegetated": 2,
    "Thick Vegetation":    3,
    "Thickly Vegetated":   3,
    "Agricultural Land":   2,
    "Built Up":            0,
    "Barren":              0,
    "Water Body":          0,
}

# Size class bins (area_sqm)
AREA_BINS   = [0, 500, 2_000, 10_000, float("inf")]
AREA_LABELS = ["small", "medium", "large", "very_large"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_cell(val) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def is_header_row(row: list) -> bool:
    text = " ".join(c for c in row if c)
    return any(p.search(text) for p in HEADER_PATTERNS)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_tables(pdf_path: Path) -> pd.DataFrame:
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[INFO] PDF file  : {pdf_path.name}")
        print(f"[INFO] PDF pages : {total}")
        for page in tqdm(pdf.pages, total=total, unit="page"):
            for table in page.extract_tables():
                for row in table:
                    cleaned = [clean_cell(c) for c in row]
                    if not any(cleaned) or is_header_row(cleaned):
                        continue
                    records.append(cleaned)
    print(f"[INFO] Raw rows collected : {len(records)}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.iloc[:, 0].str.match(r"^\d+$", na=False)].copy()
    target = len(COLUMNS)
    n = df.shape[1]
    if n < target:
        for i in range(target - n):
            df[n + i] = ""
    elif n > target:
        df = df.iloc[:, :target]
    df.columns = COLUMNS
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def geom_class(g: str) -> int:
    g = str(g)
    if "Lowly" in g:       return 1
    if "Moderately" in g:  return 2
    if "Highly" in g:      return 3
    if "Denudational" in g: return 4
    if "Structural" in g:  return 5
    return 0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline
    --------
    1.  Deduplicate on object_id
    2.  Cast numeric columns
    3.  Title-case / upper-case text columns
    4.  Drop rows missing both lat & lon
    5.  Flag coordinates outside Himachal Pradesh bounding box
    6.  Median-impute missing length_m / area_sqm
    7.  Derive area_ha, size_class, log_area, aspect_ratio
    8.  Encode veg_density (ordinal) from landuse_landcover
    9.  Encode geom_class  (ordinal) from geomorphology
    """
    print("[INFO] Preprocessing ...")

    # 1. Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["object_id"])
    print(f"  Deduplicated : -{before - len(df)} rows")

    # 2. Numeric casting
    for col in ["latitude", "longitude", "length_m", "area_sqm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sr_no"]     = pd.to_numeric(df["sr_no"],     errors="coerce")
    df["object_id"] = pd.to_numeric(df["object_id"], errors="coerce")

    # 3. Text standardisation
    for col in ["district", "landuse_landcover", "geomorphology"]:
        df[col] = df[col].str.strip().str.title()
    df["toposheet"] = df["toposheet"].str.strip().str.upper()

    # 4. Drop rows with no coordinates
    both_missing = df["latitude"].isna() & df["longitude"].isna()
    n_drop = both_missing.sum()
    if n_drop:
        print(f"  Dropped {n_drop} rows with no coordinates.")
    df = df[~both_missing].copy()

    # 5. HP bounding-box flag
    in_hp = df["latitude"].between(*HP_LAT) & df["longitude"].between(*HP_LON)
    df["coord_flag"] = (~in_hp).astype(int)
    n_flag = df["coord_flag"].sum()
    if n_flag:
        print(f"  coord_flag=1 for {n_flag} rows outside HP bounding box.")

    # 6. Median imputation for length_m and area_sqm
    for col in ["length_m", "area_sqm"]:
        med = df[col].median()
        n_miss = df[col].isna().sum()
        if n_miss:
            df[col] = df[col].fillna(med)
            print(f"  Imputed {n_miss} missing {col} with median ({med:.2f}).")

    # 7. Derived geometric features
    df["area_ha"]      = df["area_sqm"] / 10_000
    df["log_area"]     = np.log1p(df["area_sqm"])
    df["aspect_ratio"] = df["length_m"] / np.sqrt(df["area_sqm"].clip(lower=1e-6))
    df["size_class"]   = pd.cut(df["area_sqm"], bins=AREA_BINS, labels=AREA_LABELS)

    # 8. Vegetation density ordinal
    df["veg_density"] = df["landuse_landcover"].map(VEG_MAP).fillna(-1).astype(int)

    # 9. Geomorphology class ordinal
    df["geom_class"] = df["geomorphology"].apply(geom_class)

    print(f"  Final rows   : {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Landslide Inventory PDF -> CSV  (Himachal Pradesh)")
    print("=" * 60)

    raw_df  = extract_tables(PDF_PATH)
    norm_df = normalise(raw_df)
    norm_df.to_csv(RAW_CSV, index=False)
    print(f"[OK] Raw CSV           -> {RAW_CSV}  ({len(norm_df)} rows)")

    clean_df = preprocess(norm_df)
    clean_df.to_csv(CLEAN_CSV, index=False)
    print(f"[OK] Preprocessed CSV  -> {CLEAN_CSV}  ({len(clean_df)} rows)")

    print("\n-- Summary -------------------------------------------------")
    print(f"  Total sites      : {len(clean_df)}")
    print(f"  Districts        : {clean_df['district'].nunique()}")
    print(f"  District list    : {sorted(clean_df['district'].dropna().unique())}")
    print(f"  Lat range        : {clean_df['latitude'].min():.4f} to {clean_df['latitude'].max():.4f}")
    print(f"  Lon range        : {clean_df['longitude'].min():.4f} to {clean_df['longitude'].max():.4f}")
    print(f"  Area range (sqm) : {clean_df['area_sqm'].min():.2f} to {clean_df['area_sqm'].max():.2f}")
    print(f"  Length range (m) : {clean_df['length_m'].min():.2f} to {clean_df['length_m'].max():.2f}")
    print("  Size class dist  :")
    for k, v in clean_df["size_class"].value_counts().sort_index().items():
        print(f"    {k}: {v}")
    print("  Landuse types    :")
    for k, v in clean_df["landuse_landcover"].value_counts().items():
        print(f"    {k}: {v}")
    print("  Geomorphology    :")
    for k, v in clean_df["geomorphology"].value_counts().items():
        print(f"    {k}: {v}")
    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
