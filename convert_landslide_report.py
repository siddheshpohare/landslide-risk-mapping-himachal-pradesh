"""
convert_landslide_report.py
============================
Converts landslide_report.pdf  (Landslide Inventory – Field Validated, 904 pages)
into a clean, preprocessed CSV file.

Source columns (PDF):
    Sl.No. | Slide_No | State | District | Slide_Name | NH_SH_Location
    | Latitude | Longitude | Material Involved | Movement Type | History

Outputs
-------
landslide_report_raw.csv            – verbatim table extraction
landslide_report_preprocessed.csv   – cleaned & feature-engineered

Usage
-----
    pip install pdfplumber pandas tqdm
    python convert_landslide_report.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_PATH  = Path(__file__).parent / "landslide_report.pdf"
RAW_CSV   = Path(__file__).parent / "landslide_report_raw.csv"
CLEAN_CSV = Path(__file__).parent / "landslide_report_preprocessed.csv"

COLUMNS = [
    "sl_no", "slide_no", "state", "district", "slide_name",
    "nh_sh_location", "latitude", "longitude",
    "material_involved", "movement_type", "history",
]

# Header patterns that repeat on every page – skip them
HEADER_PATTERNS = [
    re.compile(r"landslide inventory", re.I),
    re.compile(r"sl\.?\s*no\.?", re.I),
    re.compile(r"field vaid", re.I),
    re.compile(r"^movement\s*type$", re.I),
    re.compile(r"^nh.?sh", re.I),
]

# India bounding box for coordinate validation
LAT_RANGE = (6.0,  38.0)
LON_RANGE = (67.0, 98.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_cell(val) -> str:
    """Strip whitespace; collapse internal newlines / spaces."""
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
    """Page-by-page table extraction using pdfplumber."""
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
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
    """Keep data rows only; align to exactly 11 columns."""
    # Only rows whose first cell is numeric (sl_no)
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

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline
    --------
    1. Deduplicate on slide_no
    2. Cast numeric columns  (sl_no, latitude, longitude)
    3. Title-case text columns
    4. Replace NA-like strings with NaN
    5. Drop rows missing both lat & lon
    6. Flag coordinates outside India bounding box
    7. Derive combined  material_movement  label
    8. Binary  has_history  flag
    """
    print("[INFO] Preprocessing ...")

    # 1. Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["slide_no"])
    print(f"  Deduplicated : -{before - len(df)} rows")

    # 2. Numeric casting
    df["sl_no"]     = pd.to_numeric(df["sl_no"],     errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # 3. Text standardisation
    for col in ["state", "district", "slide_name", "material_involved", "movement_type"]:
        df[col] = df[col].str.strip().str.title()
    for col in ["slide_no", "nh_sh_location", "history"]:
        df[col] = df[col].str.strip()

    # 4. NA-like strings -> actual NaN
    na_re = re.compile(r"^(na|n/a|not available|unknown|nil|-)$", re.I)
    for col in ["history", "nh_sh_location"]:
        mask = df[col].str.match(na_re, na=False)
        df.loc[mask, col] = pd.NA

    # 5. Drop rows with no coordinates at all
    both_missing = df["latitude"].isna() & df["longitude"].isna()
    n_drop = both_missing.sum()
    if n_drop:
        print(f"  Dropped {n_drop} rows with no coordinates.")
    df = df[~both_missing].copy()

    # 6. Coordinate validation flag
    in_india = (df["latitude"].between(*LAT_RANGE)) & (df["longitude"].between(*LON_RANGE))
    df["coord_flag"] = (~in_india).astype(int)
    n_flag = df["coord_flag"].sum()
    if n_flag:
        print(f"  coord_flag=1 for {n_flag} out-of-range rows.")

    # 7. Combined material + movement label
    df["material_movement"] = (
        df["material_involved"].fillna("") + " " + df["movement_type"].fillna("")
    ).str.strip()

    # 8. History availability flag
    df["has_history"] = df["history"].notna().astype(int)

    print(f"  Final rows   : {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PDF_PATH.exists():
        sys.exit(f"[ERROR] File not found: {PDF_PATH}")

    print("=" * 60)
    print("  Landslide Report PDF -> CSV")
    print("=" * 60)

    raw_df  = extract_tables(PDF_PATH)
    norm_df = normalise(raw_df)
    norm_df.to_csv(RAW_CSV, index=False)
    print(f"[OK] Raw CSV           -> {RAW_CSV}  ({len(norm_df)} rows)")

    clean_df = preprocess(norm_df)
    clean_df.to_csv(CLEAN_CSV, index=False)
    print(f"[OK] Preprocessed CSV  -> {CLEAN_CSV}  ({len(clean_df)} rows)")

    print("\n-- Summary -------------------------------------------------")
    print(f"  Records          : {len(clean_df)}")
    print(f"  States           : {clean_df['state'].nunique()}")
    print(f"  Districts        : {clean_df['district'].nunique()}")
    print(f"  With history     : {int(clean_df['has_history'].sum())}")
    print(f"  Lat range        : {clean_df['latitude'].min():.4f} to {clean_df['latitude'].max():.4f}")
    print(f"  Lon range        : {clean_df['longitude'].min():.4f} to {clean_df['longitude'].max():.4f}")
    print("  Material types   :")
    for k, v in clean_df["material_involved"].value_counts().items():
        print(f"    {k}: {v}")
    print("  Movement types   :")
    for k, v in clean_df["movement_type"].value_counts().items():
        print(f"    {k}: {v}")
    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
