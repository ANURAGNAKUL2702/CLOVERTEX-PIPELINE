import pandas as pd
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
RAW_DIR = Path("datalake/raw")
REFINED_DIR = Path("datalake/refined/v1")
PARTITION_PREFIX = "ingest_date="


# -------------------------------
# STANDARDIZE PATIENT ID
# -------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def standardize_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {"patient_ref", "patientid", "patient_id", "patient"}
    for col in df.columns:
        clean = col.lower().replace("-", "_")
        if clean in aliases and col != "patient_id":
            df = df.rename(columns={col: "patient_id"})
            break
    return df


# -------------------------------
# FLATTEN JSON COLUMNS
# -------------------------------
def flatten_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in list(df.columns):
        if df[col].dtype == "object":
            mask = df[col].apply(lambda x: isinstance(x, dict))

            if mask.any():
                expanded = pd.json_normalize(df.loc[mask, col])
                expanded.index = df.loc[mask].index
                expanded = expanded.add_prefix(f"{col}_")
                expanded = expanded.reindex(df.index)
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    return df


# -------------------------------
# CONVERT DATES
# -------------------------------
def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if "date" in col or col.endswith("_dt"):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# -------------------------------
# CLEAN ONE DATAFRAME
# -------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # 0. Normalize columns for consistency across sources
    df = normalize_columns(df)

    # 1. Standardize patient_id
    df = standardize_patient_id(df)

    # 2. Flatten JSON
    df = flatten_json_columns(df)

    # 3. Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # 4. Standardize missing values
    df = df.replace(["", "NA", "N/A", "null", "None"], pd.NA, regex=False)

    # 5. Strip whitespace
    for col in df.select_dtypes(include="object"):
        if not df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].astype("string").str.strip()

    # 6. Normalize categorical columns
    for col in df.columns:
        if any(key in col for key in ["category", "status", "severity"]):
            df[col] = df[col].str.lower()

    # 7. Convert date columns
    df = convert_dates(df)

    return df


# -------------------------------
# SAVE CLEANED DATA
# -------------------------------
def save_clean(df: pd.DataFrame, dataset: str, filename: str):
    output_path = REFINED_DIR / dataset
    output_path.mkdir(parents=True, exist_ok=True)

    final_file = output_path / filename
    temp_file = output_path / f"{filename}.tmp"
    df.to_parquet(temp_file, index=False)
    temp_file.replace(final_file)


def iter_raw_files():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dir not found: {RAW_DIR}")

    files = list(RAW_DIR.glob(f"**/{PARTITION_PREFIX}*/*.parquet"))

    # Fallback for non-partitioned layout
    if not files:
        files = list(RAW_DIR.glob("*/*.parquet"))

    return files


# -------------------------------
# MAIN CLEANING PIPELINE
# -------------------------------
def run_cleaning():
    print("🧹 Starting Cleaning...\n")

    raw_files = iter_raw_files()
    success, failed, total_rows = 0, 0, 0

    for file in raw_files:
        try:
            # raw/<dataset>/ingest_date=YYYY-MM-DD/<file>
            parts = file.parts
            dataset_name = parts[parts.index("raw") + 1] if "raw" in parts else file.parent.name
            ingest_partition = next((p for p in parts if p.startswith(PARTITION_PREFIX)), None)

            print(f"📂 Cleaning: {dataset_name}/{file.name}")

            df = pd.read_parquet(file)

            before_rows, before_cols = df.shape

            # Clean
            df = clean_dataframe(df)

            after_rows, after_cols = df.shape
            total_rows += after_rows

            # Save with same partitioning
            output_dir = REFINED_DIR / dataset_name
            if ingest_partition:
                output_dir = output_dir / ingest_partition

            output_dir.mkdir(parents=True, exist_ok=True)
            save_clean(df, str(output_dir.relative_to(REFINED_DIR)), file.name)

            print(f"   Rows: {before_rows} → {after_rows}, Cols: {before_cols} → {after_cols}")
            print(f"   Saved → {output_dir}/\n")

            success += 1

        except Exception as e:
            print(f"❌ Error: {file.name} → {e}\n")
            failed += 1

    # -------------------------------
    # SUMMARY
    # -------------------------------
    print("📊 Cleaning Summary")
    print(f"   Success     : {success}")
    print(f"   Failed      : {failed}")
    print(f"   Total Rows  : {total_rows}")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_cleaning()
