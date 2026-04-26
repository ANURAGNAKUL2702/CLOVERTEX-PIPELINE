import pandas as pd
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
REFINED_DIR = Path("datalake/refined/v1")
CONSUMPTION_DIR = Path("datalake/consumption/v1")
PARTITION_PREFIX = "ingest_date="


# -------------------------------
# SAFE JSON FLATTENING
# -------------------------------
def flatten_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in list(df.columns):
        if df[col].dtype in ("object", "string"):
            mask = df[col].apply(lambda x: isinstance(x, dict))

            if mask.any():
                # Normalize only dict rows
                expanded = pd.json_normalize(df.loc[mask, col])
                expanded.index = df.loc[mask].index  # preserve alignment
                expanded = expanded.add_prefix(f"{col}_")

                # Align with full dataframe index
                expanded = expanded.reindex(df.index)

                # Merge back
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    return df


# -------------------------------
# ROBUST DATE CONVERSION
# -------------------------------
def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if "date" in col or col.endswith("_dt"):
            before_na = df[col].isna().sum()

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            after_na = df[col].isna().sum()
            newly_na = after_na - before_na

            if newly_na > 0:
                print(f"⚠️ {col}: {newly_na} values could not be parsed → NaT")

    return df


# -------------------------------
# BOOLEAN CONVERSION (Y/N)
# -------------------------------
def convert_boolean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ("object", "string"):
            values = set(df[col].dropna().unique())

            if values <= {"Y", "N"}:
                df[col] = df[col].map({"Y": True, "N": False})

    return df


# -------------------------------
# TRANSFORM ONE DATAFRAME
# -------------------------------
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_json_columns(df)
    df = convert_dates(df)
    df = convert_boolean(df)
    return df


# -------------------------------
# SAVE TRANSFORMED DATA
# -------------------------------
def save_transformed(df: pd.DataFrame, dataset: str, filename: str):
    output_path = CONSUMPTION_DIR / dataset
    output_path.mkdir(parents=True, exist_ok=True)

    final_file = output_path / filename
    temp_file = output_path / f"{filename}.tmp"
    df.to_parquet(temp_file, index=False)
    temp_file.replace(final_file)


def iter_refined_files():
    if not REFINED_DIR.exists():
        raise FileNotFoundError(f"Refined dir not found: {REFINED_DIR}")

    files = list(REFINED_DIR.glob(f"**/{PARTITION_PREFIX}*/*.parquet"))

    # Fallback for non-partitioned layout
    if not files:
        files = list(REFINED_DIR.glob("*/*.parquet"))

    return files


def partition_key_from_path(path: Path) -> str:
    parts = path.parts
    partition = next((p for p in parts if p.startswith(PARTITION_PREFIX)), None)
    return partition or "unpartitioned"


def load_concat(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df = transform_dataframe(df)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_high_risk_variants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "clinical_significance" not in df.columns:
        return df

    col = df["clinical_significance"].astype("string").str.lower()
    keep = {"pathogenic", "likely pathogenic"}
    return df[col.isin(keep)]


def mark_lab_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "test_name" not in df.columns or "test_value" not in df.columns:
        df = df.copy()
        df["is_outlier"] = False
        return df

    df = df.copy()
    df["test_value"] = pd.to_numeric(df["test_value"], errors="coerce")
    df["is_outlier"] = False

    for test, group in df.groupby("test_name"):
        values = group["test_value"].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df["test_name"] == test) & (
            (df["test_value"] < lower) | (df["test_value"] > upper)
        )
        df.loc[mask, "is_outlier"] = True

    return df


def aggregate_labs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "patient_id" not in df.columns:
        return pd.DataFrame(columns=["patient_id", "labs_count", "latest_lab_date", "latest_lab_value", "latest_lab_test"]) 

    if "collection_date" in df.columns:
        agg = df.groupby("patient_id", dropna=False).agg(
            labs_count=("patient_id", "size"),
            latest_lab_date=("collection_date", "max"),
        )
    else:
        agg = df.groupby("patient_id", dropna=False).agg(
            labs_count=("patient_id", "size"),
        )
        agg["latest_lab_date"] = pd.NaT

    if "collection_date" in df.columns and "test_value" in df.columns:
        latest_idx = df.sort_values("collection_date").groupby("patient_id", dropna=False).tail(1).set_index("patient_id")
        agg["latest_lab_value"] = latest_idx["test_value"]
        if "test_name" in latest_idx.columns:
            agg["latest_lab_test"] = latest_idx["test_name"]
    else:
        agg["latest_lab_value"] = pd.NA
        agg["latest_lab_test"] = pd.NA

    if "is_outlier" in df.columns:
        outliers = df.groupby("patient_id", dropna=False)["is_outlier"].sum().astype(int)
        agg["abnormal_lab_count"] = outliers
    else:
        agg["abnormal_lab_count"] = 0

    return agg.reset_index()


def aggregate_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "patient_id" not in df.columns:
        return pd.DataFrame(columns=["patient_id", "diagnoses_count", "primary_diagnoses_count", "top_diagnosis_code"])

    agg = df.groupby("patient_id", dropna=False).agg(
        diagnoses_count=("patient_id", "size"),
    )

    if "is_primary" in df.columns:
        primary = df["is_primary"].astype("boolean").fillna(False)
        agg["primary_diagnoses_count"] = primary.groupby(df["patient_id"], dropna=False).sum().astype(int)
    else:
        agg["primary_diagnoses_count"] = 0

    if "icd10_code" in df.columns:
        top_diag = df.groupby("patient_id", dropna=False)["icd10_code"].agg(lambda x: x.value_counts().index[0])
        agg["top_diagnosis_code"] = top_diag
    else:
        agg["top_diagnosis_code"] = pd.NA

    return agg.reset_index()


def aggregate_medications(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "patient_id" not in df.columns:
        return pd.DataFrame(columns=["patient_id", "medications_count", "active_medications_count", "top_medication"])

    agg = df.groupby("patient_id", dropna=False).agg(
        medications_count=("patient_id", "size"),
    )

    if "status" in df.columns:
        status = df["status"].astype("string").str.lower()
        active = status.eq("active")
        agg["active_medications_count"] = active.groupby(df["patient_id"], dropna=False).sum().astype(int)
    else:
        agg["active_medications_count"] = 0

    if "medication_name" in df.columns:
        top_med = df.groupby("patient_id", dropna=False)["medication_name"].agg(lambda x: x.value_counts().index[0])
        agg["top_medication"] = top_med
    else:
        agg["top_medication"] = pd.NA

    return agg.reset_index()


def aggregate_variants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "patient_id" not in df.columns:
        return pd.DataFrame(columns=["patient_id", "high_risk_variants_count", "top_gene"])

    agg = df.groupby("patient_id", dropna=False).agg(
        high_risk_variants_count=("patient_id", "size"),
    )

    if "gene" in df.columns:
        top_gene = df.groupby("patient_id", dropna=False)["gene"].agg(lambda x: x.value_counts().index[0])
        agg["top_gene"] = top_gene
    else:
        agg["top_gene"] = pd.NA

    return agg.reset_index()


def build_unified_dataset(refined_files: list[Path]):
    by_partition: dict[str, dict[str, list[Path]]] = {}

    for file in refined_files:
        parts = file.parts
        dataset_name = parts[parts.index("v1") + 1] if "v1" in parts else file.parent.name
        part_key = partition_key_from_path(file)
        by_partition.setdefault(part_key, {}).setdefault(dataset_name, []).append(file)

    for part_key, datasets in by_partition.items():
        patient_files = datasets.get("patients", [])
        unified_file = next((f for f in patient_files if f.name == "patients_unified.parquet"), None)
        if unified_file:
            patients = load_concat([unified_file])
        else:
            patients = load_concat(patient_files)

        if patients.empty:
            print(f"   ⚠ No patients data for partition {part_key} — unified join skipped")
            continue

        labs = load_concat(datasets.get("labs", []))
        diagnoses = load_concat(datasets.get("diagnoses", []))
        medications = load_concat(datasets.get("medications", []))
        variants = load_concat(datasets.get("variants", []))
        variants_total = len(variants)
        variants = filter_high_risk_variants(variants)
        variants_filtered = len(variants)

        labs = mark_lab_outliers(labs)

        labs_agg = aggregate_labs(labs)
        diag_agg = aggregate_diagnoses(diagnoses)
        meds_agg = aggregate_medications(medications)
        vars_agg = aggregate_variants(variants)

        unified = patients
        for agg_df in [labs_agg, diag_agg, meds_agg, vars_agg]:
            if not agg_df.empty:
                unified = unified.merge(agg_df, on="patient_id", how="left")

        unified["abnormal_lab_count"] = unified.get("abnormal_lab_count", 0).fillna(0).astype(int)
        unified["high_risk_variants_count"] = unified.get("high_risk_variants_count", 0).fillna(0).astype(int)
        unified["high_risk_patient"] = (
            (unified["abnormal_lab_count"] > 0) & (unified["high_risk_variants_count"] > 0)
        )

        output_dir = CONSUMPTION_DIR / "unified"
        if part_key != "unpartitioned":
            output_dir = output_dir / part_key
        output_dir.mkdir(parents=True, exist_ok=True)

        out_file = output_dir / "patients_unified_analytics.parquet"
        tmp_file = output_dir / "patients_unified_analytics.tmp"
        unified.to_parquet(tmp_file, index=False)
        tmp_file.replace(out_file)

        high_risk_count = int(unified["high_risk_patient"].sum())
        print(f"   Variants filtered: {variants_filtered}/{variants_total} kept")
        print(f"   High-risk patients: {high_risk_count}")
        print(f"   Unified dataset saved → {out_file}")


# -------------------------------
# MAIN TRANSFORMATION PIPELINE
# -------------------------------
def run_transformation():
    print("⚙️ Starting Transformation...\n")

    refined_files = iter_refined_files()
    success, failed, total_rows = 0, 0, 0

    for file in refined_files:
        try:
            # refined/v1/<dataset>/ingest_date=YYYY-MM-DD/<file>
            parts = file.parts
            dataset_name = parts[parts.index("v1") + 1] if "v1" in parts else file.parent.name
            ingest_partition = next((p for p in parts if p.startswith(PARTITION_PREFIX)), None)

            print(f"📂 Transforming: {dataset_name}/{file.name}")

            df = pd.read_parquet(file)

            before_rows, before_cols = df.shape

            # Transform
            df = transform_dataframe(df)

            after_rows, after_cols = df.shape
            total_rows += after_rows

            # Skip writing per-site patients to consumption (use unified only)
            if dataset_name == "patients":
                print(f"   Rows: {before_rows} → {after_rows}, Cols: {before_cols} → {after_cols}")
                print("   Skipped writing patients to consumption (unified only)\n")
            else:
                output_dir = CONSUMPTION_DIR / dataset_name
                if ingest_partition:
                    output_dir = output_dir / ingest_partition

                output_dir.mkdir(parents=True, exist_ok=True)
                save_transformed(df, str(output_dir.relative_to(CONSUMPTION_DIR)), file.name)

                print(f"   Rows: {before_rows} → {after_rows}, Cols: {before_cols} → {after_cols}")
                print(f"   Saved → {output_dir}/\n")

            success += 1

        except Exception as e:
            print(f"❌ Error: {file.name} → {e}\n")
            failed += 1

    # -------------------------------
    # FINAL SUMMARY
    # -------------------------------
    print("📊 Transformation Summary")
    print(f"   Success     : {success}")
    print(f"   Failed      : {failed}")
    print(f"   Total Rows  : {total_rows}")

    print("🧩 Building unified analytics dataset...\n")
    build_unified_dataset(refined_files)


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_transformation()