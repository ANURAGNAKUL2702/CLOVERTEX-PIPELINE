import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import re

# -------------------------------
# CONFIG
# -------------------------------
CONSUMPTION_DIR = Path("datalake/consumption/v1")
ANALYTICS_DIR = Path("datalake/reports/analytics")
PARTITION_PREFIX = "ingest_date="


# -------------------------------
# SAFE NUMERIC STATS
# -------------------------------
def safe_numeric_stats(series: pd.Series, decimals: int = 2) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "median": None, "min": None, "max": None, "count": 0}
    return {
        "mean":   round(float(s.mean()),   decimals),
        "median": round(float(s.median()), decimals),
        "min":    round(float(s.min()),    decimals),
        "max":    round(float(s.max()),    decimals),
        "count":  int(s.count()),
    }


def stringify_keys(data: dict) -> dict:
    return {str(k): v for k, v in data.items()}


def safe_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9_.-]+", "_", name)
    return name.replace(".", "_").strip("_")


def icd10_chapter(code: str) -> str:
    if not code or pd.isna(code):
        return "unknown"

    code = str(code).upper()
    letter = code[0]
    num = None
    try:
        num = int(re.sub(r"[^0-9]", "", code)[0:2])
    except Exception:
        num = None

    if letter == "A" or letter == "B":
        return "Infectious and parasitic"
    if letter == "C":
        return "Neoplasms"
    if letter == "D" and num is not None and num <= 48:
        return "Neoplasms"
    if letter == "D":
        return "Blood/immune"
    if letter == "E":
        return "Endocrine/metabolic"
    if letter == "F":
        return "Mental/behavioral"
    if letter == "G":
        return "Nervous system"
    if letter == "H" and num is not None and num <= 59:
        return "Eye"
    if letter == "H":
        return "Ear"
    if letter == "I":
        return "Circulatory"
    if letter == "J":
        return "Respiratory"
    if letter == "K":
        return "Digestive"
    if letter == "L":
        return "Skin"
    if letter == "M":
        return "Musculoskeletal"
    if letter == "N":
        return "Genitourinary"
    if letter == "O":
        return "Pregnancy/childbirth"
    if letter == "P":
        return "Perinatal"
    if letter == "Q":
        return "Congenital"
    if letter == "R":
        return "Symptoms/signs"
    if letter == "S" or letter == "T":
        return "Injury/poisoning"
    if letter == "V" or letter == "W" or letter == "X" or letter == "Y":
        return "External causes"
    if letter == "Z":
        return "Health services"

    return "unknown"


def age_distribution(df: pd.DataFrame) -> dict:
    dob_col = next((c for c in ["date_of_birth", "birthdate", "dob"] if c in df.columns), None)
    if not dob_col:
        return {}

    dob = pd.to_datetime(df[dob_col], errors="coerce")
    dob = dob.dt.tz_localize(None)
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    age = ((today - dob).dt.days / 365.25).astype("float")
    age = age.where(age.between(0, 120))

    bins = [0, 18, 30, 45, 60, 75, 120]
    labels = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
    bucketed = pd.cut(age, bins=bins, labels=labels, right=False)
    return stringify_keys(bucketed.value_counts(dropna=False).to_dict())


# -------------------------------
# NOTES ANALYTICS
# -------------------------------
def analyze_notes(df: pd.DataFrame) -> dict:
    result = {}

    if "note_category" in df.columns:
        result["top_note_categories"] = stringify_keys(
            df["note_category"].astype(str).str.lower().value_counts().head(10).to_dict()
        )

    if "author" in df.columns:
        result["notes_per_author"] = stringify_keys(
            df["author"].value_counts().head(10).to_dict()
        )

    if "word_count" in df.columns:
        result["word_count_stats"] = safe_numeric_stats(df["word_count"], 1)

    if "is_addendum" in df.columns:
        col = df["is_addendum"]
        addendum_count = int(col.astype(str).str.upper().eq("Y").sum())
        result["addendum_count"] = addendum_count
        if len(df) > 0:
            result["addendum_rate_pct"] = round((addendum_count / len(df)) * 100, 1)
        else:
            result["addendum_rate_pct"] = 0.0

    return result


# -------------------------------
# DIAGNOSES ANALYTICS
# -------------------------------
def analyze_diagnoses(df: pd.DataFrame) -> dict:
    result = {}

    if "icd10_code" in df.columns:
        result["top_10_diagnoses"] = stringify_keys(
            df["icd10_code"].value_counts().head(10).to_dict()
        )

        chapters = df["icd10_code"].apply(icd10_chapter)
        result["icd10_chapter_distribution"] = stringify_keys(
            chapters.value_counts(dropna=False).to_dict()
        )

    if "severity" in df.columns:
        result["severity_distribution"] = stringify_keys(
            df["severity"].astype(str).str.lower().value_counts(dropna=False).to_dict()
        )

    if "status" in df.columns:
        result["status_distribution"] = stringify_keys(
            df["status"].astype(str).str.lower().value_counts().to_dict()
        )

    if "is_primary" in df.columns:
        result["primary_vs_secondary"] = stringify_keys(
            df["is_primary"].value_counts().to_dict()
        )

    return result


# -------------------------------
# VARIANTS ANALYTICS
# -------------------------------
def analyze_variants(df: pd.DataFrame) -> dict:
    result = {}

    if "clinical_significance" in df.columns:
        col = df["clinical_significance"].astype(str).str.lower()

        result["clinical_significance_distribution"] = stringify_keys(
            col.value_counts(dropna=False).to_dict()
        )

        high_risk = {"pathogenic", "likely pathogenic"}
        result["high_risk_variant_count"] = int(col.isin(high_risk).sum())

    if "gene" in df.columns:
        result["top_10_mutated_genes"] = stringify_keys(
            df["gene"].value_counts().head(10).to_dict()
        )

    if "variant_type" in df.columns:
        result["variant_type_distribution"] = stringify_keys(
            df["variant_type"].astype(str).str.lower().value_counts().to_dict()
        )

    if "allele_frequency" in df.columns:
        result["allele_frequency_stats"] = safe_numeric_stats(df["allele_frequency"], 4)

    return result


# -------------------------------
# LABS ANALYTICS
# -------------------------------
def analyze_labs(df: pd.DataFrame) -> dict:
    result = {}

    if "test_name" in df.columns:
        result["test_distribution"] = stringify_keys(
            df["test_name"].value_counts().to_dict()
        )

    if "test_value" in df.columns:
        tv = pd.to_numeric(df["test_value"], errors="coerce")
        result["missing_test_values"] = int(tv.isna().sum())

        if "test_name" in df.columns:
            outliers = {}
            for test, group in df.groupby("test_name"):
                values = pd.to_numeric(group["test_value"], errors="coerce").dropna()
                if values.empty:
                    continue
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = int(((values < lower) | (values > upper)).sum())
                if count > 0:
                    outliers[str(test)] = count

            result["out_of_range_by_test"] = stringify_keys(outliers)

    return result


# -------------------------------
# PATIENTS ANALYTICS
# -------------------------------
def analyze_patients(df: pd.DataFrame) -> dict:
    result = {}

    gender_col = next((c for c in ["sex", "gender"] if c in df.columns), None)
    if gender_col:
        result["gender_distribution"] = stringify_keys(
            df[gender_col].astype(str).str.lower().value_counts(dropna=False).to_dict()
        )

    blood_col = next((c for c in ["blood_group", "bloodtype"] if c in df.columns), None)
    if blood_col:
        result["blood_group_distribution"] = stringify_keys(
            df[blood_col].value_counts(dropna=False).to_dict()
        )

    if "site" in df.columns:
        result["patients_per_site"] = stringify_keys(
            df["site"].value_counts().to_dict()
        )

    age_dist = age_distribution(df)
    if age_dist:
        result["age_distribution"] = age_dist

    return result


# -------------------------------
# MEDICATIONS ANALYTICS
# -------------------------------
def analyze_medications(df: pd.DataFrame) -> dict:
    result = {}

    if "medication_name" in df.columns:
        result["top_10_medications"] = stringify_keys(
            df["medication_name"].value_counts().head(10).to_dict()
        )

    if "status" in df.columns:
        result["status_distribution"] = stringify_keys(
            df["status"].astype(str).str.lower().value_counts().to_dict()
        )

    return result


def analyze_unified(df: pd.DataFrame) -> dict:
    result = {}

    if "site" in df.columns:
        result["site_distribution"] = stringify_keys(
            df["site"].value_counts(dropna=False).to_dict()
        )

    age_dist = age_distribution(df)
    if age_dist:
        result["age_distribution"] = age_dist

    if "high_risk_variants_count" in df.columns:
        result["patients_with_high_risk_variants"] = int(
            pd.to_numeric(df["high_risk_variants_count"], errors="coerce").fillna(0).gt(0).sum()
        )

    if "abnormal_lab_count" in df.columns:
        result["patients_with_abnormal_labs"] = int(
            pd.to_numeric(df["abnormal_lab_count"], errors="coerce").fillna(0).gt(0).sum()
        )

    if "high_risk_patient" in df.columns:
        result["high_risk_patient_count"] = int(df["high_risk_patient"].fillna(False).astype(bool).sum())
    else:
        result["high_risk_patient_count"] = 0

    return result


# -------------------------------
# ROUTER
# -------------------------------
ANALYZERS = {
    "notes": analyze_notes,
    "diagnoses": analyze_diagnoses,
    "variants": analyze_variants,
    "labs": analyze_labs,
    "patients": analyze_patients,
    "medications": analyze_medications,
    "unified": analyze_unified,
}


# -------------------------------
# SAVE
# -------------------------------
def save_analytics(data: dict, dataset: str, row_count: int, source_file: str):
    output = {
        "_metadata": {
            "dataset": dataset,
            "row_count": row_count,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": source_file,
        },
        "analytics": data,
    }

    safe_source = safe_name(source_file)
    out_path = ANALYTICS_DIR / f"{dataset}_{safe_source}_analytics.json"

    tmp_path = ANALYTICS_DIR / f"{dataset}_{safe_source}_analytics.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    tmp_path.replace(out_path)

    print(f"   Saved → {out_path}")


# -------------------------------
# MAIN
# -------------------------------
def run_analytics():
    print("📊 Starting Analytics...\n")

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    files = list(CONSUMPTION_DIR.glob(f"**/{PARTITION_PREFIX}*/*.parquet"))
    if not files:
        files = list(CONSUMPTION_DIR.glob("*/*.parquet"))

    success, skipped = 0, 0

    for file in sorted(files):
        parts = file.parts
        dataset_name = parts[parts.index("v1") + 1] if "v1" in parts else file.parent.name

        print(f"📂 Analyzing: {dataset_name}/{file.name}")

        df = pd.read_parquet(file)

        if df.empty:
            print("   ⚠ Empty file — skipped\n")
            skipped += 1
            continue

        analyzer = ANALYZERS.get(dataset_name)

        if analyzer is None:
            print(f"   ⚠ No analyzer for '{dataset_name}' — skipped\n")
            skipped += 1
            continue

        result = analyzer(df)

        print(f"   Rows    : {len(df)}")
        print(f"   Metrics : {len(result)} computed")

        save_analytics(result, dataset_name, len(df), file.name)
        print()

        success += 1

    print("📊 Analytics Summary")
    print(f"   Completed : {success}")
    print(f"   Skipped   : {skipped}")
    print("   Output    → datalake/reports/analytics/")


if __name__ == "__main__":
    run_analytics()
