import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
CONSUMPTION_DIR = Path("datalake/consumption/v1")
PLOTS_DIR = Path("datalake/consumption/plots")
PARTITION_PREFIX = "ingest_date="


def latest_partition_dir(dataset_dir: Path) -> Path | None:
    if not dataset_dir.exists():
        return None

    parts = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(PARTITION_PREFIX)])
    return parts[-1] if parts else dataset_dir


def latest_file(dataset: str, filename_hint: str | None = None) -> Path | None:
    dataset_dir = CONSUMPTION_DIR / dataset
    part_dir = latest_partition_dir(dataset_dir)
    if part_dir is None:
        return None

    files = list(part_dir.glob("*.parquet"))
    if not files:
        return None

    if filename_hint:
        match = next((f for f in files if filename_hint in f.name), None)
        if match:
            return match

    return sorted(files)[-1]


def save_plot(fig, name: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved plot → {out_path}")


def plot_age_histogram():
    file = latest_file("unified", "patients_unified_analytics")
    if file is None:
        print("⚠ No unified dataset found for age histogram")
        return

    df = pd.read_parquet(file)
    dob_col = next((c for c in ["date_of_birth", "birthdate", "dob"] if c in df.columns), None)
    if not dob_col:
        print("⚠ No DOB column found for age histogram")
        return

    dob = pd.to_datetime(df[dob_col], errors="coerce")
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    age = ((today - dob.dt.tz_localize(None)).dt.days / 365.25).astype("float")
    age = age.where(age.between(0, 120))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(age.dropna(), bins=20, color="#4C78A8", edgecolor="white")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    save_plot(fig, "age_histogram")


def plot_gender_bar():
    file = latest_file("unified", "patients_unified_analytics")
    if file is None:
        print("⚠ No unified dataset found for gender bar chart")
        return

    df = pd.read_parquet(file)
    gender_col = next((c for c in ["sex", "gender"] if c in df.columns), None)
    if not gender_col:
        print("⚠ No gender column found for gender bar chart")
        return

    counts = df[gender_col].astype("string").str.lower().value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#F58518")
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    save_plot(fig, "gender_bar")


def plot_diagnosis_frequency():
    file = latest_file("diagnoses")
    if file is None:
        print("⚠ No diagnoses dataset found for diagnosis frequency")
        return

    df = pd.read_parquet(file)
    if "icd10_code" not in df.columns:
        print("⚠ icd10_code not found for diagnosis frequency")
        return

    counts = df["icd10_code"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", ax=ax, color="#54A24B")
    ax.set_title("Top ICD-10 Diagnoses")
    ax.set_xlabel("ICD-10")
    ax.set_ylabel("Count")
    save_plot(fig, "diagnosis_frequency")


def plot_lab_distribution():
    file = latest_file("labs")
    if file is None:
        print("⚠ No labs dataset found for lab distribution")
        return

    df = pd.read_parquet(file)
    if "test_value" not in df.columns:
        print("⚠ test_value not found for lab distribution")
        return

    values = pd.to_numeric(df["test_value"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=25, color="#72B7B2", edgecolor="white")
    ax.set_title("Lab Test Value Distribution")
    ax.set_xlabel("Test Value")
    ax.set_ylabel("Count")
    save_plot(fig, "lab_distribution")


def plot_genomics_scatter():
    file = latest_file("variants")
    if file is None:
        print("⚠ No variants dataset found for genomics scatter")
        return

    df = pd.read_parquet(file)
    required = {"allele_frequency", "read_depth"}
    if not required.issubset(set(df.columns)):
        print("⚠ allele_frequency/read_depth not found for genomics scatter")
        return

    af = pd.to_numeric(df["allele_frequency"], errors="coerce")
    rd = pd.to_numeric(df["read_depth"], errors="coerce")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(af, rd, s=14, alpha=0.6, color="#B279A2")
    ax.set_title("Genomics: Allele Frequency vs Read Depth")
    ax.set_xlabel("Allele Frequency")
    ax.set_ylabel("Read Depth")
    save_plot(fig, "genomics_scatter")


def run_visualization():
    print("📈 Starting Visualization...\n")
    plot_age_histogram()
    plot_gender_bar()
    plot_diagnosis_frequency()
    plot_lab_distribution()
    plot_genomics_scatter()


if __name__ == "__main__":
    run_visualization()
