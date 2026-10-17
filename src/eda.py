from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT_DIR / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

sns.set_theme(style="whitegrid")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # Remove exact duplicate rows, if present.
    df = df.drop_duplicates().copy()

    # Binary target: original 0 = no disease;
    # original values 1-4 = disease present.
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    print("\nTarget counts:")
    print(df["target"].value_counts().sort_index())

    print("\nTarget proportions:")
    print(df["target"].value_counts(normalize=True).sort_index())

    print("\nNumeric summary:")
    print(df[NUMERIC_FEATURES].describe())

    # Save the analysis dataset. Missing input values are retained.
    df.to_csv(
        PROCESSED_DIR / "heart_disease_cleaned.csv",
        index=False,
    )

    # 1. Histograms
    df[NUMERIC_FEATURES].hist(
        figsize=(12, 8),
        bins=15,
        edgecolor="black",
    )
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_histograms.png", dpi=160)
    plt.close("all")

    # 2. Target class balance
    counts = df["target"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
    )
    ax.set(
        title="Heart Disease Target Class Balance",
        xlabel="Target (0 = absent, 1 = present)",
        ylabel="Number of records",
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_balance.png", dpi=160)
    plt.close("all")

    # 3. Correlation heatmap for numeric columns.
    correlation_data = df[
        NUMERIC_FEATURES + ["target"]
    ].copy()

    correlation = correlation_data.corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
    )
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close("all")

    print("\nEDA completed.")
    print("Cleaned analysis data:", PROCESSED_DIR)
    print("Plots saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()