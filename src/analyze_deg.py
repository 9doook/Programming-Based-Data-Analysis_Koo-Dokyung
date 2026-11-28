
"""
Universal Differential Expression Analysis

Standardized DEG files are loaded and analyzed using
configurable statistical thresholds.
"""

from pathlib import Path
import argparse
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANDARDIZED_DIR = PROJECT_ROOT / "results" / "standardized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "deg"

DEFAULT_DATASETS = [
    "t2d",
    "combined_stress",
    "palmitate",
    "glucose",
]


def load_analysis_config(config_path=None):
    """Load analysis thresholds and dataset settings."""

    if config_path is None:
        return {
            "padj_threshold": 0.05,
            "log2fc_threshold": 0.58,
            "datasets": DEFAULT_DATASETS,
        }

    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    analysis = config.get("analysis", {})
    input_config = config.get("input", {})

    datasets = list(input_config.get("files", {}).keys())

    return {
        "padj_threshold": float(
            analysis.get("padj_threshold", 0.05)
        ),
        "log2fc_threshold": float(
            analysis.get("log2fc_threshold", 0.58)
        ),
        "datasets": datasets or DEFAULT_DATASETS,
    }


def load_deg_data(file_path):
    """Load a standardized DEG CSV file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Standardized file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "log2fc",
        "padj",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    for column in ["log2fc", "padj"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[["log2fc", "padj"]].isna().any().any():
        raise ValueError(
            "Missing or invalid numerical values detected."
        )

    return df


def filter_deg(
    df,
    padj_threshold=0.05,
    log2fc_threshold=0.58,
):
    """Identify significant, upregulated and downregulated genes."""

    significant = df[
        (df["padj"] < padj_threshold)
        & (df["log2fc"].abs() > log2fc_threshold)
    ].copy()

    upregulated = significant[
        significant["log2fc"] > log2fc_threshold
    ].copy()

    downregulated = significant[
        significant["log2fc"] < -log2fc_threshold
    ].copy()

    return significant, upregulated, downregulated


def analyze_dataset(
    dataset_name,
    input_dir=STANDARDIZED_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    padj_threshold=0.05,
    log2fc_threshold=0.58,
):
    """Analyze one standardized dataset."""

    print(f"\n[ANALYSIS] {dataset_name}")

    input_path = Path(input_dir) / f"{dataset_name}.csv"

    df = load_deg_data(input_path)

    significant, upregulated, downregulated = filter_deg(
        df,
        padj_threshold,
        log2fc_threshold,
    )

    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    significant.to_csv(
        output_dir / "all_deg.csv",
        index=False,
    )

    upregulated.to_csv(
        output_dir / "upregulated.csv",
        index=False,
    )

    downregulated.to_csv(
        output_dir / "downregulated.csv",
        index=False,
    )

    summary = {
        "dataset": dataset_name,
        "total_genes": len(df),
        "significant_deg": len(significant),
        "upregulated": len(upregulated),
        "downregulated": len(downregulated),
    }

    print(f"Total genes: {len(df):,}")
    print(f"Significant DEGs: {len(significant):,}")
    print(f"Upregulated: {len(upregulated):,}")
    print(f"Downregulated: {len(downregulated):,}")

    return summary


def analyze_all_datasets(
    config_path=None,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Run DEG analysis across configured datasets."""

    config = load_analysis_config(config_path)

    summaries = []

    for dataset_name in config["datasets"]:
        summary = analyze_dataset(
            dataset_name=dataset_name,
            padj_threshold=config["padj_threshold"],
            log2fc_threshold=config["log2fc_threshold"],
            output_dir=output_dir,
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "deg_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n[COMPLETE] DEG analysis finished.")
    print(f"Datasets analyzed: {len(summaries)}")
    print(f"Summary saved: {summary_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Universal DEG analysis pipeline"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to dataset configuration JSON",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )

    args = parser.parse_args()

    analyze_all_datasets(
        config_path=args.config,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
