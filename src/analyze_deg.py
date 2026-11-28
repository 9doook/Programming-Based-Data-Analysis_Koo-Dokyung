
"""
Universal Differential Expression Analysis Pipeline.

Supports raw TSV/TSV.GZ and standardized CSV files.
Preserves statistically significant genes even when annotations
are missing, and saves unannotated DEGs separately.
"""

from pathlib import Path
import argparse
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
STANDARDIZED_DIR = PROJECT_ROOT / "results" / "standardized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "deg"

DEFAULT_INPUT_FILES = {
    "t2d": "GSE159984_de-t2d.tsv.gz",
    "combined_stress": "GSE159984_de-d4pg.tsv.gz",
    "palmitate": "GSE159984_de-d4p.tsv.gz",
    "glucose": "GSE159984_de-d4g.tsv.gz",
}

INVALID_ANNOTATIONS = {
    "",
    "character(0)",
    "integer(0)",
    "numeric(0)",
    "nan",
    "none",
    "null",
    "na",
}


def load_analysis_config(config_path=None):
    """Load analysis thresholds and dataset input settings."""

    if config_path is None:
        return {
            "padj_threshold": 0.05,
            "log2fc_threshold": 0.58,
            "datasets": list(DEFAULT_INPUT_FILES),
            "input_dir": RAW_DATA_DIR,
            "input_files": DEFAULT_INPUT_FILES.copy(),
        }

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    analysis = config.get("analysis", {})
    input_config = config.get("input", {})

    files = input_config.get("files", {})

    input_dir = Path(
        input_config.get("directory", "data/raw")
    )

    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    return {
        "padj_threshold": float(
            analysis.get("padj_threshold", 0.05)
        ),
        "log2fc_threshold": float(
            analysis.get("log2fc_threshold", 0.58)
        ),
        "datasets": list(files) or list(DEFAULT_INPUT_FILES),
        "input_dir": input_dir,
        "input_files": files or DEFAULT_INPUT_FILES.copy(),
    }


def clean_annotation_values(series):
    """Convert invalid annotation placeholders to missing values."""

    values = series.astype("string").str.strip()

    return values.mask(
        values.str.lower().isin(INVALID_ANNOTATIONS),
        pd.NA,
    )


def load_deg_data(file_path):
    """
    Load raw TSV/TSV.GZ or standardized CSV DEG data.

    Original statistical columns are preserved where possible.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    is_tsv = (
        file_path.suffix.lower() in {".tsv", ".gz", ".txt"}
    )

    if is_tsv:
        df = pd.read_csv(
            file_path,
            sep="\t",
            compression="infer",
        )
    else:
        df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Input file is empty: {file_path}"
        )

    # Support standardized column names.
    aliases = {
        "log2fc": "log2FoldChange",
        "base_mean": "baseMean",
        "gene_symbol": "short_id",
    }

    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})

    # Clean annotation placeholders.
    annotation_columns = [
        "short_id",
        "gene_symbol",
        "ensembl_id",
        "gene_id",
        "description",
        "biotype",
    ]

    for column in annotation_columns:
        if column in df.columns:
            df[column] = clean_annotation_values(df[column])

    # Preserve the original Ensembl identifier column
    # expected by the previous analysis and test modules.
    if "ensembl_id_original" not in df.columns:
        if "ensembl_id" in df.columns:
            df.insert(
                0,
                "ensembl_id_original",
                df["ensembl_id"].copy(),
            )
        else:
            df.insert(
                0,
                "ensembl_id_original",
                pd.NA,
            )

    return df


def get_log2fc_column(df):
    """Find the available log2 fold-change column."""

    candidates = [
        "log2fc",
        "log2FoldChange",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "No log2 fold-change column found. "
        "Expected log2fc or log2FoldChange."
    )


def filter_deg(
    df,
    padj_threshold=0.05,
    log2fc_threshold=0.58,
):
    """
    Identify significant, upregulated and downregulated genes.

    Rows without gene annotations are not removed here.
    """

    if "padj" not in df.columns:
        raise ValueError("Required column missing: padj")

    log2fc_column = get_log2fc_column(df)

    working = df.copy()

    working["padj"] = pd.to_numeric(
        working["padj"],
        errors="coerce",
    )

    working[log2fc_column] = pd.to_numeric(
        working[log2fc_column],
        errors="coerce",
    )

    significant = working[
        (working["padj"] < padj_threshold)
        & (
            working[log2fc_column].abs()
            > log2fc_threshold
        )
    ].copy()

    upregulated = significant[
        significant[log2fc_column] > log2fc_threshold
    ].copy()

    downregulated = significant[
        significant[log2fc_column] < -log2fc_threshold
    ].copy()

    return significant, upregulated, downregulated


def get_unannotated_mask(df):
    """Identify rows missing both gene symbol and Ensembl ID."""

    symbol_column = next(
        (
            column
            for column in ["short_id", "gene_symbol"]
            if column in df.columns
        ),
        None,
    )

    if symbol_column is None:
        symbol_missing = pd.Series(
            True,
            index=df.index,
        )
    else:
        symbol_missing = clean_annotation_values(
            df[symbol_column]
        ).isna()

    if "ensembl_id" not in df.columns:
        ensembl_missing = pd.Series(
            True,
            index=df.index,
        )
    else:
        ensembl_missing = clean_annotation_values(
            df["ensembl_id"]
        ).isna()

    return symbol_missing & ensembl_missing


def analyze_dataset(
    dataset_name,
    input_dir=RAW_DATA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    padj_threshold=0.05,
    log2fc_threshold=0.58,
    input_filename=None,
):
    """Analyze one dataset and save all DEG results."""

    print(f"\n[ANALYSIS] {dataset_name}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if input_filename is None:
        input_filename = DEFAULT_INPUT_FILES.get(
            dataset_name,
            f"{dataset_name}.tsv.gz",
        )

    input_path = input_dir / input_filename

    df = load_deg_data(input_path)

    significant, upregulated, downregulated = filter_deg(
        df,
        padj_threshold,
        log2fc_threshold,
    )

    # Save results using the original output structure.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    significant.to_csv(
        output_dir / f"{dataset_name}_all_deg.csv",
        index=False,
    )

    upregulated.to_csv(
        output_dir / f"{dataset_name}_upregulated.csv",
        index=False,
    )

    downregulated.to_csv(
        output_dir / f"{dataset_name}_downregulated.csv",
        index=False,
    )

    # Preserve statistically significant but unannotated rows.
    unannotated_mask = get_unannotated_mask(significant)

    unannotated_deg = significant.loc[
        unannotated_mask
    ].copy()

    unannotated_dir = output_dir / "unannotated"

    unannotated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    unannotated_deg.to_csv(
        unannotated_dir
        / f"{dataset_name}_unannotated_deg.csv",
        index=False,
    )

    annotated_deg_count = (
        len(significant) - len(unannotated_deg)
    )

    summary = {
        "dataset": dataset_name,
        "total_genes": len(df),
        "significant_deg": len(significant),
        "upregulated": len(upregulated),
        "downregulated": len(downregulated),
        "annotated_significant_deg": annotated_deg_count,
        "unannotated_significant_deg": len(unannotated_deg),
    }

    print(f"Total genes: {len(df):,}")
    print(f"Significant DEGs: {len(significant):,}")
    print(f"Upregulated: {len(upregulated):,}")
    print(f"Downregulated: {len(downregulated):,}")
    print(f"Annotated DEGs: {annotated_deg_count:,}")
    print(f"Unannotated DEGs: {len(unannotated_deg):,}")

    return summary


def analyze_all_datasets(
    config_path=None,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Run DEG analysis across all configured datasets."""

    config = load_analysis_config(config_path)

    summaries = []

    for dataset_name in config["datasets"]:

        summary = analyze_dataset(
            dataset_name=dataset_name,
            input_dir=config["input_dir"],
            input_filename=config["input_files"].get(
                dataset_name
            ),
            output_dir=output_dir,
            padj_threshold=config["padj_threshold"],
            log2fc_threshold=config["log2fc_threshold"],
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_dir / "deg_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
    )

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
