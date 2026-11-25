from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "qc"

DATA_FILES = {
    "t2d": "GSE159984_de-t2d.tsv.gz",
    "combined_stress": "GSE159984_de-d4pg.tsv.gz",
    "palmitate": "GSE159984_de-d4p.tsv.gz",
    "glucose": "GSE159984_de-d4g.tsv.gz",
}

REQUIRED_COLUMNS = [
    "baseMean",
    "log2FoldChange",
    "pvalue",
    "padj",
    "gene_id",
    "ensembl_id",
    "short_id",
]


def inspect_file(label, filename):
    """개별 DEG 파일의 데이터 품질을 검사한다."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t",
        compression="gzip",
    )

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filename}: 필수 컬럼 누락 - {missing_columns}"
        )

    numeric_columns = [
        "baseMean",
        "log2FoldChange",
        "pvalue",
        "padj",
    ]

    numeric_df = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    report = {
        "label": label,
        "filename": filename,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_gene_id": int(
            df["gene_id"].duplicated().sum()
        ),
        "missing_values": {
            col: int(df[col].isna().sum())
            for col in REQUIRED_COLUMNS
        },
        "invalid_numeric_values": {
            col: int(numeric_df[col].isna().sum())
            for col in numeric_columns
        },
        "infinite_values": {
            col: int(
                np.isinf(numeric_df[col]).sum()
            )
            for col in numeric_columns
        },
        "invalid_pvalue": int(
            (
                (numeric_df["pvalue"] < 0)
                | (numeric_df["pvalue"] > 1)
            ).sum()
        ),
        "invalid_padj": int(
            (
                (numeric_df["padj"] < 0)
                | (numeric_df["padj"] > 1)
            ).sum()
        ),
        "negative_baseMean": int(
            (numeric_df["baseMean"] < 0).sum()
        ),
    }

    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    for label, filename in DATA_FILES.items():
        print(f"\n[CHECK] {label}: {filename}")

        report = inspect_file(label, filename)
        reports.append(report)

        print(f"Rows: {report['total_rows']:,}")
        print(
            "Duplicate gene IDs:",
            report["duplicate_gene_id"],
        )
        print(
            "Missing values:",
            report["missing_values"],
        )
        print(
            "Invalid p-values:",
            report["invalid_pvalue"],
        )
        print(
            "Invalid adjusted p-values:",
            report["invalid_padj"],
        )

    output_path = OUTPUT_DIR / "data_quality_report.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n[COMPLETE] Data quality inspection finished.")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
