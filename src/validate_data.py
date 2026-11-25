
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

# 실제 데이터는 24개 열이므로,
# 첫 번째 열의 이름을 명시적으로 추가한다.
EXPECTED_COLUMNS = [
    "ensembl_id_original",
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "stat",
    "pvalue",
    "padj",
    "gene_id",
    "ensembl_id",
    "short_id",
    "description",
    "biotype",
    "analysis_id",
    "seq_region_id",
    "seq_region_start",
    "seq_region_end",
    "seq_region_strand",
    "display_xref_id",
    "source",
    "is_current",
    "canonical_transcript_id",
    "version",
    "created_date",
    "modified_date",
]

REQUIRED_COLUMNS = [
    "baseMean",
    "log2FoldChange",
    "pvalue",
    "padj",
    "ensembl_id_original",
]


def load_deg_file(file_path):
    """실제 24개 열 구조에 맞춰 DEG 파일을 읽는다."""

    df = pd.read_csv(
        file_path,
        sep="\t",
        compression="gzip",
        header=0,
        names=EXPECTED_COLUMNS,
        dtype=str,
    )

    if len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"예상 컬럼 수와 다릅니다: {file_path.name}"
        )

    # R에서 생성된 빈 주석의 표시값을 결측값으로 변환한다.
    placeholder_values = [
        "integer(0)",
        "character(0)",
        "numeric(0)",
    ]

    df = df.replace(placeholder_values, np.nan)

    return df


def inspect_file(label, filename):
    """개별 DEG 파일의 데이터 품질을 검사한다."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path}"
        )

    df = load_deg_file(file_path)

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
        "lfcSE",
        "stat",
        "pvalue",
        "padj",
    ]

    numeric_df = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid_original_id = (
        df["ensembl_id_original"].notna()
        & df["ensembl_id_original"].str.startswith(
            "ENSG", na=False
        )
    )

    id_match = (
        df["ensembl_id_original"]
        == df["ensembl_id"]
    )

    report = {
        "label": label,
        "filename": filename,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),

        "duplicate_gene_id": int(
            df["gene_id"].dropna().duplicated().sum()
        ),

        "missing_values": {
            col: int(df[col].isna().sum())
            for col in REQUIRED_COLUMNS
        },

        "placeholder_annotations": {
            col: int(df[col].isna().sum())
            for col in [
                "gene_id",
                "ensembl_id",
                "short_id",
            ]
        },

        "valid_original_ensembl_id": int(
            valid_original_id.sum()
        ),

        "ensembl_id_match": int(id_match.sum()),

        "ensembl_id_mismatch": int(
            (
                df["ensembl_id_original"].notna()
                & df["ensembl_id"].notna()
                & ~id_match
            ).sum()
        ),

        "numeric_missing_or_invalid": {
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
        print(f"Columns: {report['total_columns']}")
        print(
            "Duplicate gene IDs:",
            report["duplicate_gene_id"],
        )
        print(
            "Missing values:",
            report["missing_values"],
        )
        print(
            "Ensembl ID matches:",
            report["ensembl_id_match"],
        )
        print(
            "Ensembl ID mismatches:",
            report["ensembl_id_mismatch"],
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
    
