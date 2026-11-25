
"""
GSE159984 차등 발현 유전자(DEG) 분석

4개 DEG 결과 파일을 불러와 공통 기준으로 필터링하고,
전체 DEG 및 상향/하향 조절 유전자를 저장한다.
"""

import pandas as pd

from config.analysis_config import (
    DATA_FILES,
    PADJ_THRESHOLD,
    LOG2FC_THRESHOLD,
    DEG_DIR,
)


def load_deg_data(file_path):
    """24개 필드 구조에 맞춰 DEG 파일을 불러온다."""

    columns = [
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

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=0,
        names=columns,
        compression="gzip",
        dtype=str,
    )

    # 비어 있는 주석 정보의 특수 표기를 결측값으로 변환
    df = df.replace(
        ["integer(0)", "character(0)", "numeric(0)"],
        pd.NA,
    )

    # 분석에 사용하는 수치형 열 변환
    numeric_columns = [
        "baseMean",
        "log2FoldChange",
        "pvalue",
        "padj",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def filter_deg(df):
    """통계적 유의성과 발현 변화량 기준으로 DEG를 선별한다."""

    significant = df[
        (df["padj"] < PADJ_THRESHOLD)
        & (df["log2FoldChange"].abs() > LOG2FC_THRESHOLD)
    ].copy()

    upregulated = significant[
        significant["log2FoldChange"] > LOG2FC_THRESHOLD
    ].copy()

    downregulated = significant[
        significant["log2FoldChange"] < -LOG2FC_THRESHOLD
    ].copy()

    return significant, upregulated, downregulated


def analyze_all_datasets():
    """4개 데이터셋을 분석하고 결과를 저장한다."""

    DEG_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for dataset_name, file_path in DATA_FILES.items():

        print(f"\n[ANALYSIS] {dataset_name}")

        df = load_deg_data(file_path)

        significant, upregulated, downregulated = filter_deg(df)

        # 결과 CSV 저장
        significant.to_csv(
            DEG_DIR / f"{dataset_name}_all_deg.csv",
            index=False,
        )

        upregulated.to_csv(
            DEG_DIR / f"{dataset_name}_upregulated.csv",
            index=False,
        )

        downregulated.to_csv(
            DEG_DIR / f"{dataset_name}_downregulated.csv",
            index=False,
        )

        # 데이터셋별 분석 결과 요약
        summary.append({
            "dataset": dataset_name,
            "total_genes": len(df),
            "significant_deg": len(significant),
            "upregulated": len(upregulated),
            "downregulated": len(downregulated),
        })

        print(f"전체 유전자 수: {len(df):,}")
        print(f"유의한 DEG 수: {len(significant):,}")
        print(f"상향 조절 유전자 수: {len(upregulated):,}")
        print(f"하향 조절 유전자 수: {len(downregulated):,}")

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        DEG_DIR / "deg_summary.csv",
        index=False,
    )

    print("\n[COMPLETE] DEG analysis finished.")
    print(f"Results saved in: {DEG_DIR}")


if __name__ == "__main__":
    analyze_all_datasets()

