
"""
GSE159984 GO 및 KEGG 기능적 풍부도 분석

1. 데이터셋별 배경 유전자 집합 구축
2. 전체 DEG, 상향 및 하향 조절 유전자 분리
3. GO 및 KEGG 유전자 집합 다운로드 및 보존
4. 초기하분포를 이용한 과대표현 분석
5. Benjamini-Hochberg 다중검정 보정
6. 분석 결과 및 유전자 매핑 정보 저장
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import gseapy as gp

from scipy.stats import (
    hypergeom,
    false_discovery_control,
)

from config.analysis_config import (
    DATA_FILES,
    DEG_DIR,
)

from src.analyze_deg import load_deg_data


# ========================================
# 1. 기능적 풍부도 분석 설정
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENRICHMENT_DIR = (
    PROJECT_ROOT / "results" / "enrichment"
)

LIBRARY_DIR = (
    PROJECT_ROOT / "data" / "gene_sets"
)

# 기능적 풍부도 분석의 유의성 기준
ENRICHMENT_FDR = 0.05

# 사용할 인간 유전자 집합 라이브러리
GENE_SET_LIBRARIES = {
    "go_bp": "GO_Biological_Process_2023",
    "go_mf": "GO_Molecular_Function_2023",
    "go_cc": "GO_Cellular_Component_2023",
    "kegg": "KEGG_2021_Human",
}

# 분석할 DEG 분류
GENE_GROUPS = {
    "all": "all_deg",
    "upregulated": "upregulated",
    "downregulated": "downregulated",
}

RESULT_COLUMNS = [
    "term",
    "overlap_count",
    "term_size",
    "query_size",
    "background_size",
    "gene_ratio",
    "fold_enrichment",
    "pvalue",
    "adjusted_pvalue",
    "genes",
]


# ========================================
# 2. 유전자 식별자 정리
# ========================================

def normalize_gene_symbols(values):
    """결측값과 중복을 제거하고 유전자 기호를 정리한다."""

    symbols = set()

    invalid_symbols = {
        "",
        "NAN",
        "NONE",
        "NA",
        "CHARACTER(0)",
        "INTEGER(0)",
        "NUMERIC(0)",
    }

    for value in values:

        if pd.isna(value):
            continue

        symbol = str(value).strip().upper()

        if symbol not in invalid_symbols:
            symbols.add(symbol)

    return symbols


def get_gene_symbols(df):
    """데이터프레임에서 유전자 기호를 추출한다."""

    if "short_id" not in df.columns:

        raise ValueError(
            "short_id 열이 존재하지 않습니다."
        )

    return normalize_gene_symbols(
        df["short_id"]
    )


# ========================================
# 3. 유전자 집합 라이브러리 관리
# ========================================

def load_gene_set_library(library_name):
    """유전자 집합을 다운로드하거나 저장된 자료를 재사용한다."""

    LIBRARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    library_path = (
        LIBRARY_DIR / f"{library_name}.json"
    )

    metadata_path = (
        LIBRARY_DIR / f"{library_name}_metadata.json"
    )

    # 저장된 라이브러리가 있으면 재사용
    if library_path.exists():

        print(
            f"[CACHE] {library_name}"
        )

        if not metadata_path.exists():

            raise ValueError(
                f"라이브러리 메타데이터가 없습니다: "
                f"{metadata_path}"
            )

        with metadata_path.open(
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        checksum = hashlib.sha256(
            library_path.read_bytes()
        ).hexdigest()

        if checksum != metadata["sha256"]:

            raise ValueError(
                f"라이브러리 체크섬 불일치: "
                f"{library_name}"
            )

        with library_path.open(
            encoding="utf-8"
        ) as file:

            raw_library = json.load(file)

    else:

        print(
            f"[DOWNLOAD] {library_name}"
        )

        # Enrichr에서 인간 유전자 집합 다운로드
        downloaded = gp.get_library(
            name=library_name,
            organism="Human",
        )

        raw_library = {}

        for term, genes in downloaded.items():

            normalized = normalize_gene_symbols(
                genes
            )

            if normalized:

                raw_library[term] = sorted(
                    normalized
                )

        if not raw_library:

            raise ValueError(
                f"빈 라이브러리: {library_name}"
            )

        # 다운로드한 유전자 집합을 로컬에 보존
        with library_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                raw_library,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        checksum = hashlib.sha256(
            library_path.read_bytes()
        ).hexdigest()

        metadata = {
            "library_name": library_name,
            "source": "Enrichr",
            "organism": "Human",
            "downloaded_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "gseapy_version": gp.__version__,
            "term_count": len(raw_library),
            "sha256": checksum,
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    library = {
        term: normalize_gene_symbols(genes)
        for term, genes in raw_library.items()
    }

    if not library:

        raise ValueError(
            f"사용 가능한 유전자 집합이 없습니다: "
            f"{library_name}"
        )

    print(
        f"유전자 집합 수: {len(library):,}"
    )

    return library


# ========================================
# 4. 기능적 풍부도 분석
# ========================================

def run_enrichment(
    query_genes,
    background_genes,
    gene_sets,
):
    """초기하분포를 이용한 과대표현 분석을 수행한다."""

    query_genes = set(query_genes)

    background_genes = set(background_genes)

    # 라이브러리에 포함된 전체 유전자 집합
    library_genes = set()

    for genes in gene_sets.values():

        library_genes.update(genes)

    # 실제 분석 대상 중 라이브러리에 매핑되는 유전자
    effective_background = (
        background_genes & library_genes
    )

    # 배경 유전자 및 라이브러리에 포함되는 DEG
    effective_query = (
        query_genes & effective_background
    )

    background_size = len(
        effective_background
    )

    query_size = len(
        effective_query
    )

    mapping_report = {
        "input_query_genes": len(query_genes),
        "input_background_genes": len(background_genes),
        "mapped_query_genes": query_size,
        "mapped_background_genes": background_size,
        "unmapped_query_genes": len(
            query_genes - effective_query
        ),
        "unmapped_background_genes": len(
            background_genes - effective_background
        ),
        "query_outside_background": len(
            query_genes - background_genes
        ),
        "unmapped_query_symbols": sorted(
            query_genes - effective_query
        ),
        "tested_terms": 0,
        "significant_terms": 0,
    }

    # 분석 가능한 유전자가 없는 경우
    if background_size == 0 or query_size == 0:

        return (
            pd.DataFrame(columns=RESULT_COLUMNS),
            mapping_report,
        )

    results = []

    for term, genes in gene_sets.items():

        # 배경 유전자 집합에 존재하는 유전자만 사용
        term_genes = (
            set(genes) & effective_background
        )

        term_size = len(term_genes)

        if term_size == 0:
            continue

        overlap = (
            effective_query & term_genes
        )

        overlap_count = len(overlap)

        # 초기하분포의 상측 확률 계산
        pvalue = float(
            hypergeom.sf(
                overlap_count - 1,
                background_size,
                term_size,
                query_size,
            )
        )

        gene_ratio = (
            overlap_count / query_size
        )

        expected_ratio = (
            term_size / background_size
        )

        fold_enrichment = (
            gene_ratio / expected_ratio
        )

        results.append({
            "term": term,
            "overlap_count": overlap_count,
            "term_size": term_size,
            "query_size": query_size,
            "background_size": background_size,
            "gene_ratio": gene_ratio,
            "fold_enrichment": fold_enrichment,
            "pvalue": pvalue,
            "genes": ";".join(sorted(overlap)),
        })

    if not results:

        return (
            pd.DataFrame(columns=RESULT_COLUMNS),
            mapping_report,
        )

    result_df = pd.DataFrame(results)

    # 중첩 유전자가 없는 경로도 포함하여
    # 검정한 전체 경로에 다중검정 보정 적용
    result_df["adjusted_pvalue"] = (
        false_discovery_control(
            result_df["pvalue"].to_numpy(),
            method="bh",
        )
    )

    result_df = result_df[
        RESULT_COLUMNS
    ]

    result_df = result_df.sort_values(
        by=[
            "adjusted_pvalue",
            "pvalue",
            "term",
        ],
        ascending=True,
    ).reset_index(drop=True)

    mapping_report["tested_terms"] = len(
        result_df
    )

    mapping_report["significant_terms"] = int(
        (
            result_df["adjusted_pvalue"]
            < ENRICHMENT_FDR
        ).sum()
    )

    return result_df, mapping_report


# ========================================
# 5. 데이터셋별 기능적 풍부도 분석
# ========================================

def analyze_dataset(
    dataset_name,
    file_path,
    libraries,
):
    """개별 데이터셋의 GO 및 KEGG 분석을 수행한다."""

    print(
        f"\n[ANALYSIS] {dataset_name}"
    )

    # 원본 전체 DEG 통계 자료 불러오기
    full_df = load_deg_data(
        file_path
    )

    # 유효한 통계량을 가진 분석 대상만 배경으로 사용
    valid = (
        np.isfinite(full_df["log2FoldChange"])
        & np.isfinite(full_df["padj"])
        & full_df["padj"].between(0, 1)
    )

    background_df = full_df.loc[valid]

    background_genes = get_gene_symbols(
        background_df
    )

    print(
        f"배경 데이터 행 수: {len(background_df):,}"
    )

    print(
        f"고유 배경 유전자 기호 수: "
        f"{len(background_genes):,}"
    )

    dataset_dir = (
        ENRICHMENT_DIR / dataset_name
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    mapping_reports = []

    for group_name, file_suffix in GENE_GROUPS.items():

        deg_path = (
            DEG_DIR
            / f"{dataset_name}_{file_suffix}.csv"
        )

        deg_df = pd.read_csv(
            deg_path,
            dtype={"short_id": str},
        )

        query_genes = get_gene_symbols(
            deg_df
        )

        print(
            f"\n[GROUP] {group_name}"
        )

        print(
            f"DEG 행 수: {len(deg_df):,}"
        )

        print(
            f"고유 DEG 유전자 기호 수: "
            f"{len(query_genes):,}"
        )

        for library_key, gene_sets in libraries.items():

            result_df, mapping = run_enrichment(
                query_genes,
                background_genes,
                gene_sets,
            )

            significant = result_df.loc[
                result_df["adjusted_pvalue"]
                < ENRICHMENT_FDR
            ].copy()

            # 검정한 전체 경로 저장
            all_path = (
                dataset_dir
                / f"{group_name}_{library_key}_all.csv"
            )

            result_df.to_csv(
                all_path,
                index=False,
                encoding="utf-8-sig",
            )

            # 유의한 경로만 별도로 저장
            significant_path = (
                dataset_dir
                / f"{group_name}_{library_key}_significant.csv"
            )

            significant.to_csv(
                significant_path,
                index=False,
                encoding="utf-8-sig",
            )

            mapping.update({
                "dataset": dataset_name,
                "gene_group": group_name,
                "library": library_key,
                "library_name": GENE_SET_LIBRARIES[
                    library_key
                ],
                "deg_rows": len(deg_df),
                "background_rows": len(background_df),
            })

            mapping_reports.append(mapping)

            summary_rows.append({
                "dataset": dataset_name,
                "gene_group": group_name,
                "library": library_key,
                "deg_rows": len(deg_df),
                "unique_query_symbols": len(query_genes),
                "mapped_query_genes": mapping[
                    "mapped_query_genes"
                ],
                "mapped_background_genes": mapping[
                    "mapped_background_genes"
                ],
                "tested_terms": mapping[
                    "tested_terms"
                ],
                "significant_terms": len(significant),
            })

            print(
                f"[RESULT] {library_key}: "
                f"매핑 유전자 "
                f"{mapping['mapped_query_genes']:,}개, "
                f"검정 경로 "
                f"{mapping['tested_terms']:,}개, "
                f"유의한 경로 "
                f"{len(significant):,}개"
            )

    return summary_rows, mapping_reports


# ========================================
# 6. 전체 기능적 풍부도 분석 실행
# ========================================

def main():
    """4개 데이터셋의 기능적 풍부도 분석을 실행한다."""

    print(
        "[START] GO and KEGG enrichment analysis"
    )

    ENRICHMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    libraries = {}

    # 분석에 사용할 유전자 집합 준비
    for library_key, library_name in (
        GENE_SET_LIBRARIES.items()
    ):

        libraries[library_key] = (
            load_gene_set_library(library_name)
        )

    all_summary_rows = []

    all_mapping_reports = []

    # 4개 데이터셋 반복 분석
    for dataset_name, file_path in DATA_FILES.items():

        summary_rows, mapping_reports = (
            analyze_dataset(
                dataset_name,
                file_path,
                libraries,
            )
        )

        all_summary_rows.extend(
            summary_rows
        )

        all_mapping_reports.extend(
            mapping_reports
        )

    # 전체 분석 요약 저장
    summary_df = pd.DataFrame(
        all_summary_rows
    )

    summary_path = (
        ENRICHMENT_DIR / "enrichment_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 유전자 매핑 및 분석 정보 저장
    mapping_path = (
        ENRICHMENT_DIR / "gene_mapping_report.json"
    )

    with mapping_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_mapping_reports,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n[COMPLETE] Enrichment analysis finished."
    )

    print(
        f"전체 분석 조합 수: "
        f"{len(summary_df):,}"
    )

    print(
        f"분석 요약: {summary_path}"
    )

    print(
        f"유전자 매핑 보고서: {mapping_path}"
    )

    print(
        f"결과 저장 경로: {ENRICHMENT_DIR}"
    )


if __name__ == "__main__":
    main()

