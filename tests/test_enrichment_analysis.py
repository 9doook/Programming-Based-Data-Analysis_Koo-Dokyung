import hashlib
import json
from itertools import product
from math import comb

import numpy as np
import pandas as pd
import pytest

from src import enrichment_analysis as ea
from src.analyze_deg import load_deg_data, filter_deg


# ============================================================
# 1. 유전자 기호 정규화 테스트
# ============================================================

def test_normalize_gene_symbols():
    values = [
        " INS ",
        "ins",
        "GCG",
        "gcg",
        "PDX1",
        None,
        np.nan,
        pd.NA,
        "",
        " ",
        "character(0)",
        "integer(0)",
        "numeric(0)",
        "NA",
        "None",
    ]

    result = ea.normalize_gene_symbols(values)

    assert result == {"INS", "GCG", "PDX1"}


# ============================================================
# 2. 유전자 기호 추출 테스트
# ============================================================

def test_get_gene_symbols():
    df = pd.DataFrame({
        "short_id": ["INS", "ins", "GCG", None],
        "log2FoldChange": [1.2, 1.5, -1.3, 2.0],
    })

    result = ea.get_gene_symbols(df)

    assert result == {"INS", "GCG"}


def test_missing_gene_symbol_column():
    df = pd.DataFrame({
        "log2FoldChange": [1.2, -1.5],
    })

    with pytest.raises(ValueError):
        ea.get_gene_symbols(df)


# ============================================================
# 3. 가상 유전자 집합을 이용한 통계 계산 검증
# ============================================================

@pytest.fixture
def toy_data():
    background = {
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J",
    }

    query = {"A", "B", "C"}

    gene_sets = {
        "Pathway_A": {"A", "B", "C"},
        "Pathway_B": {"A", "D", "E"},
        "Pathway_C": {"F", "G"},
        "Pathway_D": {"H", "I", "J"},
    }

    return query, background, gene_sets


# ============================================================
# 4. 초기하분포 검정 정확성 검증
# ============================================================

def test_hypergeometric_probability(toy_data):
    query, background, gene_sets = toy_data

    result, mapping = ea.run_enrichment(
        query,
        background,
        gene_sets,
    )

    row = result.set_index("term").loc["Pathway_A"]

    # 전체 10개 중 3개를 선택할 때
    # 특정 유전자 3개가 모두 포함될 확률
    expected_pvalue = 1 / comb(10, 3)

    assert row["overlap_count"] == 3
    assert row["term_size"] == 3
    assert row["query_size"] == 3
    assert row["background_size"] == 10

    assert row["pvalue"] == pytest.approx(
        expected_pvalue
    )


# ============================================================
# 5. Gene ratio 및 Fold enrichment 검증
# ============================================================

def test_enrichment_metrics(toy_data):
    query, background, gene_sets = toy_data

    result, _ = ea.run_enrichment(
        query,
        background,
        gene_sets,
    )

    rows = result.set_index("term")

    # Pathway_A
    # Gene ratio = 3 / 3 = 1
    # Fold enrichment = (3 / 3) / (3 / 10)

    assert rows.loc[
        "Pathway_A", "gene_ratio"
    ] == pytest.approx(1.0)

    assert rows.loc[
        "Pathway_A", "fold_enrichment"
    ] == pytest.approx(10 / 3)

    # Pathway_B
    # Gene ratio = 1 / 3
    # Fold enrichment = (1 / 3) / (3 / 10)

    assert rows.loc[
        "Pathway_B", "gene_ratio"
    ] == pytest.approx(1 / 3)

    assert rows.loc[
        "Pathway_B", "fold_enrichment"
    ] == pytest.approx(10 / 9)


# ============================================================
# 6. BH 다중검정 보정 검증
# ============================================================

def test_bh_adjustment(toy_data):
    query, background, gene_sets = toy_data

    result, _ = ea.run_enrichment(
        query,
        background,
        gene_sets,
    )

    rows = result.set_index("term")

    # Pathway_A의 원래 p-value = 1/120
    # 전체 검정 항목은 4개
    expected_adjusted_a = 4 / 120

    # Pathway_B에서 1개 이상 중첩될 확률
    expected_pvalue_b = (
        1 - comb(7, 3) / comb(10, 3)
    )

    expected_adjusted_b = min(
        expected_pvalue_b * 4 / 2,
        1.0,
    )

    assert rows.loc[
        "Pathway_A", "adjusted_pvalue"
    ] == pytest.approx(expected_adjusted_a)

    assert rows.loc[
        "Pathway_B", "adjusted_pvalue"
    ] == pytest.approx(expected_adjusted_b)

    assert rows.loc[
        "Pathway_C", "adjusted_pvalue"
    ] == pytest.approx(1.0)

    assert rows.loc[
        "Pathway_D", "adjusted_pvalue"
    ] == pytest.approx(1.0)


# ============================================================
# 7. 중첩 유전자 0개 처리 검증
# ============================================================

def test_zero_overlap_terms(toy_data):
    query, background, gene_sets = toy_data

    result, mapping = ea.run_enrichment(
        query,
        background,
        gene_sets,
    )

    rows = result.set_index("term")

    # 중첩 유전자가 없는 경로도 검정에 포함
    assert len(result) == 4
    assert mapping["tested_terms"] == 4

    for term in ["Pathway_C", "Pathway_D"]:
        assert rows.loc[
            term, "overlap_count"
        ] == 0

        assert rows.loc[
            term, "pvalue"
        ] == pytest.approx(1.0)

        assert rows.loc[
            term, "gene_ratio"
        ] == pytest.approx(0.0)

        assert rows.loc[
            term, "fold_enrichment"
        ] == pytest.approx(0.0)

    assert mapping["significant_terms"] == 1


# ============================================================
# 8. 배경 유전자 및 매핑 검증
# ============================================================

def test_background_and_mapping():
    background = {"A", "B", "C", "D", "E"}
    query = {"A", "B", "E", "X"}

    gene_sets = {
        "Pathway_1": {"A", "B", "C"},
        "Pathway_2": {"D", "X"},
        "Pathway_3": {"Z"},
    }

    result, mapping = ea.run_enrichment(
        query,
        background,
        gene_sets,
    )

    # 실제 배경 유전자: A, B, C, D
    # 실제 분석 유전자: A, B
    assert mapping["input_query_genes"] == 4
    assert mapping["input_background_genes"] == 5

    assert mapping["mapped_query_genes"] == 2
    assert mapping["mapped_background_genes"] == 4

    assert mapping["unmapped_query_genes"] == 2
    assert mapping["unmapped_background_genes"] == 1

    assert mapping["query_outside_background"] == 1

    assert set(
        mapping["unmapped_query_symbols"]
    ) == {"E", "X"}

    # 배경 유전자와 중첩되지 않는 경로는 제외
    assert set(result["term"]) == {
        "Pathway_1",
        "Pathway_2",
    }

    assert (
        result["background_size"] == 4
    ).all()

    assert (
        result["query_size"] == 2
    ).all()


# ============================================================
# 9. 빈 유전자 집합 처리 검증
# ============================================================

@pytest.mark.parametrize(
    "query, background, gene_sets",
    [
        (
            set(),
            {"A", "B"},
            {"Pathway": {"A"}},
        ),
        (
            {"A"},
            set(),
            {"Pathway": {"A"}},
        ),
        (
            {"A"},
            {"A", "B"},
            {},
        ),
        (
            {"X"},
            {"A", "B"},
            {"Pathway": {"A"}},
        ),
    ],
)
def test_empty_enrichment(
    query,
    background,
    gene_sets,
):
    result, mapping = ea.run_enrichment(
        query,
        background,
        gene_sets,
    )

    assert result.empty
    assert list(result.columns) == ea.RESULT_COLUMNS

    assert mapping["tested_terms"] == 0
    assert mapping["significant_terms"] == 0


# ============================================================
# 10. 라이브러리 다운로드 및 캐시 검증
# ============================================================

def test_library_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ea,
        "LIBRARY_DIR",
        tmp_path,
    )

    calls = []

    # 실제 인터넷 연결 없이 다운로드를 모의 실행
    def fake_get_library(*args, **kwargs):
        calls.append(1)

        return {
            "Pathway_1": ["a", "B", "A"],
            "Pathway_2": ["C", "D"],
        }

    monkeypatch.setattr(
        ea.gp,
        "get_library",
        fake_get_library,
    )

    library_name = "Test_Library"

    first = ea.load_gene_set_library(
        library_name
    )

    second = ea.load_gene_set_library(
        library_name
    )

    expected = {
        "Pathway_1": {"A", "B"},
        "Pathway_2": {"C", "D"},
    }

    assert first == expected
    assert second == expected

    # 최초 다운로드 이후 캐시를 재사용
    assert len(calls) == 1

    cache_path = (
        tmp_path / f"{library_name}.json"
    )

    metadata_path = (
        tmp_path / f"{library_name}_metadata.json"
    )

    assert cache_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    expected_sha256 = hashlib.sha256(
        cache_path.read_bytes()
    ).hexdigest()

    assert metadata["sha256"] == expected_sha256
    assert metadata["term_count"] == 2


# ============================================================
# 11. 캐시 파일 손상 감지 검증
# ============================================================

def test_corrupted_library_cache(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        ea,
        "LIBRARY_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        ea.gp,
        "get_library",
        lambda *args, **kwargs: {
            "Pathway": ["A", "B"],
        },
    )

    library_name = "Corrupted_Test"

    ea.load_gene_set_library(library_name)

    cache_path = (
        tmp_path / f"{library_name}.json"
    )

    # 캐시 파일 내용을 의도적으로 변경
    cache_path.write_text(
        '{"Changed_Pathway": ["X"]}',
        encoding="utf-8",
    )

    # 체크섬 불일치로 오류가 발생해야 함
    with pytest.raises(
        (ValueError, RuntimeError)
    ):
        ea.load_gene_set_library(library_name)


# ============================================================
# 12. 실제 분석 요약 및 매핑 보고서 불러오기
# ============================================================

@pytest.fixture(scope="module")
def saved_reports():
    summary_path = (
        ea.ENRICHMENT_DIR
        / "enrichment_summary.csv"
    )

    mapping_path = (
        ea.ENRICHMENT_DIR
        / "gene_mapping_report.json"
    )

    assert summary_path.exists()
    assert mapping_path.exists()

    summary = pd.read_csv(summary_path)

    mapping = json.loads(
        mapping_path.read_text(
            encoding="utf-8"
        )
    )

    return summary, mapping


# ============================================================
# 13. 전체 분석 조합 검증
# ============================================================

def test_analysis_combinations(saved_reports):
    summary, mapping = saved_reports

    expected = set(product(
        ea.DATA_FILES,
        ea.GENE_GROUPS,
        ea.GENE_SET_LIBRARIES,
    ))

    actual = set(
        summary[
            ["dataset", "gene_group", "library"]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    assert actual == expected
    assert len(summary) == len(expected)
    assert len(mapping) == len(expected)

    mapping_keys = {
        (
            row["dataset"],
            row["gene_group"],
            row["library"],
        )
        for row in mapping
    }

    assert mapping_keys == expected

    assert not summary.duplicated(
        subset=[
            "dataset",
            "gene_group",
            "library",
        ]
    ).any()


# ============================================================
# 14. 실제 DEG 파일과 원본 데이터 일관성 검증
# ============================================================

@pytest.mark.parametrize(
    "dataset",
    list(ea.DATA_FILES),
)
def test_deg_input_consistency(dataset):
    full_df = load_deg_data(
        ea.DATA_FILES[dataset]
    )

    all_deg, up_deg, down_deg = filter_deg(
        full_df
    )

    expected_groups = {
        "all": all_deg,
        "upregulated": up_deg,
        "downregulated": down_deg,
    }

    for group, suffix in ea.GENE_GROUPS.items():
        path = (
            ea.DEG_DIR
            / f"{dataset}_{suffix}.csv"
        )

        saved = pd.read_csv(path)
        expected = expected_groups[group]

        assert len(saved) == len(expected)

        assert sorted(
            saved["ensembl_id_original"].tolist()
        ) == sorted(
            expected["ensembl_id_original"].tolist()
        )

        assert ea.get_gene_symbols(
            saved
        ) == ea.get_gene_symbols(
            expected
        )


# ============================================================
# 15. 실제 결과 CSV 및 매핑 보고서 검증
# ============================================================

@pytest.mark.parametrize(
    "dataset, group, library",
    list(product(
        ea.DATA_FILES,
        ea.GENE_GROUPS,
        ea.GENE_SET_LIBRARIES,
    )),
)
def test_saved_enrichment_results(
    dataset,
    group,
    library,
    saved_reports,
):
    summary, mapping = saved_reports

    result_dir = (
        ea.ENRICHMENT_DIR / dataset
    )

    all_path = (
        result_dir
        / f"{group}_{library}_all.csv"
    )

    significant_path = (
        result_dir
        / f"{group}_{library}_significant.csv"
    )

    assert all_path.exists()
    assert significant_path.exists()

    all_results = pd.read_csv(all_path)

    significant = pd.read_csv(
        significant_path
    )

    assert list(
        all_results.columns
    ) == ea.RESULT_COLUMNS

    assert list(
        significant.columns
    ) == ea.RESULT_COLUMNS

    # 유의한 결과가 전체 결과의 부분집합인지 검증
    expected_significant = all_results.loc[
        all_results["adjusted_pvalue"]
        < ea.ENRICHMENT_FDR
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        significant.reset_index(drop=True),
        expected_significant,
        check_dtype=False,
    )

    # 요약 파일과 실제 결과 행 수 비교
    selected = summary.loc[
        (summary["dataset"] == dataset)
        & (summary["gene_group"] == group)
        & (summary["library"] == library)
    ]

    assert len(selected) == 1

    row = selected.iloc[0]

    assert len(all_results) == row["tested_terms"]

    assert len(significant) == row[
        "significant_terms"
    ]

    # 해당 분석 조합의 매핑 보고서 확인
    reports = [
        report
        for report in mapping
        if report["dataset"] == dataset
        and report["gene_group"] == group
        and report["library"] == library
    ]

    assert len(reports) == 1

    report = reports[0]

    assert report["tested_terms"] == len(
        all_results
    )

    assert report["significant_terms"] == len(
        significant
    )

    assert report["mapped_query_genes"] == row[
        "mapped_query_genes"
    ]

    assert report["mapped_background_genes"] == row[
        "mapped_background_genes"
    ]

    assert (
        report["mapped_query_genes"]
        + report["unmapped_query_genes"]
        == report["input_query_genes"]
    )

    assert (
        report["mapped_background_genes"]
        + report["unmapped_background_genes"]
        == report["input_background_genes"]
    )

    # 실제 DEG 유전자는 원본 배경에 포함되어야 함
    assert report["query_outside_background"] == 0

    # 빈 결과도 정상적인 결과로 인정
    if all_results.empty:
        return

    # 확률 범위 및 정렬 순서 검증
    assert all_results["pvalue"].between(
        0, 1
    ).all()

    assert all_results[
        "adjusted_pvalue"
    ].between(0, 1).all()

    assert all_results[
        "adjusted_pvalue"
    ].is_monotonic_increasing

    assert (
        all_results["adjusted_pvalue"]
        + 1e-12
        >= all_results["pvalue"]
    ).all()

    assert all_results["term"].is_unique

    # 유전자 수의 논리적 관계 검증
    assert (
        all_results["overlap_count"] >= 0
    ).all()

    assert (
        all_results["overlap_count"]
        <= all_results["query_size"]
    ).all()

    assert (
        all_results["overlap_count"]
        <= all_results["term_size"]
    ).all()

    assert (
        all_results["term_size"]
        <= all_results["background_size"]
    ).all()

    assert (
        all_results["query_size"]
        <= all_results["background_size"]
    ).all()

    assert (
        all_results["query_size"]
        == report["mapped_query_genes"]
    ).all()

    assert (
        all_results["background_size"]
        == report["mapped_background_genes"]
    ).all()

    # Gene ratio 및 Fold enrichment 재계산
    expected_ratio = (
        all_results["overlap_count"]
        / all_results["query_size"]
    )

    expected_fold = (
        expected_ratio
        / (
            all_results["term_size"]
            / all_results["background_size"]
        )
    )

    np.testing.assert_allclose(
        all_results["gene_ratio"],
        expected_ratio,
    )

    np.testing.assert_allclose(
        all_results["fold_enrichment"],
        expected_fold,
    )

    # 실제 중첩 유전자 목록의 개수 검증
    for result in all_results.itertuples(
        index=False
    ):
        genes = (
            set()
            if pd.isna(result.genes)
            else set(
                str(result.genes).split(";")
            )
        )

        assert len(genes) == result.overlap_count
