
"""
GSE159984 DEG 분석 자동 테스트

DEG 선별 기준, 발현 변화 방향, 결측 주석 처리,
실제 분석 결과 파일의 일관성을 검증한다.
"""

import pandas as pd
import pytest

from config.analysis_config import (
    DATA_FILES,
    DEG_DIR,
    PADJ_THRESHOLD,
    LOG2FC_THRESHOLD,
)

from src.analyze_deg import filter_deg, load_deg_data


def test_filter_deg():
    """DEG 선별 및 상향·하향 조절 분류를 검증한다."""

    df = pd.DataFrame({
        "ensembl_id_original": [
            "gene1", "gene2", "gene3", "gene4",
            "gene5", "gene6", "gene7",
        ],
        "padj": [
            0.01, 0.01, PADJ_THRESHOLD, 0.01,
            0.01, 0.01, 0.01,
        ],
        "log2FoldChange": [
            LOG2FC_THRESHOLD + 0.5,
            -LOG2FC_THRESHOLD - 0.5,
            LOG2FC_THRESHOLD + 0.5,
            LOG2FC_THRESHOLD,
            -LOG2FC_THRESHOLD,
            0.1,
            LOG2FC_THRESHOLD + 0.5,
        ],
        "gene_id": [
            "1", "2", "3", "4", "5", "6", None,
        ],
    })

    significant, upregulated, downregulated = filter_deg(df)

    # 유의한 DEG 선별 결과 확인
    assert set(significant["ensembl_id_original"]) == {
        "gene1", "gene2", "gene7",
    }

    # 상향 조절 유전자 확인
    assert set(upregulated["ensembl_id_original"]) == {
        "gene1", "gene7",
    }

    # 하향 조절 유전자 확인
    assert set(downregulated["ensembl_id_original"]) == {
        "gene2",
    }

    # 전체 DEG 수와 상향·하향 조절 유전자 수의 합 비교
    assert len(significant) == (
        len(upregulated) + len(downregulated)
    )


def test_missing_annotation():
    """유전자 주석이 없어도 DEG 기준을 충족하면 유지되는지 검증한다."""

    df = pd.DataFrame({
        "ensembl_id_original": ["ENSG00000000001"],
        "gene_id": [None],
        "short_id": [None],
        "padj": [0.01],
        "log2FoldChange": [LOG2FC_THRESHOLD + 0.5],
    })

    significant, upregulated, downregulated = filter_deg(df)

    assert len(significant) == 1
    assert len(upregulated) == 1
    assert len(downregulated) == 0

    assert significant["gene_id"].isna().all()


@pytest.mark.parametrize("dataset_name", list(DATA_FILES))
def test_saved_deg_results(dataset_name):
    """실제 데이터와 저장된 DEG 분석 결과를 비교한다."""

    summary_path = DEG_DIR / "deg_summary.csv"

    result_paths = {
        "all": DEG_DIR / f"{dataset_name}_all_deg.csv",
        "up": DEG_DIR / f"{dataset_name}_upregulated.csv",
        "down": DEG_DIR / f"{dataset_name}_downregulated.csv",
    }

    # 실제 데이터와 분석 결과 파일의 존재 여부 확인
    assert DATA_FILES[dataset_name].exists()
    assert summary_path.exists()

    for path in result_paths.values():
        assert path.exists()

    # 원본 데이터에서 예상 결과 계산
    original = load_deg_data(DATA_FILES[dataset_name])

    expected = original[
        (original["padj"] < PADJ_THRESHOLD)
        & (
            original["log2FoldChange"].abs()
            > LOG2FC_THRESHOLD
        )
    ]

    expected_up = expected[
        expected["log2FoldChange"] > LOG2FC_THRESHOLD
    ]

    expected_down = expected[
        expected["log2FoldChange"] < -LOG2FC_THRESHOLD
    ]

    # 저장된 결과 불러오기
    all_deg = pd.read_csv(result_paths["all"])
    upregulated = pd.read_csv(result_paths["up"])
    downregulated = pd.read_csv(result_paths["down"])

    summary = pd.read_csv(summary_path)

    selected = summary[summary["dataset"] == dataset_name]

    assert len(selected) == 1

    row = selected.iloc[0]

    # 분석 기준 충족 여부 확인
    assert (all_deg["padj"] < PADJ_THRESHOLD).all()

    assert (
        all_deg["log2FoldChange"].abs()
        > LOG2FC_THRESHOLD
    ).all()

    assert (
        upregulated["log2FoldChange"]
        > LOG2FC_THRESHOLD
    ).all()

    assert (
        downregulated["log2FoldChange"]
        < -LOG2FC_THRESHOLD
    ).all()

    # 원본 Ensembl ID 목록과 발현 통계 비교
    columns = [
        "ensembl_id_original",
        "padj",
        "log2FoldChange",
    ]

    for actual, reference in [
        (all_deg, expected),
        (upregulated, expected_up),
        (downregulated, expected_down),
    ]:
        pd.testing.assert_frame_equal(
            actual[columns].reset_index(drop=True),
            reference[columns].reset_index(drop=True),
            check_dtype=False,
        )

    # 전체 DEG 수의 일관성 확인
    assert len(all_deg) == (
        len(upregulated) + len(downregulated)
    )

    # 요약 파일과 실제 분석 결과 비교
    assert row["total_genes"] == len(original)
    assert row["significant_deg"] == len(all_deg)
    assert row["upregulated"] == len(upregulated)
    assert row["downregulated"] == len(downregulated)

