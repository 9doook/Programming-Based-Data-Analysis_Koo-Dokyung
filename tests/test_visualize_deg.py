"""
GSE159984 DEG 시각화 자동 테스트

1. DEG 분석 결과와 시각화 분류 결과 비교
2. 조정 p-value가 0인 경우의 처리 검증
3. 유효하지 않은 데이터 처리 검증
4. 확대 범위 및 데이터 보존 검증
5. PNG 및 PDF 결과 파일 생성 검증
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt

from config.analysis_config import (
    DATA_FILES,
    DEG_DIR,
    PADJ_THRESHOLD,
    LOG2FC_THRESHOLD,
)

from src import visualize_deg as viz


# ========================================
# 1. 실제 분석 결과와 시각화 결과 비교
# ========================================

@pytest.mark.parametrize(
    "dataset_name",
    list(DATA_FILES),
)
def test_volcano_classification(dataset_name):
    """기존 DEG 분석 결과와 시각화 분류 결과를 비교한다."""

    df, excluded = viz.prepare_volcano_data(
        DATA_FILES[dataset_name]
    )

    summary = pd.read_csv(
        DEG_DIR / "deg_summary.csv"
    )

    selected = summary.loc[
        summary["dataset"] == dataset_name
    ]

    assert len(selected) == 1

    row = selected.iloc[0]

    up_count = (
        df["regulation"] == "Upregulated"
    ).sum()

    down_count = (
        df["regulation"] == "Downregulated"
    ).sum()

    ns_count = (
        df["regulation"] == "Not selected"
    ).sum()

    assert up_count == row["upregulated"]

    assert down_count == row["downregulated"]

    assert (
        up_count + down_count
        == row["significant_deg"]
    )

    assert (
        up_count + down_count + ns_count
        == len(df)
    )

    assert (
        len(df) + excluded
        == row["total_genes"]
    )

    # 로그 변환 결과에 무한값이 없는지 확인
    assert np.isfinite(
        df["neg_log10_padj"]
    ).all()

    # 실제 DEG 선별 결과와 유전자 ID 비교
    for category, suffix in [
        ("Upregulated", "upregulated"),
        ("Downregulated", "downregulated"),
    ]:

        expected = pd.read_csv(
            DEG_DIR / f"{dataset_name}_{suffix}.csv"
        )

        actual = df.loc[
            df["regulation"] == category
        ]

        # 결측값과 문자열 ID가 혼합되어도 안전하게 비교하도록
        # 양쪽 ID를 동일한 문자열 형식으로 정규화한다.
        actual_ids = [
            "" if pd.isna(value) else str(value)
            for value in actual["ensembl_id_original"]
        ]

        expected_ids = [
            "" if pd.isna(value) else str(value)
            for value in expected["ensembl_id_original"]
        ]

        assert sorted(actual_ids) == sorted(expected_ids)


# ========================================
# 2. 조정 p-value가 0인 경우 검증
# ========================================

def test_zero_adjusted_pvalue(monkeypatch):
    """padj가 0이어도 로그 변환이 정상적으로 수행되는지 검증한다."""

    original = pd.DataFrame({
        "ensembl_id_original": [
            "gene1",
            "gene2",
        ],
        "log2FoldChange": [
            LOG2FC_THRESHOLD + 1,
            0.1,
        ],
        "padj": [
            0.0,
            0.01,
        ],
    })

    monkeypatch.setattr(
        viz,
        "load_deg_data",
        lambda _: original.copy(),
    )

    result, excluded = viz.prepare_volcano_data(
        "dummy.tsv.gz"
    )

    assert excluded == 0

    assert len(result) == 2

    assert np.isfinite(
        result["neg_log10_padj"]
    ).all()

    # 원래 조정 p-value를 변경하지 않았는지 확인
    assert result.loc[0, "padj"] == 0.0

    assert original.loc[0, "padj"] == 0.0

    # DEG 분류 결과 확인
    assert (
        result.loc[0, "regulation"]
        == "Upregulated"
    )


# ========================================
# 3. 유효하지 않은 데이터 처리 검증
# ========================================

def test_invalid_values(monkeypatch):
    """결측값, 무한값 및 잘못된 p-value 처리를 검증한다."""

    original = pd.DataFrame({
        "ensembl_id_original": [
            "valid",
            "missing_fc",
            "infinite_fc",
            "negative_padj",
            "large_padj",
            "missing_padj",
        ],
        "log2FoldChange": [
            LOG2FC_THRESHOLD + 1,
            np.nan,
            np.inf,
            1.0,
            1.0,
            1.0,
        ],
        "padj": [
            0.01,
            0.01,
            0.01,
            -0.01,
            1.5,
            np.nan,
        ],
    })

    monkeypatch.setattr(
        viz,
        "load_deg_data",
        lambda _: original.copy(),
    )

    result, excluded = viz.prepare_volcano_data(
        "dummy.tsv.gz"
    )

    assert excluded == 5

    assert len(result) == 1

    assert (
        result.iloc[0]["ensembl_id_original"]
        == "valid"
    )


# ========================================
# 4. Volcano plot 기준선 검증
# ========================================

def test_volcano_threshold_lines():
    """그래프의 DEG 기준선이 설정값과 일치하는지 검증한다."""

    df = pd.DataFrame({
        "log2FoldChange": [-1.0, 0.1, 1.0],
        "neg_log10_padj": [3.0, 0.5, 3.0],
        "regulation": [
            "Downregulated",
            "Not selected",
            "Upregulated",
        ],
    })

    fig, ax = plt.subplots()

    try:

        viz.draw_volcano(
            ax,
            df,
            "Test Volcano",
            show_legend=False,
        )

        assert len(ax.lines) == 3

        np.testing.assert_allclose(
            ax.lines[0].get_xdata(),
            [-LOG2FC_THRESHOLD] * 2,
        )

        np.testing.assert_allclose(
            ax.lines[1].get_xdata(),
            [LOG2FC_THRESHOLD] * 2,
        )

        np.testing.assert_allclose(
            ax.lines[2].get_ydata(),
            [-np.log10(PADJ_THRESHOLD)] * 2,
        )

        assert len(ax.collections) == 3

        # 산점도에 모든 데이터가 전달됐는지 확인
        plotted_count = sum(
            len(collection.get_offsets())
            for collection in ax.collections
        )

        assert plotted_count == len(df)

    finally:

        plt.close(fig)


# ========================================
# 5. 확대 그래프 및 데이터 보존 검증
# ========================================

def test_zoomed_volcano(monkeypatch):
    """확대 범위 설정 및 원본 데이터 보존을 검증한다."""

    df = pd.DataFrame({
        "log2FoldChange": [
            -1.0,
            0.1,
            1.0,
            viz.ZOOM_X_LIMIT + 1,
            1.0,
        ],
        "neg_log10_padj": [
            3.0,
            0.5,
            3.0,
            5.0,
            viz.ZOOM_Y_LIMIT + 1,
        ],
        "regulation": [
            "Downregulated",
            "Not selected",
            "Upregulated",
            "Upregulated",
            "Upregulated",
        ],
    })

    original = df.copy(deep=True)

    captured = {}

    def fake_save(fig, filename):
        captured["figure"] = fig
        captured["filename"] = filename

    monkeypatch.setattr(
        viz,
        "save_figure",
        fake_save,
    )

    try:

        viz.plot_zoomed_volcano(df)

        fig = captured["figure"]

        ax = fig.axes[0]

        assert captured["filename"] == (
            "combined_stress_volcano_zoomed"
        )

        np.testing.assert_allclose(
            ax.get_xlim(),
            [
                -viz.ZOOM_X_LIMIT,
                viz.ZOOM_X_LIMIT,
            ],
        )

        np.testing.assert_allclose(
            ax.get_ylim(),
            [
                0,
                viz.ZOOM_Y_LIMIT,
            ],
        )

        # 확대 범위 밖 데이터가 안내되는지 확인
        footer = " ".join(
            text.get_text()
            for text in fig.texts
        )

        assert "Points outside view: 2" in footer

        # 확대 범위 밖 점도 삭제되지 않았는지 확인
        plotted_count = sum(
            len(collection.get_offsets())
            for collection in ax.collections
        )

        assert plotted_count == len(df)

        pd.testing.assert_frame_equal(
            original,
            df,
        )

    finally:

        if "figure" in captured:
            plt.close(captured["figure"])


# ========================================
# 6. 통합 Volcano plot 검증
# ========================================

def test_combined_volcano(monkeypatch):
    """4개 패널의 공통 축과 데이터 보존을 검증한다."""

    dataframes = {
        name: pd.DataFrame({
            "log2FoldChange": [-1.0, 0.1, 1.0],
            "neg_log10_padj": [3.0, 0.5, 3.0],
            "regulation": [
                "Downregulated",
                "Not selected",
                "Upregulated",
            ],
        })
        for name in DATA_FILES
    }

    captured = {}

    def fake_save(fig, filename):
        captured["figure"] = fig
        captured["filename"] = filename

    monkeypatch.setattr(
        viz,
        "save_figure",
        fake_save,
    )

    try:

        viz.plot_combined_volcano(dataframes)

        fig = captured["figure"]

        assert captured["filename"] == (
            "combined_volcano_comparison"
        )

        assert len(fig.axes) == 4

        # 4개 패널 모두 동일한 축 범위를 사용하는지 확인
        for ax in fig.axes:

            np.testing.assert_allclose(
                ax.get_xlim(),
                [
                    -viz.ZOOM_X_LIMIT,
                    viz.ZOOM_X_LIMIT,
                ],
            )

            np.testing.assert_allclose(
                ax.get_ylim(),
                [
                    0,
                    viz.ZOOM_Y_LIMIT,
                ],
            )

            plotted_count = sum(
                len(collection.get_offsets())
                for collection in ax.collections
            )

            assert plotted_count == 3

        # 공통 범례 존재 여부 확인
        assert len(fig.legends) == 1

    finally:

        if "figure" in captured:
            plt.close(captured["figure"])


# ========================================
# 7. 실제 그래프 파일 생성 검증
# ========================================

def test_visualization_output(monkeypatch, tmp_path):
    """전체 시각화 코드를 실행하고 14개 파일 생성을 검증한다."""

    # 기존 연구 결과를 덮어쓰지 않도록 임시 폴더 사용
    monkeypatch.setattr(
        viz,
        "PLOT_DIR",
        tmp_path,
    )

    viz.main()

    expected_names = {
        "deg_count_comparison",
        "t2d_volcano",
        "combined_stress_volcano",
        "palmitate_volcano",
        "glucose_volcano",
        "combined_stress_volcano_zoomed",
        "combined_volcano_comparison",
    }

    expected_files = {
        f"{name}.{extension}"
        for name in expected_names
        for extension in ["png", "pdf"]
    }

    actual_files = {
        path.name
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    assert actual_files == expected_files

    assert len(actual_files) == 14

    for filename in expected_files:

        path = tmp_path / filename

        # 파일이 비어 있지 않은지 확인
        assert path.stat().st_size > 0

        # 정상적인 이미지 및 PDF 파일 형식인지 확인
        with path.open("rb") as file:
            signature = file.read(8)

        if path.suffix == ".png":
            assert signature == b"\x89PNG\r\n\x1a\n"

        elif path.suffix == ".pdf":
            assert signature.startswith(b"%PDF-")

