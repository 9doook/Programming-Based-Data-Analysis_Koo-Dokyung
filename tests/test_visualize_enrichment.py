"""
기능적 풍부도 분석 시각화 자동 테스트

- 분석 결과 데이터 검증
- Dot Plot의 수치 및 시각적 요소 검증
- 빈 결과 및 경계값 처리 검증
- 통합 비교 그래프 검증
- 요약 그래프 검증
- 전체 출력 파일 검증
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from src import visualize_enrichment as viz


# ============================================================
# 1. 테스트용 공통 데이터
# ============================================================

COMBINATIONS = list(product(
    viz.DATASET_LABELS,
    viz.GENE_GROUPS,
    viz.GENE_SET_LIBRARIES,
))


@pytest.fixture
def sample_data():
    """정상적인 풍부도 분석 결과를 생성한다."""

    return pd.DataFrame({
        "term": ["Pathway A", "Pathway B", "Pathway C"],
        "overlap_count": [10, 5, 3],
        "gene_ratio": [0.10, 0.05, 0.03],
        "adjusted_pvalue": [0.001, 0.01, 0.04],
    })


@pytest.fixture(scope="module")
def summary():
    """실제 분석 요약 데이터를 불러온다."""

    return viz.load_summary()


@pytest.fixture
def captured_figures(monkeypatch):
    """파일 저장 대신 그래프 객체를 보관한다."""

    figures = {}

    def capture(fig, filename):
        figures[filename] = fig

    monkeypatch.setattr(viz, "save_figure", capture)

    yield figures

    for fig in figures.values():
        viz.plt.close(fig)


@pytest.fixture
def temporary_results(tmp_path, monkeypatch):
    """임시 분석 결과 폴더를 생성한다."""

    monkeypatch.setattr(
        viz,
        "ENRICHMENT_DIR",
        tmp_path,
    )

    folder = tmp_path / "t2d"
    folder.mkdir()

    return folder / "all_go_bp_significant.csv"


# ============================================================
# 2. 분석 결과 불러오기 테스트
# ============================================================

def test_load_enrichment_results(
    sample_data,
    temporary_results,
):
    """유의성 기준 및 정렬 순서를 검증한다."""

    extra = pd.DataFrame({
        "term": ["Boundary", "Not significant"],
        "overlap_count": [2, 1],
        "gene_ratio": [0.02, 0.01],
        "adjusted_pvalue": [
            viz.ENRICHMENT_FDR,
            0.10,
        ],
    })

    data = pd.concat(
        [sample_data, extra],
        ignore_index=True,
    )

    data = data.iloc[::-1]

    data.to_csv(
        temporary_results,
        index=False,
    )

    result = viz.load_enrichment_results(
        "t2d", "all", "go_bp"
    )

    assert len(result) == 3

    assert result["term"].tolist() == [
        "Pathway A",
        "Pathway B",
        "Pathway C",
    ]

    assert (
        result["adjusted_pvalue"]
        < viz.ENRICHMENT_FDR
    ).all()


def test_missing_result_file(temporary_results):
    """파일이 없으면 오류를 발생시키는지 확인한다."""

    with pytest.raises(FileNotFoundError):
        viz.load_enrichment_results(
            "t2d", "all", "go_bp"
        )


def test_missing_required_column(
    sample_data,
    temporary_results,
):
    """필수 열이 누락된 경우를 검증한다."""

    sample_data.drop(
        columns=["gene_ratio"]
    ).to_csv(
        temporary_results,
        index=False,
    )

    with pytest.raises(ValueError):
        viz.load_enrichment_results(
            "t2d", "all", "go_bp"
        )


@pytest.mark.parametrize(
    "column, value",
    [
        ("adjusted_pvalue", -0.01),
        ("adjusted_pvalue", 1.1),
        ("adjusted_pvalue", np.nan),
        ("adjusted_pvalue", np.inf),
        ("gene_ratio", -0.1),
        ("gene_ratio", 1.1),
        ("gene_ratio", np.nan),
        ("overlap_count", 0),
        ("overlap_count", -1),
        ("overlap_count", 1.5),
        ("overlap_count", np.inf),
    ],
)
def test_invalid_statistics(
    sample_data,
    temporary_results,
    column,
    value,
):
    """잘못된 통계량을 거부하는지 확인한다."""

    sample_data[column] = (
        sample_data[column].astype(float)
    )

    sample_data.loc[0, column] = value

    sample_data.to_csv(
        temporary_results,
        index=False,
    )

    with pytest.raises(ValueError):
        viz.load_enrichment_results(
            "t2d", "all", "go_bp"
        )


def test_load_empty_result(
    sample_data,
    temporary_results,
):
    """유의한 항목이 없는 CSV도 정상적으로 읽는다."""

    sample_data.iloc[0:0].to_csv(
        temporary_results,
        index=False,
    )

    result = viz.load_enrichment_results(
        "t2d", "all", "go_bp"
    )

    assert result.empty


# ============================================================
# 3. Dot Plot 데이터 준비 테스트
# ============================================================

def test_top_n_selection():
    """최대 10개 항목을 선택하는지 검증한다."""

    data = pd.DataFrame({
        "term": [
            f"Pathway {i}" for i in range(15)
        ],
        "overlap_count": [3] * 15,
        "gene_ratio": [0.03] * 15,
        "adjusted_pvalue": np.linspace(
            0.001, 0.04, 15
        ),
    })

    original = data.copy(deep=True)

    result = viz.prepare_dotplot_data(data)

    assert len(result) == viz.TOP_N

    assert result["term"].tolist() == (
        data.head(viz.TOP_N)["term"].tolist()
    )

    np.testing.assert_allclose(
        result["neg_log10_padj"],
        -np.log10(
            data.head(viz.TOP_N)["adjusted_pvalue"]
        ),
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_custom_top_n(sample_data):
    """사용자가 지정한 표시 항목 수를 검증한다."""

    result = viz.prepare_dotplot_data(
        sample_data,
        top_n=2,
    )

    assert len(result) == 2


def test_zero_adjusted_pvalue(sample_data):
    """p-value가 0이어도 무한대가 발생하지 않는다."""

    sample_data.loc[0, "adjusted_pvalue"] = 0

    original = sample_data.copy(deep=True)

    result = viz.prepare_dotplot_data(
        sample_data
    )

    assert np.isfinite(
        result["neg_log10_padj"]
    ).all()

    expected = -np.log10(0.01 / 10)

    assert result.loc[
        0, "neg_log10_padj"
    ] == pytest.approx(expected)

    assert result.loc[
        0, "adjusted_pvalue"
    ] == 0

    pd.testing.assert_frame_equal(
        sample_data,
        original,
    )


def test_all_zero_adjusted_pvalues(sample_data):
    """모든 p-value가 0인 경계 상황을 검증한다."""

    sample_data["adjusted_pvalue"] = 0.0

    result = viz.prepare_dotplot_data(
        sample_data
    )

    assert np.isfinite(
        result["neg_log10_padj"]
    ).all()

    np.testing.assert_allclose(
        result["neg_log10_padj"],
        -np.log10(np.finfo(float).tiny),
    )


def test_prepare_empty_dotplot(sample_data):
    """빈 데이터의 시각화 준비를 검증한다."""

    result = viz.prepare_dotplot_data(
        sample_data.iloc[0:0]
    )

    assert result.empty
    assert "neg_log10_padj" in result.columns


# ============================================================
# 4. 색상 척도 및 점 크기 범례 테스트
# ============================================================

def test_color_normalization(sample_data):
    """공통 색상 범위를 검증한다."""

    frame = viz.prepare_dotplot_data(
        sample_data
    )

    norm = viz.create_color_norm([frame])

    assert norm.vmin == pytest.approx(
        -np.log10(viz.ENRICHMENT_FDR)
    )

    assert norm.vmax == pytest.approx(
        frame["neg_log10_padj"].max()
    )


def test_shared_color_normalization(sample_data):
    """여러 데이터의 최댓값을 사용하는지 검증한다."""

    first = viz.prepare_dotplot_data(
        sample_data
    )

    second = first.copy()

    second["neg_log10_padj"] = [4, 5, 6]

    norm = viz.create_color_norm([
        first,
        second,
    ])

    assert norm.vmax == pytest.approx(6)


def test_empty_color_normalization():
    """모든 결과가 비어 있어도 색상 범위가 유효하다."""

    norm = viz.create_color_norm([])

    assert norm.vmax > norm.vmin

    assert norm.vmax - norm.vmin == (
        pytest.approx(1)
    )


def test_size_legend(sample_data):
    """점 크기 범례가 실제 유전자 수와 일치한다."""

    frame = viz.prepare_dotplot_data(
        sample_data
    )

    handles = viz.create_size_legend([frame])

    assert [
        handle.get_label()
        for handle in handles
    ] == ["3", "5", "10"]

    for handle in handles:

        count = int(handle.get_label())

        assert handle.get_markersize() == (
            pytest.approx(
                np.sqrt(count * viz.SIZE_FACTOR)
            )
        )


def test_single_size_legend(sample_data):
    """모든 점의 크기가 같으면 범례 중복을 방지한다."""

    sample_data["overlap_count"] = 3

    handles = viz.create_size_legend([
        sample_data
    ])

    assert len(handles) == 1
    assert handles[0].get_label() == "3"


def test_empty_size_legend():
    """빈 결과에는 점 크기 범례가 없어야 한다."""

    assert viz.create_size_legend([]) == []


# ============================================================
# 5. Dot Plot 그래프 요소 검증
# ============================================================

def test_draw_dotplot(sample_data):
    """점 위치, 크기, 색상 및 축 범위를 검증한다."""

    frame = viz.prepare_dotplot_data(
        sample_data
    )

    norm = viz.create_color_norm([frame])

    fig, ax = viz.plt.subplots()

    try:

        scatter = viz.draw_dotplot(
            ax,
            frame,
            norm,
            x_limit=0.2,
        )

        offsets = scatter.get_offsets()

        np.testing.assert_allclose(
            offsets[:, 0],
            frame["gene_ratio"],
        )

        np.testing.assert_allclose(
            offsets[:, 1],
            np.arange(len(frame)),
        )

        np.testing.assert_allclose(
            scatter.get_sizes(),
            frame["overlap_count"]
            * viz.SIZE_FACTOR,
        )

        np.testing.assert_allclose(
            scatter.get_array(),
            frame["neg_log10_padj"],
        )

        assert scatter.norm is norm

        assert ax.get_xlim() == (
            pytest.approx((0, 0.2))
        )

        assert ax.yaxis_inverted()

        assert ax.get_xlabel() == "Gene Ratio"

        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

    finally:
        viz.plt.close(fig)


def test_draw_empty_dotplot(sample_data):
    """빈 그래프의 안내 문구와 축 제거를 검증한다."""

    frame = viz.prepare_dotplot_data(
        sample_data.iloc[0:0]
    )

    fig, ax = viz.plt.subplots()

    try:

        result = viz.draw_dotplot(
            ax,
            frame,
            viz.create_color_norm([]),
        )

        assert result is None

        assert not ax.axison

        assert len(ax.collections) == 0

        assert any(
            text.get_text() == "No significant terms"
            for text in ax.texts
        )

    finally:
        viz.plt.close(fig)


# ============================================================
# 6. 실제 분석 결과와의 일치 여부
# ============================================================

@pytest.mark.parametrize(
    "dataset, group, library",
    COMBINATIONS,
)
def test_saved_enrichment_data(
    dataset,
    group,
    library,
    summary,
):
    """48개 분석 조합의 실제 CSV와 시각화 데이터를 비교한다."""

    result = viz.load_enrichment_results(
        dataset,
        group,
        library,
    )

    original = pd.read_csv(
        viz.ENRICHMENT_DIR
        / dataset
        / f"{group}_{library}_significant.csv"
    )

    expected = summary.loc[
        (summary["dataset"] == dataset)
        & (summary["gene_group"] == group)
        & (summary["library"] == library)
    ]

    assert len(expected) == 1

    assert len(result) == int(
        expected["significant_terms"].iloc[0]
    )

    assert len(result) == len(original)

    sort_columns = ["adjusted_pvalue"]

    if "pvalue" in original.columns:
        sort_columns.append("pvalue")

    sort_columns.append("term")

    original = original.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result,
        original,
        check_dtype=False,
    )

    frame = viz.prepare_dotplot_data(result)

    assert len(frame) == min(
        len(result),
        viz.TOP_N,
    )

    assert frame["term"].tolist() == (
        result.head(viz.TOP_N)["term"].tolist()
    )

    assert np.isfinite(
        frame["neg_log10_padj"]
    ).all()

    if not result.empty:

        np.testing.assert_allclose(
            result["gene_ratio"],
            result["overlap_count"]
            / result["query_size"],
        )


# ============================================================
# 7. 개별 그래프 검증
# ============================================================

def test_individual_enrichment(
    captured_figures,
):
    """T2D GO BP 그래프의 표시 항목을 검증한다."""

    result = viz.load_enrichment_results(
        "t2d", "all", "go_bp"
    )

    expected = viz.prepare_dotplot_data(
        result
    )

    is_empty = viz.plot_individual_enrichment(
        "t2d", "all", "go_bp"
    )

    assert is_empty is False

    fig = captured_figures[
        "t2d_all_go_bp_dotplot"
    ]

    ax = fig.axes[0]

    scatter = ax.collections[0]

    assert len(scatter.get_offsets()) == (
        len(expected)
    )

    np.testing.assert_allclose(
        scatter.get_offsets()[:, 0],
        expected["gene_ratio"],
    )

    assert len(fig.axes) == 2

    assert len(fig.legends) == 1

    assert any(
        f"Showing {len(expected)} of "
        f"{len(result)} significant terms"
        in text.get_text()
        for text in fig.texts
    )


def test_individual_empty_enrichment(
    captured_figures,
    summary,
):
    """Glucose GO BP의 빈 결과 안내를 검증한다."""

    row = summary.loc[
        (summary["dataset"] == "glucose")
        & (summary["gene_group"] == "all")
        & (summary["library"] == "go_bp")
    ].iloc[0]

    tested_terms = int(row["tested_terms"])

    is_empty = viz.plot_individual_enrichment(
        "glucose",
        "all",
        "go_bp",
        tested_terms=tested_terms,
    )

    assert is_empty is True

    fig = captured_figures[
        "glucose_all_go_bp_dotplot"
    ]

    assert len(fig.axes) == 1

    ax = fig.axes[0]

    assert not ax.axison

    assert any(
        "No significant terms" in text.get_text()
        for text in ax.texts
    )

    assert any(
        f"Tested terms: {tested_terms:,}"
        in text.get_text()
        for text in fig.texts
    )

    assert any(
        "Significant terms: 0" in text.get_text()
        for text in fig.texts
    )


# ============================================================
# 8. 라이브러리별 통합 비교 그래프 검증
# ============================================================

@pytest.mark.parametrize(
    "library",
    list(viz.GENE_SET_LIBRARIES),
)
def test_library_comparison(
    library,
    captured_figures,
):
    """네 조건의 공통 축, 색상 및 점 크기를 검증한다."""

    viz.plot_library_comparison(library)

    fig = captured_figures[
        f"{library}_dataset_comparison"
    ]

    panels = fig.axes[:4]

    assert len(panels) == 4

    x_limits = []
    norms = []

    for ax, dataset in zip(
        panels,
        viz.DATASET_LABELS,
    ):

        result = viz.load_enrichment_results(
            dataset,
            "all",
            library,
        )

        expected = viz.prepare_dotplot_data(
            result
        )

        assert (
            f"{len(expected)} of "
            f"{len(result)} significant terms"
        ) in ax.get_title()

        if expected.empty:

            assert not ax.axison

            assert any(
                "No significant terms"
                in text.get_text()
                for text in ax.texts
            )

            continue

        scatter = ax.collections[0]

        np.testing.assert_allclose(
            scatter.get_offsets()[:, 0],
            expected["gene_ratio"],
        )

        np.testing.assert_allclose(
            scatter.get_sizes(),
            expected["overlap_count"]
            * viz.SIZE_FACTOR,
        )

        np.testing.assert_allclose(
            scatter.get_array(),
            expected["neg_log10_padj"],
        )

        x_limits.append(ax.get_xlim())
        norms.append(scatter.norm)

    if x_limits:

        for limits in x_limits:

            assert limits == pytest.approx(
                x_limits[0]
            )

        assert all(
            norm is norms[0]
            for norm in norms
        )

        assert len(fig.axes) == 5
        assert len(fig.legends) == 1

    else:

        assert len(fig.axes) == 4


# ============================================================
# 9. 요약 데이터 검증
# ============================================================

def test_summary_combinations(summary):
    """48개 분석 조합과 빈 결과 개수를 검증한다."""

    keys = [
        "dataset",
        "gene_group",
        "library",
    ]

    actual = set(
        summary[keys].itertuples(
            index=False,
            name=None,
        )
    )

    assert actual == set(COMBINATIONS)

    assert len(summary) == len(COMBINATIONS)

    assert not summary.duplicated(keys).any()

    assert (
        summary["significant_terms"]
        <= summary["tested_terms"]
    ).all()

    # 현재 분석 결과의 회귀 테스트
    assert (
        summary["significant_terms"] == 0
    ).sum() == 15


@pytest.mark.parametrize(
    "error_type",
    [
        "missing_column",
        "duplicate",
        "missing_combination",
        "negative_count",
        "fractional_count",
        "excess_count",
        "nonfinite_count",
    ],
)
def test_invalid_summary(
    summary,
    tmp_path,
    monkeypatch,
    error_type,
):
    """잘못된 요약 데이터가 거부되는지 검증한다."""

    data = summary.copy(deep=True)

    if error_type == "missing_column":

        data = data.drop(
            columns=["tested_terms"]
        )

    elif error_type == "duplicate":

        data = pd.concat(
            [data, data.iloc[[0]]],
            ignore_index=True,
        )

    elif error_type == "missing_combination":

        data = data.iloc[1:].copy()

    elif error_type == "negative_count":

        data.loc[0, "significant_terms"] = -1

    elif error_type == "fractional_count":

        data["significant_terms"] = (
            data["significant_terms"].astype(float)
        )

        data.loc[0, "significant_terms"] = 1.5

    elif error_type == "excess_count":

        data.loc[0, "significant_terms"] = (
            data.loc[0, "tested_terms"] + 1
        )

    elif error_type == "nonfinite_count":

        data["tested_terms"] = (
            data["tested_terms"].astype(float)
        )

        data.loc[0, "tested_terms"] = np.inf

    data.to_csv(
        tmp_path / "enrichment_summary.csv",
        index=False,
    )

    monkeypatch.setattr(
        viz,
        "ENRICHMENT_DIR",
        tmp_path,
    )

    with pytest.raises(ValueError):
        viz.load_summary()


# ============================================================
# 10. 요약 그래프 검증
# ============================================================

@pytest.mark.parametrize(
    "independent_y",
    [False, True],
)
def test_summary_plot(
    summary,
    captured_figures,
    independent_y,
):
    """막대 높이, 숫자 표시 및 Y축 범위를 검증한다."""

    viz.plot_enrichment_summary(
        independent_y=independent_y,
        summary=summary,
    )

    filename = (
        "enrichment_summary_comparison_independent_y"
        if independent_y
        else "enrichment_summary_comparison"
    )

    fig = captured_figures[filename]

    assert len(fig.axes) == 4

    assert len(fig.legends) == 1

    y_limits = []

    for ax, library in zip(
        fig.axes,
        viz.LIBRARY_LABELS,
    ):

        expected = []

        library_df = summary.loc[
            summary["library"] == library
        ]

        for group in viz.GROUP_LABELS:

            counts = (
                library_df.loc[
                    library_df["gene_group"] == group
                ]
                .set_index("dataset")
                .loc[
                    list(viz.DATASET_LABELS),
                    "significant_terms",
                ]
                .tolist()
            )

            expected.extend(counts)

        actual = [
            bar.get_height()
            for bar in ax.patches
        ]

        assert len(actual) == 12

        np.testing.assert_allclose(
            actual,
            expected,
        )

        assert [
            text.get_text()
            for text in ax.texts
        ] == [
            str(count)
            for count in expected
        ]

        y_limits.append(ax.get_ylim())

        if independent_y:

            expected_limit = max(
                library_df[
                    "significant_terms"
                ].max() * 1.25,
                5,
            )

        else:

            expected_limit = max(
                summary[
                    "significant_terms"
                ].max() * 1.20,
                5,
            )

        assert ax.get_ylim() == (
            pytest.approx((0, expected_limit))
        )

    if not independent_y:

        assert all(
            limits == y_limits[0]
            for limits in y_limits
        )

    assert any(
        "biological effect size or pathway activity"
        in text.get_text()
        for text in fig.texts
    )


# ============================================================
# 11. 파일 저장 테스트
# ============================================================

def test_save_figure(
    tmp_path,
    monkeypatch,
):
    """PNG와 PDF 파일의 생성 및 기본 형식을 검증한다."""

    monkeypatch.setattr(
        viz,
        "PLOT_DIR",
        tmp_path,
    )

    fig, ax = viz.plt.subplots()

    ax.plot([1, 2], [3, 4])

    figure_number = fig.number

    viz.save_figure(
        fig,
        "test_figure",
    )

    png = tmp_path / "test_figure.png"
    pdf = tmp_path / "test_figure.pdf"

    assert png.exists()
    assert pdf.exists()

    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0

    with png.open("rb") as file:

        assert file.read(8) == (
            b"\x89PNG\r\n\x1a\n"
        )

    with pdf.open("rb") as file:

        assert file.read(5) == b"%PDF-"

    assert not viz.plt.fignum_exists(
        figure_number
    )


# ============================================================
# 12. 전체 시각화 파이프라인 검증
# ============================================================

def test_main_rejects_inconsistent_results(
    summary,
    tmp_path,
    monkeypatch,
):
    """요약 결과와 개별 CSV가 다르면 실행을 중단한다."""

    incorrect = summary.copy(deep=True)

    incorrect.loc[
        0, "significant_terms"
    ] += 1

    monkeypatch.setattr(
        viz,
        "load_summary",
        lambda: incorrect,
    )

    monkeypatch.setattr(
        viz,
        "PLOT_DIR",
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="요약 결과와 개별 CSV 불일치",
    ):
        viz.main()

    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.glob("*.pdf"))


def test_complete_visualization_pipeline(
    summary,
    tmp_path,
    monkeypatch,
    capsys,
):
    """전체 54개 그래프와 108개 출력 파일을 검증한다."""

    monkeypatch.setattr(
        viz,
        "PLOT_DIR",
        tmp_path,
    )

    # 테스트 시간을 줄이기 위해 해상도만 낮춘다.
    # 실제 분석의 300 DPI 설정은 변경하지 않는다.
    monkeypatch.setattr(
        viz,
        "DPI",
        100,
    )

    viz.main()

    expected_figures = {
        f"{dataset}_{group}_{library}_dotplot"
        for dataset, group, library
        in COMBINATIONS
    }

    expected_figures.update({
        f"{library}_dataset_comparison"
        for library in viz.GENE_SET_LIBRARIES
    })

    expected_figures.update({
        "enrichment_summary_comparison",
        "enrichment_summary_comparison_independent_y",
    })

    expected_files = {
        f"{name}.{extension}"
        for name in expected_figures
        for extension in ("png", "pdf")
    }

    actual_files = {
        path.name
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    assert len(expected_figures) == 54
    assert len(expected_files) == 108

    assert actual_files == expected_files

    for path in tmp_path.iterdir():

        assert path.stat().st_size > 0

        with path.open("rb") as file:

            if path.suffix == ".png":

                assert file.read(8) == (
                    b"\x89PNG\r\n\x1a\n"
                )

            elif path.suffix == ".pdf":

                assert file.read(5) == b"%PDF-"

    empty_count = int(
        (summary["significant_terms"] == 0).sum()
    )

    output = capsys.readouterr().out

    assert "Individual plots: 48" in output

    assert (
        f"Empty-result plots: {empty_count}"
        in output
    )

    assert "Comparison plots: 4" in output
    assert "Summary plots: 2" in output
    assert "Total figures: 54" in output
    assert "Total files saved: 108" in output
