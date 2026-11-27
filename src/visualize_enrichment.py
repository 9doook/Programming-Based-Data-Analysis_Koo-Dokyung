
"""
기능적 풍부도 분석 결과 시각화

- 전체, 상향, 하향 조절 DEG의 GO 및 KEGG Dot Plot
- 라이브러리별 네 가지 분석 조건 통합 비교
- 공통 Y축 및 독립 Y축 요약 그래프
- 유의한 항목이 없는 분석 결과 처리
- PNG 및 PDF 형식 저장
"""

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from src.enrichment_analysis import (
    ENRICHMENT_DIR,
    ENRICHMENT_FDR,
    GENE_SET_LIBRARIES,
    GENE_GROUPS,
)


# ============================================================
# 1. 기본 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLOT_DIR = PROJECT_ROOT / "results" / "enrichment_plots"

TOP_N = 10
DPI = 300

DATASET_LABELS = {
    "t2d": "Type 2 Diabetes",
    "combined_stress": "Glucose + Palmitate",
    "palmitate": "Palmitate",
    "glucose": "Glucose",
}

DATASET_SHORT_LABELS = {
    "t2d": "T2D",
    "combined_stress": "Combined",
    "palmitate": "Palmitate",
    "glucose": "Glucose",
}

LIBRARY_LABELS = {
    "go_bp": "GO Biological Process",
    "go_mf": "GO Molecular Function",
    "go_cc": "GO Cellular Component",
    "kegg": "KEGG Pathway",
}

GROUP_LABELS = {
    "all": "All DEGs",
    "upregulated": "Upregulated DEGs",
    "downregulated": "Downregulated DEGs",
}

GROUP_COLORS = {
    "all": "#808080",
    "upregulated": "#D65B5B",
    "downregulated": "#5286C5",
}

CMAP = "viridis"
SIZE_FACTOR = 15

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
})


# ============================================================
# 2. 그래프 저장
# ============================================================

def save_figure(fig, filename):
    """그래프를 PNG 및 PDF 형식으로 저장한다."""

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):

        output_path = PLOT_DIR / f"{filename}.{extension}"

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
        )

        print(f"[SAVED] {output_path}")

    plt.close(fig)


# ============================================================
# 3. 분석 결과 불러오기
# ============================================================

def load_enrichment_results(dataset, group, library):
    """기존 풍부도 분석 CSV를 불러온다."""

    file_path = (
        ENRICHMENT_DIR
        / dataset
        / f"{group}_{library}_significant.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"분석 결과 파일이 없습니다: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "term",
        "overlap_count",
        "gene_ratio",
        "adjusted_pvalue",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"필수 열이 없습니다: {missing_columns}"
        )

    if df.empty:
        return df

    numeric_columns = [
        "overlap_count",
        "gene_ratio",
        "adjusted_pvalue",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    values = df[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"유효하지 않은 통계량이 있습니다: {file_path}"
        )

    if not df["adjusted_pvalue"].between(0, 1).all():
        raise ValueError("보정 p-value 범위 오류")

    if not df["gene_ratio"].between(0, 1).all():
        raise ValueError("Gene ratio 범위 오류")

    counts = df["overlap_count"]

    if ((counts <= 0) | (counts % 1 != 0)).any():
        raise ValueError("중첩 유전자 수 오류")

    df = df.loc[
        df["adjusted_pvalue"] < ENRICHMENT_FDR
    ].copy()

    sort_columns = ["adjusted_pvalue"]

    if "pvalue" in df.columns:
        sort_columns.append("pvalue")

    sort_columns.append("term")

    return df.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# 4. Dot Plot 데이터 준비
# ============================================================

def prepare_dotplot_data(df, top_n=TOP_N):
    """상위 유의 항목과 시각화용 통계량을 준비한다."""

    plot_df = df.head(top_n).copy()

    if plot_df.empty:
        plot_df["neg_log10_padj"] = pd.Series(dtype=float)
        return plot_df

    positive_values = df.loc[
        df["adjusted_pvalue"] > 0,
        "adjusted_pvalue",
    ]

    if positive_values.empty:
        minimum_pvalue = np.finfo(float).tiny
    else:
        minimum_pvalue = max(
            positive_values.min() / 10,
            np.nextafter(0.0, 1.0),
        )

    plot_pvalues = plot_df["adjusted_pvalue"].mask(
        plot_df["adjusted_pvalue"] == 0,
        minimum_pvalue,
    )

    plot_df["neg_log10_padj"] = -np.log10(
        plot_pvalues
    )

    return plot_df


# ============================================================
# 5. 색상 척도
# ============================================================

def create_color_norm(plot_frames):
    """여러 그래프에 적용할 공통 색상 범위를 계산한다."""

    minimum = -np.log10(ENRICHMENT_FDR)

    valid_frames = [
        df for df in plot_frames
        if not df.empty
    ]

    if valid_frames:
        maximum = max(
            df["neg_log10_padj"].max()
            for df in valid_frames
        )
    else:
        maximum = minimum + 1

    maximum = max(maximum, minimum + 0.01)

    return Normalize(
        vmin=minimum,
        vmax=maximum,
    )


# ============================================================
# 6. 점 크기 범례
# ============================================================

def create_size_legend(plot_frames):
    """중첩 유전자 수에 대응하는 점 크기 범례를 만든다."""

    valid_frames = [
        df for df in plot_frames
        if not df.empty
    ]

    if not valid_frames:
        return []

    counts = pd.concat(
        [
            df["overlap_count"]
            for df in valid_frames
        ],
        ignore_index=True,
    )

    legend_counts = sorted({
        int(counts.min()),
        int(counts.median()),
        int(counts.max()),
    })

    handles = []

    for count in legend_counts:

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#A0A0A0",
                markeredgecolor="#555555",
                markeredgewidth=0.6,
                markersize=np.sqrt(
                    count * SIZE_FACTOR
                ),
                label=str(count),
            )
        )

    return handles


# ============================================================
# 7. Dot Plot 그리기
# ============================================================

def draw_dotplot(
    ax,
    plot_df,
    color_norm,
    x_limit=None,
    wrap_width=43,
    label_size=10,
):
    """개별 패널에 Dot Plot을 그린다."""

    # 유의한 항목이 없는 경우
    if plot_df.empty:

        ax.set_axis_off()

        ax.text(
            0.5,
            0.5,
            "No significant terms",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="#666666",
        )

        return None

    y_positions = np.arange(len(plot_df))

    labels = [
        fill(str(term), width=wrap_width)
        for term in plot_df["term"]
    ]

    scatter = ax.scatter(
        plot_df["gene_ratio"],
        y_positions,
        s=plot_df["overlap_count"] * SIZE_FACTOR,
        c=plot_df["neg_log10_padj"],
        cmap=CMAP,
        norm=color_norm,
        edgecolors="#555555",
        linewidths=0.5,
        alpha=0.95,
        clip_on=False,
        zorder=3,
    )

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        labels,
        fontsize=label_size,
    )

    # 최상단과 최하단 점의 잘림 방지
    ax.set_ylim(
        len(plot_df) - 0.3,
        -0.7,
    )

    if x_limit is None:
        x_limit = max(
            plot_df["gene_ratio"].max() * 1.15,
            0.01,
        )

    ax.set_xlim(0, x_limit)

    ax.set_xlabel(
        "Gene Ratio",
        labelpad=8,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return scatter


# ============================================================
# 8. 개별 풍부도 분석 그래프
# ============================================================

def plot_individual_enrichment(
    dataset,
    group,
    library,
    tested_terms=None,
):
    """각 분석 조합에 대한 개별 그래프를 생성한다."""

    df = load_enrichment_results(
        dataset,
        group,
        library,
    )

    plot_df = prepare_dotplot_data(df)

    total_terms = len(df)
    displayed_terms = len(plot_df)

    filename = (
        f"{dataset}_{group}_{library}_dotplot"
    )

    title = DATASET_LABELS[dataset]

    subtitle = (
        f"{LIBRARY_LABELS[library]}"
        f" | {GROUP_LABELS[group]}"
    )

    # 유의한 결과가 없는 경우
    if plot_df.empty:

        fig, ax = plt.subplots(
            figsize=(8, 3.1),
        )

        fig.subplots_adjust(
            left=0.08,
            right=0.92,
            top=0.68,
            bottom=0.25,
        )

        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.96,
        )

        ax.set_title(
            subtitle,
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        draw_dotplot(
            ax,
            plot_df,
            create_color_norm([]),
        )

        if tested_terms is None:
            detail = "No terms met the significance threshold."
        else:
            detail = (
                f"Tested terms: {tested_terms:,}"
                " | Significant terms: 0"
            )

        fig.text(
            0.5,
            0.09,
            f"{detail}\n"
            f"Adjusted p-value < {ENRICHMENT_FDR}",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )

        save_figure(fig, filename)

        return True

    # 표시 항목 수에 따른 그래프 높이 조절
    figure_height = max(
        4.7,
        3.1 + displayed_terms * 0.52,
    )

    figure_width = (
        11.5 if displayed_terms > 5 else 10.5
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
    )

    bottom = 1.65 / figure_height
    top = 1 - 0.95 / figure_height

    fig.subplots_adjust(
        left=0.35,
        right=0.87,
        top=top,
        bottom=bottom,
    )

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        y=1 - 0.12 / figure_height,
    )

    ax.set_title(
        subtitle,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    color_norm = create_color_norm([plot_df])

    scatter = draw_dotplot(
        ax,
        plot_df,
        color_norm,
        wrap_width=43,
        label_size=10,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.045,
        pad=0.04,
    )

    colorbar.set_label(
        "-log10(Adjusted P-value)",
        fontsize=10,
    )

    handles = create_size_legend([plot_df])

    fig.legend(
        handles=handles,
        title="Overlap Count",
        loc="lower center",
        bbox_to_anchor=(
            0.60,
            0.58 / figure_height,
        ),
        ncol=len(handles),
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    fig.text(
        0.5,
        0.13 / figure_height,
        (
            f"Showing {displayed_terms} of "
            f"{total_terms} significant terms "
            f"(adjusted p-value < {ENRICHMENT_FDR}).\n"
            "Gene ratio = overlap count / "
            "mapped query genes."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    save_figure(fig, filename)

    return False


# ============================================================
# 9. 라이브러리별 통합 비교 그래프
# ============================================================

def plot_library_comparison(library):
    """전체 DEG 결과를 네 가지 조건에서 비교한다."""

    full_results = {}
    plot_results = {}

    for dataset in DATASET_LABELS:

        df = load_enrichment_results(
            dataset,
            "all",
            library,
        )

        full_results[dataset] = df

        plot_results[dataset] = (
            prepare_dotplot_data(df)
        )

    frames = list(plot_results.values())

    valid_frames = [
        df for df in frames
        if not df.empty
    ]

    color_norm = create_color_norm(frames)

    if valid_frames:

        x_limit = max(
            df["gene_ratio"].max()
            for df in valid_frames
        ) * 1.15

    else:
        x_limit = 0.1

    # 긴 항목 이름을 위한 공간 확보
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(19, 12.5),
    )

    fig.subplots_adjust(
        left=0.22,
        right=0.91,
        top=0.87,
        bottom=0.16,
        wspace=0.90,
        hspace=0.40,
    )

    fig.suptitle(
        f"{LIBRARY_LABELS[library]} Enrichment Comparison",
        fontsize=21,
        fontweight="bold",
        y=0.98,
    )

    for ax, dataset, panel in zip(
        axes.flat,
        DATASET_LABELS,
        ("A", "B", "C", "D"),
    ):

        plot_df = plot_results[dataset]

        total_terms = len(
            full_results[dataset]
        )

        displayed_terms = len(plot_df)

        draw_dotplot(
            ax,
            plot_df,
            color_norm,
            x_limit=x_limit,
            wrap_width=36,
            label_size=10,
        )

        ax.set_title(
            (
                f"{panel}. {DATASET_LABELS[dataset]}\n"
                f"{displayed_terms} of "
                f"{total_terms} significant terms"
            ),
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

    # 모든 패널에 동일한 색상 척도 적용
    if valid_frames:

        colorbar_ax = fig.add_axes(
            [0.935, 0.25, 0.014, 0.53]
        )

        mappable = ScalarMappable(
            norm=color_norm,
            cmap=CMAP,
        )

        mappable.set_array([])

        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_ax,
        )

        colorbar.set_label(
            "-log10(Adjusted P-value)",
            fontsize=11,
        )

        handles = create_size_legend(frames)

        fig.legend(
            handles=handles,
            title="Overlap Count",
            loc="lower center",
            bbox_to_anchor=(0.53, 0.065),
            ncol=len(handles),
            frameon=False,
            columnspacing=2.0,
        )

    fig.text(
        0.5,
        0.018,
        (
            "All DEGs | Up to 10 significant terms "
            "per dataset.\n"
            "Shared gene-ratio axis, color scale "
            "and marker-size scale.\n"
            "Gene ratios use dataset-specific "
            "mapped query sizes."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    save_figure(
        fig,
        f"{library}_dataset_comparison",
    )


# ============================================================
# 10. 요약 데이터 검증
# ============================================================

def load_summary():
    """48개 분석 조합의 요약 결과를 검증한다."""

    summary_path = (
        ENRICHMENT_DIR
        / "enrichment_summary.csv"
    )

    summary = pd.read_csv(summary_path)

    keys = [
        "dataset",
        "gene_group",
        "library",
    ]

    required_columns = set(keys) | {
        "tested_terms",
        "significant_terms",
    }

    if not required_columns.issubset(
        summary.columns
    ):
        raise ValueError("요약 파일의 필수 열 누락")

    if summary.duplicated(keys).any():
        raise ValueError("중복 분석 조합 발견")

    expected = {
        (dataset, group, library)
        for dataset in DATASET_LABELS
        for group in GENE_GROUPS
        for library in GENE_SET_LIBRARIES
    }

    actual = set(
        summary[keys].itertuples(
            index=False,
            name=None,
        )
    )

    if actual != expected:
        raise ValueError("분석 조합 누락 또는 불일치")

    for column in (
        "tested_terms",
        "significant_terms",
    ):

        summary[column] = pd.to_numeric(
            summary[column],
            errors="raise",
        )

        values = summary[column]

        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values % 1 != 0).any()
        ):
            raise ValueError(
                f"유효하지 않은 항목 수: {column}"
            )

    if (
        summary["significant_terms"]
        > summary["tested_terms"]
    ).any():
        raise ValueError("유의 항목 수가 검정 수를 초과함")

    return summary


# ============================================================
# 11. 풍부도 분석 요약 그래프
# ============================================================

def plot_enrichment_summary(
    independent_y=False,
    summary=None,
):
    """공통 또는 독립 Y축 요약 그래프를 생성한다."""

    if summary is None:
        summary = load_summary()

    datasets = list(DATASET_LABELS)
    groups = list(GROUP_LABELS)

    x_positions = np.arange(len(datasets))
    bar_width = 0.24

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharey=not independent_y,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.82,
        bottom=0.14,
        wspace=0.23,
        hspace=0.35,
    )

    if independent_y:
        title = (
            "Functional Enrichment Summary"
            " | Independent Y-axes"
        )
    else:
        title = "Functional Enrichment Summary"

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    global_max = int(
        summary["significant_terms"].max()
    )

    global_y_limit = max(
        global_max * 1.20,
        5,
    )

    for ax, library in zip(
        axes.flat,
        LIBRARY_LABELS,
    ):

        library_df = summary.loc[
            summary["library"] == library
        ]

        local_max = int(
            library_df["significant_terms"].max()
        )

        if independent_y:
            y_limit = max(
                local_max * 1.25,
                5,
            )
        else:
            y_limit = global_y_limit

        for group_index, group in enumerate(groups):

            group_df = library_df.loc[
                library_df["gene_group"] == group
            ]

            counts = (
                group_df
                .set_index("dataset")
                .loc[datasets, "significant_terms"]
                .to_numpy(dtype=int)
            )

            offsets = (
                x_positions
                + (group_index - 1) * bar_width
            )

            bars = ax.bar(
                offsets,
                counts,
                width=bar_width,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
                zorder=3,
            )

            # 0개인 경우에도 숫자를 표시
            for bar, count in zip(bars, counts):

                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    count + y_limit * 0.018,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_title(
            LIBRARY_LABELS[library],
            fontweight="bold",
            fontsize=13,
            pad=10,
        )

        ax.set_xticks(x_positions)

        ax.set_xticklabels(
            [
                DATASET_SHORT_LABELS[d]
                for d in datasets
            ],
            rotation=15,
            ha="right",
        )

        ax.set_ylabel("Significant Terms")

        ax.set_ylim(0, y_limit)

        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(
                integer=True,
            )
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = (
        axes.flat[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )

    if independent_y:

        axis_note = (
            "Y-axis ranges differ between libraries."
        )

        filename = (
            "enrichment_summary_comparison_independent_y"
        )

    else:

        axis_note = (
            "All panels share the same Y-axis range."
        )

        filename = "enrichment_summary_comparison"

    fig.text(
        0.5,
        0.025,
        (
            f"Significance threshold: "
            f"adjusted p-value < {ENRICHMENT_FDR}.\n"
            f"{axis_note}\n"
            "Term counts do not directly measure "
            "biological effect size or pathway activity."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    save_figure(fig, filename)


# ============================================================
# 12. 전체 시각화 실행
# ============================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = load_summary()

    # 요약 파일과 개별 분석 결과의 개수 일치 확인
    summary_lookup = summary.set_index([
        "dataset",
        "gene_group",
        "library",
    ])

    for dataset in DATASET_LABELS:

        for group in GENE_GROUPS:

            for library in GENE_SET_LIBRARIES:

                df = load_enrichment_results(
                    dataset,
                    group,
                    library,
                )

                expected_count = int(
                    summary_lookup.loc[
                        (dataset, group, library),
                        "significant_terms",
                    ]
                )

                if len(df) != expected_count:
                    raise ValueError(
                        "요약 결과와 개별 CSV 불일치: "
                        f"{dataset}, {group}, {library}"
                    )

    individual_count = 0
    empty_count = 0
    comparison_count = 0
    summary_count = 0

    # 48개 개별 그래프
    for dataset in DATASET_LABELS:

        for group in GENE_GROUPS:

            for library in GENE_SET_LIBRARIES:

                tested_terms = int(
                    summary_lookup.loc[
                        (dataset, group, library),
                        "tested_terms",
                    ]
                )

                is_empty = plot_individual_enrichment(
                    dataset,
                    group,
                    library,
                    tested_terms=tested_terms,
                )

                individual_count += 1

                if is_empty:
                    empty_count += 1

    # 4개 라이브러리 통합 비교 그래프
    for library in GENE_SET_LIBRARIES:

        plot_library_comparison(library)

        comparison_count += 1

    # 공통 Y축 요약 그래프
    plot_enrichment_summary(
        independent_y=False,
        summary=summary,
    )

    summary_count += 1

    # 독립 Y축 요약 그래프
    plot_enrichment_summary(
        independent_y=True,
        summary=summary,
    )

    summary_count += 1

    total_figures = (
        individual_count
        + comparison_count
        + summary_count
    )

    print("\n" + "=" * 60)

    print(
        "[COMPLETE] Enrichment visualization finished."
    )

    print(f"Individual plots: {individual_count}")
    print(f"Empty-result plots: {empty_count}")
    print(f"Comparison plots: {comparison_count}")
    print(f"Summary plots: {summary_count}")
    print(f"Total figures: {total_figures}")
    print(f"Total files saved: {total_figures * 2}")

    print("=" * 60)


if __name__ == "__main__":
    main()
