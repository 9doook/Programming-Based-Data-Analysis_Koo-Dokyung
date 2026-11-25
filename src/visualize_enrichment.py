import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from src.enrichment_analysis import (
    PROJECT_ROOT,
    ENRICHMENT_DIR,
    ENRICHMENT_FDR,
    DATA_FILES,
    GENE_GROUPS,
    GENE_SET_LIBRARIES,
)


# ============================================================
# 1. 시각화 설정
# ============================================================

ENRICHMENT_PLOT_DIR = (
    PROJECT_ROOT / "results" / "enrichment_plots"
)

TOP_N = 10
DPI = 300
DOT_SIZE_SCALE = 18
CMAP = "viridis"

DATASET_LABELS = {
    "t2d": "Type 2 Diabetes",
    "combined_stress": "Glucose + Palmitate",
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


# ============================================================
# 2. 그래프 저장 함수
# ============================================================

def save_figure(fig, filename):
    ENRICHMENT_PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in ["png", "pdf"]:
        output_path = (
            ENRICHMENT_PLOT_DIR
            / f"{filename}.{extension}"
        )

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"[SAVED] {output_path}")

    plt.close(fig)


# ============================================================
# 3. 분석 결과 불러오기
# ============================================================

def load_enrichment_results(
    dataset,
    group,
    library,
):
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
        "fold_enrichment",
        "pvalue",
        "adjusted_pvalue",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"필수 열이 누락되었습니다: "
            f"{missing_columns}"
        )

    return df


# ============================================================
# 4. Dot Plot 데이터 전처리
# ============================================================

def prepare_dotplot_data(df, top_n=TOP_N):
    if top_n <= 0:
        raise ValueError(
            "top_n은 양의 정수여야 합니다."
        )

    plot_df = df.copy()

    numeric_columns = [
        "overlap_count",
        "gene_ratio",
        "fold_enrichment",
        "pvalue",
        "adjusted_pvalue",
    ]

    # 숫자형 데이터 변환
    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    # 비정상 데이터 검사
    valid = (
        np.isfinite(
            plot_df[numeric_columns]
        ).all(axis=1)
        & plot_df["adjusted_pvalue"].between(0, 1)
        & plot_df["pvalue"].between(0, 1)
        & plot_df["gene_ratio"].between(0, 1)
        & (plot_df["overlap_count"] > 0)
        & (plot_df["fold_enrichment"] > 0)
        & (
            plot_df["overlap_count"]
            % 1 == 0
        )
        & plot_df["term"].notna()
    )

    if not valid.all():
        invalid_count = int((~valid).sum())

        raise ValueError(
            f"시각화 데이터에 비정상적인 "
            f"행이 {invalid_count}개 있습니다."
        )

    # 유의한 결과만 선택
    plot_df = plot_df.loc[
        plot_df["adjusted_pvalue"]
        < ENRICHMENT_FDR
    ].copy()

    # 보정 p-value 기준 상위 항목 선택
    plot_df = plot_df.sort_values(
        by=[
            "adjusted_pvalue",
            "pvalue",
            "term",
        ],
        ascending=True,
    ).head(top_n).copy()

    # p-value가 0인 경우 로그 변환 오류 방지
    # 원래 보정 p-value는 변경하지 않음
    minimum_positive = np.finfo(float).tiny

    plot_df["neg_log10_padj"] = (
        -np.log10(
            plot_df["adjusted_pvalue"].clip(
                lower=minimum_positive
            )
        )
    )

    # 긴 경로 이름 자동 줄바꿈
    plot_df["display_term"] = (
        plot_df["term"]
        .astype(str)
        .apply(
            lambda term: textwrap.fill(
                term,
                width=42,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    )

    return plot_df.reset_index(drop=True)


# ============================================================
# 5. 색상 범위 설정
# ============================================================

def create_color_norm(dataframes):
    values = []

    for df in dataframes:
        if not df.empty:
            values.extend(
                df["neg_log10_padj"].tolist()
            )

    minimum = -np.log10(ENRICHMENT_FDR)

    if not values:
        maximum = minimum + 1
    else:
        maximum = max(values)

        if maximum <= minimum:
            maximum = minimum + 1

    return Normalize(
        vmin=minimum,
        vmax=maximum,
    )


# ============================================================
# 6. 점 크기 범례 생성
# ============================================================

def create_size_legend(dataframes):
    counts = []

    for df in dataframes:
        if not df.empty:
            counts.extend(
                df["overlap_count"].tolist()
            )

    if not counts:
        return []

    legend_counts = sorted(set([
        int(min(counts)),
        int(np.median(counts)),
        int(max(counts)),
    ]))

    handles = []

    for count in legend_counts:
        handle = Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=np.sqrt(
                count * DOT_SIZE_SCALE
            ),
            markerfacecolor="#999999",
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            label=str(count),
        )

        handles.append(handle)

    return handles


# ============================================================
# 7. Dot Plot 공통 그리기 함수
# ============================================================

def draw_dotplot(
    ax,
    plot_df,
    title,
    norm,
):
    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # 유의한 항목이 없는 경우
    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No significant terms",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        return None

    y_positions = np.arange(len(plot_df))

    scatter = ax.scatter(
        plot_df["gene_ratio"],
        y_positions,
        s=(
            plot_df["overlap_count"]
            * DOT_SIZE_SCALE
        ),
        c=plot_df["neg_log10_padj"],
        cmap=CMAP,
        norm=norm,
        edgecolors="#333333",
        linewidths=0.5,
        alpha=0.9,
        zorder=3,
    )

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        plot_df["display_term"],
        fontsize=9,
    )

    # 가장 유의한 항목을 위쪽에 표시
    ax.invert_yaxis()

    ax.set_ylim(
        len(plot_df) - 0.5,
        -0.5,
    )

    ax.set_xlabel(
        "Gene Ratio",
        fontsize=11,
    )

    ax.set_xlim(left=0)

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
# 8. 개별 분석 결과 Dot Plot
# ============================================================

def plot_individual_enrichment(
    dataset,
    group,
    library,
    df,
):
    plot_df = prepare_dotplot_data(df)

    norm = create_color_norm([plot_df])

    # 경로 이름의 줄바꿈 수를 반영한 그래프 높이
    line_count = sum(
        term.count("\n") + 1
        for term in plot_df["display_term"]
    )

    figure_height = max(
        4.5,
        line_count * 0.38 + 2.5,
    )

    fig, ax = plt.subplots(
        figsize=(12, figure_height),
    )

    fig.suptitle(
        DATASET_LABELS[dataset],
        fontsize=15,
        fontweight="bold",
    )

    title = (
        f"{LIBRARY_LABELS[library]} "
        f"| {GROUP_LABELS[group]}"
    )

    scatter = draw_dotplot(
        ax,
        plot_df,
        title,
        norm,
    )

    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            pad=0.04,
            shrink=0.8,
        )

        colorbar.set_label(
            "-log10(Adjusted P-value)"
        )

        size_handles = create_size_legend(
            [plot_df]
        )

        ax.legend(
            handles=size_handles,
            title="Overlap Count",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=len(size_handles),
            frameon=False,
        )

    fig.text(
        0.5,
        0.015,
        (
            f"Top {len(plot_df)} of {len(df)} "
            f"significant terms "
            f"(adjusted p-value < "
            f"{ENRICHMENT_FDR}).\n"
            "Gene ratio = overlap count / "
            "mapped query genes."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout(
        rect=[0, 0.12, 1, 0.93]
    )

    filename = (
        f"{dataset}_{group}_{library}_dotplot"
    )

    save_figure(fig, filename)


# ============================================================
# 9. 네 가지 데이터셋 통합 비교
# ============================================================

def plot_library_comparison(
    library,
    dataframes,
):
    prepared = {
        dataset: prepare_dotplot_data(
            dataframes[dataset]
        )
        for dataset in DATA_FILES
    }

    plot_dataframes = list(
        prepared.values()
    )

    # 모든 패널에서 동일한 색상 범위 사용
    norm = create_color_norm(
        plot_dataframes
    )

    # 모든 패널에서 동일한 X축 범위 사용
    ratio_values = []

    for df in plot_dataframes:
        if not df.empty:
            ratio_values.extend(
                df["gene_ratio"].tolist()
            )

    maximum_ratio = (
        max(ratio_values) * 1.15
        if ratio_values
        else 1.0
    )

    maximum_ratio = min(
        maximum_ratio,
        1.1,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20, 15),
    )

    fig.suptitle(
        (
            f"{LIBRARY_LABELS[library]} "
            "Enrichment Comparison"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    panel_labels = ["A", "B", "C", "D"]

    for ax, dataset, panel in zip(
        axes.flat,
        DATA_FILES,
        panel_labels,
    ):
        plot_df = prepared[dataset]

        title = (
            f"{panel}. "
            f"{DATASET_LABELS[dataset]}"
        )

        draw_dotplot(
            ax,
            plot_df,
            title,
            norm,
        )

        ax.set_xlim(
            0,
            maximum_ratio,
        )

    fig.subplots_adjust(
        left=0.25,
        right=0.87,
        top=0.91,
        bottom=0.15,
        wspace=1.25,
        hspace=0.5,
    )

    # 유의한 항목이 있는 경우에만 공통 범례 생성
    if ratio_values:
        colorbar_axis = fig.add_axes([
            0.91,
            0.25,
            0.015,
            0.5,
        ])

        scalar_mappable = ScalarMappable(
            norm=norm,
            cmap=CMAP,
        )

        scalar_mappable.set_array([])

        colorbar = fig.colorbar(
            scalar_mappable,
            cax=colorbar_axis,
        )

        colorbar.set_label(
            "-log10(Adjusted P-value)"
        )

        size_handles = create_size_legend(
            plot_dataframes
        )

        fig.legend(
            handles=size_handles,
            title="Overlap Count",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=len(size_handles),
            frameon=False,
        )

    fig.text(
        0.5,
        0.02,
        (
            f"All DEGs | Top {TOP_N} significant "
            "terms per dataset.\n"
            "Shared gene-ratio axis, color scale "
            "and marker-size scale.\n"
            "Gene ratios use dataset-specific "
            "mapped query sizes."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )

    save_figure(
        fig,
        f"{library}_dataset_comparison",
    )


# ============================================================
# 10. 유의 항목 수 비교 그래프
# ============================================================

def plot_enrichment_summary():
    summary_path = (
        ENRICHMENT_DIR
        / "enrichment_summary.csv"
    )

    summary = pd.read_csv(summary_path)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        sharey=True,
    )

    x = np.arange(len(DATA_FILES))

    bar_width = 0.25

    maximum_count = int(
        summary["significant_terms"].max()
    )

    for ax, library in zip(
        axes.flat,
        GENE_SET_LIBRARIES,
    ):
        library_df = summary.loc[
            summary["library"] == library
        ]

        for i, group in enumerate(
            GENE_GROUPS
        ):
            group_df = library_df.loc[
                library_df["gene_group"]
                == group
            ]

            counts = (
                group_df
                .set_index("dataset")
                .reindex(list(DATA_FILES))
                ["significant_terms"]
            )

            # 분석 조합 누락을 0으로 처리하지 않음
            if counts.isna().any():
                raise ValueError(
                    f"분석 조합이 누락되었습니다: "
                    f"{library}, {group}"
                )

            bars = ax.bar(
                x + (i - 1) * bar_width,
                counts,
                width=bar_width,
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group],
            )

            ax.bar_label(
                bars,
                padding=3,
                fontsize=8,
                fmt="%d",
            )

        ax.set_title(
            LIBRARY_LABELS[library],
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [
                "T2D",
                "Combined",
                "Palmitate",
                "Glucose",
            ],
            rotation=15,
        )

        ax.set_ylabel(
            "Significant Terms"
        )

        ax.set_ylim(
            0,
            max(1, maximum_count * 1.18),
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
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        "Functional Enrichment Summary",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        (
            f"Significance threshold: "
            f"adjusted p-value < "
            f"{ENRICHMENT_FDR}.\n"
            "Term counts do not directly measure "
            "biological effect size or pathway activity."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=[0, 0.07, 1, 0.88]
    )

    save_figure(
        fig,
        "enrichment_summary_comparison",
    )


# ============================================================
# 11. 전체 시각화 실행
# ============================================================

def main():
    ENRICHMENT_PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_deg_results = {
        library: {}
        for library in GENE_SET_LIBRARIES
    }

    individual_count = 0
    empty_count = 0

    # 48개 분석 조합 개별 시각화
    for dataset in DATA_FILES:
        for group in GENE_GROUPS:
            for library in GENE_SET_LIBRARIES:

                df = load_enrichment_results(
                    dataset,
                    group,
                    library,
                )

                print(
                    f"[PLOT] {dataset} | "
                    f"{group} | {library} | "
                    f"{len(df)} significant terms"
                )

                plot_individual_enrichment(
                    dataset,
                    group,
                    library,
                    df,
                )

                individual_count += 1

                if df.empty:
                    empty_count += 1

                # 통합 비교 그래프에는 전체 DEG 결과 사용
                if group == "all":
                    all_deg_results[library][
                        dataset
                    ] = df

    # 라이브러리별 통합 비교 그래프
    comparison_count = 0

    for library in GENE_SET_LIBRARIES:
        plot_library_comparison(
            library,
            all_deg_results[library],
        )

        comparison_count += 1

    # 전체 풍부도 분석 요약 그래프
    plot_enrichment_summary()

    total_figures = (
        individual_count
        + comparison_count
        + 1
    )

    print("\n" + "=" * 60)

    print(
        "[COMPLETE] "
        "Enrichment visualization finished."
    )

    print(
        f"Individual plots: "
        f"{individual_count}"
    )

    print(
        f"Empty-result plots: "
        f"{empty_count}"
    )

    print(
        f"Comparison plots: "
        f"{comparison_count}"
    )

    print("Summary plots: 1")

    print(
        f"Total figures: {total_figures}"
    )

    print(
        f"Total files saved: "
        f"{total_figures * 2}"
    )

    print(
        f"Output directory: "
        f"{ENRICHMENT_PLOT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
