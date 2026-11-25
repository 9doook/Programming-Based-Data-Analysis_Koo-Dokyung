
"""
GSE159984 차등 발현 유전자(DEG) 시각화

1. 조건별 상향 및 하향 조절 유전자 수 비교
2. 개별 데이터셋의 Volcano plot 생성
3. 복합 스트레스 확대 Volcano plot 생성
4. 공통 축을 적용한 2×2 통합 Volcano plot 생성
5. 모든 그래프를 PNG 및 PDF 형식으로 저장
"""

import numpy as np
import pandas as pd
import matplotlib

# 그래픽 화면 없이도 실행할 수 있도록 설정
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config.analysis_config import (
    DATA_FILES,
    DEG_DIR,
    PLOT_DIR,
    PADJ_THRESHOLD,
    LOG2FC_THRESHOLD,
)

from src.analyze_deg import (
    load_deg_data,
    filter_deg,
)


# ========================================
# 1. 시각화 공통 설정
# ========================================

DATASET_LABELS = {
    "t2d": "Type 2 Diabetes",
    "combined_stress": "Glucose + Palmitate",
    "palmitate": "Palmitate",
    "glucose": "Glucose",
}

# 그래프 색상
UP_COLOR = "#D65B5B"
DOWN_COLOR = "#5286C5"
NS_COLOR = "#B0B0B0"

# 확대 그래프의 표시 범위
ZOOM_X_LIMIT = 3
ZOOM_Y_LIMIT = 40

# 그래프 공통 스타일
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


# ========================================
# 2. 그래프 저장 함수
# ========================================

def save_figure(fig, filename):
    """그래프를 PNG 및 PDF 형식으로 저장한다."""

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in ["png", "pdf"]:

        output_path = PLOT_DIR / (
            f"{filename}.{extension}"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"[SAVED] {output_path.name}")

    plt.close(fig)


# ========================================
# 3. DEG 개수 비교 막대그래프
# ========================================

def plot_deg_counts():
    """조건별 상향 및 하향 조절 DEG 수를 비교한다."""

    summary_path = DEG_DIR / "deg_summary.csv"

    summary = pd.read_csv(summary_path)

    labels = [
        DATASET_LABELS[name]
        for name in summary["dataset"]
    ]

    up_counts = summary["upregulated"].to_numpy()

    down_counts = (
        summary["downregulated"].to_numpy()
    )

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    # 상향 조절 유전자
    up_bars = ax.bar(
        x - width / 2,
        up_counts,
        width,
        label="Upregulated",
        color=UP_COLOR,
    )

    # 하향 조절 유전자
    down_bars = ax.bar(
        x + width / 2,
        down_counts,
        width,
        label="Downregulated",
        color=DOWN_COLOR,
    )

    # 막대 위에 실제 유전자 수 표시
    ax.bar_label(
        up_bars,
        padding=4,
    )

    ax.bar_label(
        down_bars,
        padding=4,
    )

    ax.set_title(
        "Differentially Expressed Genes Across Conditions",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    ax.set_ylabel("Number of DEGs")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 범례를 그래프 외부에 배치
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )

    ax.set_ylim(
        0,
        max(
            up_counts.max(),
            down_counts.max(),
            1,
        ) * 1.2,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        (
            f"Criteria: padj < {PADJ_THRESHOLD}, "
            f"|log2FoldChange| > {LOG2FC_THRESHOLD}"
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    save_figure(
        fig,
        "deg_count_comparison",
    )


# ========================================
# 4. Volcano plot 데이터 준비
# ========================================

def prepare_volcano_data(file_path):
    """Volcano plot에 사용할 전체 데이터를 준비한다."""

    df = load_deg_data(
        file_path
    ).copy()

    # 유효한 수치형 데이터 확인
    valid = (
        np.isfinite(df["log2FoldChange"])
        & np.isfinite(df["padj"])
        & df["padj"].between(0, 1)
    )

    excluded = int(
        (~valid).sum()
    )

    df = df.loc[valid].copy()

    if df.empty:
        raise ValueError(
            f"시각화 가능한 데이터가 없습니다: {file_path}"
        )

    # padj가 0인 경우 로그 변환 오류 방지
    positive_padj = df.loc[
        df["padj"] > 0,
        "padj",
    ]

    if positive_padj.empty:

        minimum_padj = np.finfo(
            float
        ).tiny

    else:

        minimum_padj = max(
            positive_padj.min() / 10,
            np.nextafter(0.0, 1.0),
        )

    # 원래 padj는 변경하지 않고
    # 시각화에 사용할 값만 조정
    plot_padj = df["padj"].mask(
        df["padj"] == 0,
        minimum_padj,
    )

    df["neg_log10_padj"] = (
        -np.log10(plot_padj)
    )

    # 기존 DEG 선별 함수 재사용
    _, upregulated, downregulated = filter_deg(df)

    # 기본 상태: 선별 기준 미충족
    df["regulation"] = "Not selected"

    df.loc[
        upregulated.index,
        "regulation",
    ] = "Upregulated"

    df.loc[
        downregulated.index,
        "regulation",
    ] = "Downregulated"

    return df, excluded


# ========================================
# 5. Volcano plot 공통 그리기 함수
# ========================================

def draw_volcano(
    ax,
    df,
    title,
    show_legend=True,
):
    """지정된 그래프 영역에 Volcano plot을 그린다."""

    categories = [
        ("Not selected", NS_COLOR, 9, 0.4),
        ("Downregulated", DOWN_COLOR, 16, 0.75),
        ("Upregulated", UP_COLOR, 16, 0.75),
    ]

    # 발현 변화 방향별 산점도 생성
    for category, color, size, alpha in categories:

        subset = df.loc[
            df["regulation"] == category
        ]

        ax.scatter(
            subset["log2FoldChange"],
            subset["neg_log10_padj"],
            color=color,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=f"{category} ({len(subset):,})",
        )

    # 하향 조절 기준선
    ax.axvline(
        -LOG2FC_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    # 상향 조절 기준선
    ax.axvline(
        LOG2FC_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    # 통계적 유의성 기준선
    ax.axhline(
        -np.log10(PADJ_THRESHOLD),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "log2 Fold Change"
    )

    ax.set_ylabel(
        "-log10 Adjusted P-value"
    )

    # 범례를 그래프 외부에 배치
    if show_legend:

        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=10,
        )


# ========================================
# 6. 개별 Volcano plot 생성
# ========================================

def plot_volcano(
    dataset_name,
    df,
    excluded,
):
    """전체 범위의 개별 Volcano plot을 생성한다."""

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    draw_volcano(
        ax,
        df,
        f"Volcano Plot: {DATASET_LABELS[dataset_name]}",
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"{dataset_name}_volcano",
    )

    up_count = (
        df["regulation"] == "Upregulated"
    ).sum()

    down_count = (
        df["regulation"] == "Downregulated"
    ).sum()

    print(
        f"시각화 대상: {len(df):,}개 행"
    )

    print(
        f"제외된 데이터: {excluded:,}개 행"
    )

    print(
        f"상향 조절: {up_count:,}개"
    )

    print(
        f"하향 조절: {down_count:,}개"
    )


# ========================================
# 7. 복합 스트레스 확대 Volcano plot
# ========================================

def plot_zoomed_volcano(df):
    """복합 스트레스의 주요 유전자 분포를 확대한다."""

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    draw_volcano(
        ax,
        df,
        "Volcano Plot: Glucose + Palmitate (Zoomed View)",
    )

    # 데이터는 삭제하지 않고 표시 범위만 제한
    ax.set_xlim(
        -ZOOM_X_LIMIT,
        ZOOM_X_LIMIT,
    )

    ax.set_ylim(
        0,
        ZOOM_Y_LIMIT,
    )

    # 확대 범위 밖에 위치한 데이터 수 계산
    outside = (
        (df["log2FoldChange"] < -ZOOM_X_LIMIT)
        | (df["log2FoldChange"] > ZOOM_X_LIMIT)
        | (df["neg_log10_padj"] > ZOOM_Y_LIMIT)
    )

    outside_count = int(
        outside.sum()
    )

    # 확대 범위 및 범례 기준 명시
    fig.text(
        0.5,
        0.015,
        (
            f"Zoomed axes: |log2FC| <= {ZOOM_X_LIMIT}, "
            f"-log10(padj) <= {ZOOM_Y_LIMIT}\n"
            f"Points outside view: {outside_count:,}. "
            "Legend counts refer to the full dataset."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    save_figure(
        fig,
        "combined_stress_volcano_zoomed",
    )

    print(
        f"확대 표시 범위 밖의 데이터: "
        f"{outside_count:,}개 행"
    )

    print(
        "원본 데이터는 삭제하지 않았습니다."
    )


# ========================================
# 8. 4개 조건 통합 Volcano plot
# ========================================

def plot_combined_volcano(dataframes):
    """공통 축 범위의 2×2 통합 그래프를 생성한다."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11),
        sharex=True,
        sharey=True,
    )

    panel_labels = [
        "A", "B", "C", "D"
    ]

    for ax, dataset_name, panel in zip(
        axes.flat,
        DATA_FILES,
        panel_labels,
    ):

        df = dataframes[dataset_name]

        draw_volcano(
            ax,
            df,
            (
                f"{panel}. "
                f"{DATASET_LABELS[dataset_name]}"
            ),
            show_legend=False,
        )

        # 모든 그래프에 동일한 축 범위 적용
        ax.set_xlim(
            -ZOOM_X_LIMIT,
            ZOOM_X_LIMIT,
        )

        ax.set_ylim(
            0,
            ZOOM_Y_LIMIT,
        )

        up_count = int(
            (df["regulation"] == "Upregulated").sum()
        )

        down_count = int(
            (df["regulation"] == "Downregulated").sum()
        )

        outside_count = int((
            (df["log2FoldChange"] < -ZOOM_X_LIMIT)
            | (df["log2FoldChange"] > ZOOM_X_LIMIT)
            | (df["neg_log10_padj"] > ZOOM_Y_LIMIT)
        ).sum())

        # 각 패널에 전체 DEG 수 표시
        ax.text(
            0.02,
            0.96,
            f"Up: {up_count:,}",
            transform=ax.transAxes,
            color=UP_COLOR,
            fontsize=10,
            va="top",
        )

        ax.text(
            0.02,
            0.89,
            f"Down: {down_count:,}",
            transform=ax.transAxes,
            color=DOWN_COLOR,
            fontsize=10,
            va="top",
        )

        ax.text(
            0.02,
            0.82,
            f"Outside view: {outside_count:,}",
            transform=ax.transAxes,
            color="dimgray",
            fontsize=9,
            va="top",
        )

    # 통합 그래프 공통 범례
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=NS_COLOR,
            markeredgecolor="none",
            markersize=7,
            label="Not selected",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=DOWN_COLOR,
            markeredgecolor="none",
            markersize=7,
            label="Downregulated",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=UP_COLOR,
            markeredgecolor="none",
            markersize=7,
            label="Upregulated",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        "Comparative Volcano Plots Across Conditions",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.015,
        (
            f"Criteria: padj < {PADJ_THRESHOLD}, "
            f"|log2FoldChange| > {LOG2FC_THRESHOLD}\n"
            f"Shared zoomed axes: |log2FC| <= {ZOOM_X_LIMIT}, "
            f"-log10(padj) <= {ZOOM_Y_LIMIT}. "
            "Counts refer to full datasets."
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(
        rect=[0, 0.06, 1, 0.90]
    )

    save_figure(
        fig,
        "combined_volcano_comparison",
    )


# ========================================
# 9. 전체 시각화 실행
# ========================================

def main():
    """전체 DEG 시각화 작업을 실행한다."""

    print(
        "[START] DEG visualization"
    )

    # DEG 개수 비교 막대그래프
    print(
        "\n[PLOT] DEG count comparison"
    )

    plot_deg_counts()

    # 데이터셋별 전처리 결과 보관
    dataframes = {}

    # 개별 Volcano plot 생성
    for dataset_name, file_path in DATA_FILES.items():

        print(
            f"\n[PLOT] {dataset_name} volcano"
        )

        df, excluded = (
            prepare_volcano_data(file_path)
        )

        dataframes[dataset_name] = df

        plot_volcano(
            dataset_name,
            df,
            excluded,
        )

    # 복합 스트레스 확대 그래프
    print(
        "\n[PLOT] Combined stress zoomed volcano"
    )

    plot_zoomed_volcano(
        dataframes["combined_stress"]
    )

    # 통합 Volcano plot
    print(
        "\n[PLOT] Comparative volcano plots"
    )

    plot_combined_volcano(
        dataframes
    )

    print(
        "\n[COMPLETE] DEG visualization finished."
    )

    print(
        f"Results saved in: {PLOT_DIR}"
    )


if __name__ == "__main__":
    main()
