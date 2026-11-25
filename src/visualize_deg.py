"""
GSE159984 차등 발현 유전자(DEG) 시각화

1. 데이터셋별 상향 및 하향 조절 유전자 수 비교
2. 데이터셋별 Volcano plot 생성
3. 시각화 결과를 PNG 및 PDF 형식으로 저장
"""

import numpy as np
import pandas as pd
import matplotlib

# 그래픽 화면 없이도 실행할 수 있도록 설정
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.analysis_config import (
    DATA_FILES,
    DEG_DIR,
    PLOT_DIR,
    PADJ_THRESHOLD,
    LOG2FC_THRESHOLD,
)

from src.analyze_deg import load_deg_data


# 데이터셋별 그래프 표시 이름
DATASET_LABELS = {
    "t2d": "Type 2 Diabetes",
    "combined_stress": "Glucose + Palmitate",
    "palmitate": "Palmitate",
    "glucose": "Glucose",
}


# 그래프 색상 설정
UP_COLOR = "#D65B5B"
DOWN_COLOR = "#5286C5"
NS_COLOR = "#B0B0B0"


# 그래프 공통 스타일 설정
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})


def save_figure(fig, filename):
    """그래프를 PNG 및 PDF 형식으로 저장한다."""

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ["png", "pdf"]:

        output_path = PLOT_DIR / f"{filename}.{extension}"

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"[SAVED] {output_path.name}")

    plt.close(fig)


def plot_deg_counts():
    """데이터셋별 상향 및 하향 조절 DEG 수를 비교한다."""

    summary_path = DEG_DIR / "deg_summary.csv"

    summary = pd.read_csv(summary_path)

    # 데이터셋 표시 이름 변환
    labels = [
        DATASET_LABELS[name]
        for name in summary["dataset"]
    ]

    up_counts = summary["upregulated"].to_numpy()
    down_counts = summary["downregulated"].to_numpy()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # 상향 조절 유전자 막대그래프
    up_bars = ax.bar(
        x - width / 2,
        up_counts,
        width,
        label="Upregulated",
        color=UP_COLOR,
    )

    # 하향 조절 유전자 막대그래프
    down_bars = ax.bar(
        x + width / 2,
        down_counts,
        width,
        label="Downregulated",
        color=DOWN_COLOR,
    )

    # 막대 위에 실제 유전자 수 표시
    ax.bar_label(up_bars, padding=4)
    ax.bar_label(down_bars, padding=4)

    ax.set_title(
        "Differentially Expressed Genes Across Conditions",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    ax.set_ylabel("Number of DEGs")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend(frameon=False)

    ax.set_ylim(
        0,
        max(up_counts.max(), down_counts.max(), 1) * 1.2,
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

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    save_figure(fig, "deg_count_comparison")


def plot_volcano(dataset_name, file_path):
    """개별 데이터셋의 Volcano plot을 생성한다."""

    # 선별된 DEG가 아닌 전체 유전자 발현 결과 사용
    df = load_deg_data(file_path).copy()

    # 유효한 수치형 데이터만 시각화에 사용
    valid = (
        np.isfinite(df["log2FoldChange"])
        & np.isfinite(df["padj"])
        & df["padj"].between(0, 1)
    )

    excluded = int((~valid).sum())

    df = df.loc[valid].copy()

    # padj가 0인 경우 로그 변환 오류 방지
    # 시각화용 값만 조정하며 원래 분석 통계량은 유지
    positive_padj = df.loc[df["padj"] > 0, "padj"]

    if positive_padj.empty:
        minimum_padj = np.finfo(float).tiny
    else:
        minimum_padj = max(
            positive_padj.min() / 10,
            np.nextafter(0.0, 1.0),
        )

    plot_padj = df["padj"].mask(
        df["padj"] == 0,
        minimum_padj,
    )

    df["neg_log10_padj"] = -np.log10(plot_padj)

    # 상향 및 하향 조절 유전자 분류
    up_mask = (
        (df["padj"] < PADJ_THRESHOLD)
        & (df["log2FoldChange"] > LOG2FC_THRESHOLD)
    )

    down_mask = (
        (df["padj"] < PADJ_THRESHOLD)
        & (df["log2FoldChange"] < -LOG2FC_THRESHOLD)
    )

    ns_mask = ~(up_mask | down_mask)

    fig, ax = plt.subplots(figsize=(9, 7))

    # DEG 선별 기준 미충족 유전자
    ax.scatter(
        df.loc[ns_mask, "log2FoldChange"],
        df.loc[ns_mask, "neg_log10_padj"],
        color=NS_COLOR,
        s=9,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
        label=f"Not selected ({ns_mask.sum():,})",
    )

    # 하향 조절 유전자
    ax.scatter(
        df.loc[down_mask, "log2FoldChange"],
        df.loc[down_mask, "neg_log10_padj"],
        color=DOWN_COLOR,
        s=16,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        label=f"Downregulated ({down_mask.sum():,})",
    )

    # 상향 조절 유전자
    ax.scatter(
        df.loc[up_mask, "log2FoldChange"],
        df.loc[up_mask, "neg_log10_padj"],
        color=UP_COLOR,
        s=16,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        label=f"Upregulated ({up_mask.sum():,})",
    )

    # 발현 변화량 기준선
    ax.axvline(
        x=-LOG2FC_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.axvline(
        x=LOG2FC_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    # 조정 p-value 기준선
    ax.axhline(
        y=-np.log10(PADJ_THRESHOLD),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(
        f"Volcano Plot: {DATASET_LABELS[dataset_name]}",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 Adjusted P-value")

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    save_figure(fig, f"{dataset_name}_volcano")

    print(f"시각화 대상: {len(df):,}개 행")
    print(f"제외된 데이터: {excluded:,}개 행")
    print(f"상향 조절: {up_mask.sum():,}개")
    print(f"하향 조절: {down_mask.sum():,}개")


def main():
    """전체 DEG 시각화 작업을 실행한다."""

    print("[START] DEG visualization")

    # DEG 개수 비교 그래프 생성
    print("\n[PLOT] DEG count comparison")

    plot_deg_counts()

    # 4개 데이터셋의 Volcano plot 생성
    for dataset_name, file_path in DATA_FILES.items():

        print(f"\n[PLOT] {dataset_name} volcano")

        plot_volcano(dataset_name, file_path)

    print("\n[COMPLETE] DEG visualization finished.")
    print(f"Results saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()

