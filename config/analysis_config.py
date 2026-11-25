
"""
GSE159984 유전자 발현 분석 설정 파일

분석에 사용하는 공통 기준과 파일 경로를 관리한다.
"""

from pathlib import Path

# 프로젝트 최상위 디렉터리
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 데이터 디렉터리
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# 데이터 파일 경로
DATA_FILES = {
    "t2d": RAW_DATA_DIR / "GSE159984_de-t2d.tsv.gz",
    "combined_stress": RAW_DATA_DIR / "GSE159984_de-d4pg.tsv.gz",
    "palmitate": RAW_DATA_DIR / "GSE159984_de-d4p.tsv.gz",
    "glucose": RAW_DATA_DIR / "GSE159984_de-d4g.tsv.gz",
}

# 차등 발현 유전자(DEG) 선별 기준
PADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.58

# 상향 및 하향 조절 유전자 판별 기준
UPREGULATED_THRESHOLD = LOG2FC_THRESHOLD
DOWNREGULATED_THRESHOLD = -LOG2FC_THRESHOLD

# 분석 결과 저장 디렉터리
QC_DIR = RESULTS_DIR / "qc"
DEG_DIR = RESULTS_DIR / "deg"
PLOT_DIR = RESULTS_DIR / "plots"
ENRICHMENT_DIR = RESULTS_DIR / "enrichment"
OVERLAP_DIR = RESULTS_DIR / "overlap"

