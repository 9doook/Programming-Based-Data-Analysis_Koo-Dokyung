"""
범용 DEG 데이터 표준화 모듈

다양한 GEO 데이터셋의 DEG 결과를 공통 열 구조로 변환한다.
원본 통계값은 변경하지 않는다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "gene_symbol": [
        "gene_symbol",
        "short_id",
        "gene_name",
        "symbol",
        "gene",
    ],
    "ensembl_id": [
        "ensembl_id",
        "ensembl",
        "gene_ensembl_id",
    ],
    "log2fc": [
        "log2fc",
        "log2foldchange",
        "log2_fold_change",
        "log_fc",
    ],
    "pvalue": [
        "pvalue",
        "p_value",
        "pval",
    ],
    "padj": [
        "padj",
        "adjusted_pvalue",
        "adjusted_p_value",
        "fdr",
        "qvalue",
    ],
    "base_mean": [
        "base_mean",
        "basemean",
        "mean_expression",
    ],
    "description": [
        "description",
        "gene_description",
    ],
    "biotype": [
        "biotype",
        "gene_biotype",
    ],
}


REQUIRED_COLUMNS = {
    "gene_symbol",
    "log2fc",
    "pvalue",
    "padj",
}


def normalize_column_name(name):
    """열 이름의 대소문자와 구분자를 통일한다."""

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_column_mapping(columns):
    """실제 열 이름을 표준 열 이름으로 매핑한다."""

    normalized_columns = {}

    for column in columns:
        normalized = normalize_column_name(column)

        if normalized in normalized_columns:
            raise ValueError(
                f"중복된 열 이름이 발견되었습니다: {normalized}"
            )

        normalized_columns[normalized] = column

    mapping = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        matches = [
            normalized
            for normalized in aliases
            if normalized in normalized_columns
        ]

        if len(matches) > 1:
            raise ValueError(
                f"{standard_name}에 대응하는 열이 여러 개입니다: "
                f"{matches}"
            )

        if matches:
            original_name = normalized_columns[matches[0]]
            mapping[original_name] = standard_name

    return mapping


def read_deg_file(file_path):
    """TSV 또는 압축 TSV 파일을 읽는다."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path}"
        )

    try:
        data = pd.read_csv(
            file_path,
            sep="\t",
            compression="infer",
        )

    except Exception as error:
        raise ValueError(
            f"데이터 파일을 읽을 수 없습니다: {error}"
        ) from error

    if data.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")

    return data


def standardize_deg_data(data):
    """
    DEG 데이터의 열 이름과 데이터 형식을 표준화한다.

    Parameters
    ----------
    data : pandas.DataFrame
        원본 DEG 데이터

    Returns
    -------
    pandas.DataFrame
        표준화된 DEG 데이터
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("입력은 pandas DataFrame이어야 합니다.")

    if data.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")

    result = data.copy()

    mapping = build_column_mapping(result.columns)

    result = result.rename(columns=mapping)

    missing = REQUIRED_COLUMNS - set(result.columns)

    if missing:
        raise ValueError(
            f"필수 열이 누락되었습니다: {sorted(missing)}"
        )

    # 필수 유전자 기호 확인
    
# 유전자 기호 및 식별자 표준화
result["gene_symbol"] = (
    result["gene_symbol"]
    .astype("string")
    .str.strip()
)

# GEO 데이터에서 발견되는 비정상적인 주석 표현 처리
invalid_annotations = {
    "",
    "character(0)",
    "integer(0)",
    "nan",
    "none",
    "null",
    "na",
}

for column in ["gene_symbol", "ensembl_id"]:
    if column in result.columns:
        values = (
            result[column]
            .astype("string")
            .str.strip()
        )

        invalid_mask = (
            values.str.lower()
            .isin(invalid_annotations)
        )

        result[column] = values.mask(invalid_mask)

# 유전자 기호와 Ensembl ID가 모두 없는 행 분리
unannotated_mask = result[
    "gene_symbol"
].isna()

if "ensembl_id" in result.columns:
    unannotated_mask &= result["ensembl_id"].isna()

unannotated_count = int(unannotated_mask.sum())

# 식별 가능한 유전자만 표준 분석 데이터로 사용
result = result.loc[~unannotated_mask].copy()

# 남은 유전자 기호는 필수
if result["gene_symbol"].isna().any():
    raise ValueError(
        "유전자 기호가 없고 대체 식별자도 없는 행이 남아 있습니다."
    )

# 실제 유전자 기호 중복 검사
duplicated = result["gene_symbol"].duplicated()

if duplicated.any():
    raise ValueError(
        "중복된 유전자 기호가 발견되었습니다: "
        f"{result.loc[duplicated, 'gene_symbol'].tolist()[:10]}"
    )


    # 통계값 숫자 형식 변환
    numeric_columns = [
        "log2fc",
        "pvalue",
        "padj",
    ]

    if "base_mean" in result.columns:
        numeric_columns.append("base_mean")

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if result[column].isna().any():
            raise ValueError(
                f"{column}에 결측값 또는 잘못된 숫자가 있습니다."
            )

        if not np.isfinite(result[column]).all():
            raise ValueError(
                f"{column}에 무한대 또는 비유한 값이 있습니다."
            )

    # 통계값 범위 검사
    for column in ["pvalue", "padj"]:

        if not result[column].between(0, 1).all():
            raise ValueError(
                f"{column}은 0 이상 1 이하여야 합니다."
            )

    if "base_mean" in result.columns:

        if (result["base_mean"] < 0).any():
            raise ValueError(
                "base_mean은 음수가 될 수 없습니다."
            )

    # 중복 유전자 확인
    duplicated = result["gene_symbol"].duplicated()

    if duplicated.any():
        raise ValueError(
            f"중복된 유전자 기호가 발견되었습니다: "
            f"{result.loc[duplicated, 'gene_symbol'].tolist()[:10]}"
        )

    # 표준 열 순서
    preferred_order = [
        "gene_symbol",
        "ensembl_id",
        "log2fc",
        "pvalue",
        "padj",
        "base_mean",
        "description",
        "biotype",
    ]

    ordered_columns = [
        column
        for column in preferred_order
        if column in result.columns
    ]

    remaining_columns = [
        column
        for column in result.columns
        if column not in ordered_columns
    ]

        result = result[ordered_columns + remaining_columns].reset_index(
        drop=True
    )

    result.attrs["unannotated_count"] = unannotated_count

    return result


def standardize_deg_file(input_path, output_path=None):
    """파일을 읽고 표준화한 뒤 선택적으로 저장한다."""

    data = read_deg_file(input_path)

    standardized = standardize_deg_data(data)

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        standardized.to_csv(
            output_path,
            index=False,
        )

    return standardized


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="DEG 데이터 표준화"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    result = standardize_deg_file(
        args.input,
        args.output,
    )

    print("[SUCCESS] DEG standardization completed")
    print(f"Genes: {len(result):,}")
    print(f"Columns: {len(result.columns)}")
    print(f"Output: {args.output}")
