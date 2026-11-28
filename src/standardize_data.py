"""
Universal DEG data standardization module.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "gene_symbol": ["gene_symbol", "short_id", "gene_name", "symbol", "gene"],
    "ensembl_id": ["ensembl_id", "ensembl", "gene_ensembl_id"],
    "log2fc": ["log2fc", "log2foldchange", "log2_fold_change", "log_fc"],
    "pvalue": ["pvalue", "p_value", "pval"],
    "padj": ["padj", "adjusted_pvalue", "adjusted_p_value", "fdr", "qvalue"],
    "base_mean": ["base_mean", "basemean", "mean_expression"],
    "description": ["description", "gene_description"],
    "biotype": ["biotype", "gene_biotype"],
}
REQUIRED_STAT_COLUMNS = {"log2fc", "pvalue", "padj"}
INVALID_ANNOTATIONS = {"", "character(0)", "integer(0)", "nan", "none", "null", "na"}


def normalize_column_name(name: object) -> str:
    """Normalize a column name for alias matching."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def build_column_mapping(columns: Iterable[object]) -> dict:
    """Map source column names to standard names."""
    normalized_columns = {}
    for column in columns:
        normalized = normalize_column_name(column)
        if normalized in normalized_columns:
            raise ValueError(f"중복된 열 이름이 발견되었습니다: {normalized}")
        normalized_columns[normalized] = column

    mapping = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        matches = [
            normalize_column_name(alias)
            for alias in aliases
            if normalize_column_name(alias) in normalized_columns
        ]
        if len(matches) > 1:
            raise ValueError(
                f"{standard_name}에 대응하는 열이 여러 개입니다: {matches}"
            )
        if matches:
            mapping[normalized_columns[matches[0]]] = standard_name
    return mapping


def read_deg_file(file_path: str | Path) -> pd.DataFrame:
    """Read a TSV or compressed TSV file."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
    try:
        data = pd.read_csv(file_path, sep="\t", compression="infer")
    except Exception as error:
        raise ValueError(f"데이터 파일을 읽을 수 없습니다: {error}") from error
    if data.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")
    return data


def _clean_annotation_column(series: pd.Series) -> pd.Series:
    """Convert known empty annotation placeholders to pandas NA."""
    values = series.astype("string").str.strip()
    return values.mask(values.str.lower().isin(INVALID_ANNOTATIONS), pd.NA)


def standardize_deg_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DEG columns and validate values.

    Rows without both gene_symbol and ensembl_id are excluded and counted.
    Rows with an Ensembl ID but no gene symbol are retained.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("입력은 pandas DataFrame이어야 합니다.")
    if data.empty:
        raise ValueError("입력 데이터가 비어 있습니다.")

    result = data.copy(deep=True)
    result = result.rename(columns=build_column_mapping(result.columns))

    missing_stats = REQUIRED_STAT_COLUMNS - set(result.columns)
    if missing_stats:
        raise ValueError(f"필수 열이 누락되었습니다: {sorted(missing_stats)}")

    if not ({"gene_symbol", "ensembl_id"} & set(result.columns)):
        raise ValueError("유전자 식별 열이 필요합니다: gene_symbol 또는 ensembl_id")

    # Keep both identifier columns in the standardized output schema.
    if "gene_symbol" not in result.columns:
        result["gene_symbol"] = pd.Series(
            pd.NA, index=result.index, dtype="string"
        )
    if "ensembl_id" not in result.columns:
        result["ensembl_id"] = pd.Series(
            pd.NA, index=result.index, dtype="string"
        )

    for column in ("gene_symbol", "ensembl_id"):
        if column in result.columns:
            result[column] = _clean_annotation_column(result[column])

    if "gene_symbol" in result.columns:
        missing_symbol = result["gene_symbol"].isna()
    else:
        missing_symbol = pd.Series(True, index=result.index)

    if "ensembl_id" in result.columns:
        missing_ensembl = result["ensembl_id"].isna()
    else:
        missing_ensembl = pd.Series(True, index=result.index)

    unannotated_mask = missing_symbol & missing_ensembl
    unannotated_count = int(unannotated_mask.sum())
    result = result.loc[~unannotated_mask].copy()

    if result.empty:
        raise ValueError("유효한 유전자 식별자가 있는 행이 없습니다.")

    # Check duplicate valid symbols only; retain Ensembl-only rows.
    if "gene_symbol" in result.columns:
        valid_symbols = result["gene_symbol"].dropna()
        duplicates = valid_symbols[valid_symbols.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"중복된 유전자 기호가 발견되었습니다: {duplicates[:10]}"
            )

    numeric_columns = ["log2fc", "pvalue", "padj"]
    if "base_mean" in result.columns:
        numeric_columns.append("base_mean")

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(
                f"{column}에 결측값 또는 잘못된 숫자가 있습니다."
            )
        if not np.isfinite(result[column]).all():
            raise ValueError(f"{column}에 무한대 또는 비유한 값이 있습니다.")

    for column in ("pvalue", "padj"):
        if not result[column].between(0, 1).all():
            raise ValueError(f"{column}은 0 이상 1 이하여야 합니다.")

    if "base_mean" in result.columns and (result["base_mean"] < 0).any():
        raise ValueError("base_mean은 음수가 될 수 없습니다.")

    preferred_order = [
        "gene_symbol", "ensembl_id", "log2fc", "pvalue", "padj",
        "base_mean", "description", "biotype",
    ]
    ordered_columns = [
        column for column in preferred_order if column in result.columns
    ]
    remaining_columns = [
        column for column in result.columns if column not in ordered_columns
    ]
    result = result[ordered_columns + remaining_columns].reset_index(drop=True)

    result.attrs["unannotated_count"] = unannotated_count
    result.attrs["input_row_count"] = len(data)
    result.attrs["output_row_count"] = len(result)
    result.attrs["ensembl_only_count"] = (
        int(result["gene_symbol"].isna().sum())
        if "gene_symbol" in result.columns else len(result)
    )
    return result


def standardize_deg_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read, standardize, and optionally save a DEG file."""
    data = read_deg_file(input_path)
    standardized = standardize_deg_data(data)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        standardized.to_csv(output_path, index=False)
    return standardized


def main() -> None:
    parser = argparse.ArgumentParser(description="DEG 데이터 표준화")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = standardize_deg_file(args.input, args.output)
    print("[SUCCESS] DEG standardization completed")
    print(f"Input rows: {result.attrs['input_row_count']:,}")
    print(f"Standardized rows: {len(result):,}")
    print(f"Unannotated rows excluded: {result.attrs['unannotated_count']:,}")
    print(f"Ensembl-only rows retained: {result.attrs['ensembl_only_count']:,}")
    print(f"Columns: {len(result.columns)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
