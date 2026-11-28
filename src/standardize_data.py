"""
Universal DEG data standardization module.

Standardizes DEG tables and preserves rows without gene annotations
in a separate audit CSV when processing a file.
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
INVALID_ANNOTATIONS = {
    "", "character(0)", "integer(0)", "numeric(0)",
    "nan", "none", "null", "na",
}


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


def _prepare_deg_data(data: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, normalize identifiers, and validate numeric fields."""
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

    # Always provide both identifier columns in the standardized schema.
    for column in ("gene_symbol", "ensembl_id"):
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        result[column] = _clean_annotation_column(result[column])

    numeric_columns = ["log2fc", "pvalue", "padj"]
    if "base_mean" in result.columns:
        numeric_columns.append("base_mean")

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

        if result[column].isna().any():
            raise ValueError(f"{column}에 결측값 또는 잘못된 숫자가 있습니다.")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"{column}에 무한대 또는 비유한 값이 있습니다.")

    for column in ("pvalue", "padj"):
        if not result[column].between(0, 1).all():
            raise ValueError(f"{column}은 0 이상 1 이하여야 합니다.")

    if "base_mean" in result.columns and (result["base_mean"] < 0).any():
        raise ValueError("base_mean은 음수가 될 수 없습니다.")

    return result


def _order_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Put standardized columns first and preserve all additional columns."""
    preferred_order = [
        "gene_symbol", "ensembl_id", "log2fc", "pvalue", "padj",
        "base_mean", "description", "biotype",
    ]
    ordered = [column for column in preferred_order if column in data.columns]
    remaining = [column for column in data.columns if column not in ordered]
    return data[ordered + remaining].reset_index(drop=True)


def split_annotation_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (annotated_rows, unannotated_rows).

    A row is unannotated only when both gene_symbol and ensembl_id are missing.
    Rows with either identifier are retained in annotated_rows.
    """
    prepared = _prepare_deg_data(data)

    unannotated_mask = (
        prepared["gene_symbol"].isna()
        & prepared["ensembl_id"].isna()
    )

    annotated = prepared.loc[~unannotated_mask].copy()
    unannotated = prepared.loc[unannotated_mask].copy()

    if annotated.empty:
        raise ValueError("유효한 유전자 식별자가 있는 행이 없습니다.")

    if annotated["gene_symbol"].notna().any():
        valid_symbols = annotated["gene_symbol"].dropna()
        duplicates = valid_symbols[valid_symbols.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"중복된 유전자 기호가 발견되었습니다: {duplicates[:10]}"
            )

    annotated = _order_columns(annotated)
    unannotated = _order_columns(unannotated)

    annotated.attrs["unannotated_count"] = len(unannotated)
    annotated.attrs["input_row_count"] = len(data)
    annotated.attrs["output_row_count"] = len(annotated)
    annotated.attrs["ensembl_only_count"] = int(
        annotated["gene_symbol"].isna().sum()
    )

    return annotated, unannotated


def standardize_deg_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return annotated standardized rows; audit count is stored in attrs."""
    annotated, _ = split_annotation_rows(data)
    return annotated


def standardize_deg_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    unannotated_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read, standardize, and save a DEG file.

    If output_path is supplied, unannotated rows are also written to:
    <unannotated_dir>/<output_stem>_unannotated.csv
    """
    data = read_deg_file(input_path)
    standardized, unannotated = split_annotation_rows(data)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        standardized.to_csv(output_path, index=False)

        if unannotated_dir is None:
            unannotated_dir = output_path.parent / "unannotated"

        unannotated_dir = Path(unannotated_dir)
        unannotated_dir.mkdir(parents=True, exist_ok=True)

        unannotated_path = (
            unannotated_dir / f"{output_path.stem}_unannotated.csv"
        )
        unannotated.to_csv(unannotated_path, index=False)

        standardized.attrs["unannotated_output_path"] = str(unannotated_path)

    return standardized


def main() -> None:
    parser = argparse.ArgumentParser(description="DEG 데이터 표준화")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--unannotated-dir",
        default=None,
        help="Directory for rows without gene symbol and Ensembl ID",
    )
    args = parser.parse_args()

    result = standardize_deg_file(
        args.input,
        args.output,
        args.unannotated_dir,
    )

    print("[SUCCESS] DEG standardization completed")
    print(f"Input rows: {result.attrs['input_row_count']:,}")
    print(f"Standardized rows: {len(result):,}")
    print(
        "Unannotated rows saved: "
        f"{result.attrs['unannotated_count']:,}"
    )
    print(
        "Ensembl-only rows retained: "
        f"{result.attrs['ensembl_only_count']:,}"
    )
    print(f"Columns: {len(result.columns)}")
    print(f"Output: {args.output}")
    if "unannotated_output_path" in result.attrs:
        print(f"Unannotated output: {result.attrs['unannotated_output_path']}")


if __name__ == "__main__":
    main()
