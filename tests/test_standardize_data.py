
import pandas as pd
import numpy as np
import pytest

from src.standardize_data import (
    normalize_column_name,
    build_column_mapping,
    read_deg_file,
    standardize_deg_data,
    standardize_deg_file,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "short_id": ["INS", "PPARG", "SLC2A4"],
        "ensembl_id": [
            "ENSG00000129965",
            "ENSG00000132170",
            "ENSG00000181856",
        ],
        "log2FoldChange": [1.2, -0.8, 0.9],
        "pvalue": [0.001, 0.02, 0.003],
        "padj": [0.004, 0.03, 0.008],
        "baseMean": [100.0, 200.0, 150.0],
        "description": [
            "Insulin",
            "PPARG",
            "GLUT4",
        ],
        "biotype": [
            "protein_coding",
            "protein_coding",
            "protein_coding",
        ],
    })


def test_normalize_column_name():
    assert normalize_column_name("Log2FoldChange") == (
        "log2foldchange"
    )

    assert normalize_column_name("gene-symbol") == (
        "gene_symbol"
    )

    assert normalize_column_name(" Gene Name ") == (
        "gene_name"
    )


def test_build_column_mapping():
    columns = [
        "short_id",
        "log2FoldChange",
        "pvalue",
        "padj",
    ]

    mapping = build_column_mapping(columns)

    assert mapping["short_id"] == "gene_symbol"
    assert mapping["log2FoldChange"] == "log2fc"
    assert mapping["pvalue"] == "pvalue"
    assert mapping["padj"] == "padj"


def test_standardize_deg_data(sample_data):
    result = standardize_deg_data(sample_data)

    assert list(result.columns[:6]) == [
        "gene_symbol",
        "ensembl_id",
        "log2fc",
        "pvalue",
        "padj",
        "base_mean",
    ]

    assert result["gene_symbol"].tolist() == [
        "INS",
        "PPARG",
        "SLC2A4",
    ]

    assert result["log2fc"].tolist() == [
        1.2,
        -0.8,
        0.9,
    ]

    assert len(result) == 3


def test_original_data_is_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    standardize_deg_data(sample_data)

    pd.testing.assert_frame_equal(
        sample_data,
        original,
    )


@pytest.mark.parametrize(
    "column",
    [
        "short_id",
        "log2FoldChange",
        "pvalue",
        "padj",
    ],
)
def test_missing_required_column(sample_data, column):
    data = sample_data.drop(columns=[column])

    with pytest.raises(ValueError, match="필수 열"):
        standardize_deg_data(data)


@pytest.mark.parametrize(
    "column,value",
    [
        ("log2FoldChange", np.nan),
        ("log2FoldChange", np.inf),
        ("pvalue", -0.01),
        ("pvalue", 1.1),
        ("pvalue", np.nan),
        ("padj", -0.01),
        ("padj", 1.1),
        ("padj", np.nan),
        ("baseMean", -1),
        ("baseMean", np.inf),
    ],
)
def test_invalid_numeric_values(sample_data, column, value):
    sample_data.loc[0, column] = value

    with pytest.raises(ValueError):
        standardize_deg_data(sample_data)


def test_missing_gene_symbol(sample_data):
    sample_data.loc[0, "short_id"] = None

    with pytest.raises(ValueError, match="결측값"):
        standardize_deg_data(sample_data)


def test_empty_gene_symbol(sample_data):
    sample_data.loc[0, "short_id"] = " "

    with pytest.raises(ValueError, match="비어"):
        standardize_deg_data(sample_data)


def test_duplicate_gene_symbol(sample_data):
    sample_data.loc[1, "short_id"] = "INS"

    with pytest.raises(ValueError, match="중복"):
        standardize_deg_data(sample_data)


def test_empty_dataframe():
    with pytest.raises(ValueError, match="비어"):
        standardize_deg_data(pd.DataFrame())


def test_invalid_input_type():
    with pytest.raises(TypeError):
        standardize_deg_data(["INS", "PPARG"])


def test_duplicate_column_alias(sample_data):
    sample_data["gene_symbol"] = sample_data["short_id"]

    with pytest.raises(ValueError, match="여러 개"):
        standardize_deg_data(sample_data)


def test_read_tsv_file(tmp_path, sample_data):
    path = tmp_path / "test.tsv"

    sample_data.to_csv(
        path,
        sep="\t",
        index=False,
    )

    result = read_deg_file(path)

    assert len(result) == 3
    assert "short_id" in result.columns


def test_read_gzip_file(tmp_path, sample_data):
    path = tmp_path / "test.tsv.gz"

    sample_data.to_csv(
        path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    result = read_deg_file(path)

    assert len(result) == 3


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_deg_file(tmp_path / "missing.tsv")


def test_empty_file(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        read_deg_file(path)


def test_standardize_deg_file(
    tmp_path,
    sample_data,
):
    input_path = tmp_path / "input.tsv"
    output_path = tmp_path / "output" / "standardized.csv"

    sample_data.to_csv(
        input_path,
        sep="\t",
        index=False,
    )

    result = standardize_deg_file(
        input_path,
        output_path,
    )

    assert output_path.exists()

    saved = pd.read_csv(output_path)

    assert len(saved) == 3

    assert "gene_symbol" in saved.columns

    pd.testing.assert_frame_equal(
        result,
        saved,
        check_dtype=False,
    )


def test_preserve_original_statistics(sample_data):
    result = standardize_deg_data(sample_data)

    np.testing.assert_allclose(
        result["log2fc"],
        sample_data["log2FoldChange"],
    )

    np.testing.assert_allclose(
        result["pvalue"],
        sample_data["pvalue"],
    )

    np.testing.assert_allclose(
        result["padj"],
        sample_data["padj"],
    )


def test_additional_columns_are_preserved(sample_data):
    sample_data["custom_annotation"] = [
        "A",
        "B",
        "C",
    ]

    result = standardize_deg_data(sample_data)

    assert "custom_annotation" in result.columns

    assert result["custom_annotation"].tolist() == [
        "A",
        "B",
        "C",
    ]
