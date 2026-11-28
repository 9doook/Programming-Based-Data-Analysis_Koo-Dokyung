import json

import pytest

from src.config_loader import load_config


def make_config():
    """테스트용 기본 설정을 생성한다."""

    return {
        "dataset": {
            "accession": "GSE159984",
            "title": "Test Dataset",
            "organism": "Homo sapiens",
        },
        "input": {
            "type": "precomputed_deg",
            "directory": "data/raw",
            "base_url": "https://example.com/data/",
            "files": {
                "control": "control.tsv.gz",
                "treatment": "treatment.tsv.gz",
            },
        },
        "analysis": {
            "padj_threshold": 0.05,
            "log2fc_threshold": 0.58,
        },
        "output": {
            "directory": "results/test",
        },
    }


def save_config(tmp_path, config):
    """임시 JSON 설정 파일을 저장한다."""

    path = tmp_path / "config.json"

    path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    return path


def test_load_valid_config(tmp_path):
    config = make_config()
    path = save_config(tmp_path, config)

    result = load_config(path, project_root=tmp_path)

    assert result["dataset"]["accession"] == "GSE159984"
    assert result["dataset"]["organism"] == "Homo sapiens"
    assert result["input"]["type"] == "precomputed_deg"
    assert len(result["input"]["files"]) == 2
    assert result["analysis"]["padj_threshold"] == 0.05
    assert result["analysis"]["log2fc_threshold"] == 0.58

    assert result["output"]["directory"] == (
        tmp_path / "results" / "test"
    ).resolve()


def test_resolve_input_file_paths(tmp_path):
    path = save_config(tmp_path, make_config())

    result = load_config(path, project_root=tmp_path)

    control = result["input"]["files"]["control"]

    assert control["filename"] == "control.tsv.gz"

    assert control["path"] == (
        tmp_path / "data" / "raw" / "control.tsv.gz"
    ).resolve()

    assert control["path"].is_relative_to(tmp_path.resolve())


def test_generate_download_urls(tmp_path):
    path = save_config(tmp_path, make_config())

    result = load_config(path, project_root=tmp_path)

    base_url = result["input"]["base_url"]
    filename = result["input"]["files"]["control"]["filename"]

    url = base_url + filename

    assert url == "https://example.com/data/control.tsv.gz"


def test_default_analysis_thresholds(tmp_path):
    config = make_config()
    del config["analysis"]

    path = save_config(tmp_path, config)

    result = load_config(path, project_root=tmp_path)

    assert result["analysis"]["padj_threshold"] == 0.05
    assert result["analysis"]["log2fc_threshold"] == 0.58


def test_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(
            tmp_path / "missing.json",
            project_root=tmp_path,
        )


def test_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_config(path, project_root=tmp_path)


@pytest.mark.parametrize(
    "accession",
    [
        "",
        "GSEABC",
        "GSE123ABC",
        "SRP12345",
    ],
)
def test_invalid_geo_accession(tmp_path, accession):
    config = make_config()
    config["dataset"]["accession"] = accession

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError, match="accession"):
        load_config(path, project_root=tmp_path)


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_padj_threshold(tmp_path, threshold):
    config = make_config()
    config["analysis"]["padj_threshold"] = threshold

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError):
        load_config(path, project_root=tmp_path)


def test_invalid_log2fc_threshold(tmp_path):
    config = make_config()
    config["analysis"]["log2fc_threshold"] = -0.5

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError, match="log2fc"):
        load_config(path, project_root=tmp_path)


def test_reject_path_traversal(tmp_path):
    config = make_config()
    config["input"]["directory"] = "../outside"

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError, match="외부"):
        load_config(path, project_root=tmp_path)


def test_reject_absolute_input_filename(tmp_path):
    config = make_config()
    config["input"]["files"]["control"] = str(
        tmp_path / "outside.tsv.gz"
    )

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError):
        load_config(path, project_root=tmp_path)


def test_reject_unsupported_input_type(tmp_path):
    config = make_config()
    config["input"]["type"] = "unknown"

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError, match="지원하는 입력 유형"):
        load_config(path, project_root=tmp_path)


def test_reject_empty_files(tmp_path):
    config = make_config()
    config["input"]["files"] = {}

    path = save_config(tmp_path, config)

    with pytest.raises(ValueError, match="데이터 파일"):
        load_config(path, project_root=tmp_path)
