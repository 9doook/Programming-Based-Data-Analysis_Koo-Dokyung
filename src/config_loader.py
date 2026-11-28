
"""
범용 GEO 전사체 분석 설정 로더

JSON 설정 파일을 읽고 데이터셋 정보, 입력 파일,
분석 기준 및 결과 저장 경로를 검증한다.
"""

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEO_ACCESSION_PATTERN = re.compile(r"^GSE\d+$")


def _require_mapping(value: Any, name: str) -> dict:
    """설정 항목이 JSON 객체인지 확인한다."""

    if not isinstance(value, dict):
        raise ValueError(f"{name}은 JSON 객체여야 합니다.")

    return value


def _require_string(value: Any, name: str) -> str:
    """필수 문자열 항목을 검증한다."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}은 비어 있지 않은 문자열이어야 합니다.")

    return value.strip()


def _resolve_relative_path(
    root: Path,
    value: str,
    name: str,
) -> Path:
    """프로젝트 루트를 기준으로 경로를 해석한다."""

    value = _require_string(value, name)
    relative_path = Path(value)

    if relative_path.is_absolute():
        raise ValueError(
            f"{name}에는 프로젝트 기준 상대 경로를 사용해야 합니다."
        )

    resolved = (root / relative_path).resolve()

    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(
            f"{name}이 프로젝트 폴더 외부를 가리킬 수 없습니다."
        )

    return resolved


def load_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    JSON 설정 파일을 읽고 검증한다.

    Parameters
    ----------
    config_path:
        JSON 설정 파일 경로.

    project_root:
        상대 경로를 해석할 프로젝트 루트.
        생략하면 현재 프로젝트 루트를 사용한다.

    Returns
    -------
    dict
        검증된 데이터셋 설정.
        파일 및 출력 경로는 절대 경로로 변환된다.
    """

    root = Path(
        project_root or PROJECT_ROOT
    ).resolve()

    config_file = Path(config_path)

    if not config_file.is_absolute():
        config_file = root / config_file

    config_file = config_file.resolve()

    if not config_file.is_file():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {config_file}"
        )

    try:
        with config_file.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            raw = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 형식이 올바르지 않습니다: {error}"
        ) from error

    raw = _require_mapping(raw, "설정 파일")

    # --------------------------------------------------------
    # 1. 데이터셋 정보
    # --------------------------------------------------------

    dataset = _require_mapping(
        raw.get("dataset"),
        "dataset",
    )

    accession = _require_string(
        dataset.get("accession"),
        "dataset.accession",
    ).upper()

    if not GEO_ACCESSION_PATTERN.fullmatch(accession):
        raise ValueError(
            "dataset.accession은 GSE와 숫자로 구성되어야 합니다."
        )

    title = _require_string(
        dataset.get("title"),
        "dataset.title",
    )

    organism = _require_string(
        dataset.get("organism"),
        "dataset.organism",
    )

    # --------------------------------------------------------
    # 2. 입력 데이터 정보
    # --------------------------------------------------------

    input_config = _require_mapping(
        raw.get("input"),
        "input",
    )

    input_type = _require_string(
        input_config.get("type"),
        "input.type",
    )

    if input_type != "precomputed_deg":
        raise ValueError(
            "현재 지원하는 입력 유형은 precomputed_deg입니다."
        )

    input_directory = _resolve_relative_path(
        root,
        input_config.get("directory", "data/raw"),
        "input.directory",
    )

    files = _require_mapping(
        input_config.get("files"),
        "input.files",
    )

    if not files:
        raise ValueError(
            "input.files에 하나 이상의 데이터 파일이 필요합니다."
        )

    resolved_files = {}

    for label, filename in files.items():

        label = _require_string(
            label,
            "input.files의 그룹 이름",
        )

        filename = _require_string(
            filename,
            f"input.files.{label}",
        )

        file_path = _resolve_relative_path(
            input_directory,
            filename,
            f"input.files.{label}",
        )

        resolved_files[label] = {
            "filename": filename,
            "path": file_path,
        }

    base_url = input_config.get("base_url")

    if base_url is not None:
        base_url = _require_string(
            base_url,
            "input.base_url",
        )

        if not base_url.startswith(("https://", "http://")):
            raise ValueError(
                "input.base_url은 HTTP 또는 HTTPS 주소여야 합니다."
            )

        base_url = base_url.rstrip("/") + "/"

    # --------------------------------------------------------
    # 3. 분석 기준
    # --------------------------------------------------------

    analysis = _require_mapping(
        raw.get("analysis"),
        "analysis",
    )

    try:
        padj_threshold = float(
            analysis.get("padj_threshold", 0.05)
        )

        log2fc_threshold = float(
            analysis.get("log2fc_threshold", 0.58)
        )

    except (TypeError, ValueError) as error:
        raise ValueError(
            "분석 기준은 숫자여야 합니다."
        ) from error

    import math

    if not math.isfinite(padj_threshold):
        raise ValueError("padj_threshold는 유한한 값이어야 합니다.")

    if not 0 < padj_threshold <= 1:
        raise ValueError(
            "padj_threshold는 0 초과 1 이하여야 합니다."
        )

    if not math.isfinite(log2fc_threshold):
        raise ValueError("log2fc_threshold는 유한한 값이어야 합니다.")

    if log2fc_threshold < 0:
        raise ValueError(
            "log2fc_threshold는 0 이상이어야 합니다."
        )

    # --------------------------------------------------------
    # 4. 출력 경로
    # --------------------------------------------------------

    output = _require_mapping(
        raw.get("output", {}),
        "output",
    )

    output_directory = _resolve_relative_path(
        root,
        output.get(
            "directory",
            f"results/{accession}",
        ),
        "output.directory",
    )

    # --------------------------------------------------------
    # 5. 검증된 설정 반환
    # --------------------------------------------------------

    return {
        "dataset": {
            "accession": accession,
            "title": title,
            "organism": organism,
        },
        "input": {
            "type": input_type,
            "directory": input_directory,
            "base_url": base_url,
            "files": resolved_files,
        },
        "analysis": {
            "padj_threshold": padj_threshold,
            "log2fc_threshold": log2fc_threshold,
        },
        "output": {
            "directory": output_directory,
        },
        "config_path": config_file,
        "project_root": root,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GEO 분석 설정 파일 검증"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="분석 설정 JSON 파일",
    )

    args = parser.parse_args()

    settings = load_config(args.config)

    print("[SUCCESS] Configuration loaded")
    print(f"Dataset: {settings['dataset']['accession']}")
    print(f"Organism: {settings['dataset']['organism']}")
    print(f"Input files: {len(settings['input']['files'])}")
    print(
        "Adjusted p-value threshold:",
        settings["analysis"]["padj_threshold"],
    )
    print(
        "Log2FC threshold:",
        settings["analysis"]["log2fc_threshold"],
    )
    print(f"Output: {settings['output']['directory']}")
