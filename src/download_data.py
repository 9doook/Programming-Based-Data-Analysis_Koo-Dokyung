from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE159nnn/GSE159984/suppl/"
)

DATA_FILES = {
    "t2d": "GSE159984_de-t2d.tsv.gz",
    "combined_stress": "GSE159984_de-d4pg.tsv.gz",
    "palmitate": "GSE159984_de-d4p.tsv.gz",
    "glucose": "GSE159984_de-d4g.tsv.gz",
}

TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024


def calculate_sha256(file_path: Path) -> str:
    """파일의 SHA-256 체크섬을 계산한다."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_gzip(file_path: Path) -> None:
    """gzip 파일의 무결성을 확인한다."""
    with gzip.open(file_path, "rb") as file:
        while file.read(CHUNK_SIZE):
            pass


def download_file(
    url: str,
    destination: Path,
) -> dict[str, Any]:
    """파일을 다운로드하고 검증한 뒤 저장한다."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_name(
        destination.name + ".part"
    )

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=TIMEOUT,
        )

        try:
            response.raise_for_status()

            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if chunk:
                        file.write(chunk)
        finally:
            response.close()

        if destination.suffix == ".gz":
            validate_gzip(temporary_path)

        file_size = temporary_path.stat().st_size
        checksum = calculate_sha256(temporary_path)

        os.replace(temporary_path, destination)

        return {
            "filename": destination.name,
            "url": url,
            "size_bytes": file_size,
            "sha256": checksum,
            "status": "success",
        }

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_manifest(
    records: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    """다운로드 기록을 JSON 파일로 저장한다."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = manifest_path.with_name(
        manifest_path.name + ".tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=4,
            )

        os.replace(temporary_path, manifest_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def download_all() -> list[dict[str, Any]]:
    """분석에 필요한 GEO 파일 네 개를 다운로드한다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for label, filename in DATA_FILES.items():
        url = BASE_URL + filename
        destination = DATA_DIR / filename

        print(f"[DOWNLOAD] {label}: {filename}")

        record = download_file(url, destination)
        record["label"] = label
        records.append(record)

        print(
            f"[SUCCESS] {filename} "
            f"({record['size_bytes']:,} bytes)"
        )

    manifest_path = DATA_DIR / "download_manifest.json"
    save_manifest(records, manifest_path)

    print(f"\nManifest saved: {manifest_path}")

    return records


if __name__ == "__main__":
    download_all()
