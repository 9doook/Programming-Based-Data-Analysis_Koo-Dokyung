import gzip

import pytest

from src import download_data


class FakeResponse:
    """실제 인터넷 연결 없이 다운로드 응답을 흉내 내는 객체."""

    def __init__(self, content: bytes):
        self.content = content
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        self.closed = True


def test_download_valid_gzip(monkeypatch, tmp_path):
    """정상적인 gzip 파일이 다운로드되고 저장되는지 검사한다."""

    compressed_data = gzip.compress(
        b"gene_id\tlog2FoldChange\tpadj\n"
        b"GENE1\t1.2\t0.01\n"
    )

    fake_response = FakeResponse(compressed_data)

    monkeypatch.setattr(
        download_data.requests,
        "get",
        lambda *args, **kwargs: fake_response,
    )

    destination = tmp_path / "test.tsv.gz"

    result = download_data.download_file(
        url="https://example.com/test.tsv.gz",
        destination=destination,
    )

    assert destination.exists()
    assert destination.stat().st_size > 0
    assert result["status"] == "success"
    assert len(result["sha256"]) == 64
    assert fake_response.closed

    with gzip.open(destination, "rt") as file:
        content = file.read()

    assert "GENE1" in content


def test_invalid_gzip_is_not_saved(monkeypatch, tmp_path):
    """잘못된 gzip 파일은 최종 파일로 저장되지 않는지 검사한다."""

    fake_response = FakeResponse(
        b"this is not a gzip file"
    )

    monkeypatch.setattr(
        download_data.requests,
        "get",
        lambda *args, **kwargs: fake_response,
    )

    destination = tmp_path / "invalid.tsv.gz"

    with pytest.raises(
        (gzip.BadGzipFile, EOFError, OSError)
    ):
        download_data.download_file(
            url="https://example.com/invalid.tsv.gz",
            destination=destination,
        )

    assert not destination.exists()

    temporary_path = destination.with_name(
        destination.name + ".part"
    )

    assert not temporary_path.exists()
