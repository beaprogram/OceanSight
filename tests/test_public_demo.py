import hashlib
import pytest
from oceansight.public_demo import fetch_verified


class Response:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.content


def test_verified_download_and_cache(tmp_path, monkeypatch):
    raw = b"evaluated artifact"
    monkeypatch.setattr("oceansight.public_demo.requests.get", lambda *a, **k: Response(raw))
    path = tmp_path / "model.onnx"
    sha = hashlib.sha256(raw).hexdigest()
    fetch_verified("https://example.test/model", path, sha)
    assert path.read_bytes() == raw

    def no_network(*args, **kwargs):
        raise AssertionError("Valid cache should not download again")

    monkeypatch.setattr("oceansight.public_demo.requests.get", no_network)
    assert fetch_verified("https://example.test/model", path, sha) == path


@pytest.mark.parametrize("limit,error", [(100, "fingerprint"), (2, "size")])
def test_rejected_download_keeps_existing_file(tmp_path, monkeypatch, limit, error):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old artifact")
    monkeypatch.setattr(
        "oceansight.public_demo.requests.get", lambda *a, **k: Response(b"wrong download")
    )
    with pytest.raises(ValueError, match=error):
        fetch_verified("https://example.test/model", path, "0" * 64, max_bytes=limit)
    assert path.read_bytes() == b"old artifact"
    assert list(tmp_path.iterdir()) == [path]
