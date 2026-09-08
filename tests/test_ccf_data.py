"""CCF volume resolution + download logic (no network — the download is mocked)."""

import io
from pathlib import Path

import pytest

from asvimg import ccf_data as cd


def _make_volumes(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / cd.ANNOTATION_NAME).write_bytes(b"a")
    (d / cd.TEMPLATE_NAME).write_bytes(b"t")


def _isolate(monkeypatch, tmp_path):
    """Point the repo/default/allenccf candidates at empty dirs so tests don't pick
    up the machine's real resources/ volumes."""
    monkeypatch.delenv(cd.ENV_VAR, raising=False)
    monkeypatch.setattr(cd, "_REPO_DIR", tmp_path / "_norepo")
    monkeypatch.setattr(cd, "_ALLENCCF_DIR", tmp_path / "_noallen")
    monkeypatch.setattr(cd, "DEFAULT_DIR", tmp_path / "_nodefault")


def test_candidate_dirs_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv(cd.ENV_VAR, str(tmp_path / "env"))
    dirs = cd.candidate_dirs(explicit=tmp_path / "explicit")
    assert dirs[0] == tmp_path / "explicit"       # explicit first
    assert dirs[1] == tmp_path / "env"            # then env
    assert cd.DEFAULT_DIR == dirs[-1]             # default (~/.asovi/atlas) last
    assert cd.DEFAULT_DIR == Path.home() / ".asovi" / "atlas"


def test_find_ccf_volumes(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    assert cd.find_ccf_volumes(explicit=tmp_path / "empty") is None
    _make_volumes(tmp_path)
    got = cd.find_ccf_volumes(explicit=tmp_path)
    assert got == (tmp_path / cd.ANNOTATION_NAME, tmp_path / cd.TEMPLATE_NAME)


def test_find_via_env(tmp_path, monkeypatch):
    _make_volumes(tmp_path / "cache")
    monkeypatch.setenv(cd.ENV_VAR, str(tmp_path / "cache"))
    got = cd.find_ccf_volumes()
    assert got[0].parent == tmp_path / "cache"


def test_resolve_download_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(cd.ENV_VAR, raising=False)
    assert cd.resolve_download_dir() == cd.DEFAULT_DIR
    monkeypatch.setenv(cd.ENV_VAR, str(tmp_path / "env"))
    assert cd.resolve_download_dir() == tmp_path / "env"
    assert cd.resolve_download_dir(explicit=tmp_path / "x") == tmp_path / "x"


def test_ensure_returns_existing(tmp_path):
    _make_volumes(tmp_path)
    got = cd.ensure_ccf_volumes(explicit=tmp_path, interactive=False)
    assert got == (tmp_path / cd.ANNOTATION_NAME, tmp_path / cd.TEMPLATE_NAME)


def test_ensure_noninteractive_raises_with_instructions(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as ei:
        cd.ensure_ccf_volumes(explicit=tmp_path / "empty", download=False, interactive=False)
    assert "asovi-atlas --download" in str(ei.value)
    assert cd.ANNOTATION_NAME in str(ei.value)


class _FakeResp:
    def __init__(self, data: bytes):
        self._b = io.BytesIO(data)

    def read(self, n):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_file_md5_ok_and_atomic(tmp_path, monkeypatch):
    import hashlib

    payload = b"hello world" * 1000
    md5 = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(cd.urllib.request, "urlopen", lambda url, timeout=0: _FakeResp(payload))
    dest = tmp_path / "vol.npy"
    cd._download_file("http://x", dest, expected_md5=md5, size=len(payload))
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".npy.part").exists()      # temp cleaned up


def test_download_file_md5_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cd.urllib.request, "urlopen", lambda url, timeout=0: _FakeResp(b"abc"))
    dest = tmp_path / "vol.npy"
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        cd._download_file("http://x", dest, expected_md5="deadbeef", size=3)
    assert not dest.exists() and not dest.with_suffix(".npy.part").exists()


def test_download_ccf_volumes_uses_manifest(tmp_path, monkeypatch):
    import hashlib

    payloads = {cd.ANNOTATION_NAME: b"AAA" * 100, cd.TEMPLATE_NAME: b"TTT" * 100}
    manifest = {name: {"name": name, "download_url": f"http://x/{name}",
                       "computed_md5": hashlib.md5(data).hexdigest(), "size": len(data)}
                for name, data in payloads.items()}
    monkeypatch.setattr(cd, "_figshare_manifest", lambda: manifest)
    monkeypatch.setattr(cd.urllib.request, "urlopen",
                        lambda url, timeout=0: _FakeResp(payloads[url.rsplit("/", 1)[1]]))
    seen = []
    cd.download_ccf_volumes(tmp_path, reporter=lambda l, d, t: seen.append(l))
    assert (tmp_path / cd.ANNOTATION_NAME).read_bytes() == payloads[cd.ANNOTATION_NAME]
    assert (tmp_path / cd.TEMPLATE_NAME).exists()
    assert seen                                            # progress reporter was called
