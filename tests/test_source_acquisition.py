"""Pinned public results acquisition must preserve existing source files."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


def _script(tmp_path, monkeypatch, contents):
    script_path = Path(__file__).resolve().parents[1] / "scripts/fetch_openfootball.py"
    spec = importlib.util.spec_from_file_location("fetch_openfootball", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/OPENFOOTBALL_SOURCES.json").write_text(json.dumps({"files": [
        {"season": season, "url": f"https://example.invalid/{season}",
         "sha256": hashlib.sha256(data).hexdigest()} for season, data in contents.items()
    ]}))
    return script


def test_pinned_download_then_reuse_without_network(tmp_path, monkeypatch):
    script = _script(tmp_path, monkeypatch, {"2024-25": b"synthetic public results"})
    monkeypatch.setattr(script, "urlopen", lambda *a, **kw: io.BytesIO(b"synthetic public results"))
    script.main()
    path = tmp_path / "data/raw/openfootball/2024-25.json"
    assert path.read_bytes() == b"synthetic public results"
    monkeypatch.setattr(script, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    script.main()
    assert path.read_bytes() == b"synthetic public results"


def test_differing_existing_source_is_never_replaced(tmp_path, monkeypatch):
    script = _script(tmp_path, monkeypatch, {"2024-25": b"expected"})
    path = tmp_path / "data/raw/openfootball/2024-25.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve local source")
    monkeypatch.setattr(script, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="hash mismatch"):
        script.main()
    assert path.read_bytes() == b"preserve local source"


def test_bad_remote_hash_writes_no_sources(tmp_path, monkeypatch):
    script = _script(tmp_path, monkeypatch, {"2023-24": b"valid", "2024-25": b"expected"})
    monkeypatch.setattr(script, "urlopen", lambda url, **kw:
                        io.BytesIO(b"valid" if url.endswith("2023-24") else b"changed"))
    with pytest.raises(ValueError, match="hash mismatch"):
        script.main()
    assert not (tmp_path / "data").exists()
