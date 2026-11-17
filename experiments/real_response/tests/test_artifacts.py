"""A changed protocol must not overwrite existing run provenance."""
import json
import pytest
from real_response import artifacts


def test_metadata_rejects_different_protocol_without_overwrite(tmp_path, monkeypatch):
    path = tmp_path / 'protocol_manifest.json'
    original = json.dumps({'config': {'rho': 1}})
    path.write_text(original)
    monkeypatch.setattr(artifacts, 'protocol_identity', lambda config: {'config': {'rho': 2}})
    with pytest.raises(RuntimeError, match='different scientific protocol'):
        artifacts.metadata({}, tmp_path)
    assert path.read_text() == original
