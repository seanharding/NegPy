import os

import pytest

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.scanner import ScannerProfiles


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.scanner.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))
    monkeypatch.setattr(APP_CONFIG, "scanner_dir", str(tmp_path))


def test_list_get_and_none(tmp_path):
    _write(
        os.path.join(tmp_path, "my_scanner.toml"),
        'name = "My Scanner"\nmatrix = [[1.0, -0.1, 0.0], [0.0, 1.1, -0.3], [0.0, -0.3, 1.1]]\n',
    )
    assert ScannerProfiles.list_profiles() == ["None", "My Scanner"]
    assert ScannerProfiles.get_matrix("My Scanner") == [1.0, -0.1, 0.0, 0.0, 1.1, -0.3, 0.0, -0.3, 1.1]
    assert ScannerProfiles.get_matrix("None") is None  # off entry has no matrix
    assert ScannerProfiles.is_bundled("None") is True
    assert ScannerProfiles.is_bundled("My Scanner") is False


def test_save_and_delete(tmp_path):
    ScannerProfiles.save("Setup A", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert os.path.exists(os.path.join(tmp_path, "setup_a.toml"))
    assert "Setup A" in ScannerProfiles.list_profiles()
    ScannerProfiles.delete("Setup A")
    assert "Setup A" not in ScannerProfiles.list_profiles()


def test_malformed_skipped(tmp_path):
    _write(os.path.join(tmp_path, "bad.toml"), 'name = "Bad"\nmatrix = [[1, 0], [0, 1]]\n')  # wrong shape
    assert ScannerProfiles.list_profiles() == ["None"]
