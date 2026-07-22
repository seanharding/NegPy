import os

import numpy as np
import pytest

from negpy.kernel.system.config import APP_CONFIG

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.scanner.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))
    monkeypatch.setattr(APP_CONFIG, "scanner_dir", str(tmp_path))


def _fake_captures(monkeypatch):
    """Make decode_source_negative return a synthetic bare-light image per band path,
    with a known green<->blue leakage."""
    caps = {
        "R": (0.9, 0.1, 0.03),
        "G": (0.05, 0.5, 0.15),
        "B": (0.04, 0.3, 1.0),
    }

    def fake_decode(self, path, params, fast=True):
        rgb = caps[path]  # path is just the band key in the test
        return np.tile(np.array(rgb, dtype=np.float32), (16, 16, 1))

    monkeypatch.setattr("negpy.services.rendering.image_processor.ImageProcessor.decode_source_negative", fake_decode)


def test_compute_and_save(tmp_path, monkeypatch):
    from negpy.desktop.view.widgets.scanner_calibration_dialog import ScannerCalibrationDialog
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.assets.scanner import ScannerProfiles

    _fake_captures(monkeypatch)
    dlg = ScannerCalibrationDialog(WorkspaceConfig())
    dlg._paths = {"R": "R", "G": "G", "B": "B"}
    dlg.name_edit.setText("Test Scanner")
    dlg._compute_and_save()

    assert dlg.saved_profile_name == "Test Scanner"
    assert os.path.exists(os.path.join(tmp_path, "test_scanner.toml"))
    m = np.array(ScannerProfiles.get_matrix("Test Scanner")).reshape(3, 3)
    # It should be the inverse of the normalized leakage — green/blue off-diagonals negative.
    assert m[1, 2] < 0 and m[2, 1] < 0
    assert not dlg.result_label.isHidden()


def test_compute_reports_error_on_bad_capture(monkeypatch):
    from negpy.desktop.view.widgets.scanner_calibration_dialog import ScannerCalibrationDialog
    from negpy.domain.models import WorkspaceConfig

    # A zero red-channel capture -> build_sensor_matrix raises -> surfaced, not saved.
    def fake_decode(self, path, params, fast=True):
        rgb = {"R": (0.0, 0.1, 0.03), "G": (0.05, 0.5, 0.15), "B": (0.04, 0.3, 1.0)}[path]
        return np.tile(np.array(rgb, dtype=np.float32), (16, 16, 1))

    monkeypatch.setattr("negpy.services.rendering.image_processor.ImageProcessor.decode_source_negative", fake_decode)
    dlg = ScannerCalibrationDialog(WorkspaceConfig())
    dlg._paths = {"R": "R", "G": "G", "B": "B"}
    dlg.name_edit.setText("Bad")
    dlg._compute_and_save()
    assert dlg.saved_profile_name is None
    assert "Could not build" in dlg.result_label.text()
