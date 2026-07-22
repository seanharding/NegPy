import os

import numpy as np
import pytest

from negpy.kernel.system.config import APP_CONFIG
from negpy.features.process.calibration import CANONICAL_CHROMA, CHROMA_ROLES

pytestmark = pytest.mark.usefixtures("qapp")

_GRAY = np.ones(3) / np.sqrt(3.0)


@pytest.fixture(autouse=True)
def _isolate_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))


def _chart_negative(a=None):
    """Synthetic linear negative: a horizontal strip of R/G/B/C/M/Y patches."""
    a = np.eye(3) if a is None else a
    n = len(CHROMA_ROLES)
    img = np.zeros((20, 20 * n, 3), dtype=np.float32)
    rects = []
    for i, r in enumerate(CHROMA_ROLES):
        d = a @ (1.0 * _GRAY + 0.3 * CANONICAL_CHROMA[r])
        img[:, i * 20 : (i + 1) * 20, :] = np.power(10.0, -d).astype(np.float32)
        rects.append(((i + 0.25) / n, 0.25, 0.5 / n, 0.5))  # (x, y, w, h)
    return img, rects


def test_negative_to_pixmap_valid():
    from negpy.desktop.view.widgets.chart_calibration_dialog import negative_to_pixmap

    img, _ = _chart_negative()
    pm = negative_to_pixmap(img)
    assert not pm.isNull()
    assert pm.width() > 0 and pm.height() > 0


def _frame(img, preview=None):
    from negpy.desktop.view.widgets.chart_calibration_dialog import CalibrationFrame
    from negpy.domain.models import WorkspaceConfig

    return CalibrationFrame(negative=img, base_config=WorkspaceConfig(), source_hash="test", preview_rgb=preview)


def test_rgb_to_pixmap_and_preview_path():
    from negpy.desktop.view.widgets.chart_calibration_dialog import ChartCalibrationDialog, rgb_to_pixmap

    preview = np.zeros((30, 60, 3), dtype=np.uint8)
    preview[:, :, 0] = 200
    pm = rgb_to_pixmap(preview)
    assert not pm.isNull() and pm.width() == 60 and pm.height() == 30
    # Dialog uses the supplied preview for display and keeps the negative for the worker.
    img, _ = _chart_negative()
    dlg = ChartCalibrationDialog(lambda: _frame(img, preview))
    assert dlg._frame.negative is img


def test_dialog_marks_and_save(tmp_path):
    from negpy.desktop.view.widgets.chart_calibration_dialog import ChartCalibrationDialog
    from negpy.services.assets.crosstalk import CrosstalkProfiles

    img, rects = _chart_negative()
    dlg = ChartCalibrationDialog(lambda: _frame(img))

    # Patches are auto-assigned R,G,B,C,M,Y in order — matches the strip layout.
    for x, y, w, h in rects:
        dlg._on_patch_added(x, y, w, h)
    assert [p[0] for p in dlg._patches] == list(CHROMA_ROLES)

    # Simulate the optimizer worker finishing (the real run needs the pipeline + a thread).
    dlg._on_opt_finished((1.0, -0.1, 0.05, 0.0, 1.0, -0.1, 0.0, -0.05, 1.0), 5.3, ())
    assert dlg._matrix is not None
    assert not dlg._optimizing and dlg.solve_btn.text() == "Solve"

    dlg.name_edit.setText("Chart Test")
    assert dlg.save_btn.isEnabled()
    dlg._on_save()
    assert dlg.saved_profile_name == "Chart Test"
    assert os.path.exists(os.path.join(tmp_path, "chart_test.toml"))
    assert "Chart Test" in CrosstalkProfiles.list_profiles()


def test_dialog_handles_no_frame():
    from negpy.desktop.view.widgets.chart_calibration_dialog import ChartCalibrationDialog

    dlg = ChartCalibrationDialog(lambda: None)
    assert not dlg.solve_btn.isEnabled()
    assert not dlg.result_label.isHidden()  # message shown (top-level unshown, so isVisible() is False)


def test_editor_calibrate_button_gated_on_provider():
    from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

    without = CrosstalkEditorDialog("Default", 0.0)
    assert not without.calibrate_btn.isEnabled()

    with_provider = CrosstalkEditorDialog("Default", 0.0, negative_provider=lambda: None)
    assert with_provider.calibrate_btn.isEnabled()
