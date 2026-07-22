"""Calibrate the scanner (sensor + light) crosstalk from three bare-light exposures.

Point it at red-only / green-only / blue-only captures (no film in the holder, same
settings you scan with); it decodes each raw, measures the central mean RGB, builds the
correction matrix, and saves it as a named scanner profile the Process panel can select.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.features.process.scanner import build_sensor_matrix, measure_capture
from negpy.services.assets.scanner import ScannerProfiles

_BANDS = (("R", "Red exposure"), ("G", "Green exposure"), ("B", "Blue exposure"))
_CLIP = 0.98  # a channel at/above this is treated as saturated
_CLIP_FRACTION = 0.02


class ScannerCalibrationDialog(QDialog):
    """Pick three bare-light exposures, compute the sensor-crosstalk matrix, save a profile."""

    def __init__(self, config, parent=None) -> None:
        super().__init__(parent)
        self._config = config  # decode settings (linear RAW etc.) match how scans are decoded
        self.saved_profile_name: Optional[str] = None
        self._paths = {"R": "", "G": "", "B": ""}
        self.setWindowTitle("Calibrate Scanner")
        self.resize(580, 340)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        intro = QLabel(
            "Point at three <b>bare-light</b> exposures — red-only, green-only, blue-only, with no film in the "
            "holder and the same ISO/aperture/light you scan with. NegPy measures each and builds the "
            "sensor-crosstalk correction. Avoid clipping the bright channel."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME.text_secondary};")
        root.addWidget(intro)

        grid = QGridLayout()
        self._path_edits = {}
        for i, (band, label) in enumerate(_BANDS):
            grid.addWidget(QLabel(label), i, 0)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("Choose a capture…")
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _c=False, b=band: self._browse(b))
            grid.addWidget(edit, i, 1)
            grid.addWidget(browse, i, 2)
            self._path_edits[band] = edit
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. My scanner (narrowband v4)")
        self.name_edit.textChanged.connect(self._refresh)
        name_row.addWidget(self.name_edit, 1)
        root.addLayout(name_row)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setVisible(False)
        self.result_label.setStyleSheet(
            f"background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); "
            f"border-radius: 6px; padding: 8px; color: {THEME.text_secondary};"
        )
        root.addWidget(self.result_label)
        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        self.compute_btn = QPushButton("Compute and Save")
        self.compute_btn.setDefault(True)
        self.compute_btn.clicked.connect(self._compute_and_save)
        btn_row.addWidget(close)
        btn_row.addWidget(self.compute_btn)
        root.addLayout(btn_row)
        self._refresh()

    def _browse(self, band: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Choose the {band} exposure", "", "Raw / images (*.nef *.NEF *.dng *.raf *.arw *.cr2 *.cr3 *.tif *.tiff);;All files (*)"
        )
        if path:
            self._paths[band] = path
            self._path_edits[band].setText(path)
            self._refresh()

    def _refresh(self) -> None:
        name = self.name_edit.text().strip()
        ready = all(self._paths.values()) and bool(name) and not ScannerProfiles.is_bundled(name)
        self.compute_btn.setEnabled(ready)

    def _compute_and_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or not all(self._paths.values()):
            return
        from negpy.services.rendering.image_processor import ImageProcessor

        ip = ImageProcessor()
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            measured = {}
            clipped = []
            for band in ("R", "G", "B"):
                img = ip.decode_source_negative(self._paths[band], self._config, fast=True)
                measured[band] = measure_capture(img)
                frac = float(np.mean(np.any(np.asarray(img[:, :, :3]) >= _CLIP, axis=2)))
                if frac > _CLIP_FRACTION:
                    clipped.append(band)
            matrix = build_sensor_matrix(measured["R"], measured["G"], measured["B"])
        except Exception as exc:
            self._show_result(f"Could not build the matrix: {exc}", error=True)
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        ScannerProfiles.save(name, list(matrix))
        self.saved_profile_name = name
        self._show_result(self._summary(name, measured, clipped))

    def _summary(self, name: str, measured: dict, clipped: list) -> str:
        s = np.column_stack([measured["R"], measured["G"], measured["B"]])
        s_norm = s / np.diag(s)  # leakage fractions (own channel = 1)
        gb, bg = s_norm[2, 1], s_norm[1, 2]  # green->blue, blue->green (the usual dominant terms)
        lines = [
            f"<b>Saved “{name}”.</b> Close to apply it in the Scanner dropdown.",
            f"<span style='color:{THEME.text_muted}'>Measured green↔blue leakage: {gb * 100:.0f}% / {bg * 100:.0f}%.</span>",
        ]
        if clipped:
            lines.append(
                f"<span style='color:#D9A441'>⚠ {'/'.join(clipped)} exposure(s) look clipped — re-shoot dimmer "
                "for a more accurate matrix.</span>"
            )
        return "<br>".join(lines)

    def _show_result(self, text: str, error: bool = False) -> None:
        self.result_label.setText(text)
        self.result_label.setVisible(True)
