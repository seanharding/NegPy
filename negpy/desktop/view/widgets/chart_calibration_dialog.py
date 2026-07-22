"""Derive a crosstalk profile from a photographed colour chart.

The user marks each colour patch on the current frame and tags it with a role; a
background worker then optimizes the crosstalk matrix so the *rendered* patches best
match their reference chroma (`chart_optimize` + `workers/calibrate`), and the result
is saved as a normal crosstalk profile. The frame shown is a real pipeline render and
the marked negative is the pre-crosstalk buffer it's rendered from, so boxes map 1:1.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.patch_marker_label import ROLE_COLORS, PatchMarkerLabel
from negpy.desktop.workers.calibrate import CalibrateTask, CrosstalkCalibrateWorker
from negpy.features.process.calibration import calibrate_from_marks
from negpy.features.process.chart_optimize import OFF_DIAGONAL
from negpy.services.assets.crosstalk import CrosstalkProfiles

Rect = Tuple[float, float, float, float]
_ROLES = ["R", "G", "B", "C", "M", "Y", "neutral"]
_MAX_DISPLAY = 1200  # longest edge of the display preview; sampling uses the full buffer


@dataclass
class CalibrationFrame:
    """What the caller hands the dialog: a canvas-matching preview to mark on, plus the
    pre-crosstalk negative and the (geometry-disabled) config the worker renders with."""

    negative: np.ndarray
    base_config: object  # WorkspaceConfig with geometry disabled + flatfield off
    source_hash: str
    preview_rgb: Optional[np.ndarray] = None


def rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """QPixmap from a contiguous (H, W, 3) uint8 sRGB array (a pipeline-rendered positive)."""
    buf = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    h, w = buf.shape[:2]
    img = QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def negative_to_pixmap(negative: np.ndarray) -> QPixmap:
    """Cheap display positive of a linear negative so patches read in their real
    colours. Density itself is the positive (a red patch reads as high red density).
    White-balance by a per-channel *offset* (removes the film cast) then stretch all
    channels by one shared range — independent per-channel scaling would shift hues
    (yellow toward green). Display-only; sampling reads the raw negative buffer."""
    neg = np.asarray(negative[:, :, :3], dtype=np.float32)
    longest = max(neg.shape[0], neg.shape[1])
    if longest > _MAX_DISPLAY:
        step = int(np.ceil(longest / _MAX_DISPLAY))
        neg = neg[::step, ::step]
    d = -np.log10(np.clip(neg, 1e-6, None))
    # Stats from a centre crop only — film rebate / bright surround at the edges would
    # skew them and wash the patches out (the pipeline likewise meters a centred region).
    h, w = d.shape[:2]
    cy, cx = int(h * 0.2), int(w * 0.2)
    centre = d[cy : h - cy, cx : w - cx].reshape(-1, 3)
    if not centre.size:
        centre = d.reshape(-1, 3)
    cast = np.median(centre, axis=0)  # per-channel cast (offset, not scale)
    dwb = d - (cast - cast.mean())
    lo, hi = np.percentile(centre - (cast - cast.mean()), (2.0, 98.0))  # one shared range
    dn = (dwb - lo) / max(float(hi - lo), 1e-6)
    # Hue-preserving saturation boost (scale chroma about per-pixel gray) so patches
    # read vividly for identification without the hue shift of per-channel scaling.
    gray = dn.mean(axis=2, keepdims=True)
    dn = gray + 1.6 * (dn - gray)
    pos = np.clip(dn, 0.0, 1.0) ** (1.0 / 2.2)
    buf = np.ascontiguousarray((pos * 255).astype(np.uint8))
    h, w = buf.shape[:2]
    img = QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())  # copy: detach from the temporary numpy buffer


def _role_icon(role: str) -> QIcon:
    """A small colour swatch for a role — the same colour its box gets on the image."""
    pm = QPixmap(14, 14)
    pm.fill(QColor(ROLE_COLORS.get(role, ROLE_COLORS["neutral"])))
    return QIcon(pm)


def _role_label(role: str) -> str:
    names = {"R": "Red", "G": "Green", "B": "Blue", "C": "Cyan", "M": "Magenta", "Y": "Yellow"}
    return f"{role} — {names[role]}" if role in names else "Neutral"


class ChartCalibrationDialog(QDialog):
    """Mark chart patches, solve, and save the result as a crosstalk profile."""

    def __init__(self, negative_provider: Callable[[], Optional["CalibrationFrame"]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate Crosstalk from Chart")
        self.resize(940, 640)
        self.setMinimumSize(760, 520)

        self._patches: List[List] = []  # [role, rect]
        self._matrix: Optional[Tuple[float, ...]] = None
        self.saved_profile_name: Optional[str] = None
        self._frame: Optional[CalibrationFrame] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[CrosstalkCalibrateWorker] = None
        self._optimizing = False

        self._init_ui()
        self._load_frame(negative_provider)

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.marker = PatchMarkerLabel()
        self.marker.patchAdded.connect(self._on_patch_added)
        self.marker.patchSelected.connect(self._on_patch_selected)
        root.addWidget(self.marker, 3)

        side = QWidget()
        side.setMaximumWidth(300)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        intro = QLabel(
            "Drag a box over each colour patch, then tag it. Add the six SpyderCheckr primaries "
            "(R G B C M Y); black/white/grey help. Solve optimizes the matrix against the rendered "
            "result — it takes a minute."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME.text_secondary};")
        sl.addWidget(intro)

        self.patch_list = QListWidget()
        self.patch_list.currentRowChanged.connect(self._on_row_changed)
        sl.addWidget(self.patch_list, 1)

        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("Role"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(_ROLES)
        self.role_combo.currentTextChanged.connect(self._on_role_combo_changed)
        self.delete_btn = QPushButton("Remove")
        self.delete_btn.clicked.connect(self._on_delete)
        edit_row.addWidget(self.role_combo, 1)
        edit_row.addWidget(self.delete_btn)
        sl.addLayout(edit_row)

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._on_solve_clicked)
        sl.addWidget(self.solve_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate "busy" — solve time isn't known ahead
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        sl.addWidget(self.progress_bar)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            f"background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); "
            f"border-radius: 6px; padding: 8px; color: {THEME.text_secondary};"
        )
        self.result_label.setVisible(False)
        sl.addWidget(self.result_label)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Portra 400 (my scanner)")
        self.name_edit.textChanged.connect(self._refresh_save_enabled)
        name_row.addWidget(self.name_edit, 1)
        sl.addLayout(name_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.save_btn = QPushButton("Save profile")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(cancel)
        btn_row.addWidget(self.save_btn)
        sl.addLayout(btn_row)

        root.addWidget(side)
        self._refresh_save_enabled()

    def _load_frame(self, provider: Callable[[], Optional["CalibrationFrame"]]) -> None:
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            frame = provider()
        except Exception:  # decode failure shouldn't crash the editor
            frame = None
        finally:
            QGuiApplication.restoreOverrideCursor()
        if frame is None:
            self.marker.setEnabled(False)
            self.solve_btn.setEnabled(False)
            self.result_label.setText("Could not load the current frame. Open a chart photo first.")
            self.result_label.setVisible(True)
            return
        self._frame = frame
        # Preview is a pipeline render matching the canvas; fall back to a raw positive.
        self.marker.set_frame(rgb_to_pixmap(frame.preview_rgb) if frame.preview_rgb is not None else negative_to_pixmap(frame.negative))

    # --------------------------------------------------------------- model

    def _next_role(self) -> str:
        used = [p[0] for p in self._patches]
        for r in _ROLES[:6]:
            if r not in used:
                return r
        return "neutral"

    def _on_patch_added(self, x: float, y: float, w: float, h: float) -> None:
        self._patches.append([self._next_role(), (x, y, w, h)])
        self._invalidate_result()
        self._rebuild_list(select=len(self._patches) - 1)

    def _on_patch_selected(self, index: int) -> None:
        self.patch_list.setCurrentRow(index)

    def _on_row_changed(self, row: int) -> None:
        self.marker.set_selected(row)
        valid = 0 <= row < len(self._patches)
        self.delete_btn.setEnabled(valid)
        self.role_combo.setEnabled(valid)
        if valid:
            self.role_combo.blockSignals(True)  # programmatic sync, not a user edit
            self.role_combo.setCurrentText(self._patches[row][0])
            self.role_combo.blockSignals(False)

    def _on_role_combo_changed(self, role: str) -> None:
        row = self.patch_list.currentRow()
        if 0 <= row < len(self._patches) and self._patches[row][0] != role:
            self._patches[row][0] = role
            self._invalidate_result()
            self._rebuild_list(select=row)

    def _on_delete(self) -> None:
        row = self.patch_list.currentRow()
        if 0 <= row < len(self._patches):
            del self._patches[row]
            self._invalidate_result()
            self._rebuild_list(select=min(row, len(self._patches) - 1))

    def _rebuild_list(self, select: int = -1) -> None:
        self.patch_list.blockSignals(True)
        self.patch_list.clear()
        for role, _rect in self._patches:
            self.patch_list.addItem(QListWidgetItem(_role_icon(role), _role_label(role)))
        self.patch_list.blockSignals(False)
        self.marker.set_patches([(p[0], p[1]) for p in self._patches])
        if 0 <= select < len(self._patches):
            self.patch_list.setCurrentRow(select)
        else:
            self._on_row_changed(self.patch_list.currentRow())

    # --------------------------------------------------------------- solve

    def _invalidate_result(self) -> None:
        self._matrix = None
        self.result_label.setVisible(False)
        self._refresh_save_enabled()

    def _on_solve_clicked(self) -> None:
        if self._optimizing:
            if self._worker is not None:
                self._worker.cancel()
            self.solve_btn.setEnabled(False)
            self.result_label.setText("Cancelling…")
        else:
            self._start_optimization()

    def _start_optimization(self) -> None:
        if self._frame is None or not self._patches or self._optimizing:
            return
        roles = tuple(p[0] for p in self._patches)
        rects = tuple((x, y, x + w, y + h) for _, (x, y, w, h) in self._patches)
        # Warm-start from the instant density fit so the search needs fewer renders.
        try:
            dm = np.array(calibrate_from_marks(self._frame.negative, list(zip(roles, rects))).matrix).reshape(3, 3)
            init = tuple(float(dm[i, j] / dm[i, i]) for i, j in OFF_DIAGONAL)
        except Exception:
            init = ()
        task = CalibrateTask(self._frame.negative, self._frame.base_config, self._frame.source_hash, roles, rects, init)

        self._thread = QThread()
        self._worker = CrosstalkCalibrateWorker()
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._on_opt_progress)
        self._worker.finished.connect(self._on_opt_finished)
        self._worker.error.connect(self._on_opt_error)
        self._thread.started.connect(lambda: self._worker.run(task))

        self._optimizing = True
        self._matrix = None
        self.solve_btn.setText("Cancel")
        self._set_busy(True)
        self.progress_bar.setVisible(True)
        self.result_label.setText(
            "Solving… rendering the chart repeatedly to tune the matrix (about a minute).<br>"
            "<span style='color:%s'>The window may be sluggish while this runs — that's expected.</span>" % THEME.text_muted
        )
        self.result_label.setVisible(True)
        # Force the busy state to paint *now*: once the worker starts, its GIL-holding
        # renders would otherwise stop the main thread from ever drawing it, making the
        # click look ignored.
        QApplication.processEvents()
        self._thread.start()

    def _on_opt_progress(self, evals: int, best: float) -> None:
        self.result_label.setText(f"Optimizing… best colour error {best:.1f} so far ({evals} renders)")

    def _on_opt_finished(self, matrix: object, error: float, warnings: tuple) -> None:
        self._teardown_thread()
        self._matrix = tuple(matrix)  # type: ignore[arg-type]
        lines = [f"<b>Rendered chroma error:</b> {error:.1f} (lower is better)"]
        if warnings:
            lines.append("<b>Warnings:</b><br>• " + "<br>• ".join(warnings))
        else:
            lines.append("Optimized against the rendered result — save it and raise Separation to apply.")
        self.result_label.setText("<br>".join(lines))
        self._refresh_save_enabled()

    def _on_opt_error(self, message: str) -> None:
        self._teardown_thread()
        self.result_label.setText(f"Calibration failed: {message}")

    def _teardown_thread(self) -> None:
        self._optimizing = False
        self.solve_btn.setText("Solve")
        self.solve_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._set_busy(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            if self._worker is not None:
                self._worker.deleteLater()
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _set_busy(self, busy: bool) -> None:
        """Freeze editing while the optimizer runs (marks feed the in-flight task)."""
        for w in (self.marker, self.patch_list, self.delete_btn, self.role_combo, self.name_edit):
            w.setEnabled(not busy)
        if busy:
            self.save_btn.setEnabled(False)

    def _refresh_save_enabled(self) -> None:
        name = self.name_edit.text().strip()
        ok = bool(name) and self._matrix is not None and not CrosstalkProfiles.is_bundled(name)
        self.save_btn.setEnabled(ok)

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or self._matrix is None or CrosstalkProfiles.is_bundled(name):
            return
        if name in CrosstalkProfiles.list_profiles():
            if QMessageBox.question(self, "Overwrite?", f"Replace the existing profile “{name}”?") != QMessageBox.StandardButton.Yes:
                return
        CrosstalkProfiles.save(name, list(self._matrix))
        self.saved_profile_name = name
        self.accept()

    def reject(self) -> None:
        self._cancel_if_running()
        super().reject()

    def closeEvent(self, event) -> None:
        self._cancel_if_running()
        super().closeEvent(event)

    def _cancel_if_running(self) -> None:
        if self._optimizing and self._worker is not None:
            self._worker.cancel()
            self._teardown_thread()
