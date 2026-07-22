"""Background worker for chart-based crosstalk calibration.

Runs the closed-loop optimizer (`chart_optimize.optimize_crosstalk`) off the UI thread:
each evaluation renders the marked chart through the real pipeline at reduced resolution
and measures patch chroma, so the optimizer minimizes the *rendered* colour error. Slow
(tens of seconds) by nature — hence a worker with progress + cancel.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.chart_optimize import optimize_crosstalk
from negpy.infrastructure.display.color_mgmt import apply_display_transform
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.kernel.image.logic import float_to_uint8
from negpy.services.rendering.image_processor import ImageProcessor

_OPT_RENDER_PX = 700.0  # optimize at reduced res (a matrix is resolution-independent) for speed
_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def _srgb_to_ab(rgb: np.ndarray) -> Tuple[float, float]:
    """(a*, b*) of an sRGB 0-255 triplet — the chroma the optimizer scores."""
    c = np.clip(np.asarray(rgb, float) / 255.0, 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    x, y, z = (_XYZ @ lin) / _D65

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    return (500.0 * (f(x) - f(y)), 200.0 * (f(y) - f(z)))


@dataclass(frozen=True)
class CalibrateTask:
    negative: np.ndarray  # geometry-applied, pre-crosstalk linear negative
    base_config: WorkspaceConfig  # geometry disabled + flatfield off (matches `negative`)
    source_hash: str
    roles: Tuple[str, ...]  # role per marked patch
    rects: Tuple[Tuple[float, float, float, float], ...]  # normalized (x0,y0,x1,y1) per patch
    init: Tuple[float, ...] = field(default_factory=tuple)  # optional off-diagonal warm start


class CrosstalkCalibrateWorker(QObject):
    """Optimizes the crosstalk matrix against the rendered chart; emits progress."""

    progress = pyqtSignal(int, float)  # evaluations, best chroma error
    finished = pyqtSignal(object, float, tuple)  # matrix (9 floats), error, warnings
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot(object)
    def run(self, task: CalibrateTask) -> None:
        self._cancel.clear()
        try:
            neg = task.negative
            stride = max(1, max(neg.shape[:2]) // int(_OPT_RENDER_PX))
            neg = np.ascontiguousarray(neg[::stride, ::stride])

            def render_ab(matrix9: tuple) -> dict:
                cfg = replace(
                    task.base_config,
                    process=replace(
                        task.base_config.process,
                        crosstalk_matrix=tuple(matrix9),
                        crosstalk_strength=1.0,
                        local_floors=(0.0, 0.0, 0.0),
                        local_ceils=(0.0, 0.0, 0.0),
                    ),
                )
                pos, _ = self._processor.run_pipeline(
                    neg,
                    cfg,
                    task.source_hash,
                    render_size_ref=_OPT_RENDER_PX,
                    prefer_gpu=False,
                    skip_flatfield=True,
                    readback_metrics=False,
                )
                img = float_to_uint8(apply_display_transform(np.asarray(pos)[:, :, :3], WORKING_COLOR_SPACE))
                h, w = img.shape[:2]
                out = {}
                for i, (x0, y0, x1, y1) in enumerate(task.rects):
                    cx, cy = int((x0 + x1) / 2 * w), int((y0 + y1) / 2 * h)
                    s = max(2, int(min(x1 - x0, y1 - y0) * min(w, h) * 0.25))
                    rgb = np.median(img[max(0, cy - s) : cy + s, max(0, cx - s) : cx + s, :3].reshape(-1, 3), axis=0)
                    out[i] = _srgb_to_ab(rgb)
                # The CPU render's njit kernels hold the GIL for the whole call; without a
                # yield between renders the UI thread's event loop starves and the OS shows a
                # beachball. This sleep hands the GIL back long enough for the main thread to
                # complete an event-loop pass (process events + repaint) so the dialog stays
                # responsive and its progress keeps updating.
                time.sleep(0.02)
                return out

            def on_progress(evals: int, best: float) -> bool:
                self.progress.emit(int(evals), float(best))
                return self._cancel.is_set()

            result = optimize_crosstalk(render_ab, task.roles, progress=on_progress, init=(task.init or None))
            self.finished.emit(result.matrix, result.error, result.warnings)
        except Exception as exc:  # a render/decode failure must surface, not hang the dialog
            self.error.emit(str(exc))
