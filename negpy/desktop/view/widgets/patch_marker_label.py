"""A QLabel that shows a frame and lets the user drag out labelled patch boxes.

Used by the chart-calibration dialog: the user drags a rectangle over each colour
patch and tags it with a role (R/G/B/C/M/Y or a neutral). Rects are stored as
full-frame fractions (x, y, w, h) so they map onto the decoded negative regardless
of display size. The role labels/colours are owned here only for painting; the
dialog holds the authoritative patch model and role assignment.
"""

from typing import List, Optional, Tuple

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

# Outline colour per role; neutrals share a grey. Kept local — purely cosmetic.
ROLE_COLORS = {
    "R": "#E4572E",
    "G": "#3FA34D",
    "B": "#3B7DD8",
    "C": "#2CB5B0",
    "M": "#C64B8C",
    "Y": "#D9B23A",
    "neutral": "#9A988F",
}

_MIN_DRAG = 6  # px: a drag shorter than this is ignored (stray click, not a box)

Rect = Tuple[float, float, float, float]  # (x, y, w, h) in full-frame fractions


class PatchMarkerLabel(QLabel):
    """Shows a frame; drag to add a patch box, click a box to select it."""

    patchAdded = pyqtSignal(float, float, float, float)  # (x, y, w, h) fractions
    patchSelected = pyqtSignal(int)  # index into the patch list, or -1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap: Optional[QPixmap] = None
        self._patches: List[Tuple[str, Rect]] = []  # (role, rect)
        self._selected = -1
        self._drag_start: Optional[QPoint] = None
        self._drag_now: Optional[QPoint] = None

    # ── public API ────────────────────────────────────────────────────

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_patches(self, patches: List[Tuple[str, Rect]]) -> None:
        self._patches = list(patches)
        if self._selected >= len(self._patches):
            self._selected = -1
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = index if 0 <= index < len(self._patches) else -1
        self.update()

    # ── geometry (letterboxed frame; fractions map to displayed rect) ──

    def _display(self) -> Optional[QRect]:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return None
        scale = min(self.width() / pw, self.height() / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    @staticmethod
    def _to_fraction(p: QPoint, draw_rect: QRect) -> Tuple[float, float]:
        fx = min(1.0, max(0.0, (p.x() - draw_rect.x()) / max(1, draw_rect.width())))
        fy = min(1.0, max(0.0, (p.y() - draw_rect.y()) / max(1, draw_rect.height())))
        return fx, fy

    @staticmethod
    def _rect_in_widget(rect: Rect, draw_rect: QRect) -> QRect:
        x, y, w, h = rect
        x0 = draw_rect.x() + int(x * draw_rect.width())
        y0 = draw_rect.y() + int(y * draw_rect.height())
        x1 = draw_rect.x() + int((x + w) * draw_rect.width())
        y1 = draw_rect.y() + int((y + h) * draw_rect.height())
        return QRect(QPoint(x0, y0), QPoint(x1, y1)).normalized()

    def _patch_at(self, p: QPoint, draw_rect: QRect) -> int:
        # Topmost patch (last drawn) whose box contains the point.
        for i in range(len(self._patches) - 1, -1, -1):
            if self._rect_in_widget(self._patches[i][1], draw_rect).contains(p):
                return i
        return -1

    # ── mouse ─────────────────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if self._display() is not None:
            self._drag_start = ev.pos()
            self._drag_now = ev.pos()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag_start is not None:
            self._drag_now = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        draw_rect = self._display()
        start = self._drag_start
        self._drag_start = self._drag_now = None
        if start is None or draw_rect is None:
            return
        if (ev.pos() - start).manhattanLength() < _MIN_DRAG:
            self.patchSelected.emit(self._patch_at(ev.pos(), draw_rect))  # click = select/deselect
            self.update()
            return
        x0, y0 = self._to_fraction(start, draw_rect)
        x1, y1 = self._to_fraction(ev.pos(), draw_rect)
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self.patchAdded.emit(x, y, w, h)

    # ── paint ─────────────────────────────────────────────────────────

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        draw_rect = self._display()
        if draw_rect is None:
            painter.fillRect(self.rect(), QColor("#0D0D0F"))
            painter.setPen(QColor("#888780"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame loaded")
            painter.end()
            return

        painter.drawPixmap(draw_rect, self._pixmap)
        for i, (role, rect) in enumerate(self._patches):
            box = self._rect_in_widget(rect, draw_rect)
            color = QColor(ROLE_COLORS.get(role, ROLE_COLORS["neutral"]))
            painter.setPen(QPen(color, 3 if i == self._selected else 2))
            painter.drawRect(box)
            painter.fillRect(QRect(box.left(), box.top() - 16, 22, 16), color)
            painter.setPen(QColor("#0D0D0F"))
            painter.drawText(QRect(box.left(), box.top() - 16, 22, 16), Qt.AlignmentFlag.AlignCenter, role)

        if self._drag_start is not None and self._drag_now is not None:
            painter.setPen(QPen(QColor("#1D9E75"), 1, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self._drag_start, self._drag_now).normalized())
        painter.end()
