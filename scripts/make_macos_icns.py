"""Build media/icons/icon.icns from icon.svg.

macOS reads the bundle icon from the .icns, and Qt pushes the window icon to
NSApp.applicationIconImage, which overrides the bundle icon on the Dock tile. Both
surfaces therefore need the same per-size artwork, rendered from the vector rather
than resampled from one bitmap.

Run on macOS after editing icon.svg, and commit the result:

    uv run python scripts/make_macos_icns.py
"""

import os
import subprocess
import sys
import tempfile

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QGuiApplication, QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer

# Apple's iconset ladder: (file name, pixel size). The @2x entries repeat a pixel size
# under a second name, and iconutil needs both to fill the retina slots.
ICONSET = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_SVG = os.path.join(ROOT, "media", "icons", "icon.svg")
TARGET_ICNS = os.path.join(ROOT, "media", "icons", "icon.icns")


def render(renderer: QSvgRenderer, size: int, path: str) -> None:
    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()
    if not image.save(path, "PNG"):
        raise RuntimeError(f"could not write {path}")


def main() -> int:
    if sys.platform != "darwin":
        print("iconutil is macOS-only — run this on a Mac.", file=sys.stderr)
        return 1
    if not os.path.exists(SOURCE_SVG):
        print(f"missing source {SOURCE_SVG}", file=sys.stderr)
        return 1

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QGuiApplication([])

    renderer = QSvgRenderer(SOURCE_SVG)
    if not renderer.isValid():
        print(f"could not parse {SOURCE_SVG}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as work:
        iconset = os.path.join(work, "icon.iconset")
        os.mkdir(iconset)
        for name, size in ICONSET:
            render(renderer, size, os.path.join(iconset, name))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", TARGET_ICNS], check=True)

    print(f"wrote {os.path.relpath(TARGET_ICNS, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
