"""The macOS bundle icon.

A missing .icns is silent: build.py falls back to the PNG, PyInstaller resamples one
bitmap into a partial ladder, and the Dock and Finder end up drawing different artwork.
"""

import os
import struct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ICNS = os.path.join(ROOT, "media", "icons", "icon.icns")


def _slots() -> dict[str, bytes]:
    with open(ICNS, "rb") as f:
        data = f.read()
    assert data[:4] == b"icns"
    slots = {}
    offset = 8
    while offset < len(data):
        kind = data[offset : offset + 4]
        length = struct.unpack(">i", data[offset + 4 : offset + 8])[0]
        slots[kind.decode("latin1")] = data[offset + 8 : offset + length]
        offset += length
    return slots


def test_icns_is_present() -> None:
    assert os.path.exists(ICNS), "run scripts/make_macos_icns.py"


def test_icns_covers_the_sizes_finder_and_the_dock_ask_for() -> None:
    slots = _slots()
    # ic04/ic05 are the 16 and 32 list-view sizes; ic10 is the 512@2x retina slot.
    for kind in ("ic04", "ic05", "ic07", "ic08", "ic09", "ic10"):
        assert kind in slots, f"{kind} missing from icon.icns"


def test_retina_slot_is_a_1024_render() -> None:
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(_slots()["ic10"]))
    assert image.size == (1024, 1024)
