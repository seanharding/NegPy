import numpy as np
import unittest


def get_pixel_rgb(buf, nx, ny):
    """Extracted logic from CanvasWidget.get_pixel_rgb for the numpy path."""
    if not isinstance(buf, np.ndarray):
        return None
    h, w = buf.shape[:2]
    x = int(max(0, min(w - 1, nx * w)))
    y = int(max(0, min(h - 1, ny * h)))
    px = buf[y, x]
    scale = 1.0 / 255.0 if buf.dtype == np.uint8 else 1.0
    px = np.atleast_1d(px)
    if px.shape[0] == 1:
        v = float(px[0]) * scale
        return (v, v, v)
    return (float(px[0]) * scale, float(px[1]) * scale, float(px[2]) * scale)


def to_qimage_expand(buffer: np.ndarray) -> np.ndarray:
    """Extracted monochrome-expansion logic from ImageConverter.to_qimage."""
    if buffer.dtype == np.float32:
        u8 = (np.clip(buffer, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        u8 = buffer
    if u8.ndim == 2 or (u8.ndim == 3 and u8.shape[2] == 1):
        u8 = np.stack([u8.squeeze()] * 3, axis=-1)
    if not u8.flags["C_CONTIGUOUS"]:
        u8 = np.ascontiguousarray(u8)
    return u8


class TestGetPixelRgb(unittest.TestCase):
    def test_rgb_float32(self):
        buf = np.zeros((10, 10, 3), dtype=np.float32)
        buf[5, 5] = [0.1, 0.5, 0.9]
        r, g, b = get_pixel_rgb(buf, 0.55, 0.55)
        self.assertAlmostEqual(r, 0.1, places=5)
        self.assertAlmostEqual(g, 0.5, places=5)
        self.assertAlmostEqual(b, 0.9, places=5)

    def test_rgb_uint8(self):
        buf = np.zeros((10, 10, 3), dtype=np.uint8)
        buf[5, 5] = [51, 128, 255]
        r, g, b = get_pixel_rgb(buf, 0.55, 0.55)
        self.assertAlmostEqual(r, 51 / 255.0, places=5)
        self.assertAlmostEqual(g, 128 / 255.0, places=5)
        self.assertAlmostEqual(b, 1.0, places=5)

    def test_monochrome_float32(self):
        """Monochrome DNG: 2D float array must not crash."""
        buf = np.zeros((10, 10), dtype=np.float32)
        buf[5, 5] = 0.7
        result = get_pixel_rgb(buf, 0.55, 0.55)
        self.assertIsNotNone(result)
        r, g, b = result
        self.assertAlmostEqual(r, 0.7, places=5)
        self.assertAlmostEqual(g, 0.7, places=5)
        self.assertAlmostEqual(b, 0.7, places=5)

    def test_monochrome_uint8(self):
        buf = np.zeros((10, 10), dtype=np.uint8)
        buf[5, 5] = 200
        r, g, b = get_pixel_rgb(buf, 0.55, 0.55)
        self.assertAlmostEqual(r, 200 / 255.0, places=5)
        self.assertAlmostEqual(g, 200 / 255.0, places=5)
        self.assertAlmostEqual(b, 200 / 255.0, places=5)

    def test_monochrome_single_channel_3d(self):
        buf = np.zeros((10, 10, 1), dtype=np.float32)
        buf[5, 5, 0] = 0.3
        r, g, b = get_pixel_rgb(buf, 0.55, 0.55)
        self.assertAlmostEqual(r, 0.3, places=5)
        self.assertAlmostEqual(g, 0.3, places=5)
        self.assertAlmostEqual(b, 0.3, places=5)

    def test_returns_none_for_non_array(self):
        self.assertIsNone(get_pixel_rgb("not an array", 0.5, 0.5))

    def test_clamps_coordinates(self):
        buf = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        buf[0, 0] = [0.1, 0.2, 0.3]
        r, g, b = get_pixel_rgb(buf, -1.0, -1.0)
        self.assertAlmostEqual(r, 0.1, places=5)


class TestToQimageExpand(unittest.TestCase):
    def test_rgb_passthrough(self):
        buf = np.zeros((8, 8, 3), dtype=np.uint8)
        out = to_qimage_expand(buf)
        self.assertEqual(out.shape, (8, 8, 3))

    def test_monochrome_2d_uint8_expands_to_rgb(self):
        """2D monochrome uint8 must become (H,W,3) — was causing segfault."""
        buf = np.full((8, 8), 128, dtype=np.uint8)
        out = to_qimage_expand(buf)
        self.assertEqual(out.shape, (8, 8, 3))
        self.assertTrue(np.all(out[:, :, 0] == 128))
        self.assertTrue(np.all(out[:, :, 1] == 128))
        self.assertTrue(np.all(out[:, :, 2] == 128))

    def test_monochrome_2d_float32_expands_to_rgb(self):
        """2D monochrome float32 DNG must become (H,W,3) uint8."""
        buf = np.full((8, 8), 0.5, dtype=np.float32)
        out = to_qimage_expand(buf)
        self.assertEqual(out.shape, (8, 8, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out[:, :, 0] == out[:, :, 1]))
        self.assertTrue(np.all(out[:, :, 1] == out[:, :, 2]))

    def test_monochrome_3d_single_channel_expands(self):
        buf = np.full((8, 8, 1), 200, dtype=np.uint8)
        out = to_qimage_expand(buf)
        self.assertEqual(out.shape, (8, 8, 3))
        self.assertTrue(np.all(out == 200))

    def test_output_is_contiguous(self):
        buf = np.full((8, 8), 100, dtype=np.uint8)
        out = to_qimage_expand(buf)
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_bytes_per_row_matches_w3(self):
        """Verify output stride matches w*3 — the QImage assumption."""
        buf = np.full((8, 12), 50, dtype=np.uint8)
        out = to_qimage_expand(buf)
        h, w = out.shape[:2]
        self.assertEqual(out.strides[0], w * 3)


if __name__ == "__main__":
    unittest.main()
