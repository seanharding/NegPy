import numpy as np
import pytest

from negpy.features.process.scanner import apply_sensor_correction, build_sensor_matrix, measure_capture


def test_measure_capture_center():
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[:, :, 0] = 0.8
    img[:, :, 1] = 0.3
    img[:, :, 2] = 0.05
    img[:10, :, :] = 5.0  # bright edge that a centre crop must ignore
    assert measure_capture(img) == pytest.approx((0.8, 0.3, 0.05), abs=1e-5)


def test_build_matrix_identity_captures():
    # Perfectly separated bands (no leakage) -> identity correction.
    m = np.array(build_sensor_matrix((1, 0, 0), (0, 1, 0), (0, 0, 1))).reshape(3, 3)
    assert np.allclose(m, np.eye(3), atol=1e-9)


def test_build_matrix_recovers_inverse_of_leakage():
    # Known sensor mixing S (columns = band responses, own-channel scaled arbitrarily to
    # exercise white-balance normalization). Correction should invert its normalized form.
    rgb_r = (0.9, 0.1, 0.03)
    rgb_g = (0.05, 0.5, 0.15)  # green leaks 30% into blue after own-channel normalization
    rgb_b = (0.04, 0.3, 1.0)
    m = np.array(build_sensor_matrix(rgb_r, rgb_g, rgb_b)).reshape(3, 3)
    s_norm = np.column_stack([rgb_r, rgb_g, rgb_b]) / np.diag(np.column_stack([rgb_r, rgb_g, rgb_b]))
    assert np.allclose(m @ s_norm, np.eye(3), atol=1e-9)  # correction · mixing = identity
    assert np.allclose(np.diag(m @ s_norm), 1.0)


def test_build_matrix_rejects_zero_and_singular():
    with pytest.raises(ValueError):
        build_sensor_matrix((0, 0.1, 0.03), (0.05, 0.5, 0.15), (0.04, 0.3, 1.0))  # red own-channel ~0
    with pytest.raises(ValueError):
        build_sensor_matrix((1, 1, 1), (1, 1, 1), (1, 1, 1))  # collinear


def test_apply_correction_none_is_passthrough():
    img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert apply_sensor_correction(img, None) is img


def test_apply_correction_unmixes_and_clips():
    # A capture built by mixing a clean signal with S should be recovered by C = S_norm^-1.
    rgb_r, rgb_g, rgb_b = (0.9, 0.1, 0.03), (0.05, 0.5, 0.15), (0.04, 0.3, 1.0)
    s_norm = np.column_stack([rgb_r, rgb_g, rgb_b]) / np.diag(np.column_stack([rgb_r, rgb_g, rgb_b]))
    clean = np.array([[0.6, 0.2, 0.1]], dtype=np.float32).reshape(1, 1, 3)
    mixed = np.einsum("ij,hwj->hwi", s_norm.astype(np.float32), clean)
    matrix = build_sensor_matrix(rgb_r, rgb_g, rgb_b)
    recovered = apply_sensor_correction(mixed, matrix)
    assert np.allclose(recovered[0, 0], clean[0, 0], atol=1e-5)
    assert np.all(recovered >= 0.0)  # never negative
