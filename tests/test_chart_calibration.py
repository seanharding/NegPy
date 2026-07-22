import numpy as np
import pytest

from negpy.features.exposure.normalization import resolve_crosstalk_matrix
from negpy.features.process.calibration import (
    CANONICAL_CHROMA,
    CHROMA_ROLES,
    PatchSample,
    calibrate_from_marks,
    sample_patch_density,
    solve_crosstalk_matrix,
)

_GRAY = np.ones(3) / np.sqrt(3.0)


def _off_gray(d):
    d = np.asarray(d, dtype=np.float64)
    return d - (d @ _GRAY) * _GRAY


def _ideal_density(role, luma=1.0, chroma=0.3):
    """A clean patch density: gray/luminance baseline + ideal hue chroma."""
    return luma * _GRAY + chroma * CANONICAL_CHROMA[role]


def _angular_residual(matrix, samples):
    """Mean angular error (deg) between each corrected chroma and its ideal direction."""
    m = np.array(matrix, dtype=np.float64).reshape(3, 3)
    errs = []
    for s in samples:
        out = _off_gray(m @ np.array(s.density))
        out /= np.linalg.norm(out)
        cos = np.clip(out @ CANONICAL_CHROMA[s.role], -1.0, 1.0)
        errs.append(np.degrees(np.arccos(cos)))
    return float(np.mean(errs))


def _samples_from_densities(densities):
    return [PatchSample(role=r, density=tuple(d)) for r, d in densities.items()]


# --- 1. Recovery: a known leakage is inverted from contaminated patches -------


def test_recovers_known_leakage():
    # Gray-preserving leakage (rows sum to 1): each dye leaks into its neighbours,
    # asymmetrically (green leaks strongly into the blue record).
    a = np.array([[0.82, 0.12, 0.06], [0.03, 0.92, 0.05], [0.02, 0.20, 0.78]])
    contaminated = {r: a @ _ideal_density(r) for r in CHROMA_ROLES}
    samples = _samples_from_densities(contaminated)

    result = solve_crosstalk_matrix(samples)
    m = np.array(result.matrix).reshape(3, 3)

    # The fit should substantially reduce the hue-crossing. (Per-patch-magnitude
    # targets preserve saturation rather than forcing an exact hexagon, so a purely
    # linear contamination isn't inverted to a perfect zero — it's roughly halved,
    # which is the right trade: on real charts it beats an exact-hexagon fit.)
    before = _angular_residual((1, 0, 0, 0, 1, 0, 0, 0, 1), samples)
    after = _angular_residual(result.matrix, samples)
    assert before > 1.0
    assert after < before * 0.6

    assert result.warnings == ()
    assert np.allclose(np.diag(m), 1.0, atol=0.1)
    assert np.allclose(m.sum(axis=1), 1.0)  # gray-preserving
    off = m[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() > 0.02  # a real correction, not near-identity


# --- 2. Clean input -> identity ----------------------------------------------


def test_clean_input_yields_identity():
    clean = {r: _ideal_density(r) for r in CHROMA_ROLES}
    result = solve_crosstalk_matrix(_samples_from_densities(clean))
    m = np.array(result.matrix).reshape(3, 3)
    assert np.allclose(m, np.eye(3), atol=1e-6)
    assert result.warnings == ()


def test_off_gray_cast_removed_via_centroid():
    # An orange mask is an off-gray chroma bias on every patch. With no neutrals
    # marked, the R/G/B/C/M/Y centroid estimates it — so a clean chart under a heavy
    # cast still solves to identity (bare off-gray projection would not).
    cast = _off_gray(np.array([0.25, 0.0, -0.25]))  # orange: +R, -B
    biased = {r: _ideal_density(r) + cast for r in CHROMA_ROLES}
    result = solve_crosstalk_matrix(_samples_from_densities(biased))
    m = np.array(result.matrix).reshape(3, 3)
    assert np.allclose(m, np.eye(3), atol=1e-6)
    assert result.mean_residual < 0.5


def test_neutral_patches_define_cast():
    cast = _off_gray(np.array([0.25, 0.0, -0.25]))
    samples = [PatchSample(r, tuple(_ideal_density(r) + cast)) for r in CHROMA_ROLES]
    # Neutral patches carry the cast (+ luminance) but no true chroma; at any
    # luminance they pin the same bias.
    samples.append(PatchSample("neutral", tuple(0.5 * _GRAY + cast)))
    samples.append(PatchSample("grey", tuple(1.2 * _GRAY + cast)))
    result = solve_crosstalk_matrix(samples)
    assert np.allclose(np.array(result.matrix).reshape(3, 3), np.eye(3), atol=1e-6)


def test_varying_saturation_is_preserved():
    # Patches at different saturations but ideal hues must map to identity: the fit
    # corrects hue only and never equalizes saturation. (A shared-magnitude target
    # would map these to a non-identity, saturation-flattening matrix.) A clean
    # neutral anchors zero cast so the centroid debias doesn't shift things.
    mags = {"R": 0.5, "G": 0.2, "B": 0.45, "C": 0.3, "M": 0.35, "Y": 0.25}
    samples = [PatchSample(r, tuple(1.0 * _GRAY + mags[r] * CANONICAL_CHROMA[r])) for r in CHROMA_ROLES]
    samples.append(PatchSample("neutral", tuple(1.0 * _GRAY)))
    m = np.array(solve_crosstalk_matrix(samples).matrix).reshape(3, 3)
    assert np.allclose(m, np.eye(3), atol=1e-6)


def test_physical_convention_clean_chart_is_identity():
    # Independent of CANONICAL_CHROMA: physically a red patch peaks in RED-channel
    # density (cyan dye), cyan in GREEN+BLUE, etc. Labelled by true colour, a clean
    # chart must need no correction. This pins the sign of the canonical directions
    # (a flipped convention would map each patch to its opposite and rotate 180°).
    rgb = {"R": (1, 0, 0), "G": (0, 1, 0), "B": (0, 0, 1), "C": (0, 1, 1), "M": (1, 0, 1), "Y": (1, 1, 0)}
    samples = []
    for role, v in rgb.items():
        chroma = _off_gray(np.array(v, dtype=float))
        chroma /= np.linalg.norm(chroma)
        samples.append(PatchSample(role, tuple(1.0 * _GRAY + 0.3 * chroma)))
    m = np.array(solve_crosstalk_matrix(samples).matrix).reshape(3, 3)
    assert np.allclose(m, np.eye(3), atol=1e-6)


# --- 3. Gray preservation through the real consumer --------------------------


def test_matrix_preserves_neutral_through_consumer():
    a = np.array([[0.92, 0.05, 0.03], [0.04, 0.91, 0.05], [0.02, 0.08, 0.90]])
    contaminated = {r: a @ _ideal_density(r) for r in CHROMA_ROLES}
    result = solve_crosstalk_matrix(_samples_from_densities(contaminated))

    applied = resolve_crosstalk_matrix(1.0, result.matrix)
    assert applied is not None
    neutral = np.array([0.7, 0.7, 0.7])
    assert np.allclose(applied @ neutral, neutral, atol=1e-5)


# --- 4. Robustness / warnings -------------------------------------------------


def test_missing_chroma_roles_warns():
    partial = {r: _ideal_density(r) for r in ("R", "G", "B")}
    result = solve_crosstalk_matrix(_samples_from_densities(partial))
    assert any("missing chroma" in w for w in result.warnings)


def test_collinear_input_returns_identity_with_warning():
    # Only an opposite pair (R and C are antiparallel) -> cannot span the plane.
    samples = [
        PatchSample("R", tuple(_ideal_density("R"))),
        PatchSample("C", tuple(_ideal_density("C"))),
    ]
    result = solve_crosstalk_matrix(samples)
    assert np.allclose(np.array(result.matrix).reshape(3, 3), np.eye(3))
    assert any("collinear" in w for w in result.warnings)


def test_clipped_patch_warns():
    clean = {r: _ideal_density(r) for r in CHROMA_ROLES}
    samples = _samples_from_densities(clean)
    samples[0] = PatchSample(samples[0].role, samples[0].density, clipped=True)
    result = solve_crosstalk_matrix(samples)
    assert any("clipped" in w for w in result.warnings)


# --- 5. Over-determined input lowers residual --------------------------------


def test_extra_samples_lower_residual():
    a = np.array([[0.90, 0.06, 0.04], [0.05, 0.90, 0.05], [0.03, 0.07, 0.90]])
    rng = np.random.default_rng(1234)
    noise = 0.02

    def noisy(role):
        return tuple(a @ _ideal_density(role) + rng.normal(0.0, noise, 3))

    single = [PatchSample(r, noisy(r)) for r in CHROMA_ROLES]
    many = [PatchSample(r, noisy(r)) for r in CHROMA_ROLES for _ in range(20)]

    r_single = solve_crosstalk_matrix(single)
    r_many = solve_crosstalk_matrix(many)

    clean = _samples_from_densities({r: a @ _ideal_density(r) for r in CHROMA_ROLES})
    assert _angular_residual(r_many.matrix, clean) < _angular_residual(r_single.matrix, clean)


# --- 6. sample_patch_density --------------------------------------------------


def test_sample_patch_density_uniform():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[:, :, 0] = 0.5
    img[:, :, 1] = 0.1
    img[:, :, 2] = 0.01
    d = sample_patch_density(img, (0.0, 0.0, 1.0, 1.0))
    assert d == pytest.approx((-np.log10(0.5), -np.log10(0.1), -np.log10(0.01)), abs=1e-5)


def test_sample_patch_density_respects_rect():
    img = np.full((10, 10, 3), 0.9, dtype=np.float32)
    img[:5, :5, :] = 0.2  # top-left quadrant darker
    d = sample_patch_density(img, (0.0, 0.0, 0.5, 0.5))
    assert d == pytest.approx((-np.log10(0.2),) * 3, abs=1e-5)


# --- 7. calibrate_from_marks (image -> marks -> matrix) -----------------------


def _chart_image(a, roles=CHROMA_ROLES):
    """Synthetic negative: one horizontal strip of patches, contaminated by leakage `a`."""
    n = len(roles)
    img = np.zeros((20, 20 * n, 3), dtype=np.float32)
    marks = []
    for i, r in enumerate(roles):
        lin = np.power(10.0, -(a @ _ideal_density(r)))  # density -> linear negative
        img[:, i * 20 : (i + 1) * 20, :] = lin.astype(np.float32)
        marks.append((r, ((i + 0.25) / n, 0.25, (i + 0.75) / n, 0.75)))
    return img, marks


def test_calibrate_from_marks_recovers():
    a = np.array([[0.82, 0.12, 0.06], [0.03, 0.92, 0.05], [0.02, 0.20, 0.78]])
    img, marks = _chart_image(a)
    result = calibrate_from_marks(img, marks)
    assert result.warnings == ()
    assert result.mean_residual < 1.5  # per-patch magnitude corrects hue without forcing an exact hexagon
    assert set(result.residuals) == set(CHROMA_ROLES)
    m = np.array(result.matrix).reshape(3, 3)
    assert np.allclose(m.sum(axis=1), 1.0)  # gray-preserving


def test_calibrate_from_marks_flags_clipped_patch():
    a = np.eye(3)
    img, marks = _chart_image(a)
    # Saturate the first patch (its region spans x in [0, 1/6)).
    img[:, :20, :] = 1.0
    result = calibrate_from_marks(img, marks)
    assert any("clipped" in w for w in result.warnings)
