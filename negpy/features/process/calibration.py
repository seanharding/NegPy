"""
Chart-based spectral-crosstalk calibration (Phase 1: solver only).

Derives a NegPy crosstalk matrix from measured colour-chart patches instead of
from a datasheet. The matrix is a gray-preserving 3x3 linear map applied to raw
NEGATIVE densities (d = -log10(neg)); it corrects dye crosstalk only, so it
cleans/separates the primaries rather than colour-matching them.

Math: work in the plane orthogonal to the mathematical gray axis g = (1,1,1)/sqrt(3)
(NegPy row-normalizes the matrix, so only the off-gray action matters; the film's
orange/purple mask is a gray-axis offset owned by downstream cast removal). Each
labelled colour has an ideal off-gray density-chroma direction = the gray-removed RGB
of that colour (a red scene patch forms cyan dye -> high RED-channel density); R/G/B/
C/M/Y form a regular hexagon in that plane. Fit the gray-preserving
matrix whose plane-action maps each measured chroma onto its ideal direction (least
squares); adding g·gᵀ fixes the gray axis, so rows already sum to 1 (the profile
convention) with no rescale.

Output is a flat 9-float row-major matrix, ready for `CrosstalkProfiles.save` and
`resolve_crosstalk_matrix`. No pipeline or config changes here.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from negpy.domain.types import ImageBuffer

# Roles that carry a hue target.
CHROMA_ROLES = ("R", "G", "B", "C", "M", "Y")
# Neutral roles constrain the film's colour cast (mask) but carry no hue target.
NEUTRAL_ROLES = frozenset({"NEUTRAL", "BLACK", "WHITE", "GREY", "GRAY"})

_EPSILON = 1e-6
_GRAY = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
_HIGH_RESIDUAL_DEG = 10.0  # mean angular error above this -> the fit is a poor separation
_CLIP_LEVEL = 0.98  # linear-negative value at/above which a channel is treated as saturated
_CLIP_FRACTION = 0.2  # fraction of saturated pixels above which a patch is flagged clipped


def _canonical_chroma() -> dict[str, np.ndarray]:
    """Ideal off-gray density-chroma unit direction per chroma role (gray-removed RGB).

    A patch's negative density peaks in the channels matching its own colour: a red
    scene patch exposes the red layer → cyan dye → high RED-channel density; cyan
    (green+blue) → magenta+yellow dye → high GREEN+BLUE density. So the direction is
    the patch's own RGB (off-gray), not its complement.
    """
    rgb = {"R": (1, 0, 0), "G": (0, 1, 0), "B": (0, 0, 1), "C": (0, 1, 1), "M": (1, 0, 1), "Y": (1, 1, 0)}
    out: dict[str, np.ndarray] = {}
    for role, v in rgb.items():
        d = np.asarray(v, dtype=np.float64)
        d = d - (d @ _GRAY) * _GRAY  # project off the gray axis
        out[role] = d / np.linalg.norm(d)
    return out


CANONICAL_CHROMA = _canonical_chroma()


@dataclass(frozen=True)
class PatchSample:
    """One measured chart patch: its role and its raw negative density (-log10)."""

    role: str
    density: tuple[float, float, float]
    clipped: bool = False


@dataclass(frozen=True)
class CalibrationResult:
    """Solver output: a profile-ready matrix plus fit quality."""

    matrix: tuple[float, ...]  # 9 floats, row-major, gray-preserving (rows sum to 1)
    residuals: dict[str, float]  # per-chroma-role angular error (deg) after the fit
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mean_residual(self) -> float:
        return float(np.mean(list(self.residuals.values()))) if self.residuals else 0.0


def _crop_patch(raw_negative: ImageBuffer, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Flattened (N, 3) float64 pixels of a normalized-rect region (>= 1px each axis)."""
    h, w = raw_negative.shape[:2]
    x0, y0, x1, y1 = rect
    cx0, cx1 = sorted((int(round(x0 * w)), int(round(x1 * w))))
    cy0, cy1 = sorted((int(round(y0 * h)), int(round(y1 * h))))
    cx1 = max(cx1, cx0 + 1)
    cy1 = max(cy1, cy0 + 1)
    return np.asarray(raw_negative[cy0:cy1, cx0:cx1, :3], dtype=np.float64).reshape(-1, 3)


def sample_patch_density(raw_negative: ImageBuffer, rect: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """
    Trimmed-mean negative density of a patch region.

    `rect` is (x0, y0, x1, y1) normalized to [0, 1] over the raw negative (same
    convention as ProcessConfig.analysis_rect). Per channel, average the values
    between the 10th and 90th percentile (robust to dust/edges), then -log10.
    """
    patch = _crop_patch(raw_negative, rect)
    out = []
    for ch in range(3):
        col = patch[:, ch]
        lo, hi = np.percentile(col, (10.0, 90.0))
        inner = col[(col >= lo) & (col <= hi)]
        mean = float(inner.mean()) if inner.size else float(col.mean())
        out.append(-np.log10(max(mean, _EPSILON)))
    return (out[0], out[1], out[2])


def _patch_clipped(raw_negative: ImageBuffer, rect: tuple[float, float, float, float]) -> bool:
    """True if much of the patch is saturated (near the linear ceiling) — an
    unreliable density measurement, since -log10 flattens near 1.0."""
    patch = _crop_patch(raw_negative, rect)
    frac = float(np.mean(np.any(patch >= _CLIP_LEVEL, axis=1)))
    return frac > _CLIP_FRACTION


def calibrate_from_marks(raw_negative: ImageBuffer, marks: Sequence[tuple[str, tuple[float, float, float, float]]]) -> CalibrationResult:
    """
    End-to-end chart calibration: sample each labelled patch region on the raw
    negative and solve for the crosstalk matrix.

    `marks` pairs a role ("R"/"G"/"B"/"C"/"M"/"Y" or a neutral) with a normalized
    rect. The negative must be the pre-crosstalk, pre-normalization linear buffer.
    """
    samples = [
        PatchSample(role=role, density=sample_patch_density(raw_negative, rect), clipped=_patch_clipped(raw_negative, rect))
        for role, rect in marks
    ]
    return solve_crosstalk_matrix(samples)


def _off_gray(d: np.ndarray) -> np.ndarray:
    """Component of a density vector orthogonal to the gray axis."""
    return d - (d @ _GRAY) * _GRAY


def _solve_plane_map(chroma: np.ndarray, targets: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Least-squares 2x2 map T s.t. T·(chroma in basis) ~= (targets in basis)."""
    p = chroma @ basis  # (N, 2) measured off-gray chroma in plane coords
    q = targets @ basis  # (N, 2) target chroma in plane coords
    # lstsq(p, q) -> X (2x2) with p·X ~= q; T = X^T so that T·p_i ~= q_i.
    x, *_ = np.linalg.lstsq(p, q, rcond=None)
    return x.T


def solve_crosstalk_matrix(samples: Sequence[PatchSample]) -> CalibrationResult:
    """
    Fit a gray-preserving crosstalk matrix from labelled chart patches.

    Multiple samples per chroma role are averaged (over-determined charts). Falls
    back to identity with a warning when the chroma directions can't span the plane.
    """
    warnings: list[str] = []

    by_role: dict[str, list[np.ndarray]] = {}
    neutrals: list[np.ndarray] = []
    for s in samples:
        role = s.role.upper()
        density = np.asarray(s.density, dtype=np.float64)
        if role in NEUTRAL_ROLES:
            neutrals.append(density)
            continue
        if role not in CHROMA_ROLES:
            continue
        if s.clipped:
            warnings.append(f"patch {role} is clipped; its measurement is unreliable")
        by_role.setdefault(role, []).append(density)

    roles = sorted(by_role)
    missing = [r for r in CHROMA_ROLES if r not in by_role]
    if missing:
        warnings.append(f"missing chroma patches: {', '.join(missing)} (fit is less constrained)")

    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if len(roles) < 2:
        warnings.append("need at least two non-opposite chroma patches to solve; returning identity")
        return CalibrationResult(matrix=identity, residuals={}, warnings=tuple(warnings))

    chroma = np.array([_off_gray(np.mean(by_role[r], axis=0)) for r in roles])
    # Remove the film's off-gray colour cast (the orange/purple mask is itself a
    # chroma bias, not a gray-axis offset, so `_off_gray` alone leaves it in and a
    # gray-preserving matrix can't subtract a constant). Estimate the bias from the
    # neutral patches if marked, else from the chroma centroid (R/G/B/C/M/Y are
    # antipodal, so their true chroma sums to zero → the centroid is the bias).
    if neutrals:
        bias = _off_gray(np.mean(neutrals, axis=0))
    else:
        bias = np.mean(chroma, axis=0)
    chroma = chroma - bias
    # Target = each patch's ideal hue direction at *its own* measured magnitude, so the
    # fit corrects hue only and leaves saturation alone. Targeting a shared (mean)
    # magnitude instead equalizes the primaries — it drags the naturally-saturated ones
    # (e.g. blue) toward gray and muddies the result (validated on a real chart: a mean-
    # magnitude fit came out *less* saturated than no correction; per-patch beat even a
    # hand-tuned matrix on hue accuracy). Clean input still maps to identity.
    targets = np.array([np.linalg.norm(chroma[i]) * CANONICAL_CHROMA[r] for i, r in enumerate(roles)])

    # Orthonormal basis of the gray-orthogonal plane; require the measured chroma
    # to actually span it (opposite colours are antiparallel, not independent).
    u = _off_gray(np.array([1.0, -1.0, 0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(_GRAY, u)
    basis = np.column_stack((u, v))  # (3, 2)
    if np.linalg.matrix_rank(chroma @ basis, tol=1e-9) < 2:
        warnings.append("chroma patches are collinear (all opposite pairs); returning identity")
        return CalibrationResult(matrix=identity, residuals={}, warnings=tuple(warnings))

    t = _solve_plane_map(chroma, targets, basis)
    # Adding g·gᵀ fixes the gray axis, so m·(1,1,1) = (1,1,1): rows already sum to 1
    # (the profile convention) and the consumer's row-normalization is a no-op — no
    # diagonal rescale, which would break that gray-preservation.
    m = basis @ t @ basis.T + np.outer(_GRAY, _GRAY)

    residuals: dict[str, float] = {}
    for i, r in enumerate(roles):
        out = _off_gray(m @ chroma[i])
        norm = np.linalg.norm(out)
        if norm < _EPSILON:
            residuals[r] = 90.0
            continue
        cos = float(np.clip((out / norm) @ CANONICAL_CHROMA[r], -1.0, 1.0))
        residuals[r] = float(np.degrees(np.arccos(cos)))

    mean_res = float(np.mean(list(residuals.values()))) if residuals else 0.0
    if mean_res > _HIGH_RESIDUAL_DEG:
        worst = max(residuals, key=residuals.__getitem__)
        warnings.append(
            f"high fit residual ({mean_res:.1f}° mean, worst {worst} at {residuals[worst]:.0f}°); "
            "recheck that patch's label and that its box sits on flat, evenly-lit colour"
        )

    return CalibrationResult(
        matrix=tuple(float(x) for x in m.reshape(-1)),
        residuals=residuals,
        warnings=tuple(warnings),
    )
