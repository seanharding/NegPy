"""Scanner (sensor + light) crosstalk calibration and correction.

A narrowband RGB capture system's sensor channels cross-respond to the other bands —
the colour-filter passbands overlap (green/blue especially), so even a pure red/green/
blue exposure leaks into the wrong channels. That's a fixed property of the sensor+light,
independent of any film. Calibrate it once from three bare-light exposures and correct
every capture with a 3x3 matrix in the LINEAR domain, before the log/inversion where the
film-dye crosstalk lives (a density-domain matrix can only approximate a linear one).

Calibration: three bare-light exposures (red-only, green-only, blue-only, no film) give
the sensor's response to each band — the columns of the mixing matrix S. Normalize each
column so its own channel reads 1 (leaving the band-intensity/white-balance on the
diagonal for downstream normalization to handle), then invert to un-mix.
"""

import numpy as np

from negpy.domain.types import ImageBuffer

_EPS = 1e-6


def measure_capture(img: ImageBuffer, center: float = 0.5) -> tuple[float, float, float]:
    """Mean linear RGB over the central `center` fraction of a bare-light exposure
    (centre-weighted to dodge vignetting / non-uniform edges)."""
    h, w = img.shape[:2]
    my, mx = (1.0 - center) / 2.0, (1.0 - center) / 2.0
    region = np.asarray(img[int(h * my) : int(h * (1 - my)), int(w * mx) : int(w * (1 - mx)), :3], dtype=np.float64)
    m = region.reshape(-1, 3).mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def build_sensor_matrix(
    rgb_red: tuple[float, float, float], rgb_green: tuple[float, float, float], rgb_blue: tuple[float, float, float]
) -> tuple[float, ...]:
    """
    Correction matrix (9 floats, row-major) from the three bare-light exposures' mean
    RGB. Columns of S are each band's sensor response; normalize so every own-channel = 1
    (white balance preserved, left for downstream normalization), then invert to un-mix.

    Raises ValueError on a band with ~zero own-channel response (bad capture / mislabel)
    or a singular system, rather than emitting a garbage matrix.
    """
    s = np.column_stack([rgb_red, rgb_green, rgb_blue]).astype(np.float64)  # columns = R/G/B exposures
    diag = np.diag(s).copy()
    if np.any(np.abs(diag) < _EPS):
        raise ValueError("a band's own-channel response is ~zero — check the captures and R/G/B labelling")
    s_norm = s / diag  # each column / its own-channel value -> unit diagonal
    if abs(np.linalg.det(s_norm)) < _EPS:
        raise ValueError("sensor response is singular (bands too collinear to invert)")
    correction = np.linalg.inv(s_norm)
    return tuple(float(x) for x in correction.reshape(-1))


def apply_sensor_correction(img: ImageBuffer, matrix) -> ImageBuffer:
    """Un-mix a linear (H, W, 3) capture with the 3x3 correction; identity when `matrix`
    is None. Applied to linear light before the log — the domain sensor crosstalk lives in."""
    if matrix is None:
        return img
    m = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    out = np.einsum("ij,hwj->hwi", m, img[:, :, :3].astype(np.float32, copy=False))
    return np.clip(out, 0.0, None)


def scanner_token(process) -> str:
    """Identity of the active scanner matrix, folded into the render source hash so the
    engine cache (and the meters) re-run when the scanner profile changes."""
    m = getattr(process, "scanner_matrix", None)
    return "|sc:" + (",".join(f"{v:.4g}" for v in m) if m else "0")
