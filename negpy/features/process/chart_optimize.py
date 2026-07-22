"""Closed-loop crosstalk calibration: optimize the matrix against the *rendered* result.

The density-domain fit in `calibration.py` minimizes hue error in negative-density space
— a proxy for what the eye sees, since normalization + the print curve + toning sit
between it and the final image. This module instead optimizes the six off-diagonal
matrix terms to minimize colour error of the *rendered* chart patches against known
reference values (Nelder–Mead, pipeline in the loop). Because a hand-tuned matrix is
just one point in the same six-parameter space, the optimum is guaranteed no worse.

Pure/decoupled: the caller supplies a `render_ab` callback that renders a candidate
matrix through the real pipeline and returns each patch's measured (a*, b*). The
optimizer never imports the rendering service, so it is unit-testable with a fake
callback.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

# Off-diagonal positions of a 3x3 matrix, row-major — the six free parameters (the
# diagonal is pinned to 1 and the pipeline row-normalizes on apply).
OFF_DIAGONAL = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))

# SpyderCheckr 24 reference chroma (CIELAB a*, b*, D65) for the saturated primaries
# (Datacolor published values). Neutrals target (0, 0). Hue *and* saturation targets —
# that's what lets the fit restore saturation the density-domain proxy left on the table.
SPYDERCHECKR_AB = {
    "R": (60.75, 31.17),
    "G": (-40.8, 34.75),
    "B": (13.78, -49.48),
    "C": (-32.5, -28.75),
    "M": (53.45, -13.55),
    "Y": (3.36, 87.02),
}
_NEUTRAL_ROLES = frozenset({"NEUTRAL", "BLACK", "WHITE", "GREY", "GRAY"})


def matrix_from_params(offdiag: Sequence[float]) -> tuple[float, ...]:
    """Nine row-major floats for a unit-diagonal matrix with the given off-diagonals."""
    m = np.eye(3)
    for k, (i, j) in enumerate(OFF_DIAGONAL):
        m[i, j] = offdiag[k]
    return tuple(float(x) for x in m.reshape(-1))


def target_ab(role: str) -> Optional[tuple[float, float]]:
    """Reference (a*, b*) for a role, or None if the role carries no colour target."""
    r = role.upper()
    if r in _NEUTRAL_ROLES:
        return (0.0, 0.0)
    return SPYDERCHECKR_AB.get(r)


@dataclass(frozen=True)
class OptimizeResult:
    matrix: tuple[float, ...]  # 9 floats, row-major, unit diagonal (pipeline row-normalizes)
    error: float  # final mean chroma error (CIELAB a*b* distance) across targeted patches
    evaluations: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _nelder_mead(
    f: Callable[[np.ndarray], float], x0: np.ndarray, step: float, max_evals: int, on_eval: Optional[Callable[[int, float], bool]]
) -> tuple[np.ndarray, float, int]:
    """Minimal Nelder–Mead. Explicit simplex (a zero start would otherwise get a
    microscopic default step and stall). `on_eval(count, best)` may return True to stop."""
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = [x0.astype(float)] + [x0 + step * np.eye(n)[k] for k in range(n)]
    fvals = [f(x) for x in simplex]
    evals = n + 1
    stop = False
    while evals < max_evals and not stop:
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if on_eval is not None and on_eval(evals, fvals[0]):
            break
        if float(np.max(np.abs(np.array(simplex[1:]) - simplex[0]))) < 1e-3 and (fvals[-1] - fvals[0]) < 1e-2:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        evals += 1
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            evals += 1
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            evals += 1
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
                    evals += 1
    best = int(np.argmin(fvals))
    return simplex[best], float(fvals[best]), evals


def optimize_crosstalk(
    render_ab: Callable[[tuple[float, ...]], dict[int, tuple[float, float]]],
    roles: Sequence[str],
    max_evals: int = 260,
    progress: Optional[Callable[[int, float], bool]] = None,
    init: Optional[Sequence[float]] = None,
) -> OptimizeResult:
    """
    Optimize the crosstalk matrix so the rendered chart patches best match their
    reference chroma.

    `render_ab(matrix9)` renders the candidate matrix through the pipeline and returns
    `{patch_index: (a*, b*)}`. `roles[i]` names patch i (R/G/B/C/M/Y or a neutral);
    patches with no reference target are ignored. `progress(evals, best_error)` may
    return True to cancel. `init` seeds the six off-diagonals (e.g. the fast density-fit
    result), so the search converges in fewer renders than from identity.
    """
    targets = {i: target_ab(role) for i, role in enumerate(roles)}
    scored = {i: t for i, t in targets.items() if t is not None}
    warnings: list[str] = []
    if not scored:
        warnings.append("no patches with known reference colours; cannot optimize")
        return OptimizeResult(matrix=matrix_from_params(np.zeros(6)), error=0.0, evaluations=0, warnings=tuple(warnings))

    def objective(offdiag: np.ndarray) -> float:
        measured = render_ab(matrix_from_params(offdiag))
        errs = [(measured[i][0] - t[0]) ** 2 + (measured[i][1] - t[1]) ** 2 for i, t in scored.items() if i in measured]
        return float(np.sqrt(np.mean(errs))) if errs else 1e6

    x0 = np.zeros(6) if init is None else np.asarray(init, dtype=float)
    x, err, evals = _nelder_mead(objective, x0, step=0.15, max_evals=max_evals, on_eval=progress)
    return OptimizeResult(matrix=matrix_from_params(x), error=err, evaluations=evals, warnings=tuple(warnings))
