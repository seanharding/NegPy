import numpy as np

from negpy.features.process.chart_optimize import (
    OFF_DIAGONAL,
    SPYDERCHECKR_AB,
    matrix_from_params,
    optimize_crosstalk,
    target_ab,
)


def test_matrix_from_params_places_off_diagonals():
    m = np.array(matrix_from_params([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])).reshape(3, 3)
    assert np.allclose(np.diag(m), 1.0)
    for k, (i, j) in enumerate(OFF_DIAGONAL):
        assert m[i, j] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][k]


def test_target_ab_roles():
    assert target_ab("R") == SPYDERCHECKR_AB["R"]
    assert target_ab("neutral") == (0.0, 0.0)
    assert target_ab("grey") == (0.0, 0.0)
    assert target_ab("unknown") is None


def _synthetic_render(star, seed=0):
    """Fake pipeline: rendered (a*,b*) is linear in the matrix params with the optimum
    at `star` (error 0 there). B has full column rank so the optimum is unique."""
    roles = ["R", "G", "B", "C", "M", "Y"]
    targets = np.array([SPYDERCHECKR_AB[r] for r in roles])  # (6, 2)
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 6.0, size=(12, 6))  # 2 outputs * 6 patches x 6 params

    def render_ab(matrix9):
        m = np.array(matrix9).reshape(3, 3)
        offdiag = np.array([m[i, j] for i, j in OFF_DIAGONAL])
        delta = B @ (offdiag - star)  # (12,)
        meas = targets + delta.reshape(6, 2)
        return {i: (float(meas[i, 0]), float(meas[i, 1])) for i in range(6)}

    return roles, render_ab


def test_optimizer_recovers_known_optimum():
    star = np.array([-0.18, 0.09, 0.01, -0.44, 0.05, 0.07])
    roles, render_ab = _synthetic_render(star)
    result = optimize_crosstalk(render_ab, roles, max_evals=400)
    assert result.error < 0.5  # drives rendered chroma error to ~zero
    m = np.array(result.matrix).reshape(3, 3)
    found = np.array([m[i, j] for i, j in OFF_DIAGONAL])
    assert np.allclose(found, star, atol=0.05)


def test_optimizer_beats_identity():
    star = np.array([-0.2, 0.1, 0.0, -0.4, 0.05, 0.08])
    roles, render_ab = _synthetic_render(star, seed=3)

    def err_at(offdiag):
        meas = render_ab(matrix_from_params(offdiag))
        t = [SPYDERCHECKR_AB[r] for r in roles]
        return np.sqrt(np.mean([(meas[i][0] - t[i][0]) ** 2 + (meas[i][1] - t[i][1]) ** 2 for i in range(6)]))

    result = optimize_crosstalk(render_ab, roles, max_evals=400)
    assert result.error < err_at(np.zeros(6))  # better than no correction


def test_init_warm_start_converges():
    star = np.array([-0.18, 0.09, 0.01, -0.44, 0.05, 0.07])
    roles, render_ab = _synthetic_render(star)
    # Seeding near the optimum should still converge (and not blow up).
    result = optimize_crosstalk(render_ab, roles, max_evals=200, init=star + 0.03)
    assert result.error < 0.5


def test_no_reference_roles_warns():
    result = optimize_crosstalk(lambda m: {}, ["foo", "bar"], max_evals=50)
    assert any("no patches" in w for w in result.warnings)
    assert np.allclose(np.array(result.matrix).reshape(3, 3), np.eye(3))


def test_progress_can_cancel():
    star = np.array([-0.2, 0.1, 0.0, -0.4, 0.05, 0.08])
    roles, render_ab = _synthetic_render(star)
    calls = {"n": 0}

    def progress(evals, best):
        calls["n"] += 1
        return calls["n"] >= 2  # cancel almost immediately

    result = optimize_crosstalk(render_ab, roles, max_evals=400, progress=progress)
    assert result.evaluations < 100  # stopped early, didn't run the full budget
