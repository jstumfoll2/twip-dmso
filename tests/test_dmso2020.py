"""Tests for the DMSO of the 2020 ACC paper (:class:`twip.observers.DMSO2020`)."""

import numpy as np
import pytest

from twip.control import c2d_zoh
from twip.dynamics import linear_model
from twip.experiments import OBSERVER_CASES, design_lqr, make_estimators, make_truth, run_observer
from twip.observers import B_MSO, DMSO, DMSO2020, TanhBasis

A, B = linear_model()
F, G = c2d_zoh(A, B, 0.01)
G = G.ravel()


def test_paper_defaults():
    d = DMSO2020(F, G)
    np.testing.assert_array_equal(d.Km, np.eye(4))  # K_DMSO = I_4x4
    assert d.alpha == 0.01
    np.testing.assert_array_equal(d.basis.scales, np.ones(4))  # phi = [1, tansig(x)]
    np.testing.assert_array_equal(d.Bm, B_MSO)


@pytest.mark.parametrize("a2", [0.0, 0.05, 0.9, 1.5])
def test_leak_factor_is_the_matrix_norm(a2):
    phi = np.array([1.0, 0.3, -0.2, 0.5, 0.1])
    alpha = a2 / (phi @ phi)
    M = np.eye(5) - alpha * np.outer(phi, phi)
    assert DMSO2020(F, G, alpha=alpha).leak_factor(phi) == pytest.approx(np.linalg.norm(M, 2))
    assert DMSO2020(F, G, alpha=alpha, leak_norm="fro").leak_factor(phi) == pytest.approx(np.linalg.norm(M, "fro"))
    assert DMSO2020(F, G, alpha=alpha).leak_factor(phi) == 1.0  # alpha ||phi||^2 <= 2


def test_two_norm_law_is_rev8_sigma_modification():
    """With the 2-norm, eq. (6) is the revision-8 sigma-mod with Gamma_8 = alpha, kappa = Gamma / alpha."""
    tr = make_truth(OBSERVER_CASES["O2"])
    rev8 = DMSO(tr.F, tr.G, kappa=0.1)
    assert rev8.gamma == pytest.approx(0.01)
    acc = DMSO2020(tr.F, tr.G, alpha=rev8.gamma, gamma=rev8.gamma * 0.1, Km=rev8.Km, basis=TanhBasis())
    r8, ra = run_observer(rev8, tr), run_observer(acc, tr)
    np.testing.assert_allclose(ra.xpred, r8.xpred, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ra.fhat, r8.fhat, rtol=0, atol=1e-12)


def test_identifies_constant_uncertainty():
    f = np.array([0.001, -0.002])
    K = design_lqr()
    d = DMSO2020(F, G, gamma=1e-4)
    x = np.array([0.0, 0.0, 0.02, 0.0])
    d.initialize(x)
    for _ in range(6000):
        u = float(-K @ x)
        d.step(x, u)
        x = F @ x + G * u + B_MSO @ f
    np.testing.assert_allclose(d.fhat, f, rtol=0.02)  # the leakage biases fhat toward zero, ~0.6% here


def test_conditions_hold_with_paper_gains_on_the_observer_cases():
    tr = make_truth(OBSERVER_CASES["O2"])
    est = make_estimators(tr, noise=False, acc2020=True)["DMSO (ACC 2020)"]
    c = run_observer(est, tr).conditions
    assert c.satisfied and c.weights_contractive
    assert c.sigma_max_A == pytest.approx(np.linalg.norm(tr.F - np.eye(4), 2))


def test_condition_10_does_not_bound_the_weights():
    """Gamma just below 2 satisfies (9)-(11), yet with ||phi||^2 > 1 the weight error is
    multiplied by 1 - Gamma - alpha ||phi||^2 < -1 along phi each step."""
    d = DMSO2020(F, G, gamma=1.99, Km=F)  # A = 0, so (11) holds trivially
    y = np.full(4, 3.0)  # held measurement: phi = [1, tanh(3), ...], ||phi||^2 = 4.96
    with np.errstate(all="ignore"):
        for _ in range(400):
            d.step(y, 0.0)
    c = d.conditions()
    assert c.satisfied
    assert not c.weights_contractive
    assert np.abs(d.W).max() > 1e3
    ok = DMSO2020(F, G, gamma=1.9, Km=F)  # 1.9 + 0.0496 < 2
    for _ in range(400):
        ok.step(y, 0.0)
    assert ok.conditions().weights_contractive
    assert np.abs(ok.W).max() < 1.0


def test_rejects_unknown_norm():
    with pytest.raises(ValueError):
        DMSO2020(F, G, leak_norm="1")
