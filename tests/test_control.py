import numpy as np
import pytest
from scipy.linalg import expm
from scipy.signal import cont2discrete

from twip.control import c2d_zoh, dlqr, lqr, lqrd
from twip.legacy.dynamics import linear_model
from twip.integrators import rk4

A, B = linear_model()
Q_THESIS = np.diag([100, 50, 1e-4, 1e-4])
R_THESIS = 1000.0


def test_c2d_matches_scipy():
    F, G = c2d_zoh(A, B, 0.01)
    Fs, Gs, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), 0.01, method="zoh")
    np.testing.assert_allclose(F, Fs, atol=1e-12)
    np.testing.assert_allclose(G, Gs, atol=1e-12)


def test_lqr_stabilizes():
    K = lqr(A, B, Q_THESIS, R_THESIS)
    assert np.all(np.linalg.eigvals(A - B @ K).real < 0)


def test_dlqr_stabilizes():
    F, G = c2d_zoh(A, B, 0.01)
    K = dlqr(F, G, Q_THESIS, R_THESIS)
    assert np.all(np.abs(np.linalg.eigvals(F - G @ K)) < 1)


@pytest.mark.parametrize("dt", [0.001, 0.01, 0.013])
def test_lqrd_stabilizes(dt):
    F, G = c2d_zoh(A, B, dt)
    K = lqrd(A, B, Q_THESIS, R_THESIS, dt)
    assert K.shape == (1, 4)
    assert np.all(np.abs(np.linalg.eigvals(F - G @ K)) < 1)


def test_lqrd_approaches_continuous_lqr_as_dt_shrinks():
    Kc = lqr(A, B, Q_THESIS, R_THESIS)
    Kd = lqrd(A, B, Q_THESIS, R_THESIS, 1e-4)
    np.testing.assert_allclose(Kd, Kc, rtol=2e-3)


def test_lqrd_cost_discretization_is_exact():
    """Check the Van Loan integral against brute-force quadrature."""
    dt = 0.01
    n = 4
    Aa = np.zeros((5, 5))
    Aa[:4, :4], Aa[:4, 4:] = A, B
    Wa = np.zeros((5, 5))
    Wa[:4, :4], Wa[4, 4] = Q_THESIS, R_THESIS
    taus = np.linspace(0, dt, 2001)
    vals = np.array([expm(Aa * t).T @ Wa @ expm(Aa * t) for t in taus])
    Wd_quad = np.trapezoid(vals, taus, axis=0)
    # rebuild via the module path and compare the resulting gains
    Kd = lqrd(A, B, Q_THESIS, R_THESIS, dt)
    F, G = c2d_zoh(A, B, dt)
    Kq = dlqr(F, G, Wd_quad[:n, :n], Wd_quad[n:, n:], Wd_quad[:n, n:])
    np.testing.assert_allclose(Kd, Kq, rtol=1e-5)


def test_rk4_matches_matrix_exponential():
    Acl = A - B @ lqr(A, B, Q_THESIS, R_THESIS)
    x0 = np.array([1.0, 0.3, 0.17, 0.02])
    x = rk4(lambda t, x: Acl @ x, x0, 0.0, 0.5, 0.001)
    np.testing.assert_allclose(x, expm(Acl * 0.5) @ x0, rtol=1e-8, atol=1e-10)


def test_rk4_fourth_order():
    f = lambda t, x: -x  # noqa: E731
    errs = [abs(rk4(f, np.array([1.0]), 0, 1, h)[0] - np.exp(-1)) for h in (0.1, 0.05)]
    assert errs[0] / errs[1] == pytest.approx(16, rel=0.1)
