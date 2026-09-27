import numpy as np
import pytest

from twip.legacy.dynamics import linear_model, twip_nonlinear, uncert_extra_terms
from twip.params import THESIS, RobotParams


def jacobian(p=THESIS, h=1e-6):
    x0 = np.zeros(4)
    J = np.column_stack(
        [(twip_nonlinear(x0 + h * e, 0.0, p) - twip_nonlinear(x0 - h * e, 0.0, p)) / (2 * h) for e in np.eye(4)]
    )
    Jb = (twip_nonlinear(x0, h, p) - twip_nonlinear(x0, -h, p)) / (2 * h)
    return J, Jb


@pytest.mark.parametrize("p", [THESIS, THESIS.with_thesis_uncertainty()])
def test_hand_linearization_matches_nonlinear_jacobian(p):
    A, B = linear_model(p)
    J, Jb = jacobian(p)
    np.testing.assert_allclose(J, A, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(Jb, B.ravel(), rtol=1e-6, atol=1e-8)


def test_up_and_down_conventions_differ_by_pi():
    rng = np.random.default_rng(1)
    for _ in range(20):
        X = rng.normal(size=4)
        v = rng.normal() * 5
        up = twip_nonlinear(X, v, convention="up")
        down = twip_nonlinear(X + [0, 0, np.pi, 0], v, convention="down")
        np.testing.assert_allclose(up, down, atol=1e-12)


def test_equilibria():
    # atol covers sin(pi) ~ 1e-16 in the "up" convention
    np.testing.assert_allclose(twip_nonlinear(np.zeros(4), 0.0), 0, atol=1e-12)
    np.testing.assert_allclose(twip_nonlinear(np.zeros(4), 0.0, convention="down"), 0, atol=1e-12)


def test_upright_is_unstable_hanging_is_stable():
    A, _ = linear_model()
    assert np.max(np.linalg.eigvals(A).real) > 1  # ~ +6.3 rad/s
    h = 1e-6
    Jd = np.column_stack(
        [
            (twip_nonlinear(h * e, 0.0, convention="down") - twip_nonlinear(-h * e, 0.0, convention="down")) / (2 * h)
            for e in np.eye(4)
        ]
    )
    assert np.max(np.linalg.eigvals(Jd).real) <= 1e-9


def test_positive_voltage_accelerates_forward():
    xdd = twip_nonlinear(np.zeros(4), 1.0)[1]
    assert xdd > 0


def test_params_defaults_and_uncertainty():
    p = RobotParams()
    assert p.Iw == pytest.approx(p.Mw * p.r**2 / 2)
    assert p.Ip == pytest.approx(0.02497, rel=1e-4)
    q = p.with_thesis_uncertainty()
    assert q.Mp == pytest.approx(1.1 * p.Mp)
    assert q.Mw == pytest.approx(1.5 * p.Mw)
    assert q.Iw == pytest.approx(q.Mw * p.r**2 / 2)
    assert q.l == pytest.approx(0.9 * p.l)
    assert (q.km, q.ke, q.R) == pytest.approx((2.5 * p.km, 0.95 * p.ke, 1.2 * p.R))
    assert p.with_thesis_uncertainty(l_scale=1.0).l == p.l


def test_extra_terms_hook():
    X = np.array([0.3, 0.2, 0.0, 0.0])
    base = twip_nonlinear(X, 0.0)
    extra = twip_nonlinear(X, 0.0, extra=uncert_extra_terms)
    assert extra[1] - base[1] == pytest.approx(0.5 * np.sin(0.3) * 0.2)
    assert extra[3] - base[3] == pytest.approx(0.25 * 0.2**2)
