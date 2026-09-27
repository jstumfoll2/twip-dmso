from dataclasses import replace

import numpy as np
import pytest

from twip.extra_control import _nn_update, preset, run
from twip.integrators import rk4

DEG = np.pi / 180
R2D = 180 / np.pi


@pytest.mark.parametrize("name", ["unmodeled_dynamics", "parameter_uncertainty", "deadzone", "deadzone_and_backlash"])
def test_figure_presets_extra_control_beats_lqr(name):
    r = run(preset(name))
    assert np.all(np.isfinite(r.xextra))
    assert r.error_extra < 0.7 * r.error_lqr


def test_as_saved_script_diverges_under_extra_control():
    """Documents that ExtraControl_v5.m as saved (ke0 = 1) is unstable with extra control."""
    r = run(preset("as_saved"))
    assert abs(r.xextra[2, -1] - np.pi) > 1.0  # tilt has run away
    assert abs(r.xtrue[2, -1] - np.pi) < 0.2  # LQR-only is fine


def test_extra_off_matches_lqr_only():
    r = run(replace(preset("unmodeled_dynamics"), extra=False, steps=200))
    np.testing.assert_array_equal(r.xextra, r.xtrue)
    assert np.all(r.u_e == 0)


def test_unmodeled_dynamics_matches_saved_thesis_figure():
    """Values read off Extra control/Unmodeled Dynamics/figure1.jpg (nominal and LQR-only curves)."""
    r = run(preset("unmodeled_dynamics"))
    nom_tilt = r.xnom[2] * R2D + 180
    assert nom_tilt.min() == pytest.approx(177.8, abs=0.3)
    assert r.xtrue[0].min() == pytest.approx(-1.08, abs=0.05)
    assert r.time[np.argmin(r.xtrue[0])] == pytest.approx(1.5, abs=0.1)
    assert r.xtrue[1].min() == pytest.approx(-2.3, abs=0.1)


def test_deadzone_matches_saved_thesis_figure():
    """Values read off Extra control/Deadzone/figure1.jpg."""
    r = run(preset("deadzone"))
    assert r.xnom[0].min() == pytest.approx(-0.2, abs=0.03)
    assert r.xtrue[0].min() == pytest.approx(-0.3, abs=0.03)
    assert r.xextra[0].min() == pytest.approx(0.14, abs=0.05)
    assert r.xtrue[1].min() == pytest.approx(-1.75, abs=0.05)


def test_desired_tilt_holds_desired_velocity_in_linear_model():
    from twip.dynamics import linear_model

    r = run(replace(preset("unmodeled_dynamics"), steps=5))
    A, B = linear_model()
    th = r.x_des[2, 0] - np.pi
    v = r.x_des[1, 0]
    # there is a voltage u that makes xddot = thetaddot = 0 at (v, th)
    u = -(A[1, 1] * v + A[1, 2] * th) / B[1, 0]
    assert A[3, 1] * v + A[3, 2] * th + B[3, 0] * u == pytest.approx(0, abs=1e-12)


def test_nn_update_matches_generic_rk4():
    rng = np.random.default_rng(3)
    n_in, n_hid = 4, 3
    W1, W2 = rng.normal(size=(n_in, n_hid)), rng.normal(size=n_hid)
    phi1, phi2, err = rng.normal(size=n_in), rng.normal(size=n_hid), 0.7
    g1, s1, g2, s2 = 0.1, 0.01, 0.2, 0.05
    w2, w1 = _nn_update(W2, W1, phi1, phi2, err, g1, s1, g2, s2, 0.0, 0.01, 0.002)

    def f(_t, z):  # NN1v3.m, written directly
        a, b = z[:n_hid], z[n_hid:].reshape(n_in, n_hid)
        db = -g1 * (phi1[:, None] * (b.T @ phi1 + err)[None, :] + s1 * b)
        return np.concatenate([g2 * (phi2 * err - s2 * a), db.ravel()])

    z = rk4(f, np.concatenate([W2, W1.ravel()]), 0.0, 0.01, 0.002)
    np.testing.assert_allclose(w2, z[:n_hid])
    np.testing.assert_allclose(w1, z[n_hid:].reshape(n_in, n_hid))
