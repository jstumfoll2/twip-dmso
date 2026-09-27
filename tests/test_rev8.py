"""Tests for the thesis revision-8 observer and controllers."""

import numpy as np
import pytest

from twip.control import c2d_zoh
from twip.controllers import (
    CFBGains,
    CommandFilter,
    CommandFilteredBackstepping,
    LQRTracking,
    TrackingModel,
    TwoStepExtraControl,
    position_zeros,
)
from twip.dynamics import linear_model
from twip.experiments import OBSERVER_CASES, design_lqr, make_estimators, make_truth, rms, run_observer
from twip.observers import B_MSO, DMSO, KalmanPredictor, TanhBasis, gamma_bound

A, B = linear_model()
F, G = c2d_zoh(A, B, 0.01)
G = G.ravel()


# --- DMSO (Chapter 3) --------------------------------------------------------


def test_gamma_bound_matches_corollary():
    assert gamma_bound(0.5, 5.0) == pytest.approx(min(0.5, 0.75 / 0.5) / 5)
    assert gamma_bound(0.9, 5.0) == pytest.approx((1 - 0.81) / (2 * 0.81) / 5)


def test_dmso_places_A_and_satisfies_conditions():
    d = DMSO(F, G, sigma_bar=0.6)
    np.testing.assert_allclose(F - d.Km, 0.6 * np.eye(4), atol=1e-15)
    assert d.gamma == pytest.approx(0.1 * gamma_bound(0.6, TanhBasis().phi_max2))
    assert TanhBasis()(np.array([1e6, -1e6, 1e6, 0.0])) @ TanhBasis()(np.array([1e6, -1e6, 1e6, 0.0])) <= 5 + 1e-12


def test_dmso_identifies_constant_uncertainty():
    f = np.array([0.001, -0.002])
    # regulate the open-loop plant with LQR so the trajectory stays in range
    K = design_lqr()
    d = DMSO(F, G)
    x = np.array([0.0, 0.0, 0.02, 0.0])
    d.initialize(x)
    for k in range(6000):
        u = float(-K @ x)
        d.step(x, u)
        x = F @ x + G * u + B_MSO @ f
    np.testing.assert_allclose(d.fhat, f, rtol=0.02, atol=2e-5)


def test_augmented_kalman_identifies_constant_uncertainty():
    f = np.array([0.001, -0.002])
    K = design_lqr()
    Gw = np.array([[5e-5, 0], [0.01, 0], [0, 5e-5], [0, 0.01]])
    kf = KalmanPredictor(F, G, Gw @ Gw.T * 1e-4 + 1e-12 * np.eye(4), 1e-10 * np.eye(4), augmented=True, q_f=1e-8)
    x = np.array([0.0, 0.0, 0.02, 0.0])
    kf.initialize(x)
    for k in range(3000):
        u = float(-K @ x)
        kf.step(x, u)
        x = F @ x + G * u + B_MSO @ f
    np.testing.assert_allclose(kf.fhat, f, rtol=0.05, atol=2e-5)


def test_rev8_dmso_beats_augmented_kf_on_uncertainty_in_case_O2():
    tr = make_truth(OBSERVER_CASES["O2"])
    ests = make_estimators(tr, noise=False)
    f_dmso = rms(run_observer(ests["DMSO (rev. 8)"], tr).fhat - tr.f, 500)
    f_aug = rms(run_observer(ests["Augmented-state KF"], tr).fhat - tr.f, 500)
    assert np.all(f_dmso < f_aug / 5)


# --- controllers (Chapter 6) -------------------------------------------------


def test_command_filter_has_unit_dc_gain_and_rate_output():
    cf = CommandFilter(20.0, 0.9, 0.01)
    for _ in range(500):
        cf.step(1.0)
    assert cf.value == pytest.approx(1.0, abs=1e-6)
    assert cf.rate == pytest.approx(0.0, abs=1e-6)


def test_cfb_applied_control_does_not_depend_on_lqr_gain():
    """Substituting (ue) into u = u_nom + u_e cancels u_nom."""
    m = TrackingModel.build()
    x0 = np.array([0.0, 0.0, 0.05, 0.0])
    xhat = np.array([0.01, 0.02, 0.03, 0.1])
    u1 = CommandFilteredBackstepping(design_lqr(), m, x0, networks=False)(xhat, 3)
    u2 = CommandFilteredBackstepping(10 * design_lqr() + 1, m, x0, networks=False)(xhat, 3)
    assert u1 == pytest.approx(u2, abs=1e-9)


def test_position_output_is_non_minimum_phase():
    from twip.params import THESIS

    assert np.max(position_zeros(A, B).real) == pytest.approx(7.2333, abs=1e-3)
    At, Bt = linear_model(THESIS, "xdot")
    assert np.max(position_zeros(At, Bt).real) == pytest.approx(6.1133, abs=1e-3)  # not caused by the corrections


def test_cfb_closed_loop_is_unstable_at_the_rhp_zero():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("thesis_rev8", Path(__file__).parents[1] / "scripts" / "thesis_rev8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for wn in ((20, 40, 60), (200, 400, 600)):
        ev = np.linalg.eigvals(mod._cfb_jacobian(CFBGains(wn=wn)))
        assert ev.real.max() > 5.0
    assert ev.real.max() == pytest.approx(7.23, abs=0.05)  # converges to the zero as bandwidth grows


def test_lqr_and_two_step_track_the_velocity_command():
    from twip.baseline import KalmanEstimator, simulate
    from twip.experiments import CONTROL_CASES, control_sim_config

    m = TrackingModel.build()
    cfg = control_sim_config(CONTROL_CASES["C1"])
    for c in (LQRTracking(design_lqr(), m), TwoStepExtraControl(design_lqr(), m)):
        r = simulate(c, KalmanEstimator(sensors=cfg.sensors), cfg)
        assert not r.fell
        assert np.mean(r.x[1, -300:]) == pytest.approx(0.1, abs=0.03)


# --- corrected (flat-output) command-filtered design ------------------------


def test_flat_coordinates_remove_the_control_and_motor_constants():
    from twip.params import CORRECTED

    for p in (CORRECTED, CORRECTED.scaled(km=0.7, ke=2.0, R=1.3)):
        Ap, Bp = linear_model(p)
        b = Bp[1, 0] / Bp[3, 0]
        assert Ap[1, 1] - b * Ap[3, 1] == pytest.approx(0, abs=1e-9)  # a1 = 0
        assert Ap[1, 3] - b * Ap[3, 3] == pytest.approx(0, abs=1e-9)  # a4 = 0
        assert Ap[1, 2] - b * Ap[3, 2] == pytest.approx(-38.746, abs=1e-3)  # a2 independent of motor constants
        # y = x1 - b x3 has no finite transmission zeros (it is the flat output)
        C = np.array([[1.0, 0, -b, 0]])
        from scipy.linalg import eigvals

        M = np.block([[Ap, Bp], [C, np.zeros((1, 1))]])
        N = np.block([[np.eye(4), np.zeros((4, 1))], [np.zeros((1, 5))]])
        z = eigvals(M, N)
        assert np.all(np.abs(z[np.isfinite(z)]) > 1e6)


def _load_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("thesis_rev8", Path(__file__).parents[1] / "scripts" / "thesis_rev8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flat_design_closed_loop_is_stable_across_bandwidths():
    from twip.controllers import FLAT_GAINS

    mod = _load_script()
    for s in (0.5, 1.0, 3.0, 10.0):
        g = CFBGains(c=FLAT_GAINS.c, wn=tuple(s * w for w in FLAT_GAINS.wn))
        assert np.linalg.eigvals(mod._cfb_jacobian(g, flat=True)).real.max() < 0
    # the design as written in Section 6.4 is not
    assert np.linalg.eigvals(mod._cfb_jacobian(CFBGains(c=FLAT_GAINS.c, wn=FLAT_GAINS.wn), flat=False)).real.max() > 0


@pytest.mark.parametrize("case", ["C2", "C4"])
def test_flat_design_balances_and_beats_lqr(case):
    from twip.baseline import KalmanEstimator, simulate
    from twip.controllers import FLAT_GAINS
    from twip.experiments import CONTROL_CASES, control_sim_config

    m = TrackingModel.build()
    cfg = control_sim_config(CONTROL_CASES[case])
    errs = {}
    for name, c in (
        ("lqr", LQRTracking(design_lqr(), m)),
        ("flat", CommandFilteredBackstepping(design_lqr(), m, np.array(cfg.x0), gains=FLAT_GAINS, output="flat")),
    ):
        r = simulate(c, KalmanEstimator(sensors=cfg.sensors), cfg)
        assert not r.fell
        xd = np.array([m.x_des(k) for k in range(len(r.time))]).T
        errs[name] = np.sqrt(np.mean((r.x[0, 500:] - xd[0, 500:]) ** 2))
    assert errs["flat"] < errs["lqr"] / 3


def test_flat_design_control_does_not_depend_on_lqr_gain():
    m = TrackingModel.build()
    x0 = np.array([0.0, 0.0, 0.05, 0.0])
    xhat = np.array([0.01, 0.02, 0.03, 0.1])
    u1 = CommandFilteredBackstepping(design_lqr(), m, x0, networks=False, output="flat")(xhat, 3)
    u2 = CommandFilteredBackstepping(10 * design_lqr() + 1, m, x0, networks=False, output="flat")(xhat, 3)
    assert u1 == pytest.approx(u2, abs=1e-9)
