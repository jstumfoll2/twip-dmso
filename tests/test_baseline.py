"""Tests for the corrected plant, sensors, actuator and baseline controller."""

import numpy as np
import pytest

from twip.actuators import Actuator, ActuatorConfig
from twip.baseline import (
    HARDWARE_DEFAULT_GAINS,
    ComplementaryEstimator,
    KalmanEstimator,
    LQRController,
    SimConfig,
    design_lqr,
    simulate,
)
from twip.control import c2d_zoh
from twip.dynamics import accelerations, derivative, linear_model
from twip.legacy.dynamics import twip_nonlinear
from twip.params import CORRECTED, THESIS, ke_from_free_run, km_from_stall
from twip.sensors import SensorConfig, SensorSuite, specific_force

DEG = np.pi / 180


# --- parameters -------------------------------------------------------------


def test_corrected_parameters_come_from_thesis_data():
    assert ke_from_free_run() == pytest.approx((12 - 0.3 * 2.5) / (350 * 2 * np.pi / 60))
    assert CORRECTED.ke == pytest.approx(0.3069, rel=1e-3)
    assert CORRECTED.ke / THESIS.ke > 80  # the thesis value is ~85x too small
    assert CORRECTED.Ip == pytest.approx(THESIS.Ip - THESIS.Mp * THESIS.l**2)
    assert CORRECTED.km == THESIS.km
    assert km_from_stall() == pytest.approx(0.1554, rel=1e-3)


def test_scaled_keeps_wheel_inertia_consistent():
    q = CORRECTED.scaled(Mw=1.5, Mp=1.1)
    assert q.Iw == pytest.approx(q.Mw * q.r**2 / 2)
    assert q.Mp == pytest.approx(1.1 * CORRECTED.Mp)


# --- dynamics ---------------------------------------------------------------


def test_xdot_back_emf_reproduces_thesis_eom():
    rng = np.random.default_rng(0)
    for _ in range(50):
        X, v = rng.normal(size=4), rng.normal() * 5
        np.testing.assert_allclose(derivative(X, v, THESIS, "xdot"), twip_nonlinear(X, v, THESIS), atol=1e-12)


@pytest.mark.parametrize("back_emf", ["relative", "xdot", "rev8"])
@pytest.mark.parametrize("p", [CORRECTED, THESIS, CORRECTED.scaled(Mp=1.2, l=0.8, ke=0.5)])
def test_linear_model_is_the_jacobian(p, back_emf):
    A, B = linear_model(p, back_emf)
    h = 1e-6
    J = np.column_stack(
        [(derivative(h * e, 0.0, p, back_emf) - derivative(-h * e, 0.0, p, back_emf)) / (2 * h) for e in np.eye(4)]
    )
    Jb = (derivative(np.zeros(4), h, p, back_emf) - derivative(np.zeros(4), -h, p, back_emf)) / (2 * h)
    np.testing.assert_allclose(J, A, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(Jb, B.ravel(), rtol=1e-6, atol=1e-8)


def test_relative_back_emf_adds_thetadot_coupling():
    A, _ = linear_model(CORRECTED)
    A0, _ = linear_model(CORRECTED, "xdot")
    assert A0[1, 3] == 0 and A0[3, 3] == 0
    # the motor sees xdot / r + thetadot, so with no viscous term the thetadot
    # coefficients are r times the xdot coefficients
    assert A[1, 3] == pytest.approx(CORRECTED.r * A[1, 1])
    assert A[3, 3] == pytest.approx(CORRECTED.r * A[3, 1])


def test_motor_torque_is_internal_to_the_robot():
    """The motor torque acts between body and wheels, so it cannot change the robot's
    angular momentum about the wheel contact point: at rest and upright the voltage
    produces accelerations with zero net moment about that point."""
    for p in (CORRECTED, CORRECTED.scaled(Mp=1.3, l=0.8, km=1.7)):
        beta = 2 * p.Mw + 2 * p.Iw / p.r**2 + p.Mp
        xdd, thdd = accelerations(np.zeros(4), 5.0, p)
        # d/dt of angular momentum about the contact point (CW positive), upright, at rest
        dH = (beta * p.r + p.Mp * p.l) * xdd - (p.Ip + p.Mp * p.l**2 + p.Mp * p.l * p.r) * thdd
        assert abs(xdd) > 1e-3 and dH == pytest.approx(0.0, abs=1e-9)
        # the input direction therefore fixes b = B2 / B4 from the inertias alone
        A, B = linear_model(p)
        assert B[1, 0] / B[3, 0] == pytest.approx((p.Ip + p.Mp * p.l**2 + p.Mp * p.l * p.r) / (p.Mp * p.l + p.r * beta))


def test_positive_voltage_drives_forward_and_tilts_back():
    """A forward wheel torque pushes the base forward and reacts on the body, tilting it back (CCW)."""
    xdd, thdd = accelerations(np.zeros(4), 5.0, CORRECTED)
    assert xdd > 0 and thdd > 0


def test_back_emf_is_dissipative():
    """With v = 0 the motors can only remove energy (the relative-speed form guarantees it)."""
    p = CORRECTED
    beta = 2 * p.Mw + 2 * p.Iw / p.r**2 + p.Mp
    rng = np.random.default_rng(1)
    for _ in range(50):
        X = rng.normal(size=4) * [1, 1, 0.5, 2]
        xd, thd = X[1], X[3]
        xdd, thdd = accelerations(X, 0.0, p)
        c, s = np.cos(X[2]), np.sin(X[2])
        M = np.array([[beta, -p.Mp * p.l * c], [-p.Mp * p.l * c, p.Ip + p.Mp * p.l**2]])
        qd, qdd = np.array([xd, thd]), np.array([xdd, thdd])
        dT = qd @ M @ qdd + p.Mp * p.l * s * thd * xd * thd  # d/dt of kinetic energy
        dV = -p.Mp * p.g * p.l * s * thd  # d/dt of potential energy
        assert dT + dV <= 1e-9


# --- sensors ----------------------------------------------------------------


def test_accelerometer_reads_tilt_at_rest():
    for th in (-0.3, 0.0, 0.2):
        ax, az = specific_force(np.array([0, 0, th, 0]), 0.0, 0.0, h=0.02)
        assert np.arctan2(-ax, az) == pytest.approx(th)


def test_accelerometer_reads_apparent_vertical_when_accelerating():
    """Leaning into a steady acceleration a (theta = -atan(a/g)) reads as level."""
    a = 2.0
    th = -np.arctan(a / 9.81)
    ax, az = specific_force(np.array([0, 0, th, 0]), a, 0.0, h=0.0)
    assert np.arctan2(-ax, az) == pytest.approx(0.0, abs=1e-12)


def test_encoders_measure_rotation_relative_to_body():
    s = SensorSuite(SensorConfig(noise=False, quantize=False), CORRECTED)
    # rotating the body counterclockwise (theta > 0) over a wheel fixed on the ground
    # turns the wheel clockwise relative to the body, the same sense as rolling forward
    m = s.measure(np.array([0.0, 0.0, 0.1, 0.0]), 0.0, 0.0, 0.01)
    assert m.pos == pytest.approx(CORRECTED.r * 0.1)
    # rolling forward turns the wheel clockwise relative to an upright body
    m = s.measure(np.array([0.02, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.01)
    assert m.pos == pytest.approx(0.02)


def test_encoder_quantization_and_differenced_velocity():
    s = SensorSuite(SensorConfig(noise=False), CORRECTED)
    m0 = s.measure(np.array([0.0, 0, 0, 0]), 0, 0, 0.01)
    m1 = s.measure(np.array([0.00145, 0, 0, 0]), 0, 0, 0.01)
    assert m0.vel == 0.0
    assert m1.counts == 10 and m1.pos == pytest.approx(10 * 0.000147)
    assert m1.vel == pytest.approx(m1.pos / 0.01)


def test_gyro_noise_matches_allan_analysis():
    s = SensorSuite(SensorConfig(bias_init="zero", quantize=False), CORRECTED, np.random.default_rng(0))
    gy = np.array([s.measure(np.zeros(4), 0, 0, 0.01).gy for _ in range(20000)])
    assert np.std(np.diff(gy)) / np.sqrt(2) == pytest.approx(np.sqrt(0.0012), rel=0.05)


# --- actuator ---------------------------------------------------------------


def test_actuator_matches_firmware_voltage_path():
    act = Actuator()
    assert act.apply(25.0) == pytest.approx(10.0)
    v = act.apply(1.23456)
    assert v <= 1.23456 and (1.23456 - v) < 1 / 5904.5
    assert Actuator(ActuatorConfig(gain=0.9)).apply(5.0) == pytest.approx(0.9 * np.trunc(5 * 5904.5) / 5904.5)


def test_actuator_backlash_holds_output():
    act = Actuator(ActuatorConfig(quantize=False, backlash=(0.5, -0.5)))
    out = [act.apply(u) for u in (0.0, 0.3, 1.0, 0.8)]
    assert out[1] == 0.0  # still inside the band
    assert out[2] == pytest.approx(0.5)
    assert out[3] == pytest.approx(0.5)  # small reversal: held


# --- baseline controller ----------------------------------------------------


def test_lqr_stabilizes_corrected_linear_model():
    A, B = linear_model()
    F, G = c2d_zoh(A, B, 0.01)
    K = design_lqr()
    assert np.max(np.abs(np.linalg.eigvals(F - G @ K[None, :]))) < 1


def test_hardware_default_gains_do_not_stabilize_corrected_model():
    """Documented finding: the firmware's compiled-in gains are not a working baseline."""
    A, B = linear_model()
    F, G = c2d_zoh(A, B, 0.01)
    assert np.max(np.abs(np.linalg.eigvals(F - G @ HARDWARE_DEFAULT_GAINS[None, :]))) > 1


@pytest.mark.parametrize("noise", [False, True])
def test_baseline_balances_with_kalman(noise):
    cfg = SimConfig(duration=12.0)
    cfg.sensors.noise = noise
    r = simulate(LQRController(design_lqr()), KalmanEstimator(), cfg)
    assert not r.fell
    m = r.metrics(settle_from=6.0)
    assert m["rms_tilt_deg"] < 0.05
    assert m["est_rms_tilt_deg"] < 0.05


def test_firmware_style_complementary_estimator_balances():
    """The accelerometer tilt shifts by ~xddot/g, but with the corrected motor-torque
    sign the complementary filter and this LQR balance.  (The first corrected baseline
    found a runaway here; it was an artifact of the sign error.)"""
    r = simulate(LQRController(design_lqr()), ComplementaryEstimator(), SimConfig(duration=10.0))
    assert not r.fell
    assert r.metrics(settle_from=5.0)["est_rms_tilt_deg"] < 0.1


def test_thesis_initial_condition_is_recoverable_within_10V():
    cfg = SimConfig(x0=(1.0, 0.3, 10 * DEG, 1 * DEG), duration=5.0)
    r = simulate(LQRController(design_lqr()), KalmanEstimator(), cfg)
    assert not r.fell
    assert np.max(np.abs(r.v_applied)) < 10.0


def test_one_step_delay_is_tolerated():
    r = simulate(LQRController(design_lqr()), KalmanEstimator(), SimConfig(duration=10.0, delay_steps=1))
    assert not r.fell


def test_model_residual_is_zero_for_tiny_noise_free_motion():
    """The corrected nonlinear plant matches its own linear model for tiny motion,
    and clearly does not match the thesis model."""
    cfg = SimConfig(x0=(0, 0, 0.05 * DEG, 0), duration=1.0)
    cfg.sensors.noise = False
    r = simulate(LQRController(design_lqr()), KalmanEstimator(), cfg)
    assert np.max(np.abs(r.model_residual())) < 1e-6
    assert np.max(np.abs(r.model_residual(THESIS, "xdot"))) > 1e-5


def test_play_operator_backlash():
    from twip.actuators import Backlash

    bl = Backlash(0.0, 1.0, 0.5, -0.5)
    assert [bl.step(v) for v in (0.3, 1.0, 0.8, 0.2, -0.4)] == pytest.approx([0.0, 0.5, 0.5, 0.5, 0.1])
