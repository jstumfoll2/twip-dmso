import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from twip.analysis import replay_filters
from twip.control import c2d_zoh, lqrd
from twip.dynamics import linear_model
from twip.estimators import DMSO, LinearKalman, tansig

A, B = linear_model()
DT = 0.01
F, G = c2d_zoh(A, B, DT)


def test_tansig_definition():
    n = np.linspace(-5, 5, 101)
    np.testing.assert_allclose(tansig(n), 2 / (1 + np.exp(-2 * n)) - 1, atol=1e-15)


def test_kalman_reaches_steady_state_gain():
    Q = np.diag([1e-4, 1e-2, 1e-4, 1e-2])
    R = np.diag([1e-6, 1e-5, 4e-6, 2e-5])
    kf = LinearKalman(F, G, Q, R)
    for _ in range(2000):
        kf.step(np.zeros(4))
    Pinf = solve_discrete_are(F.T, np.eye(4), Q, R)  # a-priori covariance
    Ppost = Pinf - Pinf @ np.linalg.inv(Pinf + R) @ Pinf
    np.testing.assert_allclose(kf.P, Ppost, rtol=1e-6, atol=1e-14)


def test_kalman_scalar_R_broadcasts_like_matlab():
    kf = LinearKalman(F, G, np.eye(4) * 1e-3, 0.1)
    kf.step(np.ones(4))  # must not raise; S = H P H' + 0.1 on every element


def _closed_loop_with_disturbance(d, steps=3000):
    K = lqrd(A, B, np.diag([100, 50, 1e-4, 1e-4]), 1000, DT).ravel()
    m = DMSO(F, G, np.eye(4), gamma=0.1)
    x = np.array([0.1, 0.0, 0.05, 0.0])
    u = 0.0
    for _ in range(steps):
        innov = x - m.xhat
        xh, f = m.step(x, u)
        m.update_weights(innov)
        x = F @ x + G.ravel() * u + d
        u = -K @ xh
    return x, xh, f


def test_dmso_tracks_state_without_disturbance():
    x, xh, f = _closed_loop_with_disturbance(np.zeros(4))
    np.testing.assert_allclose(xh, x, atol=1e-9)
    np.testing.assert_allclose(f, 0, atol=1e-9)


def test_dmso_recovers_matched_constant_disturbance():
    d = np.array([0, 0.002, 0, -0.003])
    x, xh, f = _closed_loop_with_disturbance(d)
    np.testing.assert_allclose(f, d[[1, 3]], rtol=1e-6)
    np.testing.assert_allclose(xh, x, atol=1e-8)


# --- regression against real robot logs (twip_v4 firmware, filteringtest12) ---


@pytest.fixture(scope="module")
def replay(filtering_log):
    return replay_filters(filtering_log, firmware_bug=True)


def test_complementary_filter_matches_robot(filtering_log, replay):
    # Logged values are printed with 3 decimals, so allow ~0.02 deg of drift.
    assert np.max(np.abs(replay["thetac"] - filtering_log["pitchc"])) < 0.02


def test_angle_bias_kalman_matches_robot(filtering_log, replay):
    assert np.max(np.abs(replay["thetak"] - filtering_log["pitchk"])) < 0.02
    assert np.max(np.abs(replay["bias"] - filtering_log["bias"])) < 0.03


def test_dmso2_with_firmware_bug_matches_robot(filtering_log, replay):
    assert np.max(np.abs(replay["mso"] - filtering_log["pitchmso"])) < 0.005
    assert np.max(np.abs(replay["msodot"] - filtering_log["pitchdotmso"])) < 0.002
    assert np.max(np.abs(replay["fhat1"] - filtering_log["fhat1"])) < 0.002
    assert np.max(np.abs(replay["fhat2"] - filtering_log["fhat2"])) < 0.002


def test_documented_quirks_are_what_the_robot_ran(filtering_log):
    """The "fixed" versions do NOT reproduce the robot, confirming both quirks."""
    from twip.estimators import AngleBiasKalman

    fixed = replay_filters(filtering_log, firmware_bug=False)
    assert np.max(np.abs(fixed["mso"] - filtering_log["pitchmso"])) > 1.0

    kf = AngleBiasKalman(literal=False)
    d = filtering_log
    bias = np.array([kf.step(d["gy"][i], d["ax"][i], d["az"][i], d["dt"][i])[1] for i in range(len(d["dt"]))])
    assert np.max(np.abs(bias - d["bias"])) > 0.5
