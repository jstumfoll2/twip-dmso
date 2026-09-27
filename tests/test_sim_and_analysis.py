from dataclasses import replace

import numpy as np
import pytest

from twip.analysis import allan, gauss_markov_bias_variance, replay_implementation
from twip.sim_dmso import preset, run

DEG = np.pi / 180


def test_nominal_closed_loop_balances_and_settles():
    r = run(replace(preset("no_noise_no_uncertainty"), steps=3000))  # 30 s; the position loop is slow
    assert np.max(np.abs(r.xtrue[2])) < 15 * DEG
    np.testing.assert_allclose(r.xtrue[:, -1], 0, atol=5e-3)
    # noise-free, so after convergence the DMSO tracks truth closely
    assert np.max(np.abs(r.xtrue[2, 200:] - r.xhat[2, 200:])) < 1e-3


@pytest.mark.parametrize(
    "name", ["no_noise_no_uncertainty", "noise_with_uncertainty", "noise_with_uncertainty_perturbed"]
)
def test_presets_run_and_stay_upright(name):
    r = run(replace(preset(name), steps=min(preset(name).steps, 1000)))
    assert np.all(np.isfinite(r.xtrue))
    assert np.max(np.abs(r.xtrue[2])) < 20 * DEG
    assert np.max(np.abs(r.u)) <= preset(name).saturation + 1e-12


def test_seed_makes_noisy_sim_reproducible():
    cfg = replace(preset("noise_with_uncertainty"), steps=200)
    np.testing.assert_array_equal(run(cfg).y, run(cfg).y)


def test_backlash_flag_has_no_effect_like_the_matlab_script():
    cfg = replace(preset("noise_with_uncertainty"), steps=300)
    a, b = run(cfg), run(replace(cfg, backlash=False))
    np.testing.assert_array_equal(a.xtrue, b.xtrue)


def test_dmso_beats_kalman_on_position_in_noise_scenario():
    """The thesis headline result (qualitatively): DMSO position error << Kalman's."""
    r = run(replace(preset("noise_with_uncertainty"), steps=500))
    s = slice(50, None)
    e_dmso = np.sqrt(np.mean((r.xtrue[0, s] - r.xhat[0, s]) ** 2))
    e_kf = np.sqrt(np.mean((r.xtrue[0, s] - r.kalman[0, s]) ** 2))
    assert e_dmso < e_kf / 10


# --- Allan variance ---


def test_allan_white_noise_slope():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    tau = np.array([0.01, 0.1, 1.0, 10.0])
    r = allan(x, 100.0, tau)
    # white noise: sigma(tau) = sigma / sqrt(tau * rate)
    np.testing.assert_allclose(r.osig, 1 / np.sqrt(tau * 100), rtol=0.1)


def test_allan_matches_stored_matlab_results(allan_mat):
    data, av, tau = allan_mat["data"], allan_mat["avar"], allan_mat["tau"]
    idx = np.arange(0, len(tau), 37)  # subsample tau to keep the test fast
    r = allan(data.freq, data.rate, tau[idx])
    for k in ("sig", "sig2", "osig", "osigerr"):
        ref = np.asarray(getattr(av, k))[idx]
        got = getattr(r, k)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        ok = ~np.isnan(ref)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=1e-9)


def test_gauss_markov_variance_of_white_noise_shrinks_with_binning():
    x = np.random.default_rng(0).standard_normal(140 * 5000)
    assert gauss_markov_bias_variance(x) == pytest.approx(1 / 140, rel=0.1)


# --- hardware replay ---


def test_implementation_replay_runs(implementation_log):
    r = replay_implementation(implementation_log)
    assert r.kalman.shape[1] == len(implementation_log["dt"])
    assert np.all(np.isfinite(r.xhatmso))


def test_R_overwrite_bug_degrades_kalman(implementation_log):
    buggy = replay_implementation(implementation_log, reproduce_R_bug=True)
    fixed = replay_implementation(implementation_log, reproduce_R_bug=False)
    s = slice(310, -1)
    acc = buggy.accpitch[s]
    rms = lambda r: np.sqrt(np.mean((r.kalman[2, 1:][s] / DEG - acc) ** 2))  # noqa: E731
    assert rms(fixed) < 0.1 < 1.0 < rms(buggy)


def test_firmware_v9_dmso_reproduces_robot_log(v9_log):
    """The on-board DMSO behind the hardware results, recomputed from the raw log columns."""
    from twip.analysis import replay_firmware_v9

    r = replay_firmware_v9(v9_log)
    # logged with 3-4 decimals
    for k, tol in (("xhat", 2e-3), ("xhatdot", 2e-3), ("pitchm", 2e-3), ("gyhat", 2e-3)):
        assert np.max(np.abs(r[k] - v9_log[k])) < tol, k


def test_v8_log_format():
    from twip.data import V8_COLUMNS, load_log, thesis_path

    path = thesis_path("Thesis", "Thesis Programs", "Implementation", "implementationtest1.txt")
    if not path.exists():
        pytest.skip("data not found")
    log = load_log(path, V8_COLUMNS)
    assert len(log["dt"]) > 1000
    assert np.all(np.abs(log["dt"]) < 0.1)
