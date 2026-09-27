import numpy as np
import pytest

from twip.actuators import deadzone, saturate
from twip.legacy.actuators import Backlash, backlash_legacy
from twip.legacy.sensors import ENCODER_POS_RES, SensorModel, quantize

DEG = np.pi / 180


def test_saturate_and_deadzone():
    assert saturate(20, 11.1) == 11.1
    assert saturate(-20, 11.1) == -11.1
    assert saturate(3, 11.1) == 3
    assert deadzone(1.9, 2) == 0
    assert deadzone(-2.5, 2) == -2.5


def test_legacy_backlash_is_a_no_op_for_thesis_settings():
    rng = np.random.default_rng(0)
    u_prev = 0.0
    for u in rng.normal(scale=5, size=500):
        assert backlash_legacy(u, u_prev, 1.0, 10 * DEG, -10 * DEG) == u
        u_prev = u


def test_backlash_holds_output_inside_dead_band():
    bl = Backlash(y0=0.0, m=1.0, d_plus=0.5, d_minus=-0.5)
    v = np.concatenate([np.linspace(0, 2, 21), np.linspace(2, 1.2, 9), np.linspace(1.2, 0, 13)])
    out = [bl.step(v[k], v[k - 1]) if k else bl.y for k in range(len(v))]
    out = np.array(out)
    # output only starts moving once the input has cleared d_plus
    assert out[v.tolist().index(0.5)] == 0.0
    assert out[20] == pytest.approx(2 - 0.5)
    # small reversal (2 -> 1.2, less than the 1.0 band width) leaves output held
    assert np.all(out[21:30] == out[20])


@pytest.mark.parametrize(
    "value, expected",
    [(0.0, 0.0), (0.0001, 0.000147), (0.00007, 0.0), (-0.0001, -0.000147), (0.000147 * 3.4, 0.000147 * 3)],
)
def test_quantize_rounds_to_nearest_count(value, expected):
    assert quantize(value, ENCODER_POS_RES) == pytest.approx(expected, abs=1e-12)


def test_sensor_disabled_returns_truth():
    s = SensorModel(0.01, enabled=False, rng=np.random.default_rng(0))
    x = np.array([0.1234567, 0.3, 0.2, -0.1])
    np.testing.assert_array_equal(s.measure(x), x)
    np.testing.assert_array_equal(s.initial(x), x)


def test_sensor_white_noise_level():
    s = SensorModel(0.01, rng=np.random.default_rng(0))
    ys = np.array([s.measure(np.zeros(4)) for _ in range(20000)])
    # First differences remove the slowly walking bias, leaving sqrt(2) * white noise.
    white = lambda col: np.std(np.diff(ys[:, col])) / np.sqrt(2)  # noqa: E731
    assert white(2) == pytest.approx(np.sqrt(4.0474e-06), rel=0.05)
    assert white(3) == pytest.approx(np.sqrt(0.0012) * DEG, rel=0.05)
    assert np.all(ys[:, 0] == 0) and np.all(ys[:, 1] == 0)
