"""Simulated sensor model used in the DMSO simulations.

Noise parameters come from the Allan variance / autocorrelation analysis in
``Tests/Allan Variance/allanprocess.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENCODER_POS_RES = 0.000147  # m per encoder count
ENCODER_VEL_RES = 0.0113  # m/s per count per sample


def quantize(value: float, q: float) -> float:
    """Round to the nearest multiple of ``q`` (literal port of the ``mod`` blocks).

    Python's ``%`` matches MATLAB ``mod`` for a positive divisor, including for
    negative ``value``.
    """
    rem = value % q
    return value - rem if rem <= q / 2 else value - rem + q


@dataclass
class SensorNoise:
    # gyroscope
    gyro_bias_var: float = 1.0986e-04  # walking bias variance
    gyro_bias_tau: float = 1610.0  # walking bias time constant
    gyro_white_var: float = 0.0012  # white noise variance, (deg/s)^2
    # accelerometer-derived tilt angle
    acc_bias_var: float = 1.8929e-07
    acc_bias_tau: float = 2160.0
    # The scripts define accvar = 4.0387e-06 but actually use 4.0474e-06.
    acc_white_var: float = 4.0474e-06  # rad^2


class SensorModel:
    """Produces ``y = [x, xdot, theta, thetadot]`` measurements from the true state.

    * Position/velocity: quantized to encoder resolution, no random noise.
    * Tilt: truth + white noise + first-order Gauss-Markov bias.
    * Tilt rate: truth + white noise (deg/s converted to rad/s) + Gauss-Markov bias.

    The bias sample added at step k+1 is the one from step k, as in the MATLAB code.
    ``gyro_bias_var`` comes from deg/s data but is added to a rad/s signal
    without conversion, again as in the MATLAB code.
    """

    def __init__(self, dt: float, noise: SensorNoise | None = None, rng: np.random.Generator | None = None, enabled: bool = True):
        self.taus = 1 / dt
        self.n = noise or SensorNoise()
        self.rng = rng or np.random.default_rng()
        self.enabled = enabled
        self.ba = 0.0
        self.bg = 0.0

    def initial(self, x: np.ndarray) -> np.ndarray:
        """Measurement at k=0: exact position/velocity, noisy angles, no bias."""
        if not self.enabled:
            return np.array(x[:4], float)
        return np.array(
            [
                x[0],
                x[1],
                x[2] + self.rng.standard_normal() * np.sqrt(self.n.acc_white_var),
                x[3] + self.rng.standard_normal() * np.sqrt(self.n.gyro_white_var) * np.pi / 180,
            ]
        )

    def measure(self, x: np.ndarray) -> np.ndarray:
        n, taus, rng = self.n, self.taus, self.rng
        ba_prev, bg_prev = self.ba, self.bg
        # Draw in the MATLAB order (accel bias, accel white, gyro bias, gyro
        # white) so runs with a given seed stay reproducible.
        self.ba = ba_prev + 1 / taus * (
            -1 / n.acc_bias_tau * ba_prev + np.sqrt(2 * taus * n.acc_bias_var / n.acc_bias_tau) * rng.standard_normal()
        )
        theta = x[2] + rng.standard_normal() * np.sqrt(n.acc_white_var) + ba_prev
        self.bg = bg_prev + 1 / taus * (
            -1 / n.gyro_bias_tau * bg_prev + np.sqrt(2 * taus * n.gyro_bias_var / n.gyro_bias_tau) * rng.standard_normal()
        )
        thetadot = x[3] + rng.standard_normal() * np.sqrt(n.gyro_white_var) * np.pi / 180 + bg_prev

        if not self.enabled:
            return np.array(x[:4], float)
        return np.array(
            [
                quantize(x[0], ENCODER_POS_RES),
                quantize(x[1], ENCODER_VEL_RES),
                theta,
                thetadot,
            ]
        )
