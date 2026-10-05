"""Sensor model of the robot (corrected baseline).

Models what the v9 firmware actually read each loop, in the model's sign
convention:

* **MPU-9150 accelerometer** (``ax``, ``az`` in g): the specific force at the
  IMU location, a point ``imu_height`` above the axle on the body axis.  This
  includes the robot's own acceleration, not just gravity, so the firmware's
  tilt estimate ``atan2(-ax, az)`` is corrupted while the robot accelerates, as
  on the real robot.
* **MPU-9150 gyroscope** (``gy`` in deg/s): tilt rate.
* **Motor encoders**: they count wheel rotation *relative to the body*.  Rolling
  forward turns the wheel clockwise at ``xdot / r`` while ``theta`` is
  counterclockwise, so the relative angle is ``x / r + theta`` and
  ``pos = counts * 0.000147`` is ``x + r * theta``, quantized (see
  :mod:`twip.dynamics`).  Velocity is the firmware's backward difference
  ``(pos - pos_prev) / dt``.  The firmware's own logged position is the mirror
  image, ``-(x + r * theta)``, as inferred from the switch-on transients in
  ``scripts/validate_plant.py`` (thesis Section 5.3.3); that frame is not yet
  confirmed in closed loop (see ``docs/BASELINE.md``).

Noise levels are from the thesis's Allan-variance analysis (``allandata.mat``,
recomputed in :mod:`twip.analysis`).  Each inertial sensor gets white noise plus
a first-order Gauss-Markov bias, discretized exactly, and is quantized to its
LSB.

Corrections vs. :mod:`twip.legacy.sensors`: accelerometer kinematics, relative
encoders, velocity from differenced position (the thesis quantized velocity
directly at 0.0113 m/s, the 13 ms step), and gyro bias units (the thesis added a
deg/s bias to a rad/s signal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .params import CORRECTED, RobotParams

DEG = math.pi / 180


@dataclass
class SensorConfig:
    imu_height: float = 0.02  # m above the axle (electronics on the bottom plate)

    # gyroscope (deg/s); thesis Allan analysis: white variance 0.0012, bias variance 1.0986e-4
    gyro_white_std: float = math.sqrt(0.0012)
    gyro_bias_std: float = math.sqrt(1.0986e-4)
    gyro_bias_tau: float = 1610.0  # s
    gyro_lsb: float = 1 / 131  # deg/s, +-250 deg/s range

    # accelerometer (g, per axis).  The thesis characterized noise on the
    # tilt angle atan2(-ax, az) (variance 4.0474e-6 rad^2); near upright that is
    # the ax noise in g, so the same numbers apply per axis.
    accel_white_std: float = math.sqrt(4.0474e-6)
    accel_bias_std: float = math.sqrt(1.8929e-7)
    accel_bias_tau: float = 2160.0  # s
    accel_lsb: float = 1 / 16384  # g, +-2 g range

    # encoders: 64 CPR x 30:1 gearbox = 1920 counts per wheel revolution
    counts_per_rev: int = 1920
    meters_per_count: float = 0.000147  # firmware conversion (true value 2*pi*r/1920 = 0.00014726)

    noise: bool = True
    quantize: bool = True
    bias_init: str = "stationary"  # "stationary" draws the initial biases; "zero" starts them at 0


@dataclass
class Measurement:
    """One loop's worth of raw sensor data, in the firmware's units."""

    ax: float  # g
    az: float  # g
    gy: float  # deg/s
    counts: int  # encoder counts (relative wheel rotation)
    pos: float  # m, counts * meters_per_count  (~ x + r*theta; the firmware logged the mirror image)
    vel: float  # m/s, backward difference of pos
    dt: float  # s, time since the previous sample

    @property
    def accel_tilt(self) -> float:
        """The firmware's accelerometer tilt, ``atan2(-ax, az)`` in rad."""
        return math.atan2(-self.ax, self.az)

    @property
    def gyro_rate(self) -> float:
        """Tilt rate in rad/s."""
        return self.gy * DEG


def specific_force(X: np.ndarray, xdd: float, thdd: float, h: float, g: float = 9.81) -> tuple[float, float]:
    """Noise-free accelerometer reading ``(ax, az)`` in g at height ``h`` on the body axis.

    Axes are oriented so that ``atan2(-ax, az) == theta`` when the robot is at rest,
    which is the firmware's convention.
    """
    th, thd = X[2], X[3]
    s, c = math.sin(th), math.cos(th)
    # IMU position (x - h sin th, h cos th); acceleration in the world frame
    a_x = xdd - h * c * thdd + h * s * thd**2
    a_z = -h * s * thdd - h * c * thd**2
    f_x, f_z = a_x, a_z + g
    ax = -(f_x * c + f_z * s) / g
    az = (-f_x * s + f_z * c) / g
    return ax, az


class _GaussMarkov:
    def __init__(self, std: float, tau: float, rng: np.random.Generator, stationary: bool):
        self.std, self.tau, self.rng = std, tau, rng
        self.b = std * rng.standard_normal() if stationary else 0.0

    def step(self, dt: float) -> float:
        phi = math.exp(-dt / self.tau)
        self.b = phi * self.b + self.std * math.sqrt(1 - phi * phi) * self.rng.standard_normal()
        return self.b


class SensorSuite:
    """Generates :class:`Measurement` samples from the true plant state."""

    def __init__(self, cfg: SensorConfig | None = None, params: RobotParams = CORRECTED, rng: np.random.Generator | None = None):
        self.cfg = cfg or SensorConfig()
        self.p = params
        self.rng = rng or np.random.default_rng()
        stationary = self.cfg.bias_init == "stationary"
        c = self.cfg
        self._gb = _GaussMarkov(c.gyro_bias_std, c.gyro_bias_tau, self.rng, stationary)
        self._axb = _GaussMarkov(c.accel_bias_std, c.accel_bias_tau, self.rng, stationary)
        self._azb = _GaussMarkov(c.accel_bias_std, c.accel_bias_tau, self.rng, stationary)
        self._pos_prev: float | None = None

    def _q(self, value: float, lsb: float) -> float:
        return round(value / lsb) * lsb if self.cfg.quantize else value

    def measure(self, X: np.ndarray, xdd: float, thdd: float, dt: float) -> Measurement:
        """Sample the sensors at state ``X`` with body accelerations ``xdd``, ``thdd``."""
        c, rng = self.cfg, self.rng
        ax, az = specific_force(X, xdd, thdd, c.imu_height, self.p.g)
        gy = X[3] / DEG
        if c.noise:
            ax += self._axb.step(dt) + c.accel_white_std * rng.standard_normal()
            az += self._azb.step(dt) + c.accel_white_std * rng.standard_normal()
            gy += self._gb.step(dt) + c.gyro_white_std * rng.standard_normal()
        ax, az = self._q(ax, c.accel_lsb), self._q(az, c.accel_lsb)
        gy = self._q(gy, c.gyro_lsb)

        phi_rel = X[0] / self.p.r + X[2]  # wheel angle relative to the body
        counts_f = phi_rel * c.counts_per_rev / (2 * math.pi)
        counts = int(round(counts_f)) if c.quantize else counts_f
        pos = counts * c.meters_per_count if c.quantize else (X[0] + self.p.r * X[2])
        vel = 0.0 if self._pos_prev is None else (pos - self._pos_prev) / dt
        self._pos_prev = pos
        return Measurement(ax, az, gy, counts, pos, vel, dt)
