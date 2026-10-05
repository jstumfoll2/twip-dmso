"""Baseline controller, reference state estimators and closed-loop simulation.

This is the reference that a new controller or observer design is compared
against.  Everything runs on the corrected plant (:mod:`twip.dynamics`), sensor
(:mod:`twip.sensors`) and actuator (:class:`twip.actuators.Actuator`) models.

* :func:`design_lqr` - the primary baseline: discrete LQR (``lqrd``) designed on the
  corrected linear model.  The default weights are the ones in the final thesis
  simulation, so differences come from the plant correction, not a retune.
* :data:`HARDWARE_DEFAULT_GAINS` - the gains compiled into firmware v7-v9,
  applied here in the model frame.  On the robot potentiometers overrode them;
  the gains actually used are recovered from the logs by
  ``scripts/validate_plant.py`` (see ``docs/BASELINE.md``).
* :class:`ComplementaryEstimator` - the firmware's measurement pipeline without
  the DMSO: complementary-filtered tilt, gyro rate, tilt-compensated encoders.
* :class:`KalmanEstimator` - a textbook 4-state Kalman filter on the corrected model.
* :func:`simulate` - closed loop with the firmware's loop order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .actuators import Actuator, ActuatorConfig
from .control import c2d_zoh, lqrd
from .dynamics import BackEMF, Disturbance, accelerations, derivative, linear_model
from .integrators import rk4
from .params import CORRECTED, RobotParams
from .sensors import DEG, Measurement, SensorConfig, SensorSuite

HARDWARE_DEFAULT_GAINS = np.array([-1.5811, -2.3951, 93.13, 14.9217])
LQR_Q = np.diag([100.0, 50.0, 1e-4, 1e-4])  # final thesis DMSO simulation
LQR_R = 1000.0


def design_lqr(
    p: RobotParams = CORRECTED,
    dt: float = 0.01,
    Q: np.ndarray = LQR_Q,
    R: float = LQR_R,
    back_emf: BackEMF = "relative",
) -> np.ndarray:
    """Discrete LQR gain ``K`` (for ``u = -K x``) on the linearized plant."""
    A, B = linear_model(p, back_emf)
    return lqrd(A, B, Q, R, dt).ravel()


# ---------------------------------------------------------------------------
# estimators


class Estimator(Protocol):
    def initialize(self, x0: np.ndarray) -> None:
        """Warm start at a known state (the firmware's filters had converged before balancing)."""
        ...

    def update(self, m: Measurement, v_prev: float) -> np.ndarray:
        """State estimate ``[x, xdot, theta, thetadot]`` from this loop's measurement.

        ``v_prev`` is the voltage the firmware commanded over the previous interval.
        """
        ...


class ComplementaryEstimator:
    """Tilt from a complementary filter (alpha = 0.99 as in v9), rate from the gyro,
    and position/velocity from the encoders corrected for body tilt (``pos - r*theta``).

    The filter is initialized from the first accelerometer reading.  On the robot
    it ran continuously while idle, so it had converged before balancing started.

    ``pos`` is the model-frame encoder reading of :mod:`twip.sensors`.  The firmware
    used its own logged position, which the logs indicate is mirrored relative to it
    (not yet confirmed in closed loop; see ``docs/BASELINE.md``).
    """

    def __init__(self, alpha: float = 0.99, r: float = CORRECTED.r):
        self.alpha, self.r = alpha, r
        self.tilt: float | None = None

    def initialize(self, x0: np.ndarray) -> None:
        self.tilt = float(x0[2])

    def update(self, m: Measurement, v_prev: float) -> np.ndarray:
        rate = m.gyro_rate
        if self.tilt is None:
            self.tilt = m.accel_tilt
        else:
            self.tilt = self.alpha * (self.tilt + rate * m.dt) + (1 - self.alpha) * m.accel_tilt
        return np.array([m.pos - self.r * self.tilt, m.vel - self.r * rate, self.tilt, rate])


class KalmanEstimator:
    """Textbook discrete Kalman filter on the corrected linear model.

    Measurements ``z = [pos, vel, accel_tilt, gyro_rate]`` with
    ``pos = x + r*theta`` and ``vel = xdot + r*thetadot`` (relative encoders).

    The accelerometer tilt is modeled as ``theta + (xddot - h*thetaddot)/g``
    (``accel_model=True``).  The robot's own acceleration shifts the apparent
    vertical, and since ``xddot`` and ``thetaddot`` are linear in the state and the
    applied voltage, this is a linear measurement with feedthrough.  Treating the
    accelerometer as ``theta + noise`` instead (``accel_model=False``) biases the
    tilt estimate by about ``xddot/g`` whenever the robot accelerates.

    ``R`` comes from the sensor model (encoder quantization, accelerometer and gyro
    white noise); ``tilt_std`` adds margin for linearization and model error.
    ``Q`` is white acceleration noise on both degrees of freedom.  The covariance
    update uses the Joseph form.
    """

    def __init__(
        self,
        dt: float = 0.01,
        p: RobotParams = CORRECTED,
        sensors: SensorConfig | None = None,
        tilt_std: float = 0.1,  # rad; accelerometer model error (robustness sweep in docs/BASELINE.md)
        accel_std: float = 1.0,  # m/s^2, process noise on xddot
        ang_accel_std: float = 5.0,  # rad/s^2, process noise on thetaddot
        back_emf: BackEMF = "relative",
        accel_model: bool = True,
    ):
        s = sensors or SensorConfig()
        A, B = linear_model(p, back_emf)
        self.F, G = c2d_zoh(A, B, dt)
        self.G = G.ravel()
        r = p.r
        self.H = np.array([[1, 0, r, 0], [0, 1, 0, r], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.Dv = np.zeros(4)  # measurement feedthrough from the applied voltage
        if accel_model:
            h = s.imu_height
            self.H[2] += (A[1] - h * A[3]) / p.g
            self.Dv[2] = (B[1, 0] - h * B[3, 0]) / p.g
        q = s.meters_per_count
        tilt_var = s.accel_white_std**2 + tilt_std**2
        self.R = np.diag([q**2 / 12, 2 * q**2 / 12 / dt**2, tilt_var, (s.gyro_white_std * DEG) ** 2])
        Gw = np.array([[dt**2 / 2, 0], [dt, 0], [0, dt**2 / 2], [0, dt]])
        self.Q = Gw @ np.diag([accel_std**2, ang_accel_std**2]) @ Gw.T
        self.x: np.ndarray | None = None
        self.P = np.diag([1e-4, 1e-2, 1e-2, 1e-2])

    def initialize(self, x0: np.ndarray) -> None:
        self.x = np.asarray(x0, float).copy()
        self.P = np.diag([1e-6, 1e-4, 1e-4, 1e-4])

    def update(self, m: Measurement, v_prev: float) -> np.ndarray:
        z = np.array([m.pos, m.vel, m.accel_tilt, m.gyro_rate])
        if self.x is None:  # no warm start: initialize from the first measurement
            th, w = z[2], z[3]
            self.x = np.array([z[0] - self.H[0, 2] * th, z[1] - self.H[1, 3] * w, th, w])
            return self.x.copy()
        x = self.F @ self.x + self.G * v_prev
        P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (P @ self.H.T).T).T
        # v_prev is also the voltage on the motors at this sample (it drives xddot now)
        self.x = x + K @ (z - self.H @ x - self.Dv * v_prev)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()


# ---------------------------------------------------------------------------
# controller


class LQRController:
    def __init__(self, K: np.ndarray, x_ref: np.ndarray | None = None):
        self.K = np.asarray(K, float).ravel()
        self.x_ref = np.zeros(4) if x_ref is None else np.asarray(x_ref, float)

    def __call__(self, xhat: np.ndarray, k: int) -> float:
        return float(-self.K @ (xhat - self.x_ref))


# ---------------------------------------------------------------------------
# closed-loop simulation


@dataclass
class SimConfig:
    dt: float = 0.01  # v9 loop period
    duration: float = 10.0
    # 5 deg from rest.  The thesis initial condition (1 m, 0.3 m/s, 10 deg, 1 deg/s) is
    # also recoverable on the corrected plant, peaking near 2.7 V.
    x0: tuple[float, float, float, float] = (0.0, 0.0, 5 * DEG, 0.0)
    plant: RobotParams = CORRECTED
    back_emf: BackEMF = "relative"
    disturbance: Disturbance | None = None  # additive (xddot, thetaddot) on the plant
    rk4_substeps: int = 10
    delay_steps: int = 0  # extra whole-sample delay between computing and applying u
    sensors: SensorConfig = field(default_factory=SensorConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    fall_angle: float = 45 * DEG  # stop the run if |theta| exceeds this
    warm_start: bool = True  # call estimator.initialize(x0) before the first loop
    seed: int | None = 0


@dataclass
class SimResult:
    time: np.ndarray  # (N+1,)
    x: np.ndarray  # (4, N+1) true state
    xhat: np.ndarray  # (4, N+1) estimate available at each loop
    u_cmd: np.ndarray  # (N,) controller output
    v_known: np.ndarray  # (N,) voltage the firmware believes it applied (clamped, quantized)
    v_applied: np.ndarray  # (N,) voltage reaching the motors
    meas: list[Measurement]
    fell: bool
    dt: float

    def model_residual(self, p_model: RobotParams = CORRECTED, back_emf: BackEMF = "relative") -> np.ndarray:
        """Lumped model uncertainty ``d[k] = x[k+1] - (F x[k] + G v_known[k])``, shape (4, N).

        This is the quantity a DMSO-type observer's ``fhat[k]`` should estimate
        when it propagates ``xhat[k+1] = F xhat[k] + G v[k] + fhat[k]``.  It is
        aligned so that ``d[k]`` belongs to the step *out of* ``k`` (the thesis
        plots were one sample off).
        """
        A, B = linear_model(p_model, back_emf)
        F, G = c2d_zoh(A, B, self.dt)
        n = len(self.v_known)
        return self.x[:, 1 : n + 1] - (F @ self.x[:, :n] + np.outer(G.ravel(), self.v_known))

    def metrics(self, settle_from: float = 0.0) -> dict[str, float]:
        k0 = int(round(settle_from / self.dt))
        x = self.x[:, k0:]
        err = self.xhat[:, k0:] - x
        v = self.v_applied[k0:]
        return {
            "fell": float(self.fell),
            "rms_tilt_deg": float(np.sqrt(np.mean(x[2] ** 2)) / DEG),
            "rms_position_m": float(np.sqrt(np.mean(x[0] ** 2))),
            "rms_velocity_mps": float(np.sqrt(np.mean(x[1] ** 2))),
            "control_energy_V2s": float(np.sum(v**2) * self.dt),
            "max_abs_voltage": float(np.max(np.abs(v))) if len(v) else 0.0,
            "saturated_fraction": float(np.mean(np.abs(self.v_known[k0:]) >= 10.0 - 1e-3)) if len(v) else 0.0,
            "est_rms_x": float(np.sqrt(np.mean(err[0] ** 2))),
            "est_rms_xdot": float(np.sqrt(np.mean(err[1] ** 2))),
            "est_rms_tilt_deg": float(np.sqrt(np.mean(err[2] ** 2)) / DEG),
            "est_rms_rate_dps": float(np.sqrt(np.mean(err[3] ** 2)) / DEG),
        }


def simulate(
    controller: Callable[[np.ndarray, int], float],
    estimator: Estimator,
    cfg: SimConfig | None = None,
) -> SimResult:
    """Run the closed loop.  Each loop ``k`` follows firmware v9:

    1. sample the sensors at ``x[k]`` (the accelerometer sees the acceleration
       produced by the voltage currently on the motors),
    2. estimate ``xhat[k]``,
    3. compute ``u[k]`` and send it to the motors, where it is held until the next loop.
    """
    cfg = cfg or SimConfig()
    dt = cfg.dt
    N = int(round(cfg.duration / dt))
    rng = np.random.default_rng(cfg.seed)
    sensors = SensorSuite(cfg.sensors, cfg.plant, rng)
    act = Actuator(cfg.actuator)

    x = np.full((4, N + 1), np.nan)
    xhat = np.full((4, N + 1), np.nan)
    u_cmd = np.zeros(N)
    v_known = np.zeros(N)
    v_applied = np.zeros(N)
    meas: list[Measurement] = []
    x[:, 0] = cfg.x0
    pending = [0.0] * cfg.delay_steps  # commands computed but not yet applied
    v_now_known, v_now = 0.0, 0.0  # voltage on the motors before the run starts
    fell = False
    if cfg.warm_start and hasattr(estimator, "initialize"):
        estimator.initialize(np.array(cfg.x0, float))

    def f(_t, X, v):
        return derivative(X, v, cfg.plant, cfg.back_emf, cfg.disturbance)

    last = N
    for k in range(N + 1):
        xdd, thdd = accelerations(x[:, k], v_now, cfg.plant, cfg.back_emf, cfg.disturbance)
        m = sensors.measure(x[:, k], xdd, thdd, dt)
        meas.append(m)
        xhat[:, k] = estimator.update(m, v_now_known)
        if k == N:
            break
        if abs(x[2, k]) > cfg.fall_angle:
            fell, last = True, k
            break
        u_cmd[k] = controller(xhat[:, k], k)
        pending.append(u_cmd[k])
        u_apply = pending.pop(0)
        v_now_known = act.command(u_apply)
        v_now = act.apply(u_apply)
        v_known[k], v_applied[k] = v_now_known, v_now
        x[:, k + 1] = rk4(lambda t, X: f(t, X, v_now), x[:, k], k * dt, (k + 1) * dt, dt / cfg.rk4_substeps)

    n = last
    return SimResult(
        time=np.arange(n + 1) * dt,
        x=x[:, : n + 1],
        xhat=xhat[:, : n + 1],
        u_cmd=u_cmd[:n],
        v_known=v_known[:n],
        v_applied=v_applied[:n],
        meas=meas[: n + 1],
        fell=fell,
        dt=dt,
    )
