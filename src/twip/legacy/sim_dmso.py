"""Closed-loop DMSO vs. Kalman simulation (port of ``main_bala_discrete.m``).

The nonlinear plant is driven by LQR acting on the DMSO state estimate.  A
4-state linear Kalman filter runs alongside for comparison, and the DMSO's
uncertainty estimate ``fhat`` is compared with the "actual" uncertainty
(true state minus the one-step linear prediction).

The thesis has three copies of this script, one per results section.  They
differ in more than their folder names suggest (see :func:`preset`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..actuators import deadzone, saturate
from .actuators import backlash_legacy
from ..control import c2d_zoh, lqrd
from .dynamics import linear_model, twip_nonlinear
from ..estimators import LinearKalman
from .observers import DMSO
from ..integrators import rk4
from ..params import THESIS, RobotParams
from .sensors import SensorModel, SensorNoise

DEG = np.pi / 180


@dataclass
class DMSOSimConfig:
    dt: float = 0.01
    steps: int = 1000
    x_init: tuple[float, float, float, float] = (1.0, 0.3, 10 * DEG, 1 * DEG)
    x_des: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # plant ("truth"); the controller and observers always use THESIS
    plant: RobotParams = THESIS
    rk4_substeps: int = 5

    # measurement noise
    noise: bool = True
    seed: int | None = 0

    # actuator nonlinearities
    saturation: float | None = 11.1
    deadzone: float | None = None
    backlash: bool = False  # legacy (no-op) backlash, see actuators.backlash_legacy
    backlash_m: float = 1.0
    backlash_d_plus: float = 10 * DEG
    backlash_d_minus: float = -10 * DEG

    # LQR
    Qlqr: np.ndarray = field(default_factory=lambda: np.diag([100, 50, 1e-4, 1e-4]))
    Rlqr: float = 1000.0

    # DMSO
    Kmso: np.ndarray = field(default_factory=lambda: np.eye(4))
    gamma: float = 0.1
    sigma: float = 0.0
    # "current": innovation y[k] - xhat[k] (final thesis scripts)
    # "next":    innovation y[k+1] - xhat[k+1] (both "no noise" scripts)
    weight_innovation: Literal["current", "next"] = "current"

    # 4-state Kalman for comparison
    Qkf: np.ndarray = field(default_factory=lambda: np.diag([1e-4, 1e-2, 1e-4, 1e-2]))
    Rkf: np.ndarray = field(default_factory=lambda: np.diag([1e-6, 1e-5, 4e-6, 2e-5]))
    P0kf: np.ndarray = field(default_factory=lambda: 0.001 * np.eye(4))


@dataclass
class DMSOSimResult:
    time: np.ndarray  # (N+1,)
    xtrue: np.ndarray  # (4, N+1)
    u: np.ndarray  # (N+1,) control actually applied (after nonlinearities)
    u_unshaped: np.ndarray  # (N+1,) control before backlash/deadzone ("unobs")
    xhat: np.ndarray  # (4, N+1) DMSO estimate
    kalman: np.ndarray  # (4, N+1) Kalman estimate
    fhat: np.ndarray  # (2, N) DMSO uncertainty estimate
    uncertainty: np.ndarray  # (4, N) true state minus one-step linear prediction
    y: np.ndarray  # (4, N+1) measurements
    W: np.ndarray  # (N+1, 5, 2) weight history
    Klqr: np.ndarray


def preset(name: str) -> DMSOSimConfig:
    """Configurations matching the thesis folders.

    ``"no_noise_no_uncertainty"``
        ``Thesis Programs/no noise no uncertainty``. 100 steps at dt=.01,
        nominal plant, no noise, Q=diag(10,10,1e-5,1e-4), sat=11.
    ``"no_noise_with_uncertainty"``
        ``Thesis Programs/no noise with uncertainty``. 15000 steps at dt=.001,
        perturbed plant, gamma=.5.
    ``"noise_with_uncertainty"``
        ``Thesis Programs/Noise with uncertainty`` *as it is on disk*.  The
        parameter-uncertainty block in its ``twipnonlinear.m`` is commented
        out, so the plant is nominal.  Backlash is enabled, but the legacy
        backlash is a no-op, so the only "uncertainty" left is
        saturation, sensor noise and linearization error.
    ``"noise_with_uncertainty_perturbed"``
        The ``Noise with uncertainty/New folder`` copy (a day older).  It has
        the perturbed plant, dt=.001 and backlash off.
    """
    no_noise_Q = np.diag([10, 10, 1e-5, 1e-4])
    if name == "no_noise_no_uncertainty":
        return DMSOSimConfig(
            dt=0.01, steps=100, noise=False, saturation=11.0, Qlqr=no_noise_Q, gamma=0.1, weight_innovation="next"
        )
    if name == "no_noise_with_uncertainty":
        return DMSOSimConfig(
            dt=0.001,
            steps=15000,
            noise=False,
            saturation=11.0,
            Qlqr=no_noise_Q,
            gamma=0.5,
            weight_innovation="next",
            plant=THESIS.with_thesis_uncertainty(),
        )
    if name == "noise_with_uncertainty":
        return DMSOSimConfig(dt=0.01, steps=1000, noise=True, backlash=True)
    if name == "noise_with_uncertainty_perturbed":
        return DMSOSimConfig(dt=0.001, steps=1000, noise=True, plant=THESIS.with_thesis_uncertainty())
    raise ValueError(f"unknown preset {name!r}")


def run(cfg: DMSOSimConfig) -> DMSOSimResult:
    dt, N = cfg.dt, cfg.steps
    time = np.arange(N + 1) * dt

    A, B = linear_model(THESIS)  # controller/observer model is always nominal
    F, G = c2d_zoh(A, B, dt)
    g = G.ravel()
    Klqr = lqrd(A, B, cfg.Qlqr, cfg.Rlqr, dt).ravel()
    x_des = np.asarray(cfg.x_des, float)

    rng = np.random.default_rng(cfg.seed)
    sensor = SensorModel(dt, SensorNoise(), rng, enabled=cfg.noise)
    mso = DMSO(F, G, cfg.Kmso, cfg.gamma, cfg.sigma)
    kf = LinearKalman(F, G, cfg.Qkf, cfg.Rkf, P0=cfg.P0kf.copy())

    xtrue = np.zeros((4, N + 1))
    u = np.zeros(N + 1)
    u_unshaped = np.zeros(N + 1)
    xhat = np.zeros((4, N + 1))
    kal = np.zeros((4, N + 1))
    fhat = np.zeros((2, N))
    unc = np.zeros((4, N))
    y = np.zeros((4, N + 1))
    Whist = np.zeros((N + 1, 5, 2))
    x_dt = np.zeros((4, N + 1))

    xtrue[:, 0] = cfg.x_init
    x_dt[:, 0] = cfg.x_init
    y[:, 0] = sensor.initial(xtrue[:, 0])

    def plant(u_k: float):
        return lambda t, x: twip_nonlinear(x, u_k, cfg.plant, convention="up")

    for i in range(N):
        # 4-state Kalman, using the measurement from step i
        kal[:, i + 1] = kf.step(y[:, i], u[i])

        # DMSO
        innov_now = y[:, i] - xhat[:, i]
        xhat[:, i + 1], fhat[:, i] = mso.step(y[:, i], u[i])

        # control computed from xhat[k+1]; applied over the *next* interval
        uk = -Klqr @ (xhat[:, i + 1] - x_des)
        if cfg.saturation is not None:
            uk = saturate(uk, cfg.saturation)
        u_unshaped[i + 1] = uk
        if cfg.backlash:
            uk = backlash_legacy(uk, u[i], cfg.backlash_m, cfg.backlash_d_plus, cfg.backlash_d_minus)
        if cfg.deadzone is not None:
            uk = deadzone(uk, cfg.deadzone)
        u[i + 1] = uk

        # truth: the plant over [t_i, t_i+1] is driven by u[i] (xtrue(5,i) in MATLAB)
        xtrue[:, i + 1] = rk4(plant(u[i]), xtrue[:, i], time[i], time[i + 1], dt / cfg.rk4_substeps)
        x_dt[:, i + 1] = F @ xtrue[:, i] + g * u[i]
        unc[:, i] = xtrue[:, i] - x_dt[:, i]

        y[:, i + 1] = sensor.measure(xtrue[:, i + 1])

        if cfg.weight_innovation == "current":
            mso.update_weights(innov_now)
        else:
            mso.update_weights(y[:, i + 1] - xhat[:, i + 1])
        Whist[i + 1] = mso.W

    return DMSOSimResult(time, xtrue, u, u_unshaped, xhat, kal, fhat, unc, y, Whist, Klqr)

