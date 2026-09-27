"""The Chapter 8 test matrix (thesis revision 8), run on the corrected plant.

Observer study (O1-O4): a truth trajectory is generated once per case by the
baseline LQR acting on the *true* state while tracking a slow position
sinusoid, which keeps the system excited.  Every estimator then processes the
same measurement and input sequences, so they are compared on identical
information.

Control study (C1-C4): each controller closes the loop itself, with the state
supplied by the baseline Kalman estimator (``twip.baseline.KalmanEstimator``),
tracking a constant 0.1 m/s velocity command as in the thesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .actuators import Actuator, ActuatorConfig
from .baseline import ComplementaryEstimator, KalmanEstimator, LQRController, SimConfig, design_lqr, simulate
from .control import c2d_zoh
from .dynamics import accelerations, derivative, linear_model
from .integrators import rk4
from .legacy.dynamics import uncert_extra_terms
from .observers import B_MSO, DMSO, KalmanPredictor, LegacyDMSOPredictor, TanhBasis, kalman_injection_gain
from .params import CORRECTED, RobotParams
from .sensors import DEG, SensorConfig, SensorSuite

DT = 0.01

# Parameter error used for the "uncertainty" cases: the perturbation of
# twipnonlinear_uncert.m (the thesis extra-control plant: l unchanged) with
# km x1.25 instead of x2.5.  The full thesis set, including km x2.5 or l x0.9,
# cannot be stabilized by the nominal LQR on the corrected plant even with
# perfect state knowledge.
PERTURBATION = dict(Mp=1.1, Mw=1.5, km=1.25, ke=0.95, R=1.2)


def perturbed_plant() -> RobotParams:
    return CORRECTED.scaled(**PERTURBATION)


def unmodeled(X: np.ndarray) -> tuple[float, float]:
    """Unmodeled terms of the thesis (twipnonlinear_uncert.m)."""
    return uncert_extra_terms(X)


# ---------------------------------------------------------------------------
# observer study


@dataclass
class ObserverCase:
    name: str
    noise: bool
    plant: RobotParams = CORRECTED
    disturbance: Callable | None = None
    duration: float = 30.0
    seed: int = 0


OBSERVER_CASES = {
    "O1": ObserverCase("O1", noise=False),
    "O2": ObserverCase("O2", noise=False, plant=perturbed_plant()),
    "O3": ObserverCase("O3", noise=True, plant=perturbed_plant()),
    "O4": ObserverCase("O4", noise=True, plant=perturbed_plant(), disturbance=unmodeled),
}


@dataclass
class Truth:
    time: np.ndarray
    x: np.ndarray  # (4, N+1)
    u: np.ndarray  # (N,) voltage applied over [k, k+1]
    y: np.ndarray  # (4, N+1) full-state measurement
    f: np.ndarray  # (2, N) true uncertainty Bm^T (x[k+1] - F x[k] - G u[k])
    F: np.ndarray
    G: np.ndarray


def reference(t: float) -> np.ndarray:
    w = 2 * math.pi / 8.0
    return np.array([0.3 * math.sin(w * t), 0.3 * w * math.cos(w * t), 0.0, 0.0])


def make_truth(case: ObserverCase) -> Truth:
    """Closed-loop truth: LQR on the true state, tracking ``reference``.

    The measurement ``y`` is the robot's pipeline: tilt-compensated encoder position
    and velocity, the complementary-filtered tilt (the "pre-filter" of Chapter 4),
    and the gyro rate.  Without noise, ``y = x``.
    """
    N = int(round(case.duration / DT))
    A, B = linear_model(CORRECTED)
    F, G = c2d_zoh(A, B, DT)
    K = design_lqr()
    rng = np.random.default_rng(case.seed)
    sensors = SensorSuite(SensorConfig(), case.plant, rng)
    comp = ComplementaryEstimator()
    act = Actuator(ActuatorConfig())

    x = np.zeros((4, N + 1))
    y = np.zeros((4, N + 1))
    u = np.zeros(N)
    x[:, 0] = (0.0, 0.0, 5 * DEG, 0.0)
    comp.initialize(x[:, 0])
    v = 0.0
    for k in range(N + 1):
        if case.noise:
            xdd, thdd = accelerations(x[:, k], v, case.plant, "relative", case.disturbance)
            m = sensors.measure(x[:, k], xdd, thdd, DT)
            y[:, k] = comp.update(m, v)
        else:
            y[:, k] = x[:, k]
        if k == N:
            break
        v = act.apply(float(-K @ (x[:, k] - reference(k * DT))))
        u[k] = v
        x[:, k + 1] = rk4(
            lambda t, X: derivative(X, v, case.plant, "relative", case.disturbance), x[:, k], k * DT, (k + 1) * DT, DT / 10
        )
    d = x[:, 1:] - (F @ x[:, :-1] + np.outer(G.ravel(), u))
    return Truth(np.arange(N + 1) * DT, x, u, y, B_MSO.T @ d, F, G.ravel())


def make_estimators(truth: Truth, noise: bool) -> dict[str, object]:
    """The five estimators of Table 8.2, all on the nominal corrected model."""
    F, G = truth.F, truth.G
    s = SensorConfig()
    if noise:
        # measurement noise of the y pipeline: encoder quantization, complementary tilt, gyro
        q = s.meters_per_count
        R = np.diag([q**2 / 12, 2 * q**2 / 12 / DT**2, (0.5 * DEG) ** 2, (s.gyro_white_std * DEG) ** 2])
    else:
        R = np.diag([1e-10] * 4)
    Gw = np.array([[DT**2 / 2, 0], [DT, 0], [0, DT**2 / 2], [0, DT]])
    Q_nom = Gw @ np.diag([0.05**2, 0.2**2]) @ Gw.T + 1e-12 * np.eye(4)  # sensor-noise level only
    Q_inf = Gw @ np.diag([2.0**2, 10.0**2]) @ Gw.T + 1e-10 * np.eye(4)  # inflated for parameter error
    return {
        "KF, nominal Q": KalmanPredictor(F, G, Q_nom, R),
        "KF, inflated Q": KalmanPredictor(F, G, Q_inf, R),
        "Augmented-state KF": KalmanPredictor(F, G, Q_nom, R, augmented=True, q_f=1e-7),
        "DMSO (rev. 7)": LegacyDMSOPredictor(F, G),
        "DMSO (rev. 8)": DMSO(F, G),
        "DMSO (rev. 8) + sigma-mod": DMSO(F, G, kappa=0.1),
        "DMSO (rev. 8), Kalman-gain injection": _kalman_injected_dmso(F, G, Q_nom, R),
    }


def _kalman_injected_dmso(F, G, Q, R) -> DMSO:
    Km = kalman_injection_gain(F, Q, R)
    d = DMSO(F, G, Km=Km)
    if d.gamma == 0.0:  # outside Theorem 3.1 (sigma_max(A) >= 1): empirical gain
        d.gamma = 0.002
    return d


@dataclass
class ObserverRun:
    xpred: np.ndarray  # (4, N) prediction of x[k+1] after processing y[k], u[k]
    fhat: np.ndarray  # (2, N) uncertainty estimate at step k (nan if the estimator has none)
    conditions: object | None = None


def run_observer(est, truth: Truth) -> ObserverRun:
    N = len(truth.u)
    xpred = np.zeros((4, N))
    fhat = np.full((2, N), np.nan)
    est.initialize(truth.y[:, 0])
    for k in range(N):
        xpred[:, k] = est.step(truth.y[:, k], truth.u[k])
        fh = getattr(est, "fhat", None)
        if fh is not None and len(fh):
            fhat[:, k] = fh
    cond = est.conditions() if hasattr(est, "conditions") else None
    return ObserverRun(xpred, fhat, cond)


def observer_error(run: ObserverRun, truth: Truth) -> np.ndarray:
    """``x[k+1] - xpred[k]``, shape (4, N)."""
    return truth.x[:, 1:] - run.xpred


def rms(a: np.ndarray, start: int) -> np.ndarray:
    return np.sqrt(np.nanmean(a[..., start:] ** 2, axis=-1))


# ---------------------------------------------------------------------------
# control study


@dataclass
class ControlCase:
    name: str
    plant: RobotParams = CORRECTED
    deadzone: float | None = None
    backlash: float | None = None  # half-width, V
    duration: float = 20.0
    seed: int = 0


CONTROL_CASES = {
    "C1": ControlCase("C1"),
    "C2": ControlCase("C2", plant=perturbed_plant()),
    "C3": ControlCase("C3", plant=perturbed_plant(), deadzone=0.5),
    "C4": ControlCase("C4", plant=perturbed_plant(), deadzone=0.5, backlash=0.25),
}

V_DES = 0.1  # m/s, constant velocity command (thesis extra-control study)


def control_sim_config(case: ControlCase) -> SimConfig:
    act = ActuatorConfig(deadzone=case.deadzone, backlash=(case.backlash, -case.backlash) if case.backlash else None)
    return SimConfig(
        dt=DT, duration=case.duration, x0=(0.0, 0.0, 5 * DEG, 0.0), plant=case.plant, actuator=act, seed=case.seed
    )
