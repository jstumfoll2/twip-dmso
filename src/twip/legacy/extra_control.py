"""Neural-network "extra control" on top of LQR (port of ``ExtraControl_v5.m`` + ``NN1v3.m``).

Three systems are simulated side by side from the same initial state:

* ``xnom``: the *linear* discrete model under LQR (``u_opt``), the ideal reference.
* ``xtrue``: the perturbed nonlinear plant under LQR only (``u_lqr``).
* ``xextra``: the same perturbed plant under LQR plus extra control (``u = u_nom + u_e``).

The extra control is a backstepping-style design. The velocity loop picks a
virtual tilt ``xbar3`` using NN1's estimate ``Fhat1`` of the unknown velocity
dynamics. The tilt loop then computes ``u_e`` using NN2's estimate ``Fhat2``.
Both NNs are two-layer tansig networks: a fixed random input layer ``V`` and
adaptive weights ``We1`` (hidden) and ``We2`` (output), updated by ``NN1v3.m``'s
continuous-time laws integrated with RK4 over each sample.

The whole script uses the *down* angle convention (upright = pi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from ..actuators import deadzone, saturate
from .actuators import Backlash
from ..control import c2d_zoh, lqrd
from .dynamics import linear_model, twip_nonlinear, uncert_extra_terms
from .observers import DMSO, tansig
from ..integrators import rk4
from ..params import THESIS, RobotParams
from .sensors import SensorModel, SensorNoise

DEG = np.pi / 180
PI_E3 = np.array([0.0, 0.0, np.pi, 0.0])

# The "true" plant in ExtraControl_v5.m is twipnonlinear_uncert.m: perturbed
# parameters (l not scaled) plus two unmodeled terms.
UNCERT_PLANT = THESIS.with_thesis_uncertainty(l_scale=1.0)


def constant_velocity(v: float = 0.1) -> Callable[[int, float], tuple[float, float]]:
    """Desired velocity profile: returns ``(xdot_des, xddot_des)`` for step ``i``."""
    return lambda i, dt: (v, 0.0)


def sinusoidal_velocity(amp: float = 0.3, w: float = 0.01) -> Callable[[int, float], tuple[float, float]]:
    """The commented-out alternative in the script: ``.3*sin(.01*i)`` (i is the step index)."""
    return lambda i, dt: (amp * np.sin(w * i), w * amp * np.cos(w * i))


@dataclass
class ExtraControlConfig:
    dt: float = 0.001
    steps: int = 1000
    x_init: tuple[float, float, float, float] = (1.0, 0.3, np.pi + 10 * DEG, 1 * DEG)
    velocity: Callable[[int, float], tuple[float, float]] = field(default_factory=constant_velocity)

    # plant
    nonlinear: bool = True
    plant: RobotParams = UNCERT_PLANT
    plant_extra_terms: bool = True  # 0.5*sin(x)*xdot and 0.25*xdot^2 from twipnonlinear_uncert.m
    rk4_substeps: int = 10
    # Additive disturbance for the *linear* plant option (``uncertainty`` in the script, zeros by default).
    linear_uncertainty: Callable[[int, np.ndarray], np.ndarray] | None = None

    # controller design model; ``flag.parameteruncertainty`` perturbs this, not the plant
    model_parameter_uncertainty: bool = False

    # sensors
    noise: bool = False
    seed: int | None = 0

    # nonlinearities
    saturation: float | None = 10.0
    deadzone: float | None = None  # the script's value when enabled: 0.5
    tilt_backlash: tuple[float, float] | None = None  # (d_plus, d_minus); the script uses +-0.1 deg

    # LQR
    Qlqr: np.ndarray = field(default_factory=lambda: np.diag([1, 1, 0.1, 0.1]))
    Rlqr: float = 1.0

    # extra control
    extra: bool = True
    ke0: float = 1.0
    ke1: float = 2.5  # e2 gain (NN1 loop)
    ke3: float = 1.0  # e1 gain
    ke2: float = 2.5  # ebar4 gain (NN2 loop)
    ke4: float = 1.0  # ebar3 gain
    gamma11: float = 0.1
    sigma11: float = 0.01
    gamma12: float = 0.1
    sigma12: float = 0.01
    gamma21: float = 0.1
    sigma21: float = 0.01
    gamma22: float = 0.1
    sigma22: float = 0.01
    hidden1: int = 10
    hidden2: int = 10
    nn_substeps: int = 5
    # The script trains NN2 on e(4) = x4 - x4des rather than ebar(4) = x4 - xbar4.
    nn2_error: Literal["e4", "ebar4"] = "e4"

    # DMSO running alongside on the LQR-only plant (its output is not used for control)
    mso_gamma: float = 0.2
    mso_sigma: float = 0.0


@dataclass
class ExtraControlResult:
    time: np.ndarray
    x_des: np.ndarray  # (4, N+1), down convention
    xtrue: np.ndarray  # (4, N+1) LQR only
    xextra: np.ndarray  # (4, N+1) LQR + extra control
    xnom: np.ndarray  # (4, N+1) linear model, angle relative to upright
    u: np.ndarray  # (N,) total control applied to xextra
    u_nom: np.ndarray
    u_e: np.ndarray
    u_lqr: np.ndarray
    u_opt: np.ndarray
    Fhat: np.ndarray  # (2, N)
    Ftrue: np.ndarray  # (2, N)
    xbar: np.ndarray  # (4, N)
    xhat: np.ndarray  # (4, N+1) DMSO estimate (upright-relative)
    fhat: np.ndarray  # (2, N)
    error_lqr: float
    error_extra: float
    error_nom: float
    control_effort: float


def preset(name: str) -> ExtraControlConfig:
    """Configurations for ``ExtraControl_v5.m`` and the figure sets beside it.

    ``"as_saved"``
        The script exactly as saved (29 May 2015): ``ke0 = 1``, 1 s at dt=.001, perturbed
        plant from ``twipnonlinear_uncert.m``.  **The extra-controlled system diverges
        with these settings.** The virtual tilt rate ``xbar4 = ke0 * d(xbar3)/dt`` spikes to
        ~20 rad/s while the control is saturated, and ``u_e`` then drives the tilt away.
        The LQR-only and nominal systems are unaffected.

    Figure presets, reconstructed for the folders in ``Thesis Programs/Extra control/``.
    The figures (26 May 2015) predate the saved script, and the intermediate version that
    made them is not in the folder.  These settings were found by matching the saved
    figures: ``ke0 = 0`` (as in ``ExtraControl_v4.m``) and dt = .01.  The deadzone
    figures used the *nominal* nonlinear plant.

    ``"unmodeled_dynamics"``
        Perturbed plant + unmodeled terms, 7.5 s.  Nominal and LQR-only match
        ``Unmodeled Dynamics/figure1.jpg``; the extra-control curve has the same shape.
    ``"parameter_uncertainty"``
        Perturbed plant without the unmodeled terms, 7.5 s.
    ``"deadzone"``
        Nominal plant, 0.5 V deadzone, 7.5 s.  All three curves match ``Deadzone/figure1.jpg``.
    ``"deadzone_and_backlash"``
        Nominal plant, deadzone + +-0.1 deg tilt backlash, 15 s.  Matches
        ``Deadzone and Backlash/figure1.jpg``.  The backlash acts on the tilt *state*, so
        it only releases once ``|thetadot| * dt`` exceeds the band, which makes it very
        sensitive to dt.
    """
    if name == "as_saved":
        return ExtraControlConfig()
    fig = dict(dt=0.01, steps=750, ke0=0.0)
    if name == "unmodeled_dynamics":
        return ExtraControlConfig(**fig)
    if name == "parameter_uncertainty":
        return ExtraControlConfig(plant_extra_terms=False, **fig)
    nominal_plant = dict(plant=THESIS, plant_extra_terms=False)
    if name == "deadzone":
        return ExtraControlConfig(deadzone=0.5, **nominal_plant, **fig)
    if name == "deadzone_and_backlash":
        return ExtraControlConfig(
            deadzone=0.5, tilt_backlash=(0.1 * DEG, -0.1 * DEG), **nominal_plant, **{**fig, "steps": 1500}
        )
    raise ValueError(f"unknown preset {name!r}")


def _nn_update(
    W2: np.ndarray, W1: np.ndarray, phi1: np.ndarray, phi2: np.ndarray, err: float,
    g1: float, s1: float, g2: float, s2: float, t0: float, t1: float, h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the ``NN1v3.m`` weight laws over one sample with RK4.

    ``W1dot = -g1*(phi1 (W1' phi1 + 1*err)' + s1*W1)`` and ``W2dot = g2*(phi2*err - s2*W2)``,
    with ``phi1``, ``phi2`` and ``err`` held constant over the sample.
    """
    n_in, n_hid = W1.shape
    ones = np.ones(n_hid)

    def f(_t, z):
        w2 = z[:n_hid]
        w1 = z[n_hid:].reshape(n_in, n_hid)
        dw1 = -g1 * (np.outer(phi1, w1.T @ phi1 + ones * err) + s1 * w1)
        dw2 = g2 * (phi2 * err - s2 * w2)
        return np.concatenate([dw2, dw1.ravel()])

    z = rk4(f, np.concatenate([W2, W1.ravel()]), t0, t1, h)
    return z[:n_hid], z[n_hid:].reshape(n_in, n_hid)


def run(cfg: ExtraControlConfig) -> ExtraControlResult:
    dt, N = cfg.dt, cfg.steps
    time = np.arange(N + 1) * dt
    rng = np.random.default_rng(cfg.seed)

    p_model = THESIS.with_thesis_uncertainty() if cfg.model_parameter_uncertainty else THESIS
    A, B = linear_model(p_model)
    F, G = c2d_zoh(A, B, dt)
    g = G.ravel()
    b2, b4 = B[1, 0], B[3, 0]
    K = lqrd(A, B, cfg.Qlqr, cfg.Rlqr, dt).ravel()

    # Steady tilt (and voltage) needed to hold a constant forward velocity.
    Asub = np.array([[A[1, 2], b2], [A[3, 2], b4]])
    Bsub = np.array([-A[1, 1], -A[3, 1]])

    def tilt_for_velocity(v: float) -> float:
        return float(np.linalg.solve(Asub, Bsub * v)[0])

    extra_terms = uncert_extra_terms if cfg.plant_extra_terms else None

    def plant_step(x: np.ndarray, u: float, t0: float, k: int) -> np.ndarray:
        if cfg.nonlinear:
            f = lambda t, X: twip_nonlinear(X, u, cfg.plant, convention="down", extra=extra_terms)  # noqa: E731
            return rk4(f, x, t0, t0 + dt, dt / cfg.rk4_substeps)
        d = cfg.linear_uncertainty(k, x) if cfg.linear_uncertainty else np.zeros(4)
        return F @ (x - PI_E3) + g * u + PI_E3 + dt * d

    # NN setup: fixed random input layers, zero adaptive weights
    n_in1, n_in2 = 10, 12
    V1 = rng.random((n_in1, n_in1))
    V2 = rng.random((n_in2, n_in2))
    We11 = np.zeros((n_in1, cfg.hidden1))
    We12 = np.zeros(cfg.hidden1)
    We21 = np.zeros((n_in2, cfg.hidden2))
    We22 = np.zeros(cfg.hidden2)

    sensor = SensorModel(dt, SensorNoise(), rng, enabled=cfg.noise)
    mso = DMSO(F, G, np.eye(4), cfg.mso_gamma, cfg.mso_sigma)

    x_des = np.zeros((4, N + 1))
    x_des[:, 0] = [0.0, cfg.velocity(0, dt)[0], 0.0, 0.0]
    x_des[2, 0] = tilt_for_velocity(x_des[1, 0]) + np.pi

    xtrue = np.zeros((4, N + 1))
    xextra = np.zeros((4, N + 1))
    xnom = np.zeros((4, N + 1))
    xtrue[:, 0] = xextra[:, 0] = cfg.x_init
    xnom[:, 0] = np.asarray(cfg.x_init) - PI_E3
    xhat = np.zeros((4, N + 1))
    y = np.zeros((4, N + 1))
    y[:, 0] = sensor.initial(xtrue[:, 0])

    u = np.zeros(N)
    u_nom = np.zeros(N)
    u_e = np.zeros(N)
    u_lqr = np.zeros(N)
    u_opt = np.zeros(N)
    Fhat = np.zeros((2, N))
    Ftrue = np.zeros((2, N))
    xbar = np.zeros((4, N))
    x4bard = np.zeros(N)
    fhat = np.zeros((2, N))
    u_lqr_prev = 0.0  # xtrue(5,i) in the script, used by the DMSO

    bl_true = bl_extra = bl_nom = None
    if cfg.tilt_backlash is not None:
        dp, dm = cfg.tilt_backlash
        bl_true = Backlash(xtrue[2, 0], 1.0, dp, dm)
        bl_extra = Backlash(xextra[2, 0], 1.0, dp, dm)
        bl_nom = Backlash(xnom[2, 0], 1.0, dp, dm)
    # Pre-backlash plant angles (xtrue2 / xextra2 / xnom2 in the script)
    th_true_raw, th_extra_raw, th_nom_raw = xtrue[2, 0], xextra[2, 0], xnom[2, 0]

    err_lqr = err_extra = err_nom = 0.0
    effort = 0.0

    for i in range(N):
        # DMSO on the LQR-only system (upright-relative), not used for control
        # (in long runs this DMSO can overflow to inf/nan; the script never uses it)
        innov = (y[:, i] - PI_E3) - xhat[:, i]
        with np.errstate(over="ignore", invalid="ignore"):
            xhat[:, i + 1], fhat[:, i] = mso.step(y[:, i] - PI_E3, u_lqr_prev)

        # desired trajectory
        v_next, vdot = cfg.velocity(i + 1, dt)
        x_des[0, i + 1] = x_des[0, i] + x_des[1, i] * dt
        x_des[1, i + 1] = v_next
        x_des[2, i + 1] = tilt_for_velocity(v_next) + np.pi
        x_des[3, i + 1] = 0.0

        # LQR for each of the three systems
        u_lqr[i] = -K @ (xtrue[:, i] - x_des[:, i])
        u_nom[i] = -K @ (xextra[:, i] - x_des[:, i])
        u_opt[i] = -K @ (xnom[:, i] - (x_des[:, i] - PI_E3))

        if cfg.extra:
            e = xextra[:, i] - x_des[:, i]

            # NN1: velocity-loop uncertainty -> virtual tilt command xbar3
            in1 = np.array([1, *xextra[:, i], e[0], e[1], x_des[0, i], x_des[1, i], u[i - 1] if i else 0.0])
            phie11 = tansig(V1.T @ in1)
            phie12 = tansig(We11.T @ phie11)
            Fhat[0, i] = We12 @ phie12

            xbar[0:2, i] = x_des[0:2, i]
            xbar[2, i] = (x_des[2, i] - np.pi) + (-cfg.ke1 * e[1] - Fhat[0, i] - cfg.ke3 * e[0]) / A[1, 2]
            if i == 0:
                xbar[3, i] = x_des[3, i]
            else:
                xbar[3, i] = x_des[3, i] + cfg.ke0 * (xbar[2, i] - xbar[2, i - 1]) / dt
                x4bard[i] = (xbar[3, i] - xbar[3, i - 1]) / dt
            ebar = xextra[:, i] - xbar[:, i] - PI_E3

            # NN2: tilt-loop uncertainty -> extra control u_e
            in2 = np.array([1, *xextra[:, i], ebar[2], ebar[3], e[0], e[1], x_des[0, i], x_des[1, i], u_nom[i]])
            phie21 = tansig(V2.T @ in2)
            phie22 = tansig(We21.T @ phie21)
            Fhat[1, i] = We22 @ phie22

            u_e[i] = -(-cfg.ke2 * ebar[3] - Fhat[1, i] - cfg.ke4 * ebar[2]) / b4

            h = dt / cfg.nn_substeps
            We12, We11 = _nn_update(
                We12, We11, phie11, phie12, e[1],
                cfg.gamma11, cfg.sigma11, cfg.gamma12, cfg.sigma12, time[i], time[i + 1], h,
            )
            nn2_err = e[3] if cfg.nn2_error == "e4" else ebar[3]
            We22, We21 = _nn_update(
                We22, We21, phie21, phie22, nn2_err,
                cfg.gamma21, cfg.sigma21, cfg.gamma22, cfg.sigma22, time[i], time[i + 1], h,
            )

        u[i] = u_nom[i] + u_e[i]

        # actuator nonlinearities (applied to all three controls)
        for arr in (u, u_opt, u_lqr):
            if cfg.saturation is not None:
                arr[i] = saturate(arr[i], cfg.saturation)
            if cfg.deadzone is not None:
                arr[i] = deadzone(arr[i], cfg.deadzone)
        effort += u[i] ** 2

        # propagate the three systems; unlike the DMSO sim, u[i] acts immediately
        nxt = plant_step(xextra[:, i], u[i], time[i], i)
        if bl_extra is not None:
            raw = nxt[2]
            nxt[2] = bl_extra.step(raw, th_extra_raw)
            th_extra_raw = raw
        xextra[:, i + 1] = nxt

        nxt = plant_step(xtrue[:, i], u_lqr[i], time[i], i)
        if bl_true is not None:
            raw = nxt[2]
            nxt[2] = bl_true.step(raw, th_true_raw)
            th_true_raw = raw
        xtrue[:, i + 1] = nxt
        u_lqr_prev = u_lqr[i]

        nxt = F @ xnom[:, i] + g * u_opt[i]
        if bl_nom is not None:
            raw = nxt[2]
            nxt[2] = bl_nom.step(raw, th_nom_raw)
            th_nom_raw = raw
        xnom[:, i + 1] = nxt

        # "true" control uncertainties (as defined in the script, scaled by -dt)
        d = cfg.linear_uncertainty(i, xextra[:, i]) if (cfg.linear_uncertainty and not cfg.nonlinear) else np.zeros(4)
        Ftrue[0, i] = -dt * (A[1, 1] * xextra[1, i] + A[1, 2] * (x_des[2, i] - np.pi) + b2 * u[i] + d[1])
        Ftrue[1, i] = -dt * (A[3, 1] * xextra[1, i] + A[3, 2] * (xextra[2, i] - np.pi) + b4 * u_nom[i] - x4bard[i] + d[3])

        err_extra += np.linalg.norm(xextra[:, i] - x_des[:, i])
        err_lqr += np.linalg.norm(xtrue[:, i] - x_des[:, i])
        err_nom += np.linalg.norm(xnom[:, i] - x_des[:, i] + PI_E3)

        y[:, i + 1] = sensor.measure(xtrue[:, i + 1])
        with np.errstate(over="ignore", invalid="ignore"):
            mso.update_weights(innov)

    return ExtraControlResult(
        time, x_des, xtrue, xextra, xnom, u, u_nom, u_e, u_lqr, u_opt, Fhat, Ftrue, xbar,
        xhat, fhat, err_lqr, err_extra, err_nom, effort,
    )
