"""Controllers for the Chapter 8 control study (thesis revision 8).

* :class:`LQRTracking` - LQR on the tracking error with the nominal steady-state
  feedforward (Section 6.1).
* :class:`TwoStepExtraControl` - the revision-7 two-step neural extra control
  (``ExtraControl_v5.m`` with the figure settings, ``ke0 = 0``), ported to the
  upright-zero convention and the corrected nominal model.
* :class:`CommandFilteredBackstepping` - the revision-8 four-step command-filtered
  neural backstepping (Sections 6.3-6.5).

All controllers are called once per 10 ms loop as ``u = ctrl(xhat, k)`` and track
``x1d(t) = v_des * t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .dynamics import linear_model
from .legacy.extra_control import _nn_update
from .params import CORRECTED, RobotParams


@dataclass
class TrackingModel:
    """The plant form (eq. plantform) read off the nominal linear model.

    The corrected model also has ``thetadot`` terms (``A24``, ``A44``) from the
    relative back-EMF; they are kept as known terms.
    """

    A: np.ndarray
    B: np.ndarray
    v_des: float
    dt: float

    @classmethod
    def build(cls, p: RobotParams = CORRECTED, v_des: float = 0.1, dt: float = 0.01) -> TrackingModel:
        A, B = linear_model(p)
        return cls(A, B.ravel(), v_des, dt)

    @property
    def A1(self):
        return self.A[1, 1]

    @property
    def A2(self):
        return self.A[1, 2]

    @property
    def A3(self):
        return self.A[3, 1]

    @property
    def A4(self):
        return self.A[3, 2]

    @property
    def B1(self):
        return self.B[1]

    @property
    def B2(self):
        return self.B[3]

    def steady_state(self) -> tuple[float, float]:
        """Tilt and voltage that hold ``v_des`` in the nominal linear model."""
        M = np.array([[self.A2, self.B1], [self.A4, self.B2]])
        th, u = np.linalg.solve(M, -np.array([self.A1, self.A3]) * self.v_des)
        return float(th), float(u)

    def x_des(self, k: int) -> np.ndarray:
        th, _ = self.steady_state()
        return np.array([self.v_des * k * self.dt, self.v_des, th, 0.0])


def position_zeros(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Transmission zeros of ``x1(s)/u(s)``: finite generalized eigenvalues of the
    Rosenbrock system matrix.  A right-half-plane zero makes position tracking by
    plant inversion (as in the command-filtered design) internally unstable."""
    from scipy.linalg import eigvals

    n = A.shape[0]
    C = np.zeros((1, n))
    C[0, 0] = 1.0
    M = np.block([[A, B.reshape(n, 1)], [C, np.zeros((1, 1))]])
    N = np.block([[np.eye(n), np.zeros((n, 1))], [np.zeros((1, n + 1))]])
    z = eigvals(M, N)
    return np.sort_complex(z[np.isfinite(z)])


class LQRTracking:
    def __init__(self, K: np.ndarray, model: TrackingModel, u_limit: float = 10.0):
        self.K, self.m, self.u_limit = np.asarray(K, float).ravel(), model, u_limit
        self.u_ss = model.steady_state()[1]

    def nominal(self, xhat: np.ndarray, k: int) -> float:
        return self.u_ss - float(self.K @ (xhat - self.m.x_des(k)))

    def __call__(self, xhat: np.ndarray, k: int) -> float:
        return self.nominal(xhat, k)


# ---------------------------------------------------------------------------
# revision 7: two-step extra control


class TwoStepExtraControl(LQRTracking):
    """Revision-7 extra control (``ExtraControl_v5.m`` logic with ``ke0 = 0``).

    ``xbar3 = x3des + A2^-1 (-ke1 e2 - Fhat1 - ke3 e1)`` is the virtual tilt; ``u_e =
    B2^-1 (ke2 ebar4 + Fhat2 + ke4 ebar3)``.  Two-layer networks with a fixed random
    input layer; the weights follow ``NN1v3.m``, integrated with RK4 over each sample.
    NN1 is driven by ``e2`` and NN2 by ``e4``, as in the saved script.
    """

    def __init__(self, K, model: TrackingModel, seed: int = 0, ke1=2.5, ke3=1.0, ke2=2.5, ke4=1.0,
                 gamma=0.1, sigma=0.01, hidden=10, u_limit=10.0):
        super().__init__(K, model, u_limit)
        rng = np.random.default_rng(seed)
        self.V1, self.V2 = rng.random((10, 10)), rng.random((12, 12))
        self.W11, self.W12 = np.zeros((10, hidden)), np.zeros(hidden)
        self.W21, self.W22 = np.zeros((12, hidden)), np.zeros(hidden)
        self.ke1, self.ke3, self.ke2, self.ke4 = ke1, ke3, ke2, ke4
        self.g, self.s = gamma, sigma
        self.u_prev = 0.0
        self.log: dict[str, list] = {"Fhat1": [], "Fhat2": [], "u_e": []}

    def __call__(self, xhat: np.ndarray, k: int) -> float:
        m, dt = self.m, self.m.dt
        xd = m.x_des(k)
        e = xhat - xd
        u_nom = self.nominal(xhat, k)
        in1 = np.array([1, *xhat, e[0], e[1], xd[0], xd[1], self.u_prev])
        p11 = np.tanh(self.V1.T @ in1)
        p12 = np.tanh(self.W11.T @ p11)
        F1 = float(self.W12 @ p12)
        xbar3 = xd[2] + (-self.ke1 * e[1] - F1 - self.ke3 * e[0]) / m.A2
        xbar4 = xd[3]  # ke0 = 0
        eb3, eb4 = xhat[2] - xbar3, xhat[3] - xbar4
        in2 = np.array([1, *xhat, eb3, eb4, e[0], e[1], xd[0], xd[1], u_nom])
        p21 = np.tanh(self.V2.T @ in2)
        p22 = np.tanh(self.W21.T @ p21)
        F2 = float(self.W22 @ p22)
        u_e = -(-self.ke2 * eb4 - F2 - self.ke4 * eb3) / m.B2
        h = dt / 5
        self.W12, self.W11 = _nn_update(self.W12, self.W11, p11, p12, e[1], self.g, self.s, self.g, self.s, 0.0, dt, h)
        self.W22, self.W21 = _nn_update(self.W22, self.W21, p21, p22, e[3], self.g, self.s, self.g, self.s, 0.0, dt, h)
        u = u_nom + u_e
        self.u_prev = float(np.clip(u, -self.u_limit, self.u_limit))
        self.log["Fhat1"].append(F1)
        self.log["Fhat2"].append(F2)
        self.log["u_e"].append(u_e)
        return u


# ---------------------------------------------------------------------------
# revision 8: command-filtered neural backstepping


class CommandFilter:
    """Second-order filter (eq. cmdfilt), discretized exactly for a held input."""

    def __init__(self, wn: float, zeta: float, dt: float, q0: float = 0.0):
        Ac = np.array([[0.0, 1.0], [-wn**2, -2 * zeta * wn]])
        Bc = np.array([0.0, wn**2])
        M = np.zeros((3, 3))
        M[:2, :2], M[:2, 2] = Ac, Bc
        E = expm(M * dt)
        self.Ad, self.Bd = E[:2, :2], E[:2, 2]
        self.q = np.array([q0, 0.0])

    @property
    def value(self) -> float:
        return float(self.q[0])

    @property
    def rate(self) -> float:
        return float(self.q[1])

    def step(self, alpha: float) -> None:
        self.q = self.Ad @ self.q + self.Bd * alpha


class TwoLayerNet:
    """``Fhat = W2^T tanh(W1^T tanh(V^T P))`` with the revision-8 update laws
    (eqs. upd2, upd1), integrated with forward Euler at the loop rate."""

    def __init__(self, n_in: int, h1: int, h2: int, rng: np.random.Generator, gamma1: float, gamma2: float,
                 kappa1: float, kappa2: float, v_scale: float = 1.0):
        self.V = rng.normal(scale=v_scale, size=(n_in, h1))
        # Small random first layer: with W1 = W2 = 0 both regressors vanish and nothing is learned.
        self.W1 = rng.normal(scale=0.1, size=(h1, h2))
        self.W2 = np.zeros(h2)
        self.g1, self.g2, self.k1, self.k2 = gamma1, gamma2, kappa1, kappa2
        self.enabled = True

    def forward(self, P: np.ndarray) -> float:
        self.s1 = np.tanh(self.V.T @ P)
        self.s2 = np.tanh(self.W1.T @ self.s1)
        self.ds2 = 1 - self.s2**2  # diagonal of sigma_2'
        return float(self.W2 @ self.s2) if self.enabled else 0.0

    def update(self, rho: float, dt: float) -> None:
        if not self.enabled:
            return
        lam = self.s2 - self.ds2 * (self.W1.T @ self.s1)  # Lambda_i
        dW2 = self.g2 * (lam * rho - self.k2 * self.W2)
        dW1 = self.g1 * (np.outer(self.s1, rho * (self.ds2 * self.W2)) - self.k1 * self.W1)
        self.W2 = self.W2 + dt * dW2
        self.W1 = self.W1 + dt * dW1


@dataclass
class CFBGains:
    c: tuple[float, float, float, float] = (1.0, 2.0, 6.0, 12.0)  # Theorem 6.1: c2 > 1, others > 1/2
    wn: tuple[float, float, float] = (20.0, 40.0, 60.0)  # filter bandwidths varpi_i, rad/s
    zeta: tuple[float, float, float] = (0.9, 0.9, 0.9)
    gamma: tuple[float, float] = (0.5, 2.0)  # (gamma_i1, gamma_i2)
    kappa: tuple[float, float] = (0.01, 0.01)
    hidden: tuple[int, int] = (8, 8)


class CommandFilteredBackstepping(LQRTracking):
    """Revision-8 extra control, executed in the order of Remark 6.1.

    ``Fhat_i`` = known part of ``F_i`` from the nominal model + network estimate of
    the unknown part (``d_i``), so each network only has to learn the uncertainty.
    ``u = u_nom + u_e`` exactly as eq. (ue).  Substituting (ue) shows that ``u_nom``
    cancels: ``B2 u = -Fhat2 - c4 z4 - z3``.  The LQR gain therefore has no effect
    on the applied control (see ``tests/test_controllers.py``).
    """

    def __init__(self, K, model: TrackingModel, x0: np.ndarray, gains: CFBGains | None = None,
                 networks: bool = True, seed: int = 0, u_limit: float = 10.0):
        super().__init__(K, model, u_limit)
        g = gains or CFBGains()
        self.g = g
        dt = model.dt
        rng = np.random.default_rng(seed)
        self.f = [CommandFilter(g.wn[i], g.zeta[i], dt, q0=float(x0[i + 1])) for i in range(3)]
        self.nn = [
            TwoLayerNet(4, 10, g.hidden[0], rng, g.gamma[0], g.gamma[1], g.kappa[0], g.kappa[1]),
            TwoLayerNet(5, 10, g.hidden[1], rng, g.gamma[0], g.gamma[1], g.kappa[0], g.kappa[1]),
        ]
        for n in self.nn:
            n.enabled = networks
        self.log: dict[str, list] = {k: [] for k in ("z", "chi", "alpha", "Fnn1", "Fnn2", "u_e", "u_nom")}

    def __call__(self, xhat: np.ndarray, k: int) -> float:
        m, g, dt = self.m, self.g, self.m.dt
        x1, x2, x3, x4 = xhat
        x1d, x1d_dot = m.v_des * k * dt, m.v_des
        # (1)-(2) error coordinates from state and filter states
        x2c, x2c_dot = self.f[0].value, self.f[0].rate
        x3c, x3c_dot = self.f[1].value, self.f[1].rate
        x4c, x4c_dot = self.f[2].value, self.f[2].rate
        z = np.array([x1 - x1d, x2 - x2c, x3 - x3c, x4 - x4c])
        # (3) approximations: known model part + network estimate of d_i
        n1 = self.nn[0].forward(np.array([x2, x3, x2c_dot, 1.0]))
        n2 = self.nn[1].forward(np.array([x2, x3, x4, x4c_dot, 1.0]))
        F1 = m.A1 * x2 + m.A[1, 3] * x4 - x2c_dot + n1
        F2 = m.A3 * x2 + m.A4 * x3 + m.A[3, 3] * x4 - x4c_dot + n2
        # (4) control
        u_nom = self.nominal(xhat, k)
        u_e = (-F2 - m.B2 * u_nom - g.c[3] * z[3] - z[2]) / m.B2
        u = u_nom + u_e
        u_applied = float(np.clip(u, -self.u_limit, self.u_limit))
        # (5) virtual controls, alpha_2 using the control just computed
        a1 = x1d_dot - g.c[0] * z[0]
        a2 = (-F1 - m.B1 * u_applied - g.c[1] * z[1] - z[0]) / m.A2
        a3 = x3c_dot - g.c[2] * z[2] - m.A2 * z[1]
        # (6) propagate filters and networks
        for filt, a in zip(self.f, (a1, a2, a3)):
            filt.step(a)
        self.nn[0].update(z[1], dt)
        self.nn[1].update(z[3], dt)
        L = self.log
        L["z"].append(z)
        L["chi"].append(np.array([x2c - a1, x3c - a2, x4c - a3]))
        L["alpha"].append(np.array([a1, a2, a3]))
        L["Fnn1"].append(n1)
        L["Fnn2"].append(n2)
        L["u_e"].append(u_e)
        L["u_nom"].append(u_nom)
        return u
