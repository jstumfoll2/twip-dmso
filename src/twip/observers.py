"""Observers from thesis revision 8: the corrected DMSO (Chapter 3) and the three
Kalman baselines (Section 7, "Three Baselines").  :class:`DMSO2020` is the DMSO of
the 2020 ACC paper, kept as an alternative to the revision-8 DMSO.

All observers here use the thesis's full-state measurement ``y_k`` (``y = x`` plus
noise), and are compared on the same information set: after processing ``y_k``
and ``u_k`` each returns its prediction of ``x[k+1]``.  For the DMSO this is its
native output ``xhat_{k+1}``.  For the Kalman filters it is ``F x_{k|k} + G u_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

B_MSO = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])  # eq. (Bmso)


class TanhBasis:
    """``phi(y) = [1, tanh(y_1/s_1), ..., tanh(y_n/s_n)]``, bounded by construction:
    ``||phi||^2 <= 1 + n`` for every ``y`` (Appendix A, Stage 2, step 2)."""

    def __init__(self, scales=(1.0, 1.0, 0.2, 1.0)):
        self.scales = np.asarray(scales, float)
        self.phi_max2 = 1.0 + len(self.scales)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], np.tanh(y / self.scales)))


def kalman_injection_gain(F: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``Km = F L`` with ``L`` the steady-state (a-posteriori) Kalman gain for ``H = I``.

    Then ``A = F (I - L)``.  Without measurement noise ``L -> I`` and ``A -> 0``
    (the deadbeat design of Corollary 3.3).  With noise ``A`` is Schur but strongly
    non-normal, so Theorem 3.1 does not apply in the Euclidean norm.
    """
    from scipy.linalg import solve_discrete_are

    P = solve_discrete_are(F.T, np.eye(F.shape[0]), Q, R)
    L = P @ np.linalg.inv(P + R)
    return F @ L


def gamma_bound(sigma_bar: float, phi_max2: float) -> float:
    """Corollary 3.2: ``Gamma < (1/phi_max^2) * min(1/2, (1 - s^2) / (2 s^2))``."""
    if sigma_bar <= 0:
        return 0.5 / phi_max2
    return min(0.5, (1 - sigma_bar**2) / (2 * sigma_bar**2)) / phi_max2


@dataclass
class DMSOConditions:
    sigma_max_A: float
    phi_max2_design: float
    phi_max2_realized: float
    gamma: float
    gamma_bound: float

    @property
    def satisfied(self) -> bool:
        """Corollary 3.2 with the *realized* phi_max."""
        return self.sigma_max_A < 1 and self.gamma < gamma_bound(self.sigma_max_A, self.phi_max2_realized)

    def ultimate_bound(self, eps_N: float) -> float:
        """``b_e`` of Theorem 3.1 with ``theta = 1`` and the realized ``phi_max``."""
        s, g = self.sigma_max_A, self.gamma * self.phi_max2_realized
        c1 = 1 + 2 * g
        rho = s * np.sqrt(c1)
        return np.sqrt(c1) * eps_N / (1 - rho) if rho < 1 else np.inf


class DMSO:
    """Discrete modified state observer, revision 8 (eqs. obs, update, sigmod).

    ``xhat_{k+1} = F xhat_k + G u_k + Bm W_k^T phi(y_k) + Km (y_k - xhat_k)``
    ``W_{k+1}   = (1 - Gamma kappa) W_k + Gamma phi(y_k) e_{k+1}^T Bm``,  ``e = y - xhat``.

    The basis is evaluated on the measurement ``y_k`` (revision 7 used ``xhat_k``),
    and the weights are updated with the *next* error ``e_{k+1}``, which is formed at
    step ``k+1``.  ``Km = F - sigma_bar I`` places ``A = F - Km = sigma_bar I``.
    ``kappa > 0`` adds sigma-modification.
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        sigma_bar: float = 0.5,
        gamma_frac: float = 0.1,
        kappa: float = 0.0,
        basis: TanhBasis | None = None,
        Bm: np.ndarray = B_MSO,
        Km: np.ndarray | None = None,
        gamma: float | None = None,
    ):
        n = F.shape[0]
        self.F, self.G, self.Bm = F, G.reshape(n), Bm
        self.Km = F - sigma_bar * np.eye(n) if Km is None else Km
        self.basis = basis or TanhBasis()
        s_max = float(np.linalg.norm(F - self.Km, 2))
        self.gamma_max = gamma_bound(s_max, self.basis.phi_max2) if s_max < 1 else 0.0
        self.gamma = gamma if gamma is not None else gamma_frac * self.gamma_max
        self.kappa = kappa
        self.W = np.zeros((len(self.basis.scales) + 1, Bm.shape[1]))
        self.xhat: np.ndarray | None = None
        self._phi_prev: np.ndarray | None = None
        self._phi2_seen = 0.0
        self.fhat = np.zeros(Bm.shape[1])

    def initialize(self, x0: np.ndarray) -> None:
        self.xhat = np.asarray(x0, float).copy()

    def step(self, y: np.ndarray, u: float) -> np.ndarray:
        """Process ``y_k`` and ``u_k``; return ``xhat_{k+1}``."""
        if self.xhat is None:
            self.xhat = y.copy()
        e = y - self.xhat  # e_k, available now
        if self._phi_prev is not None:  # W_k from phi_{k-1} and e_k  (eq. update / sigmod)
            self.W = (1 - self.gamma * self.kappa) * self.W + self.gamma * np.outer(self._phi_prev, e @ self.Bm)
        phi = self.basis(y)
        self._phi2_seen = max(self._phi2_seen, float(phi @ phi))
        self.fhat = self.W.T @ phi
        self.xhat = self.F @ self.xhat + self.G * u + self.Bm @ self.fhat + self.Km @ e
        self._phi_prev = phi
        return self.xhat.copy()

    def conditions(self) -> DMSOConditions:
        A = self.F - self.Km
        return DMSOConditions(
            float(np.linalg.norm(A, 2)), self.basis.phi_max2, self._phi2_seen, self.gamma, self.gamma_max
        )


@dataclass
class DMSO2020Conditions:
    """Conditions (9)-(11) of Theorem 1 of the ACC 2020 paper, with ``||phi_k||^2`` at
    its realized maximum and ``Q = I - A^T A`` (``lambda_min(Q) = 1 - sigma_max(A)^2``).
    """

    sigma_max_A: float
    alpha: float
    gamma: float
    phi_max2_design: float
    phi_max2_realized: float
    leak_norm_max: float  # largest ||I - alpha phi phi^T|| along the run

    @property
    def eta(self) -> float:
        """Eq. (34), worst case over the realized ``||phi_k||^2``."""
        a, p2 = self.alpha, self.phi_max2_realized
        if not 0 < a * p2 < 1:
            return float("inf")
        return p2 + (a * p2 + self.gamma * self.leak_norm_max) ** 2 / (a * (1 - a * p2))

    @property
    def paper_conditions(self) -> tuple[bool, bool, bool]:
        """(9) ``alpha ||phi||^2 < 1``, (10) ``0 < Gamma < 2``,
        (11) ``lambda_min(Q) - alpha sigma_max(A)^2 eta > 0``."""
        c9 = self.alpha * self.phi_max2_realized < 1
        c10 = 0 < self.gamma < 2
        s2 = self.sigma_max_A**2
        c11 = c9 and (1 - s2) - self.alpha * s2 * self.eta > 0
        return c9, c10, c11

    @property
    def satisfied(self) -> bool:
        return all(self.paper_conditions)

    @property
    def weights_contractive(self) -> bool:
        """``Gamma ||M|| + alpha ||phi||^2 < 2``, ``M = I - alpha phi phi^T``.

        With the regressor frozen, the weight error evolves as
        ``((1 - Gamma ||M||) I - alpha phi phi^T) W~`` plus forcing, whose
        eigenvalues are ``1 - Gamma ||M||`` and ``1 - Gamma ||M|| - alpha ||phi||^2``.
        This condition is necessary for the weights to stay bounded, and (9)-(11)
        do not imply it: they admit ``Gamma`` up to 2 regardless of ``alpha ||phi||^2``.
        """
        g = self.gamma * self.leak_norm_max
        return 0 < g and g + self.alpha * self.phi_max2_realized < 2


class DMSO2020:
    """DMSO of Stumfoll, Yao and Balakrishnan, "Neural Network Based Discrete Time
    Modified State Observer: Stability Analysis and Case Study", ACC 2020, pp. 2484-2489.

    ``xhat_{k+1} = F xhat_k + G u_k + Bm W_k^T phi(y_k) + Km (y_k - xhat_k)``    (3), (5)
    ``W_{k+1} = W_k + alpha phi(y_k) e_{k+1}^T Bm - Gamma ||I - alpha phi phi^T|| W_k``  (6)

    Defaults follow the paper's TWIP case study (Section IV): ``Km = I``,
    ``alpha = 0.01``, ``phi = [1, tanh(y)]`` and uncertainty assigned to the velocity
    and tilt-rate rows only (``Bm``; the theory of Section II has ``Bm = I``).
    The paper does not report ``Gamma`` or how it normalized the basis.  The default
    ``gamma = 1e-3`` makes the leakage per step equal to that of the revision-8 DMSO
    with sigma-modification (``Gamma kappa = 0.01 * 0.1``), and the basis is unscaled.

    ``leak_norm`` picks the matrix norm in (6), which the paper does not define.
    In the induced 2-norm ``||I - alpha phi phi^T|| = 1`` whenever
    ``alpha ||phi||^2 <= 2``, so the law is the revision-8 sigma-modification
    ``W_{k+1} = (1 - Gamma_8 kappa) W_k + Gamma_8 phi e_{k+1}^T Bm`` with
    ``Gamma_8 = alpha`` and ``kappa = Gamma / alpha``.  With ``"fro"`` the norm is
    ``sqrt(l - 1 + (1 - alpha ||phi||^2)^2)``, about 2.2 for ``l = 5``.

    The interface (``initialize``, ``step``, ``fhat``, ``conditions``) matches :class:`DMSO`.
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        alpha: float = 0.01,
        gamma: float = 1e-3,
        Km: np.ndarray | None = None,
        basis: TanhBasis | None = None,
        Bm: np.ndarray = B_MSO,
        leak_norm: str = "2",
    ):
        if leak_norm not in ("2", "fro"):
            raise ValueError("leak_norm must be '2' or 'fro'")
        n = F.shape[0]
        self.F, self.G, self.Bm = F, G.reshape(n), Bm
        self.Km = np.eye(n) if Km is None else Km
        self.basis = basis or TanhBasis(scales=(1.0,) * n)
        self.alpha, self.gamma, self.leak_norm = alpha, gamma, leak_norm
        self.W = np.zeros((len(self.basis.scales) + 1, Bm.shape[1]))
        self.xhat: np.ndarray | None = None
        self._phi_prev: np.ndarray | None = None
        self._phi2_seen = 0.0
        self._leak_seen = 0.0
        self.fhat = np.zeros(Bm.shape[1])

    def initialize(self, x0: np.ndarray) -> None:
        self.xhat = np.asarray(x0, float).copy()

    def leak_factor(self, phi: np.ndarray) -> float:
        """``||I - alpha phi phi^T||``: its eigenvalues are 1 (``l - 1`` times) and
        ``1 - alpha ||phi||^2``."""
        a2 = self.alpha * float(phi @ phi)
        if self.leak_norm == "2":
            return max(1.0, abs(1 - a2))
        return float(np.sqrt(len(phi) - 1 + (1 - a2) ** 2))

    def step(self, y: np.ndarray, u: float) -> np.ndarray:
        """Process ``y_k`` and ``u_k``; return ``xhat_{k+1}``."""
        if self.xhat is None:
            self.xhat = y.copy()
        e = y - self.xhat  # e_k, available now
        if self._phi_prev is not None:  # W_k from phi_{k-1}, e_k and W_{k-1}  (eq. 6)
            m = self.leak_factor(self._phi_prev)
            self._leak_seen = max(self._leak_seen, m)
            self.W = (1 - self.gamma * m) * self.W + self.alpha * np.outer(self._phi_prev, e @ self.Bm)
        phi = self.basis(y)
        self._phi2_seen = max(self._phi2_seen, float(phi @ phi))
        self.fhat = self.W.T @ phi
        self.xhat = self.F @ self.xhat + self.G * u + self.Bm @ self.fhat + self.Km @ e
        self._phi_prev = phi
        return self.xhat.copy()

    def conditions(self) -> DMSO2020Conditions:
        return DMSO2020Conditions(
            float(np.linalg.norm(self.F - self.Km, 2)),
            self.alpha,
            self.gamma,
            self.basis.phi_max2,
            self._phi2_seen,
            self._leak_seen or self.leak_factor(np.ones(len(self.basis.scales) + 1)),
        )


class KalmanPredictor:
    """Discrete Kalman filter with full-state measurement ``H = I`` (eqs. kf1-kf5,
    Joseph form), reporting the one-step prediction ``F x_{k|k} + G u_k``.

    ``augmented=True`` is baseline (c): the uncertainty ``f`` is appended as a
    random-walk state, ``x_aug = [x; f]``, ``F_aug = [[F, Bm], [0, I]]``, with
    process noise ``q_f`` on the appended block.  ``fhat`` then plays the same
    role as the DMSO's.
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        augmented: bool = False,
        q_f: float = 1e-6,
        Bm: np.ndarray = B_MSO,
    ):
        n = F.shape[0]
        p = Bm.shape[1] if augmented else 0
        self.n, self.p = n, p
        self.F = np.block([[F, Bm], [np.zeros((p, n)), np.eye(p)]]) if augmented else F
        self.G = np.concatenate([G.reshape(n), np.zeros(p)])
        self.H = np.hstack([np.eye(n), np.zeros((n, p))])
        self.Q = np.block([[Q, np.zeros((n, p))], [np.zeros((p, n)), q_f * np.eye(p)]]) if augmented else Q
        self.R = R
        self.x: np.ndarray | None = None
        self.P = np.eye(n + p) * 1e-4
        self.fhat = np.zeros(p)

    def initialize(self, x0: np.ndarray) -> None:
        self.x = np.concatenate([np.asarray(x0, float), np.zeros(self.p)])

    def step(self, y: np.ndarray, u: float) -> np.ndarray:
        if self.x is None:
            self.initialize(y)
        H = self.H
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ (y - H @ self.x)
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.fhat = self.x[self.n :].copy()
        # predict to k+1
        self.x = self.F @ self.x + self.G * u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[: self.n].copy()


class LegacyDMSOPredictor:
    """The thesis (revision 7) DMSO in the same interface, for comparison:
    basis on ``xhat``, ``K = I``, innovation ``y_k - xhat_k``, ``gamma = 0.1``."""

    def __init__(self, F: np.ndarray, G: np.ndarray, gamma: float = 0.1):
        from .legacy.observers import DMSO as _D

        self._d = _D(F, G, np.eye(4), gamma)
        self.fhat = np.zeros(2)
        self._started = False

    def initialize(self, x0: np.ndarray) -> None:
        self._d.xhat = np.asarray(x0, float).copy()

    def step(self, y: np.ndarray, u: float) -> np.ndarray:
        innov = y - self._d.xhat
        xh, self.fhat = self._d.step(y, u)
        self._d.update_weights(innov)
        return xh
