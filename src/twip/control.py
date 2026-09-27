"""Replacements for the MATLAB Control System Toolbox calls used in the thesis:
``c2d`` (zero-order hold), ``lqr``, ``dlqr`` and ``lqrd``."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are


def c2d_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretization, same as ``c2d(ss(A,B,C,D), dt)``."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray | float) -> np.ndarray:
    """Continuous-time LQR gain ``K`` for ``u = -K x`` (MATLAB ``lqr``)."""
    R = np.atleast_2d(R).astype(float)
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def dlqr(
    F: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray | float,
    N: np.ndarray | None = None,
) -> np.ndarray:
    """Discrete-time LQR gain with optional cross term ``N`` (MATLAB ``dlqr``)."""
    R = np.atleast_2d(R).astype(float)
    if N is None:
        N = np.zeros((F.shape[0], G.shape[1]))
    P = solve_discrete_are(F, G, Q, R, s=N)
    return np.linalg.solve(R + G.T @ P @ G, G.T @ P @ F + N.T)


def lqrd(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray | float, dt: float
) -> np.ndarray:
    """Discrete LQR gain designed from a *continuous* cost (MATLAB ``lqrd``).

    The plant and the continuous quadratic cost are both discretized over one
    ZOH sample, then ``dlqr`` is solved with the resulting cross term.  The
    cost integral is evaluated with Van Loan's block-matrix exponential.
    """
    n, m = B.shape
    R = np.atleast_2d(R).astype(float)

    # Augmented state z = [x; u] with u held constant over the sample.
    Aa = np.zeros((n + m, n + m))
    Aa[:n, :n] = A
    Aa[:n, n:] = B
    Wa = np.zeros((n + m, n + m))
    Wa[:n, :n] = Q
    Wa[n:, n:] = R

    k = n + m
    C = np.zeros((2 * k, 2 * k))
    C[:k, :k] = -Aa.T
    C[:k, k:] = Wa
    C[k:, k:] = Aa
    E = expm(C * dt)
    Phi = E[k:, k:]
    Wd = Phi.T @ E[:k, k:]
    Wd = 0.5 * (Wd + Wd.T)

    Ad, Bd = Phi[:n, :n], Phi[:n, n:]
    Qd, Nd, Rd = Wd[:n, :n], Wd[:n, n:], Wd[n:, n:]
    return dlqr(Ad, Bd, Qd, Rd, Nd)
