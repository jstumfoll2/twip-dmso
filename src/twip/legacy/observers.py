"""The thesis observers: 4-state DMSO, 2-state DMSO and the v9 on-board DMSO.

These are kept exactly as designed in the thesis so that a new observer can be
compared against them.
"""

from __future__ import annotations

import numpy as np

RAD2DEG = 180 / np.pi


def tansig(n: np.ndarray) -> np.ndarray:
    """MATLAB ``tansig`` (numerically identical to ``tanh``)."""
    return np.tanh(n)


class DMSO:
    """Discrete modified state observer with a single-layer tansig basis.

    ``xhat[k+1] = F xhat + G u + Bmso fhat + K (y - H xhat)``, with
    ``fhat = W' phi(xhat)``, ``phi = [1, tansig(xhat)]``, and the weight update
    ``W += gamma * phi (innovation' Bmso) - sigma * W``.

    The thesis scripts differ in *which* innovation drives the weight update
    (``y[k] - H xhat[k]``, ``y[k+1] - H xhat[k+1]`` or ``y[k] - H xhat[k+1]``).
    The innovation is therefore passed to :meth:`update_weights` explicitly,
    and each caller reproduces its own script.
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        K: np.ndarray,
        gamma: float,
        sigma: float = 0.0,
        Bmso: np.ndarray | None = None,
        H: np.ndarray | None = None,
    ):
        n = F.shape[0]
        self.F, self.G, self.K = F, G.reshape(n), K
        self.gamma, self.sigma = gamma, sigma
        self.Bmso = Bmso if Bmso is not None else np.array([[0, 0], [1, 0], [0, 0], [0, 1]], float)
        self.H = H if H is not None else np.eye(n)
        self.xhat = np.zeros(n)
        self.W = np.zeros((n + 1, self.Bmso.shape[1]))
        self._phi = self.basis(self.xhat)

    @staticmethod
    def basis(xhat: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], tansig(xhat)))

    def step(self, y: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Propagate one sample. Returns ``(xhat_next, fhat)``."""
        self._phi = self.basis(self.xhat)
        fhat = self.W.T @ self._phi
        innov = y - self.H @ self.xhat
        self.xhat = self.F @ self.xhat + self.G * u + self.Bmso @ fhat + self.K @ innov
        return self.xhat.copy(), fhat

    def update_weights(self, innovation: np.ndarray) -> None:
        """Weight update using the basis vector from the most recent :meth:`step`."""
        self.W = (
            self.W
            + self.gamma * np.outer(self._phi, innovation @ self.Bmso)
            - self.sigma * self.W
        )


class DMSO2:
    """2-state (tilt deg, tilt rate deg/s) DMSO used for sensor filtering on the robot.

    Default gains are the tuned values in ``DMSO.m`` and ``MSO.ino``:
    ``K = diag(.0632, .3964)``, ``gamma = 1.8e-5``, ``P = -.007 I``,
    ``sigma = .5116``.

    ``firmware_bug=True`` reproduces ``twip_v4/MSO.ino`` exactly.  That code
    uses ``F[1][0]`` (always 0) where it should use ``F[0][1] = deltat``, so
    tilt is never propagated from tilt rate. ``DMSO.m`` uses the correct F.
    """

    def __init__(
        self,
        K1: float = 0.0632,
        K2: float = 0.3964,
        gamma: float = 0.000018,
        P: float = -0.007,
        sigma: float = 0.5116,
        firmware_bug: bool = False,
    ):
        self.K = np.diag([K1, K2])
        self.gamma, self.P, self.sigma = gamma, P, sigma
        self.firmware_bug = firmware_bug
        self.x = np.zeros(2)
        self.W = np.zeros((3, 2))

    def step(self, gy: float, ax: float, az: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """One update from raw gyro rate (deg/s) and accel (g). Returns ``(xhat, fhat)``."""
        F = np.array([[1.0, 0.0 if self.firmware_bug else dt], [0.0, 1.0]])
        phi = np.array([1.0, self.x[0], self.x[1]])
        y = np.array([np.arctan2(-ax, az) * RAD2DEG, gy])
        ytilde = y - self.x
        fhat = self.W.T @ phi
        self.x = F @ self.x + fhat + self.K @ ytilde
        self.W = self.W + dt * self.gamma * (np.outer(phi, ytilde) * self.P - self.sigma * self.W)
        return self.x.copy(), fhat


class FirmwareDMSO:
    """4-state DMSO exactly as run on the robot in ``twip_v9`` (``filters.ino``, March 2015).

    This is the estimator behind the hardware results (``implementationtest4/5.txt``).
    It differs from the MATLAB replay (``lqrkalmantest1v2.m``) in three ways:

    * ``F`` and ``G`` are hard-coded (rounded, dt = 0.01), not computed from the model.
    * ``Kmso = 0.5 I`` is declared but never used, so the correction gain is effectively ``I``.
    * The tilt measurement is the complementary-filter output ``pitchc``, not the raw
      accelerometer angle.

    The weight update uses the post-update innovation ``meas - est[k+1]``, like the MATLAB replay.
    """

    F = np.array(
        [
            [1, 0.010, 0.0001, 0],
            [0, 0.999, 0.0239, 0.0001],
            [0, -0.000005, 1.002, 0.010],
            [0, -0.00094, 0.3970, 1.002],
        ]
    )
    G = np.array([0.0001, 0.0122, 0.0001, 0.0118])
    C = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=float)
    COUNTS_PER_VOLT = 5904.5  # "5904.5 PWM input/V" (~65535 counts / 11.1 V battery)

    def __init__(self, gamma: float = 0.05):
        self.gamma = gamma
        self.est = np.zeros(4)
        self.W = np.zeros((5, 2))

    def step(self, pos: float, xdot: float, pitchc_deg: float, gy_deg: float, drive_counts: float) -> tuple[np.ndarray, np.ndarray]:
        """One loop iteration. ``drive_counts`` is the motor command from the *previous* loop."""
        phi = np.concatenate(([1.0], tansig(self.est)))
        meas = np.array([pos, xdot, pitchc_deg * 0.01745, gy_deg * 0.01745])
        fhat = self.W.T @ phi
        self.est = self.F @ self.est + self.G * (drive_counts / self.COUNTS_PER_VOLT) + self.C @ fhat + (meas - self.est)
        self.W = self.W + self.gamma * np.outer(phi, (meas - self.est) @ self.C)
        return self.est.copy(), fhat
