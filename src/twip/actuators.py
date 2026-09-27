"""Actuator nonlinearities (saturation, deadzone, backlash) and the firmware voltage path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def saturate(u: float, limit: float) -> float:
    """``if abs(u) > sat, u = sign(u)*sat``."""
    return float(np.clip(u, -limit, limit))


def deadzone(u: float, width: float) -> float:
    """``if abs(u) < dz, u = 0``: a hard deadzone with a jump."""
    return 0.0 if abs(u) < width else u


class Backlash:
    """Backlash (play) with slope ``m`` and dead band ``[d_minus, d_plus]``.

    The output follows the input only when the input pushes it from one side of
    the band: ``y = clamp(y_prev, m*(v - d_plus), m*(v - d_minus))``.

    The thesis version (:class:`twip.legacy.actuators.Backlash`) decides whether to
    engage by looking at the *previous* input, so it engages one step late and
    can miss a fast crossing completely.
    """

    def __init__(self, y0: float = 0.0, m: float = 1.0, d_plus: float = 0.0, d_minus: float = 0.0):
        self.y, self.m, self.d_plus, self.d_minus = y0, m, d_plus, d_minus

    def step(self, v: float) -> float:
        lo, hi = self.m * (v - self.d_plus), self.m * (v - self.d_minus)
        self.y = min(max(self.y, lo), hi)
        return self.y


@dataclass
class ActuatorConfig:
    """Voltage path from controller output to motor (firmware v9 ``LQR()`` + ``Drive_Motor2``).

    Defaults reproduce the firmware: clamp to +-10 V, convert to PWM counts at
    5904.5 counts/V, truncate to ``int``.  ``gain`` scales the voltage actually
    delivered (for example battery sag, or the 0.93 left-motor factor in
    ``Drive_Motor2``, averaged).  Deadzone and backlash are off by default,
    because no measured values exist for the 12 V motors.
    """

    v_limit: float = 10.0
    counts_per_volt: float = 5904.5
    quantize: bool = True
    gain: float = 1.0
    deadzone: float | None = None  # V
    backlash: tuple[float, float] | None = None  # (d_plus, d_minus) in V


class Actuator:
    def __init__(self, cfg: ActuatorConfig | None = None):
        self.cfg = cfg or ActuatorConfig()
        c = self.cfg
        self._bl = Backlash(0.0, 1.0, *c.backlash) if c.backlash else None

    def command(self, u_cmd: float) -> float:
        """The voltage the firmware *believes* it applies: clamped, then truncated to PWM counts."""
        c = self.cfg
        v = saturate(u_cmd, c.v_limit)
        if c.quantize:
            v = math.trunc(v * c.counts_per_volt) / c.counts_per_volt
        return v

    def apply(self, u_cmd: float) -> float:
        """Return the voltage reaching the motors for controller output ``u_cmd`` (V)."""
        c = self.cfg
        v = self.command(u_cmd)
        if c.deadzone is not None:
            v = deadzone(v, c.deadzone)
        if self._bl is not None:
            v = self._bl.step(v)
        return c.gain * v
