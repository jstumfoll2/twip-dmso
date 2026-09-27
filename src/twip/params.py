"""Physical parameters of the two-wheeled inverted pendulum (TWIP) robot.

Ported from the parameter block that is copy-pasted at the top of every
MATLAB script (e.g. ``Thesis/Thesis Programs/*/main_bala_discrete.m`` and
``twipnonlinear.m``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def _pendulum_inertia() -> float:
    # Body modeled as two plates (0.4 kg and 0.8 kg, 0.16 m tall) offset from
    # the wheel axle by 0.235 m and 0.02 m.
    return 1 / 12 * 0.4 * 0.16**2 + 0.4 * 0.235**2 + 1 / 12 * 0.8 * 0.16**2 + 0.8 * 0.02**2


def _yaw_inertia() -> float:
    return 1 / 12 * 0.4 * (0.16**2 + 0.21**2) + 1 / 12 * 0.8 * (0.16**2 + 0.21**2)


@dataclass(frozen=True)
class RobotParams:
    Mp: float = 1.432  # kg, measured mass of body
    Mw: float = 0.1168  # kg, measured mass of each wheel (motor 1 115.3 g, motor 2 118.3 g)
    r: float = 0.045  # m, wheel radius (90 mm wheel)
    Ip: float = _pendulum_inertia()  # kg m^2, body inertia about the axle (pitch)
    Iz: float = _yaw_inertia()  # kg m^2, body inertia about the vertical axis (unused in planar model)
    Iw: float | None = None  # kg m^2, wheel inertia; None -> solid disk Mw*r^2/2
    g: float = 9.81  # m/s^2
    l: float = 0.075  # m, axle to body CG
    d: float = 0.5 * 0.263  # m, half the wheel track (unused in planar model)
    km: float = 0.11541  # N m / A, motor torque constant
    ke: float = 0.00361  # V s / rad, motor back-EMF constant
    R: float = 2.5  # ohm, motor armature resistance
    b: float = 0.0  # viscous friction (always 0 in the thesis code)

    def __post_init__(self) -> None:
        if self.Iw is None:
            object.__setattr__(self, "Iw", self.Mw * self.r**2 / 2)

    def with_thesis_uncertainty(self, l_scale: float = 0.9) -> RobotParams:
        """Perturbed "true" plant used for the uncertainty studies.

        Matches the ``% Modeling uncertainties`` block in ``twipnonlinear.m``:
        Mp*1.1, Mw*1.5 (Iw recomputed from the new Mw), l*0.9, km*2.5,
        ke*0.95, R*1.2.  ``twipnonlinear_uncert.m`` (Extra Control chapter)
        leaves ``l`` unscaled, so pass ``l_scale=1.0`` for that variant.
        """
        Mw = 1.5 * self.Mw
        return replace(
            self,
            Mp=1.1 * self.Mp,
            Mw=Mw,
            Iw=Mw * self.r**2 / 2,
            l=l_scale * self.l,
            km=2.5 * self.km,
            ke=0.95 * self.ke,
            R=1.2 * self.R,
        )


NOMINAL = RobotParams()
