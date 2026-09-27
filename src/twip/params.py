"""Physical parameters of the two-wheeled inverted pendulum (TWIP) robot.

Two parameter sets:

* :data:`THESIS` - the values copy-pasted at the top of every thesis MATLAB script.
* :data:`CORRECTED` - the baseline for new work.  Two values change, both derived
  from the thesis's own data (see ``docs/BASELINE.md``):

  - ``ke``: the thesis value (0.00361 V s/rad) is ~85x smaller than its own Table
    4.2 free-run data implies.  Recomputed with :func:`ke_from_free_run`.
  - ``Ip``: the thesis formula gives inertia about the *axle* (it includes
    parallel-axis terms), but the equations of motion need inertia about the
    body *CG*.  Corrected by subtracting ``Mp * l**2``.

  ``km`` stays at the thesis value.
"""

from __future__ import annotations

import math
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
    Ip: float = _pendulum_inertia()  # kg m^2, body pitch inertia. The EOM treat it as about the CG; the thesis value is about the axle
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


    def scaled(self, **factors: float) -> RobotParams:
        """Multiply named parameters, e.g. ``p.scaled(Mp=1.1, km=0.9)``.

        ``Iw`` follows ``Mw`` (solid disk) unless ``Iw`` itself is scaled.
        """
        values = {k: getattr(self, k) * f for k, f in factors.items()}
        if "Mw" in factors and "Iw" not in factors:
            values["Iw"] = values["Mw"] * self.r**2 / 2
        return replace(self, **values)


# Thesis Table 4.2: Pololu 12 V 37D motor with 30:1 gearbox.
MOTOR_RATED_VOLTAGE = 12.0  # V
MOTOR_FREE_RUN_SPEED_RPM = 350.0
MOTOR_FREE_RUN_CURRENT = 0.300  # A
MOTOR_STALL_TORQUE = 110 * 0.00706155  # oz-in -> N m
MOTOR_STALL_CURRENT = 5.0  # A


def ke_from_free_run(
    V: float = MOTOR_RATED_VOLTAGE,
    I_free: float = MOTOR_FREE_RUN_CURRENT,
    rpm_free: float = MOTOR_FREE_RUN_SPEED_RPM,
    R: float = 2.5,
) -> float:
    """Back-EMF constant from free-run data: ``V = R*I + ke*omega`` at steady state."""
    return (V - R * I_free) / (rpm_free * 2 * math.pi / 60)


def km_from_stall(tau_stall: float = MOTOR_STALL_TORQUE, I_stall: float = MOTOR_STALL_CURRENT) -> float:
    """Torque constant from stall data (~0.155 N m/A for Table 4.2).  Not used by
    :data:`CORRECTED`; it is kept here so the discrepancy with 0.11541 is visible."""
    return tau_stall / I_stall


THESIS = RobotParams()  # parameter values as used throughout the thesis

CORRECTED = RobotParams(
    ke=ke_from_free_run(),  # ~0.307 V s/rad
    Ip=THESIS.Ip - THESIS.Mp * THESIS.l**2,  # ~0.0169 kg m^2, about the body CG
)
