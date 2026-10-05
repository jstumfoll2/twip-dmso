# Corrected baseline: plant, sensors, actuator, controller

This is the reference that new controller and observer designs are compared
against. It fixes the problems found while porting the thesis code
([`CODE_INVENTORY.md`](CODE_INVENTORY.md)). The thesis-faithful code is still in
`twip.legacy`, so thesis results remain reproducible.

| Piece | Module | Thesis version |
|---|---|---|
| Parameters | `twip.params.CORRECTED` | `twip.params.THESIS` |
| Plant | `twip.dynamics` | `twip.legacy.dynamics` |
| Sensors | `twip.sensors` | `twip.legacy.sensors` |
| Actuator | `twip.actuators.Actuator` | inline in each script |
| Controller + estimators + closed loop | `twip.baseline` | `twip.legacy.sim_dmso`, `twip.legacy.extra_control` |

```bash
uv run python scripts/run_baseline.py
```

---

## 1. Plant

**Structure.** Planar TWIP, both wheels lumped, derived by Lagrange's equations
(mass-matrix form in `twip.dynamics`). `theta` is counterclockwise-positive with
the CG at `(x - l sin(theta), l cos(theta))`; rolling forward turns the wheels
clockwise at `xdot/r`. With `back_emf="xdot"` it reproduces the thesis EOM to
1e-12, sign error included (below).

**Correction 0: motor reaction sign.** The motor torque `T` turns the wheels
clockwise and reacts on the body counterclockwise, so the generalized forces are
`(T/r, +T)`. The thesis pendulum moment balance had `-(C_L + C_R)`,
i.e. `(T/r, -T)`. Check: the torque is internal, so it cannot change the angular
momentum about the wheel contact point; at rest and upright that requires
`B2/B4 = (Ip + Mp l^2 + Mp l r) / (r beta + Mp l)`, which the corrected model
satisfies and the thesis model does not (`test_motor_torque_is_internal_to_the_robot`).

**Correction 1: back-EMF kinematics.** The motors are fixed to the body, so their
speed is `w_rel = xdot/r + thetadot`. The input terms already imply this (virtual
work of `T` through `x/r + theta`), but the thesis back-EMF used `xdot/r` only,
which drops the A24/A44 coupling. The corrected form is also the one that makes
the motors strictly dissipative at zero voltage (tested).

**Correction 2: `ke`.** Thesis Table 4.2 lists the 12 V 37D 30:1 motor as 350 rpm and
0.3 A free-run, and 110 oz-in and 5 A stall, with R = 2.5 ohm. Free-run steady state
gives `ke = (12 - 0.3*2.5) / 36.65 rad/s = 0.307 V s/rad`. The thesis used 0.00361,
~85x smaller, which removes almost all back-EMF damping.

**Correction 3: `Ip` about the CG.** The thesis formula
`1/12*.4*.16^2 + .4*.235^2 + 1/12*.8*.16^2 + .8*.02^2` includes parallel-axis terms
for plates 23.5 cm and 2 cm above the axle, so it is the inertia about the *axle*.
The EOM need it about the *CG*: `Ip_cg = Ip_axle - Mp*l^2 = 0.0169 kg m^2` (was 0.0250).

**Not changed:** `km = 0.11541`. The stall data give ~0.155, but the source of 0.11541
isn't recorded (`twip.params.km_from_stall` keeps the discrepancy visible). All masses
and `l` are the thesis's measured values.

Effect on the linearized model (dt = 0.01):

| | thesis | corrected |
|---|---|---|
| A22, A24 | -0.098, 0 | -12.65, -0.569 |
| A42, A43, A44 | -0.095, 39.7, 0 | -79.6, 57.0, -3.58 |
| B1, B2 | 1.22, 1.18 | 1.86, 11.67 |
| unstable pole | 6.30 rad/s | 5.95 rad/s |
| RHP zero of x/u | 6.11 rad/s | 5.95 rad/s |

In the corrected model the unstable mode barely moves the base: at zero voltage
the back-EMF torque on a falling body is close to the torque that holds the base
still, so the pole and the zero of `x/u` nearly coincide (5.953 vs 5.946).

### What the plant is (and isn't) validated against

- **Validated:** the mechanics (independent Lagrangian vs. thesis symbolic EOM); the
  linearization (Jacobian); energy dissipation of the motor model; and that the
  corrected parameters follow from the thesis's own Table 4.2.
- **Not validated by data.** The hardware logs (`implementationtest4/5.txt`) cannot
  discriminate between the thesis and corrected parameters:
  - One-step regression explains only 15-34% of the variance, with unstable
    coefficients (closed-loop data under `u = -Kx` is nearly collinear).
  - Over 100-200 ms, open-loop prediction from logged states and voltages is *worse*
    for both models than a naive "tilt rate stays constant" predictor (1.72 deg vs.
    1.93 deg thesis and 2.26 deg corrected, RMS tilt error at 200 ms).
  - A simulation-error parameter fit diverges (`Ip -> infinity`).

  The logs are dominated by effects neither model contains (frame flex, motor
  mismatch, battery voltage, timing, the filtered tilt signal).
- **Recommended bench tests** (if the robot still exists). Each takes minutes and pins
  down the parameters that matter most:
  1. *Motor free-run:* wheels off the ground, step the PWM through 2-10 V and log the
     encoder speed. The slope gives `ke` directly; the current sense gives `km` and `R`.
  2. *Pendulum swing:* hold the wheels, let the body swing hanging *down*, and time
     the period. `T = 2*pi*sqrt((Ip + Mp*l^2) / (Mp*g*l))` gives `Ip` at the measured `l`.
  3. *Balance-point check:* the tilt at which the robot balances on the bench (motors
     off) gives any CG offset from the body axis.

## 2. Sensors (`twip.sensors`)

Models what firmware v9 read each 10 ms loop:

| Sensor | Model | Thesis model |
|---|---|---|
| Accelerometer | specific force at the IMU, **2 cm above the axle**, in the firmware's axes; firmware tilt = `atan2(-ax, az)`; white noise + Gauss-Markov bias per axis; 1/16384 g LSB | tilt = truth + noise |
| Gyroscope | tilt rate + white noise + Gauss-Markov bias (deg/s throughout); 1/131 deg/s LSB | bias variance in deg/s added to rad/s |
| Encoders | counts of wheel rotation **relative to the body** (1920/rev); `pos = counts * 0.000147` (= `x + r*theta`); velocity = backward difference | `x` quantized; velocity quantized at 0.0113 m/s (13 ms) |

Noise levels are the thesis Allan-variance results (reproduced in `twip.analysis`).
Biases start from their stationary distribution; set `bias_init="zero"` to start at zero.

**The accelerometer correction matters most.** An accelerometer measures tilt
relative to *apparent* gravity. While the robot accelerates at `a`, its accelerometer
tilt is off by about `a/g` (0.1 rad at 1 m/s^2). In closed loop this biases any
estimator that treats it as `theta + noise`. The thesis's idealized sensor model
could not show this. (An earlier version of this baseline found that the
firmware's complementary-filter pipeline runs away within 1-3 s; that was an
artifact of the motor-torque sign error. On the corrected plant it balances,
with a tilt-estimate error of about 0.02 deg after recovery.)

## 3. Actuator (`twip.actuators.Actuator`)

Firmware v9 path: clamp to +-10 V, then `int(V * 5904.5)` PWM counts. 5904.5 counts/V
is the firmware comment "5904.5 PWM input/V", about 65535/11.1 V. Optional: `gain`
(battery sag, the 0.93 left-motor factor in `Drive_Motor2`), deadzone, and backlash.
Backlash is now a correct play operator; the thesis rule engaged one step late, and
the DMSO sim's version was a no-op. No measured deadzone or backlash exists for the
12 V motors, so both are off by default.

## 4. Baseline controller (`twip.baseline`)

**Primary: `design_lqr()`**, discrete LQR (`lqrd`) on the corrected linear model with
the final thesis weights `Q = diag(100, 50, 1e-4, 1e-4)`, `R = 1000`, dt = 10 ms.
Keeping the weights means differences from the thesis come from the plant correction.

`K = [-0.298, -13.39, 22.13, 3.121]` (thesis plant: `[-0.296, -0.904, 74.9, 11.9]`).
The velocity gain grows to work against the back-EMF damping, and the tilt gains
shrink because the corrected tilt input gain B4 is ten times larger. The position
loop stays slow (a ~22 s closed-loop mode, |lambda| = 0.9995), so position creeps
slowly after recovery.

**Estimator: `KalmanEstimator`**, a textbook 4-state Kalman filter (Joseph form) on the
corrected model:
- The encoder rows include the `+r*theta` terms.
- The accelerometer row is `theta + (xddot - h*thetaddot)/g`. Both accelerations are
  linear in state and voltage, so this is a linear measurement with feedthrough.
- `R` comes from the sensor model; `tilt_std = 0.1 rad` of extra accelerometer
  uncertainty was chosen by a robustness sweep (below).

**Second baseline: `HARDWARE_DEFAULT_GAINS = [-1.5811, -2.3951, 93.13, 14.9217]`**, the
gains compiled into firmware v7-v9. They are *not* a working baseline:
- They don't stabilize the corrected model (spectral radius 1.001), and are
  marginal (0.990) on the thesis model.
- On the robot, potentiometers overrode them. The v8 logs show the gains actually
  used, e.g. `[-0.22, -0.381, 72.5, 2.62]`, with runs of 2-3 s; the v9 runs behind the
  thesis results didn't log their gains.
- They match the MATLAB comment "Modified gains used for simulation", i.e. they came
  from simulation.

**Loop timing** (`simulate`) follows v9: sample sensors, estimate, compute `u[k]`, and
hold it on the motors until the next loop. `delay_steps` adds whole-sample delays.
The accelerometer sees the acceleration produced by the voltage currently on the motors.

### Baseline performance (15 s runs, noise on, seed 0)

| Case | Result |
|---|---|
| 5 deg from rest, nominal | recovers; peak 0.12 m/s, 1.9 V; RMS tilt after 5 s 0.005 deg; estimate error 0.02 deg |
| + 1 sample delay | recovers |
| + actuator gain 0.93 / 0.5 V deadzone | recovers (deadzone: 0.26 deg RMS limit-cycle) |
| IMU on the top plate (0.24 m) | recovers |
| Thesis initial condition (1 m, 0.3 m/s, 10 deg, 1 deg/s) | recovers; peak 2.7 V |
| Complementary estimator (firmware style) | recovers; estimate error 0.02 deg |

Robustness to plant error (true plant perturbed, estimator and controller nominal;
starts of 2 deg and 5 deg, seeds 0-3, 8 runs each): survives `Mp x1.1 km x0.9 l x0.9`,
`l x0.9`, `l x1.1`, `Ip x1.3`, `km x1.2` and the thesis's `km x2.5`, all with a
tilt-estimate error of about 0.05 deg. `ke` is the weak point: `ke x0.5` survives with a
1.4 deg tilt-estimate error, and `ke x2` falls in 4 of 8 runs. The `ke` sensitivity is in
model-based tilt estimation: the accelerometer row depends on the modeled `xddot`. That
is where an uncertainty-estimating observer should help, and it matters because `ke`
is the least certain parameter.

The Kalman noise setting `tilt_std = 0.1 rad` came from a sweep over the `ke` runs on
the plant before the sign correction, and has not been re-swept.

## 5. Plugging in a new design

```python
from twip.baseline import SimConfig, simulate

class MyObserver:            # anything with these two methods
    def initialize(self, x0): ...
    def update(self, m, v_prev):   # m: twip.sensors.Measurement; returns [x, xdot, theta, thetadot]
        ...

def my_controller(xhat, k):  # returns volts
    ...

r = simulate(my_controller, MyObserver(), SimConfig(plant=CORRECTED.scaled(ke=0.5)))
r.metrics(settle_from=5)
r.model_residual()           # the true lumped uncertainty, aligned with the step out of k
```

`Measurement` gives raw firmware-unit data (`ax`, `az`, `gy`, `counts`, `pos`, `vel`, `dt`)
plus `accel_tilt` and `gyro_rate` helpers. `SimConfig.disturbance` adds unmodeled
`(xddot, thetaddot)` terms (for example `twip.legacy.dynamics.uncert_extra_terms`), and
`CORRECTED.scaled(...)` perturbs parameters.
