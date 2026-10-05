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
isn't recorded (`twip.params.km_from_stall` keeps the discrepancy visible). The masses
are the thesis's measured values; `l`, like `Ip`, is a lumped-mass estimate and has not
been measured.

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

- **Mechanics:** independent Lagrangian vs. thesis symbolic EOM; the
  linearization (Jacobian); energy dissipation of the motor model; and that the
  corrected parameters follow from the thesis's own Table 4.2.
- **Measured data** (`scripts/validate_plant.py` on `implementationtest1, 2, 4, 5`;
  results in `docs/thesis/generated/plant_validation.json`, thesis Section 5.3).
  The script runs against the thesis data folder, which is not in this repo.
  The first three tests below decide the reaction sign, all in favour of the corrected
  `+T` model; the last item lists what does not discriminate or favours `-T`:
  - *Input-free momentum test* (`p_ddot = c1 theta_ddot + c2 theta`, no voltage or
    motor constants). On the two v9 switch-on transients, `+T` predicts the encoder
    from the tilt history to 1.3-1.8 mm RMS, against 5.8-12.3 mm for `-T` and
    8.5-18.9 mm for the thesis model; on the encoder and the double-integrated
    accelerometer, its errors are 3 to 16 times smaller. Free fit: `c1 = 0.205` and
    `0.203` (`+T` 0.204, `-T` 0.786, thesis 1.082). The accelerometer's axle
    coefficient is 20-50% above `+T`'s, which points to an `Ip` above the
    0.0169 kg m^2 lumped-mass estimate.
  - *Voltage response.* At switch-on the command sits at -10 V. The logged tilt falls
    15.1 deg in 100 ms. `+T` predicts -12.7 / -14.3 deg; `-T` and the thesis model
    predict that the robot keeps falling, and `-T` cannot match it for any `Ip`, `l`,
    `km` tried. The in-situ B4 is 12.6-14.7 rad/s^2 per V (`+T` 11.67, 8-26% low).
  - *Closed loop* (`scripts/validate_closed_loop.py`, which needs no robot data;
    results in `docs/thesis/generated/plant_closedloop.json`). With the v9 gains
    recovered from the logs, a simulated `-T` plant falls within 5.3 s for every one
    of 160 parameter sets (`Ip`, `l`, `km`, `ke`) and in either encoder frame, and the
    thesis model never holds the base, while the robot balanced for 12.6 s and 21.2 s.
  - *Gains used.* The gains were recovered from the logs by least squares (RMS
    residual <= 2.4 mV). With the v8 test-1 gains the base ran away at 1.49 /s; `+T`
    predicts 1.48, `-T` 5.48, thesis 2.72, though with the encoder not mirrored (and no
    estimator) the thesis model gives 1.48 too.
  - *Not discriminating, or favouring `-T`:* the steady state, whose coefficients
    contradict all three models. The robot sits in a stick-slip / backlash limit
    cycle with an encoder frozen for 0.3-0.8 s, possibly with a motor deadzone (about
    2 V on the earlier gearmotors, not measured on the present ones). No linear model
    captures this, and none predicts the tilt better than holding the state. Some
    predictions in the short v8 runs and of the tilt near upright favour the `-T`
    models, and the input regression is inconclusive.
- **Firmware frame** (inferred from the switch-on transients, encoder vs. tilt, with
  the accelerometer agreeing): logged tilt and command have the model's sign; the
  logged position is mirrored, `pos = -(x + r*theta)`, so `x = -pos - r*theta`. In
  this frame the position and velocity gains the robot used have the opposite sign
  to the LQR design (the potentiometers spanned only [-15, 0] in the firmware frame).
  The frame is not yet confirmed in closed loop: in simulation the corrected model
  reproduces the v9 runs (base held within a few mm for 21 s) in the mirrored frame
  only if the floor's rolling resistance is at least about 4% of the weight;
  unmirrored, or with the tilt gains alone, it does so with 2%.
- **Still unmeasured:** `Ip`, `l`, `km`, the reflected armature inertia, deadzone and
  backlash of the 12 V motors, and the floor's rolling resistance. The quick bench
  tests:
  1. *Wheels blocked, +1 V:* the body must pitch backward (`+T`) at about 3.7 rad/s^2
     per V. `-T` predicts forward.
  2. *Motor free-run / back-drive:* `V = R i + ke w` gives `ke`. Back-driving a wheel at
     1 rev/s should read about 1.9 V open-circuit.
  3. *Pendulum swing* with the wheels clamped:
     `T = 2*pi*sqrt((Ip + Mp*l^2) / (Mp*g*l))` gives `Ip` at a measured `l`
     (predicted 0.967 s).
  4. *Encoder frame:* motors off, lean the robot so the logged tilt is positive and
     roll it toward the lean; the logged position rises if the encoder is mirrored.
     The force needed to push it slowly over the floor it balanced on gives the
     rolling resistance.

## 2. Sensors (`twip.sensors`)

Models what firmware v9 read each 10 ms loop, in the model's sign convention. The
firmware's own logged position is the mirror image, `-(x + r*theta)` (inferred from the
switch-on transients and not yet confirmed in closed loop; see *Firmware frame* above).

| Sensor | Model | Thesis model |
|---|---|---|
| Accelerometer | specific force at the IMU, **2 cm above the axle**, in the firmware's axes; firmware tilt = `atan2(-ax, az)`; white noise + Gauss-Markov bias per axis; 1/16384 g LSB | tilt = truth + noise |
| Gyroscope | tilt rate + white noise + Gauss-Markov bias (deg/s throughout); 1/131 deg/s LSB | bias variance in deg/s added to rad/s |
| Encoders | counts of wheel rotation **relative to the body** (1920/rev); `pos = counts * 0.000147` (= `x + r*theta`; the firmware logged `-(x + r*theta)`); velocity = backward difference | `x` quantized; velocity quantized at 0.0113 m/s (13 ms) |

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
- Applied in the model frame, as `run_baseline.py --gains hardware` does, they don't
  stabilize the corrected model (spectral radius 1.001), and are marginal (0.990) on
  the thesis model.
- On the robot, potentiometers overrode them. The gains actually used are recovered
  from the logs by least squares (RMS residual <= 2.4 mV, `scripts/validate_plant.py`).
  In the firmware frame they are v8 `[-0.22, -0.38, 26.9 (test 1) or 72.5 (test 2), 2.61]`,
  with runs of 3.0 s and 3.9 s, and v9 `[-5.49, -5.86, 118.4, 0.90]`, the runs of 12.6 s
  and 21.2 s behind the thesis results. None is linearly stable on any candidate
  model. In the model frame the position and velocity entries flip sign if the
  encoder is mirrored (*Firmware frame*, Section 1).
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
