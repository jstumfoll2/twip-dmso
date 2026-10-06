# Two-Wheeled Robot Thesis: Code Inventory

A map of the original thesis folder (MATLAB + Arduino, Aug 2014 - Jun 2015): what's
in it, which files matter, which are copies or older versions, and what the Python
port found. Paths below are relative to that folder (`N:\Two wheeled robot` on
the author's machine), which is not part of this repository.

**Project in one line:** a two-wheeled inverted pendulum (TWIP) robot (Arduino
Due, MPU-9150 IMU, Pololu VNH5019 driver, 30:1 gearmotors with 64 CPR encoders
(34:1, 48 CPR before firmware v7), 3S LiPo) used to study a **discrete modified
state observer (DMSO)**. The DMSO estimates state and model uncertainty with a
neural-network basis, and the thesis compares it against Kalman and
complementary filters. A later chapter adds neural-network **"extra control"** on
top of LQR.

---

## 1. Where the important code is

The final thesis MATLAB code is in **`Thesis/Thesis Programs/`**. The robot firmware
is in **`twip_v1/`** (all versions). Almost everything else is an earlier version,
an exact copy, or an experiment.

| Thesis topic | Canonical file(s) | What it does | Python port |
|---|---|---|---|
| Plant model | `Thesis Programs/*/twipnonlinear.m` | Nonlinear EOM, state `[x, xdot, theta, thetadot, v]` | `twip.legacy.dynamics.twip_nonlinear` |
| Linearized model | parameter block at the top of every `main_*.m` | A, B about upright (symbolic results hard-coded) | `twip.legacy.dynamics.linear_model` |
| Integrator | `RK4.m` (8 identical copies) | Fixed-step RK4 | `twip.integrators.rk4` |
| EOM validation | `Thesis Programs/EOM Test/EOMtest_discrete.m` | LQR on nonlinear vs. discrete linear plant | covered by `test_legacy_dynamics.py`, `test_control.py` |
| **DMSO simulation** (main result) | `Thesis Programs/{no noise no uncertainty, no noise with uncertainty, Noise with uncertainty}/main_bala_discrete.m` | LQR on the DMSO estimate; DMSO vs. 4-state Kalman; uncertainty estimation; sensor noise; saturation, deadzone, backlash | `twip.legacy.sim_dmso` |
| **Extra control** (NN) | `Thesis Programs/Extra control/ExtraControl_v5.m`, `NN1v3.m`, `twipnonlinear_uncert.m` | Two 2-layer NN "extra control" terms on top of LQR, tracking a velocity trajectory | `twip.legacy.extra_control` |
| Hardware results | `Thesis Programs/Implementation/lqrkalmantest1v2.m` + `implementationtest1-5.txt` | Replays logged LQR runs through Kalman/DMSO offline | `twip.analysis.replay_implementation` |
| On-board DMSO (hardware) | `twip_v1/twip_v9/twip_v9/filters.ino` | The DMSO that actually ran during the hardware tests | `twip.legacy.observers.FirmwareDMSO`, `analysis.replay_firmware_v9` |
| Sensor filter comparison | `Tests/Filtering Test/filteringtest.m`, `DMSO.m`, `kalmanFilter.m`, `complementaryFilter.m` | Offline complementary / Kalman / 2-state DMSO on raw IMU logs | `twip.analysis.replay_filters` |
| Sensor noise | `Tests/Allan Variance/allan.m`, `allanprocess.m` | Allan deviation + Gauss-Markov bias fit used in the sim noise model | `twip.analysis.allan` |
| Plant validation (rev 8, Section 5.3) | `implementationtest1,2,4,5.txt`, `Tests/Motor Tests/motortest9.txt` | Tests the plant model against the logged runs; recovers the gains actually used | `scripts/validate_plant.py`, `scripts/validate_closed_loop.py` |

### The three DMSO-simulation folders are not just flag changes

| | no noise no uncertainty | no noise with uncertainty | Noise with uncertainty (final) | Noise w/ unc. `New folder` (1 day older) |
|---|---|---|---|---|
| steps x dt | 100 x 0.01 | 15000 x 0.001 | 1000 x 0.01 | 1000 x 0.001 |
| Plant uncertainty in `twipnonlinear.m` | off | **on** | **off (commented out)** | on |
| Sensor noise | off | off | on | on |
| Backlash flag | – | – | on (but see bug 1) | off |
| LQR Q | diag(10,10,1e-5,1e-4) | same | diag(100,50,1e-4,1e-4) | same |
| DMSO gamma | 0.1 | 0.5 | 0.1 | 0.1 |
| Weight-update innovation | `y(i+1)-xhat(i+1)` | `y(i+1)-xhat(i+1)` | `y(i)-xhat(i)` | `y(i)-xhat(i)` |
| Saturation | 11 V | 11 V | 11.1 V | 11.1 V |

The AIAA copies (`AIAA Student Conference/AIAA programs/`) are older again:
dt = 0.013, R x 1.0 instead of 1.2, and different gamma values.

### Extra control: the saved script is not what made the figures

`ExtraControl_v5.m` was last saved on 29 May 2015. The figure folders beside it
(`Unmodeled Dynamics/`, `Deadzone/`, `Deadzone and Backlash/`,
`NL with parameter uncertainty/`) are from 26 May, made by an intermediate
version that no longer exists. Findings:

- **v5 as saved diverges under extra control.** `ke0 = 1` makes
  `xbar4 = d(xbar3)/dt`, which spikes to ~20 rad/s while the control is saturated.
  `u_e = B4^-1 (ke2*ebar4 + ...)` then drives the tilt away. LQR-only and nominal are fine.
- With `ke0 = 0` (the value in `ExtraControl_v4.m`) and dt = 0.01, the Python port
  reproduces the saved figures. Nominal and LQR-only match `Unmodeled Dynamics/figure1.jpg`,
  and all three curves match `Deadzone/figure1.jpg` and `Deadzone and Backlash/figure1.jpg`.
  The deadzone figures used the **nominal** nonlinear plant, not `twipnonlinear_uncert.m`,
  which was edited on 28 May. These settings are the `twip.legacy.extra_control` presets.
- **Backlash acts on the tilt state**, not the input. The angle stays frozen until
  `|thetadot|*dt` exceeds the 0.1 deg band, so results change completely with dt.
- **NN2's weight law is driven by `e(4)`** (tilt rate minus desired rate). The design
  suggests `ebar(4)`, which `ExtraControl_v4.m` used. Switch: `nn2_error`.
- The first NN layers `V1`, `V2` are `rand(...)` with no seed, so MATLAB runs weren't
  repeatable either. The port seeds them.
- `Ftrue` (the "actual" uncertainty in figure 3) is scaled by `-dt` in v5 but not in v4.
- The DMSO that v5 runs alongside is never used for control, and it overflows on long runs.

---

## 2. Robot firmware (`twip_v1/`)

| Version | Date | Balancing controller | Filters / estimator | Notes |
|---|---|---|---|---|
| v1 | Oct 2014 | PID | – | single 57 KB sketch |
| v2 | Oct 2014 | PID | – | serial keys tune gains live |
| v3 | Nov 2014 | PID | complementary, Kalman | |
| v4 (+ `msotest/`, `../twip_v4.zip`) | Nov 2014 | PID | + 2-state MSO | files split; **MSO F bug** (bug 2) |
| v5 | Jan 2015 | PID, sliding mode | 2-state MSO in Eigen | "to correct MSO bug" |
| v6 | Jan 2015 | same | same | 20 kHz PWM, potentiometer gain knobs |
| v7 | Feb 2015 | **LQR**, sliding mode | Eigen MSO | native USB + Bluetooth serial |
| v8 | Mar 2015 | LQR on MSO estimate | 4-state MSO (Kalman tilt in) | logged `implementationtest1-2.txt` |
| **v9** | 27 Mar 2015 | LQR on MSO estimate | 4-state MSO (complementary tilt in) | logged `implementationtest4-5.txt` → **thesis hardware results** |

- LQR gains compiled into v7-v9: `[-1.5811, -2.3951, 93.13, 14.9217]`, matching the "Modified gains
  used for simulation" comment in `main_bala_discrete.m`. On the robot, potentiometer knobs read in the
  idle loop overrode them; the gains actually used are recovered from the logs (item 18 below).
- `5904.5`: "PWM input/V" (about 65535 counts / 11.1 V battery). Control is clamped to ±10 V first.
- Loop period is 10 ms in v9 (13 ms in v8). Encoder: 0.000147 m/count.

---

## 3. Bugs and surprises

All confirmed by running the Python port. Those marked (log) were checked against
real robot logs.

1. **Backlash in the DMSO sim does nothing.** `u(i) <= m*u(i) - m*d_plus` reduces to
   `0 <= -d_plus` when `m = 1`. The "No Backlash" and "Backlash" traces in
   `Noise with uncertainty/main_bala_discrete.m` figure 11 are identical.
2. **(log) v4 firmware MSO never propagated tilt from tilt rate.** `MSO.ino` uses `F[1][0]` (= 0)
   instead of `F[0][1]` (= dt). With the bug, the port reproduces the logged `pitchmso` to 0.002 deg.
   Without it, the error is 47 deg. Fixed in v5 (Jan 2015). The Nov 2014 filter-comparison data
   (`Tests/Filtering Test/filteringtest*.csv`) was logged with the bug.
3. **(log) Kalman covariance update uses already-updated values** (`kalmanFilter.m`, `filters.ino`).
   It is not the textbook update, but it is what the robot ran.
4. **(log) The hardware Kalman in `lqrkalmantest1v2.m` used the wrong R.** `R = .1` (meant for an unused
   2-state filter) overwrites the 4x4 R. The Kalman tilt is then 3.2 deg RMS from the accelerometer,
   against 0.02 deg with the intended R, so the thesis's hardware DMSO-vs-Kalman comparison is against
   a mis-tuned Kalman filter.
5. **(log) The robot's DMSO is not the one the MATLAB replay models.** v9 hard-codes F and G for dt = .01,
   declares `Kmso = 0.5 I` but never applies it (so the gain is effectively I), and feeds in the
   complementary-filtered tilt. `lqrkalmantest1v2.m` uses `Kmso = 0.5 I` and the raw accelerometer tilt.
   `FirmwareDMSO` reproduces the logged `pitchm`, `xhat`, `xhatdot` and `gyhat` to print precision.
6. **Mixed log formats.** `implementationtest1-2.txt` are v8 logs (17 columns),
   `implementationtest3.txt` is a 15-column format that none of the builds here produce,
   and only `implementationtest4-5.txt` have the 12-column layout that `implementationimport*.m`
   assumes. v8 also logs tilt rate x 59.2958 instead of 57.2958 while balancing.
7. **Extra control as saved diverges** (see above).
8. **Uncertainty plotted one sample off** in the DMSO sim (`uncertainty(:,i)` is the residual *into* step i).
9. **Gyro walking-bias units:** estimated in deg/s, added to a rad/s signal.
10. **Input timing differs between sims:** the DMSO sim applies `u(i)` over `[t_i, t_i+1]` (one-step delay);
    the extra-control sim applies the control computed at step i immediately.
11. **Two angle conventions:** `EOM Test`, `Extra control` and `Tests/LQR Tests` use upright = pi.
    The DMSO sims use upright = 0 (`cos(-pi/2-theta)` for sin).

Found while building the corrected baseline (details in [`BASELINE.md`](BASELINE.md)):

12. **Back-EMF uses `xdot` only.** The motor turns at `xdot/r + thetadot` relative to the body.
13. **`ke` is ~85x too small** for the thesis's own Table 4.2 motor data (0.00361 vs. 0.307 V s/rad).
14. **`Ip` is about the axle**, but the EOM need it about the CG.
15. **Accelerometer modeled as tilt + noise.** A real accelerometer reads tilt relative to apparent
    gravity, off by about `xddot/g` while the robot accelerates.
16. **Encoders measure `x + r*theta`**, not `x`, because they count rotation relative to the body.
    The switch-on transients (encoder vs. tilt, with the accelerometer agreeing) indicate that the
    firmware's logged position is the mirror image, `pos = -(x + r*theta)`, so `x = -pos - r*theta`
    (`scripts/validate_plant.py`, thesis Section 5.3.3); the Section 8.4 hardware replay uses this frame.
    The closed loop does not decide it: in simulation the corrected model holds the base of the v9 runs
    only if the floor resists rolling, in either frame (about 6% of the weight mirrored, 2% unmirrored or
    with the tilt gains alone). Bench check: motors off, lean the robot so the logged tilt is positive
    and roll it toward the lean; the logged position rises if the encoder is mirrored.
17. **The thesis's discrete backlash engages one step late** (it tests the previous input).
18. **The firmware's compiled-in LQR gains were overridden by potentiometers** on the robot. The gains
    actually used, recovered from the logs by least squares (RMS residual <= 2.4 mV,
    `scripts/validate_plant.py`), are, in the firmware frame, v8 `[-0.22, -0.38, K3, 2.61]` with
    `K3` = 26.9 (test 1) or 72.5 (test 2), and v9 `[-5.49, -5.86, 118.4, 0.90]`. The v9 gains balanced
    the robot for 12.6 s and 21.2 s, although, in the mirrored frame, none of these gain sets is
    linearly stable on any candidate model (unmirrored, only the test-2 gains on the thesis model are).
19. **Motor reaction torque has the wrong sign.** The thesis pendulum moment balance has
    `-(C_L + C_R)`; the torque that turns the wheels clockwise reacts on the body
    counterclockwise, so it is `+(C_L + C_R)`. This flips the back-EMF and input terms of the
    tilt equation (`B2/B4` must equal the angular-momentum ratio about the contact point; see
    `BASELINE.md`). Items 12 and 16 were first written with the opposite signs, inherited
    from this error.

### Status

| # | Issue | Corrected baseline | Legacy (thesis reproduction) |
|---|---|---|---|
| 1 | no-op backlash | `actuators.Backlash` (play operator) | `legacy.actuators.backlash_legacy` |
| 2 | v4 firmware F bug | n/a (hardware history) | `replay_filters(firmware_bug=True)` |
| 3 | in-place Kalman covariance | `AngleBiasKalman()` textbook by default | `literal=True` (used by `replay_filters`) |
| 4 | R overwrite | `replay_implementation()` intended R by default | `reproduce_R_bug=True` |
| 5 | firmware DMSO differs | n/a (observer is being redesigned) | `legacy.observers.FirmwareDMSO` |
| 6 | mixed log formats | `data.V8_COLUMNS`, `data.fix_v8_gyhat` | |
| 7 | extra control diverges | n/a (controller is being redesigned) | `legacy.extra_control` presets |
| 8 | uncertainty one sample off | `SimResult.model_residual()` aligned to the step out of k | `legacy.sim_dmso` |
| 9 | gyro bias units | `sensors.SensorConfig` in deg/s throughout | `legacy.sensors` |
| 10 | input timing | `SimConfig.delay_steps`, v9 loop order | per legacy sim |
| 11 | angle conventions | upright = 0 only | `legacy.dynamics(convention=...)` |
| 12-14 | back-EMF, `ke`, `Ip` | `dynamics`, `params.CORRECTED` | `params.THESIS`, `back_emf="xdot"` |
| 15-16 | accelerometer, encoders | `sensors.SensorSuite`, `baseline.KalmanEstimator` | `legacy.sensors` |
| 17 | backlash timing | `actuators.Backlash` | `legacy.actuators.Backlash` |
| 18 | hardware gains | `baseline.HARDWARE_DEFAULT_GAINS` is the compiled-in set, not what ran; the gains used are recovered by `scripts/validate_plant.py` | |
| 19 | motor reaction sign | `dynamics` (`back_emf="relative"`) | `back_emf="xdot"` keeps the thesis sign |

## 4. Still missing
- Firmware producing `implementationtest3.txt` (15 columns).
- The intermediate `ExtraControl` version that produced the 26 May figures (reconstructed; see above).

---

## 5. Full folder map

| Folder / file | Size | Contents | Keep? |
|---|---|---|---|
| `Thesis/` | 784 MB | Thesis drafts (**`Jason Stumfoll Thesisv7.pdf`**, 11 Jun 2015, is the latest), derivation notes (`DMSO derivation.docx`, `Extra Control*.docx`, `Robot Model.docx`), diagrams, defense slides, **`Thesis Programs/`** | Core |
| ↳ `Thesis Programs/Noise with uncertainty/myPeaks.avi` | 630 MB | Animation render; the bulk of the folder size | Can delete |
| `twip_v1/` | – | **All firmware versions** v1-v9 + `msotest/` + `twip_v7.zip` | Core |
| `Tests/` | 196 MB | Hardware test logs + post-processing. `Filtering Test/`, `LQR Tests/` (~25 near-duplicate scripts), `Motor Tests/`, `IMU Tests/`, `Allan Variance/` | Core data |
| `Model/` | 7.8 MB | **Earliest** (Aug 2014-Jan 2015) Simulink models and drivers. `slprj/` and `*.mexw64` are build artifacts. `parameters.mat` is a different, borrowed robot (m_b = 15 kg). | Historical |
| `AIAA Student Conference/` | 20 MB | Paper, slides, and an earlier **copy** of the DMSO programs | Superseded |
| `Latex/` | 0.5 MB | `DiscreteMSO(v2).tex`: DMSO derivation write-up | Reference |
| root `*.m` | – | Exploration: `slidingmode*.m` (adaptive sliding mode; also on the robot in v5-v9), `deadzone431*.m` (NN deadzone textbook example), `discretekalman.m`, `*tuning*.m` (fminsearch DMSO tuning), `fminsearchbnd.m` (third-party) | Not thesis results |
| root `mso_v1.ino`, `mso_v2.ino`, `twip_v4.zip` | – | MSO fragments; `mso_v2` gains = `TuningParameters.mat` `copt`. `twip_v4.zip` duplicates `twip_v1/twip_v4` | Reference |
| `Other Peoples Code/` | 39 MB | Third-party AHRS, Balanduino, K_Bot, MPU-9150 libs, others' robots | Not yours |
| `Arduino Libraries/`, `Data sheet/`, `Serial-Oscilloscope-v1.5/` | – | Vendor libs, datasheets, a serial plotter | Reference |
| `Papers/` | 172 MB | 90 PDFs + EndNote library | Reference |
| `Pictures/`, `Movies/` | 262 MB | Robot photos and videos | Media |

### Duplicates (identical hashes)
- `RK4.m` x8, `savefigures.m` x26, `implementationimport(2).m` in AIAA and Thesis, `lqrkalmantest1v23/24.m` in `Tests/LQR Tests` and its subfolder.
- `twipnonlinear.m`: 11 copies, **all different**. `main_bala_discrete.m`: 8 copies, all different.

## 6. Data formats
- `.mat` files are MATLAB v5 (`scipy.io.loadmat` reads them).
- `filteringtest*.csv/.mat` (v4): `pitchc, pitchk, bias, pitchmso, pitchdotmso, fhat1, fhat2, dt, gy, ax, az`.
- `implementationtest4-5.txt` (v9): `pitchc, pitchm, x, xdot, dt, ax, az, gy, xhat, xhatdot, gyhat, control`.
- `implementationtest1-2.txt` (v8): `pitchk, pitchm, x, xdot, dt, ax, az, gy, bias, xhat, xhatdot, gyhat, control, lqr1-4`.
- Angles are in deg, rates in deg/s, accelerations in g, control in PWM counts (/5904.5 = V).
- `allandata.mat`: 1.95 M samples at 140 Hz, stationary, plus MATLAB's `avar` result.
