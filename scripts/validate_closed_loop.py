"""Can each candidate plant balance with the gains the robot actually ran?

    uv run python scripts/validate_closed_loop.py           # everything (a few minutes)
    uv run python scripts/validate_closed_loop.py --quick   # coarser grid

The v9 runs balanced for 12.6 s and 21.2 s with the base held within a few
millimeters, on the gains recovered from the logs by ``validate_plant.py``.  This
script runs the simulated v9 loop of :func:`validate_plant.simulate` (sensors,
complementary filter, firmware DMSO, saturation, PWM truncation) with those gains
for 21 s on each candidate plant (+T, -T, rev7) and asks whether it stays up, and
whether the base stays put:

1. over a grid of the uncertain parameters (Ip, l, km, ke), clean and with the
   nuisance terms of ``simulate`` (gearbox friction, backlash, rolling resistance),
   with the encoder mirrored (the frame inferred from the switch-on transients),
   not mirrored, and with the position and velocity gains removed;
2. at nominal parameters, over the floor's rolling resistance (0 to 15 % of the
   weight) with and without gearbox friction;
3. whether the switch-on test of ``validate_plant.integral_test`` still ranks a
   simulated plant correctly when the floor resists rolling that much (each
   plant with its own stabilizing LQR, as in the method validation there).

Needs no robot data.  Writes ``docs/thesis/generated/plant_closedloop.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "thesis" / "generated" / "plant_closedloop.json"

K_V9 = (-5.491, -5.86, 118.36, 0.893)  # firmware frame, validate_plant.recover_gains
FRAMES = {  # name -> (pos_sign, gains)
    "mirrored": (-1.0, K_V9),
    "not mirrored": (+1.0, K_V9),
    "tilt gains only": (-1.0, (0.0, 0.0, K_V9[2], K_V9[3])),
}
T_RUN = 21.0
GRID_COLUMNS = ("model", "frame", "nuisance", "Ip", "l", "km_scale", "ke_scale", "fell_at_s", "base_range_mm", "base_range_last_10s_mm", "tilt_rms_deg")
BASE_TOL_MM = 10.0  # "base held": range over the last 10 s


def _vp():
    spec = importlib.util.spec_from_file_location("validate_plant", HERE / "validate_plant.py")
    vp = importlib.util.module_from_spec(spec)
    sys.modules["validate_plant"] = vp
    spec.loader.exec_module(vp)
    return vp


vp = None


def _init() -> None:
    global vp
    vp = _vp()


def _run(model: str, frame: str, nuisance: bool, seed: int = 1, mu_rr: float = 0.02, vc: float = 0.6, **par) -> dict:
    base, be = vp.MODELS[model]
    if par:
        p = replace(base, Ip=par["Ip"], l=par["l"], km=base.km * par["km_scale"], ke=base.ke * par["ke_scale"])
        vp.MODELS["_trial"] = (p, be)
        name = "_trial"
    else:
        name = model
    pos_sign, K = FRAMES[frame]
    try:
        rn = vp.simulate(name, seed, T=T_RUN, theta0_deg=3.0, nuisance=nuisance, K=np.array(K), mu_rr=mu_rr, vc=vc, pos_sign=pos_sign)
    except RuntimeError as e:
        return {"fell_at_s": float(str(e).split("t=")[1])}
    finally:
        vp.MODELS.pop("_trial", None)  # integral_test iterates over MODELS
    last = rn.t > T_RUN - 10
    return {
        "fell_at_s": None,
        "base_range_mm": round(float(np.ptp(rn.P) * 1e3), 1),
        "base_range_last_10s_mm": round(float(np.ptp(rn.P[last]) * 1e3), 1),
        "tilt_rms_deg": round(float(np.sqrt(np.mean(rn.th[rn.t > 2] ** 2)) / vp.DEG), 2),
    }


def _grid_job(a):
    model, frame, nuisance, Ip, cg, ks, kes = a
    r = _run(model, frame, nuisance, Ip=Ip, l=cg, km_scale=ks, ke_scale=kes)
    return {"model": model, "frame": frame, "nuisance": nuisance, "Ip": Ip, "l": cg, "km_scale": ks, "ke_scale": kes} | r


def _floor_job(a):
    model, frame, mu, vc, seed = a
    r = _run(model, frame, True, seed=seed, mu_rr=mu, vc=vc)
    return {"model": model, "frame": frame, "mu_rr": mu, "vc": vc, "seed": seed} | r


def _method_job(a):
    model, mu, seed0 = a
    alpha = 0.99 if model == "+T" else 1.0  # as in validate_plant.main
    for seed in range(seed0, seed0 + 20):
        try:
            rn = vp.simulate(model, seed, nuisance=True, alpha=alpha, mu_rr=mu)
            break
        except RuntimeError:
            continue
    else:
        return {"model": model, "mu_rr": mu, "seed": None, "balancing_run": False}
    it = vp.integral_test(rn, 0.1, 0.8, 300, np.random.default_rng(seed))
    res = {"model": model, "mu_rr": mu, "seed": seed, "balancing_run": True, "travel_mm": round(it["P_range_mm"], 1)}
    for key, coef in (("encoder", "c1"), ("accelerometer", "a")):
        q = it[key]
        rms = {k: round(v["rms_mm"], 2) for k, v in q["models"].items()}
        res[key] = {"rms_mm": rms, "ranked_first": min(rms, key=rms.get), coef: round(q["free_fit"][coef], 3), coef + "_ci": [round(c, 3) for c in q["free_fit"][coef + "_ci"]]}
    return res


def _held(o: dict) -> bool:
    return o["fell_at_s"] is None and o["base_range_last_10s_mm"] < BASE_TOL_MM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarser grid")
    args = ap.parse_args()
    models = ("+T", "-T", "rev7")
    if args.quick:
        Ips, ls, kss, kes = (0.008, 0.017, 0.030), (0.05, 0.075, 0.15), (0.5, 1.0, 4.0), (1.0,)
    else:
        Ips, ls, kss, kes = (0.008, 0.012, 0.017, 0.025, 0.030), (0.05, 0.075, 0.10, 0.15), (0.5, 1.0, 2.0, 4.0), (1.0, 1.7)
    grid_jobs = list(itertools.product(models, FRAMES, (False, True), Ips, ls, kss, kes))
    floor_jobs = list(itertools.product(models, FRAMES, (0.0, 0.02, 0.04, 0.06, 0.10, 0.15), (0.0, 0.6), (1, 2)))
    method_jobs = list(itertools.product(("+T", "-T"), (0.06, 0.10), (101, 121)))
    with Pool(initializer=_init) as pool:
        print(f"1. parameter grid: {len(grid_jobs)} runs")
        grid = pool.map(_grid_job, grid_jobs, chunksize=8)
        print(f"2. floor rolling resistance: {len(floor_jobs)} runs")
        floor = pool.map(_floor_job, floor_jobs, chunksize=4)
        print(f"3. switch-on test under floor resistance: {len(method_jobs)} runs")
        method = pool.map(_method_job, method_jobs)

    summary = {}
    for model, frame, nuis in itertools.product(models, FRAMES, (False, True)):
        rows = [o for o in grid if (o["model"], o["frame"], o["nuisance"]) == (model, frame, nuis)]
        up = [o for o in rows if o["fell_at_s"] is None]
        falls = sorted(o["fell_at_s"] for o in rows if o["fell_at_s"] is not None)
        summary[f"{model}|{frame}|{'nuisance' if nuis else 'clean'}"] = {
            "parameter_sets": len(rows),
            "up_for_21s": len(up),
            "up_and_base_held": sum(_held(o) for o in rows),
            "longest_up_s": T_RUN if up else max(falls),
            "median_fall_s": float(np.median(falls)) if falls else None,
            "smallest_base_range_last_10s_mm_if_up": min((o["base_range_last_10s_mm"] for o in up), default=None),
        }
    floor_table = {}
    for model, frame, vc in itertools.product(models, FRAMES, (0.0, 0.6)):
        row = {}
        for mu in (0.0, 0.02, 0.04, 0.06, 0.10, 0.15):
            v = [o for o in floor if (o["model"], o["frame"], o["vc"], o["mu_rr"]) == (model, frame, vc, mu)]
            row[f"{mu:.2f}"] = [o["fell_at_s"] if o["fell_at_s"] is not None else f"up, base {o['base_range_last_10s_mm']:.0f} mm" for o in v]
        floor_table[f"{model}|{frame}|gearbox friction {vc} V"] = row
    out = {
        "settings": {
            "gains_fw_frame": K_V9,
            "T_s": T_RUN,
            "theta0_deg": 3.0,
            "base_held_if_range_last_10s_below_mm": BASE_TOL_MM,
            "grid": {"Ip": Ips, "l": ls, "km_scale": kss, "ke_scale": kes},
            "quick": args.quick,
        },
        "grid_summary": summary,
        "floor_rolling_resistance": floor_table,
        "switch_on_test_under_floor_resistance": method,
        "grid_columns": GRID_COLUMNS,
        "grid_runs": [[o.get(c) for c in GRID_COLUMNS] for o in grid],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    rows = ",\n".join("  " + json.dumps(r) for r in out["grid_runs"])  # one run per line
    text = text[: text.index('"grid_runs": [') + len('"grid_runs": [')] + "\n" + rows + "\n ]\n}"
    OUT.write_text(text)
    for k, v in summary.items():
        print(f"   {k:34s} up {v['up_for_21s']:3d}/{v['parameter_sets']}, base held {v['up_and_base_held']:3d}, longest {v['longest_up_s']:.1f} s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
