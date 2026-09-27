"""Loaders for the robot's logged test data.

Replaces the auto-generated MATLAB ``*import*.m`` scripts, which all parse
comma-separated serial logs with hard-coded ``J:\\`` / ``H:\\`` paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io as sio

# The original thesis folder (MATLAB code and robot logs).  By default this repo is
# expected to sit inside it; set TWIP_THESIS_ROOT to point elsewhere.
THESIS_ROOT = Path(os.environ.get("TWIP_THESIS_ROOT", Path(__file__).resolve().parents[3]))

# Serial log column orders, taken from the matching import scripts / firmware prints.
IMPLEMENTATION_COLUMNS = (  # twip_v9 serial print -> implementationtest4.txt, implementationtest5.txt
    "pitchc", "pitchm", "x", "xdot", "dt", "ax", "az", "gy", "xhat", "xhatdot", "gyhat", "control",
)
# twip_v8 serial print -> implementationtest1.txt, implementationtest2.txt.  While balancing,
# v8 logs thetahatdot * 59.2958 (typo for 57.2958); in the idle loop it uses 57.2958.
# The last four columns are the potentiometer-set LQR gains (idle) or zeros (balancing).
V8_COLUMNS = (
    "pitchk", "pitchm", "x", "xdot", "dt", "ax", "az", "gy", "bias", "xhat", "xhatdot", "gyhat", "control",
    "lqr1", "lqr2", "lqr3", "lqr4",
)
# implementationtest3.txt has 15 columns and matches none of the firmware builds in twip_v1/.
FILTERING_COLUMNS = (  # twip_v4.ino serial print -> filteringtest*.csv
    "pitchc", "pitchk", "bias", "pitchmso", "pitchdotmso", "fhat1", "fhat2", "dt", "gy", "ax", "az",
)


def load_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Load a MATLAB v5 .mat file, flattening column/row vectors to 1-D."""
    raw = sio.loadmat(str(path), squeeze_me=False)
    out = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and 1 in v.shape and v.dtype != object:
            v = v.ravel()
        out[k] = v
    return out


def load_log(path: str | Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Parse a comma-separated serial log into named columns.

    Rows that don't parse to the expected number of numbers are dropped.
    Those are serial glitches and the text banners printed between runs.
    (The MATLAB import scripts turn them into NaN rows instead.)
    """
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = [p for p in line.strip().split(",") if p != ""]
            if len(parts) != len(columns):
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    arr = np.array(rows, dtype=float).reshape(-1, len(columns))
    return {name: arr[:, i] for i, name in enumerate(columns)}


def thesis_path(*parts: str) -> Path:
    return THESIS_ROOT.joinpath(*parts)
