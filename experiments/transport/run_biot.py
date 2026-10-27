"""Transport pipeline, step 1: run the Biot annulus with both models and save the conservative
velocity `cflux` (and the raw `flux`, for comparison) at every step of the last period.

    python run_biot.py      # -> results/annulus_biot{2,3}.npz and results/annulus_biot{2,3}_fields.npz

The problem is experiments/annulus_biot{2,3}.yaml; only what is recorded is set here. The run starts
from rest and is periodic after two cycles, so the last period, t in [2, 3] s, is the periodic flow.
"""
from pathlib import Path

import biotx

HERE = Path(__file__).resolve().parent
RECORD = {
    "time.T": 3.0,
    "output.dir": str(HERE / "results"),
    "output.fields": ["p", "flux", "cflux"],
    "output.snapshots": {"from": 2.0, "to": 3.0},      # every step of the last period
}

for model in ("biot2", "biot3"):
    exp = biotx.load(HERE.parent / f"annulus_{model}.yaml", RECORD)
    biotx.simulate(exp)