"""Transport pipeline, step 1, with a transmantle loading on the annulus: both walls pulse, and only
a small zero-mean difference between them drives the flow across the tissue.

    outer wall   p = 947 + 160.5 sin(2 pi t) Pa,  d = 0
    cavity       p = 947 + 162.1 sin(2 pi t) Pa,  total traction = -alpha p n (zero effective stress)

The model is linear and the 947 Pa baseline is a static equilibrium (uniform pressure, no displacement,
no flow), so the solution is the baseline plus the response to the oscillating part. Only the
oscillating part is solved, from rest (sin(0) = 0); the pressure is relative to 947 Pa.

    python run_biot_transmantle.py   # -> results/annulus_biot{2,3}_transmantle.npz and ..._fields.npz
"""
from pathlib import Path

import biotx

HERE = Path(__file__).resolve().parent
TRANSMANTLE = {
    "params": {"P_wall": 160.5, "P_cavity": 162.1, "f": 1.0, "alpha_b": 0.49},  # alpha_b = material.alpha
    "loads": {"p_wall": "P_wall * sin(2*pi*f*t)", "p_cavity": "P_cavity * sin(2*pi*f*t)"},
    "boundaries": {"cavity": {"pressure": "p_cavity", "normal_traction": "-alpha_b * p_cavity"},
                   "wall": {"pressure": "p_wall", "displacement": [0, 0]}},
}
RECORD = {
    "time.T": 3.0,
    "output.dir": str(HERE / "results"),
    "output.fields": ["p", "flux", "cflux"],
    "output.snapshots": {"from": 2.0, "to": 3.0},      # every step of the last period
}

for model in ("biot2", "biot3"):
    exp = biotx.load(HERE.parent / f"annulus_{model}.yaml",
                     {**TRANSMANTLE, **RECORD, "output.name": f"annulus_{model}_transmantle"})
    biotx.simulate(exp)
