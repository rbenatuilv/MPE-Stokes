"""biotx against the original spx code (PE_analysis/pulsatile.py, archived) on the same coarse
mesh: same forms, same schemes, so the probe time series agree to round-off. The spx results
were frozen in data/spx_pulsatile.npz (scaled units, steps after t0) before spx was removed."""
from pathlib import Path

import numpy as np
import pytest

import biotx

REPO = Path(__file__).resolve().parents[1]
REF = np.load(Path(__file__).parent / "data" / "spx_pulsatile.npz")
H, T = float(REF["H"]), float(REF["T"])      # non-dimensional mesh size (L0 = 1 cm), final time


@pytest.mark.parametrize("model,deg", [("biot2", 1), ("biot2", 2), ("biot3", 2)])
def test_pulsatile_matches_spx(model, deg):
    exp = biotx.load(REPO / "experiments" / "pulsatile.yaml",
                     {"model": model, "discretization.deg": deg, "geometry.h": H * 1e-2, "time.T": T,
                      "output.snapshots": [], "output.lines": {}, "output.grid": None})
    m = biotx.make_model(exp)
    rec = m.run(biotx.Recorder(m))
    s = exp.scales
    q = np.array(rec.data["probe_flux"])[1:] * s.T0 / s.L0       # drop t0, back to scaled units
    p = np.array(rec.data["probe_p"])[1:] / s.P0
    q_ref, p_ref = REF[f"{model}_{deg}_q"], REF[f"{model}_{deg}_p"]
    assert np.abs(q - q_ref).max() < 1e-10 * np.abs(q_ref).max()
    assert np.abs(p - p_ref).max() < 1e-10 * np.abs(p_ref).max()
