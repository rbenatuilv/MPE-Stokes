from spx.cases.stokes_mms import solve
from spx.config import load
from spx.errors import convergence_table


def test_stokes_rates():
    """On three meshes, the Stokes errors fall at the rates theory predicts."""
    cfg = load("configs/mms_stokes_2d.yaml")
    last = convergence_table([solve(N, cfg) for N in (4, 8, 16)]).iloc[-1]
    assert last["rate_u_L2"] > 1.8
    assert last["rate_u_H1"] > 0.85
    assert last["rate_p_L2"] > 0.85
