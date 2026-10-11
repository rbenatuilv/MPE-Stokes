from spx.cases.biot_mms import solve
from spx.config import load
from spx.errors import convergence_table


def test_biot_rates():
    """On three meshes, the Biot errors fall at the rates theory predicts."""
    cfg = load("configs/mms_biot_2d.yaml")
    last = convergence_table([solve(N, cfg) for N in (4, 8, 16)]).iloc[-1]
    assert last["rate_d_L2"] > 1.8 and last["rate_d_H1"] > 0.85
    assert last["rate_p_L2"] > 1.8 and last["rate_p_H1"] > 0.85
