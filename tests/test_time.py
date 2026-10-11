# tests/test_time.py
import pytest

from spx.cases.coupled_time import time_study
from spx.config import load


@pytest.mark.slow
def test_first_order_in_time():
    """Halving dt halves every error: the scheme is first order in time."""
    df = time_study(load("configs/mms_time_2d.yaml"), N=4, levels=3)
    last = df.iloc[-1]
    for f in ("uS", "pS", "d", "pP"):
        assert last[f"rate_{f}_L2"] > 0.9, f
