import pytest
from spx.config import load


def test_loads():
    cfg = load("configs/mms_stokes_2d.yaml")
    assert cfg.mesh.dim == 2 and cfg.discretization.deg == 1
    assert cfg.time is None


def test_typo_is_an_error(tmp_path):
    text = open("configs/mms_stokes_2d.yaml").read().replace("muS:", "mu_S:")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text)
    with pytest.raises(TypeError):
        load(bad)
