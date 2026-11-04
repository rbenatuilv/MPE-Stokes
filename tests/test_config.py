"""YAML parsing, overrides, unit scaling and the boundary-condition checks."""
from pathlib import Path

import pytest

import biotx
from biotx.biot2 import TwoField
from biotx.config import from_dict

PULSATILE = Path(__file__).parent / "data" / "pulsatile.yaml"


def test_scaling_of_the_brain_parameters():
    exp = biotx.load(PULSATILE)
    c = exp.material.scaled(exp.scales)
    assert c.mu == pytest.approx(1.0) and c.lam == pytest.approx(11567 / 1500)
    assert c.kappa == pytest.approx(1.57e-11 / 3.5e-3 * 1500 / 1e-4)
    assert c.rhoW == pytest.approx(c.rhoF * 1.6 / 0.2)


def test_overrides_and_with_():
    exp = biotx.load(PULSATILE, {"time.dt": 0.02, "model": "biot3"})
    assert exp.time.dt == 0.02 and exp.model == "biot3" and exp.time.nsteps == 150
    assert exp.with_({"discretization.deg": 1}).discretization.deg == 1


def test_E_nu():
    raw = biotx.load(PULSATILE).raw
    raw["material"] = {"E": 3.0, "nu": 0.25, "alpha": 1, "c0": 0, "K": 1, "muF": 1}
    m = from_dict(raw).material
    assert m.mu == pytest.approx(1.2) and m.lam == pytest.approx(1.2)


@pytest.mark.parametrize("bc", [{"pressure": 0}, {"pressure": 0, "flux": 0, "displacement": [0, 0]},
                                {"pressure": 0, "dispalcement": [0, 0]}])
def test_bad_boundaries(bc):
    raw = biotx.load(PULSATILE).raw
    raw["boundaries"]["pia"] = bc
    with pytest.raises(ValueError):
        from_dict(raw)


def test_unknown_boundary_name():
    exp = biotx.load(PULSATILE, {"geometry.h": 4e-3, "boundaries.skull": {"pressure": 0, "displacement": [0, 0]}})
    with pytest.raises(ValueError, match="unknown boundaries"):
        biotx.make_model(exp)


@pytest.mark.parametrize("key, value, match", [
    ("sources", {"fluid_sorce": 1.0}, "not used by biot2"),
    ("sources", {"mass_source": 1.0}, "not used by biot2"),
    ("geometry.raduis", 1.0, "unknown key"),
    ("geometry.inner", {"center": [0, 0], "r": 1.0}, "inner takes"),
    ("geometry.shape", "circle", "unknown geometry shape"),
    ("params.t", 1.0, "hide built-in"),
    ("params.e", 1.0, "hide built-in"),
    ("loads.r", "x", "hide built-in"),
    ("loads.A", "1", "both params and loads"),
])
def test_names_are_checked(key, value, match):
    with pytest.raises(ValueError, match=match):
        biotx.load(PULSATILE, {key: value})


def test_sources_depend_on_the_model():
    biotx.load(PULSATILE, {"model": "biot3", "sources": {"mass_source": 0, "fluid_force": [0, 0]}})
    with pytest.raises(ValueError, match="not used by biot3"):
        biotx.load(PULSATILE, {"model": "biot3", "sources": {"fluid_source": 0}})


def test_unused_material_is_reported():
    with pytest.warns(UserWarning, match=r"\['rho_f', 'phi', 'tortuosity'\] not used by biot2"):
        biotx.load(PULSATILE)
    with pytest.warns(UserWarning, match=r"\['beta'\] not used by biot3"):
        biotx.load(PULSATILE, {"model": "biot3", "material.beta": 1e-9})


class Renamed(TwoField):
    """A model defined outside biotx: the class alone registers it."""
    name = "test_renamed"


def test_a_new_model_registers_itself():
    exp = biotx.load(PULSATILE, {"model": "test_renamed", "geometry.h": 4e-3, "time.T": 0.01})
    model = biotx.make_model(exp)
    assert type(model) is Renamed and model.sd is model.nmk["d"]
    assert next(model.steps()) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="test_renamed"):
        biotx.load(PULSATILE, {"model": "nonexistent"})
