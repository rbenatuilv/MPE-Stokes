"""Saved fields, run comparison, 3D shapes and slices, SI ParaView output."""
import numpy as np
import pytest

import biotx
from biotx.compare import compare
from biotx.config import from_dict
from biotx.geometry import build

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0,
            "rho": 1.0, "rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}
SQUARE = {"shape": "rectangle", "corners": [[0, 0], [2, 1]], "n": [6, 3]}
BOX = {"shape": "box", "corners": [[0, 0, 0], [2, 1, 1]], "n": [2, 1, 1]}


def experiment(tmp_path, name, geometry=SQUARE, model="biot2", deg=2, **output):
    dim = len(geometry["corners"][0]) if "corners" in geometry else 3
    zero = [0] * dim
    sides = list(build(geometry).names)
    bnd = {s: {"pressure": 0, "displacement": zero} for s in sides}
    bnd[sides[0]] = {"pressure": "sin(pi*t) * (1 + y)", "normal_traction": "-sin(pi*t)"}
    return from_dict({"model": model, "material": MATERIAL, "geometry": geometry,
                      "scales": {"L0": 2.0, "T0": 0.5, "P0": 3.0},
                      "boundaries": bnd, "discretization": {"deg": deg},
                      "time": {"T": 0.2, "dt": 0.05, "theta": 0.5},
                      "output": {"snapshots": [0.1, 0.2], "dir": str(tmp_path), "name": name, **output}},
                     name, tmp_path)


def run(exp):
    model = biotx.make_model(exp)
    rec = model.run(biotx.Recorder(model))
    path = rec.save()
    return model, path


def test_fields_round_trip_and_self_comparison(tmp_path):
    model, path = run(experiment(tmp_path, "a", fields=["d", "p", "flux", "velocity"]))
    fl = biotx.load_fields(tmp_path / "a_fields.npz")
    assert np.allclose(fl.snap_t, [0.1, 0.2]) and fl.names == ["d", "p", "flux", "velocity"]
    pts = [[0.3, 0.2], [1.7, 0.9], [1.01, 0.5]]
    for name in fl.names:                 # last snapshot = final state of the model, in SI
        f, fac = model.quantity(name)
        mine = biotx.Points(model.mesh, pts, model.exp.scales.L0)(f, fac)
        back = biotx.Points(fl.mesh, pts)(fl.function(name, 1))
        assert np.allclose(back, mine, rtol=1e-13, atol=1e-300)
    err = compare(path, [path])["a"]
    assert all(err["fields"][q].max() < 1e-13 for q in fl.names) and max(err["flux"].values()) == 0


def test_comparison_of_different_runs(tmp_path):
    _, ref = run(experiment(tmp_path, "ref", fields=True))
    _, p1 = run(experiment(tmp_path, "p1", deg=1, fields=True))
    _, f3 = run(experiment(tmp_path, "f3", model="biot3", fields=True))
    err = compare(ref, [p1, f3])
    for e in err.values():
        assert np.allclose(e["fields"]["t"], [0.1, 0.2])
        assert all(0 < e["fields"][q].max() < 1 for q in ("d", "p", "flux"))


def test_paraview_output_leaves_the_run_unchanged(tmp_path):
    _, a = run(experiment(tmp_path, "pv0", grid=11))
    _, b = run(experiment(tmp_path, "pv1", grid=11, paraview=1))
    ra, rb = biotx.load_results(a), biotx.load_results(b)
    # (two identical runs already differ at round-off: the LU solve is not bitwise reproducible)
    assert np.allclose(ra["grid_p"], rb["grid_p"], rtol=0, atol=1e-13, equal_nan=True)
    assert (tmp_path / "pv1" / "p.bp").exists()


@pytest.mark.parametrize("g, names, ncells", [
    (BOX, ["left", "right", "front", "back", "bottom", "top"], 12),
    ({"shape": "ball", "radius": 1.0, "h": 0.5}, ["outer"], None),
    ({"shape": "shell", "radius": 1.0, "inner": {"center": [0.3, 0, 0], "radius": 0.3}, "h": 0.4},
     ["outer", "inner"], None)])
def test_3d_shapes(g, names, ncells):
    d = build(g)
    assert d.mesh.topology.dim == 3 and list(d.names) == names
    tags = set(np.unique(d.facet_tags.values))
    assert tags <= set(d.names.values())
    if ncells:
        assert d.mesh.topology.index_map(3).size_global == ncells      # 2 cubes, 6 tetrahedra each


def test_3d_run_with_slices(tmp_path):
    exp = experiment(tmp_path, "box", geometry=BOX, deg=1, fields=True,
                     probes={"c": [1.0, 0.5, 0.5]},
                     slices={"mid": {"normal": "y", "at": 0.5, "n": 5}})
    _, path = run(exp)
    r = biotx.load_results(path)
    assert r["slice_mid"].shape == (25, 3) and np.allclose(r["slice_mid"][:, 1], 0.5)
    assert r["slice_mid_p"].shape == (2, 25) and r["slice_mid_flux"].shape == (2, 25, 3)
    assert not np.isnan(r["slice_mid_p"]).any() and r["probe_p"].shape == (5, 1)
    assert compare(path, [path])["box"]["fields"]["p"].max() < 1e-13


def test_grid_and_slices_by_dimension(tmp_path):
    with pytest.raises(ValueError, match="slices"):
        biotx.Recorder(biotx.make_model(experiment(tmp_path, "x", geometry=BOX, deg=1, grid=5)))
    with pytest.raises(ValueError, match="needs 3 components"):
        exp = experiment(tmp_path, "x", geometry=BOX, deg=1)
        biotx.make_model(exp.with_({"boundaries.top": {"pressure": 0, "displacement": [0, 0]}}))
