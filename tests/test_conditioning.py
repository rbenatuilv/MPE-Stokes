"""Condition numbers (biotx.conditioning): SLEPc against the dense SVD for both models, both
scalings, 2D and 3D; the matrix of the requested theta; the study driver and its checks."""
import numpy as np
import pytest

import biotx
from biotx import conditioning as cn
from biotx.config import from_dict

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0,
            "rho": 1.0, "rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}
SQUARE = {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [3, 3]}
BOX = {"shape": "box", "corners": [[0, 0, 0], [2, 1, 1]], "n": [2, 1, 1]}


def experiment(model="biot2", deg=1, geometry=SQUARE, material=MATERIAL, **time):
    dim = len(geometry["corners"][0])
    sides = list(biotx.build_geometry(geometry).names)
    bnd = {s: {"traction": [0] * dim, "flux": 0} for s in sides}
    bnd[sides[0]] = {"displacement": [0] * dim, "pressure": 0}
    return from_dict({"model": model, "material": material, "geometry": geometry, "boundaries": bnd,
                      "discretization": {"deg": deg}, "time": {"T": 0.1, "dt": 0.05, **time}}, "cond")


@pytest.mark.parametrize("model, deg, geometry", [("biot2", 1, SQUARE), ("biot2", 2, SQUARE),
                                                  ("biot3", 2, SQUARE), ("biot2", 1, BOX)])
@pytest.mark.parametrize("scaling", ["raw", "jacobi", "schur"])
def test_slepc_matches_the_dense_svd(model, deg, geometry, scaling):
    c = cn.condition(biotx.make_model(experiment(model, deg, geometry)), scaling, dense_check=True)
    assert c.cond == pytest.approx(c.dense, rel=1e-6)
    for share in (c.share_min, c.share_max):
        assert sum(share.values()) == pytest.approx(1.0) and min(share.values()) >= 0


def test_the_matrix_of_the_requested_theta():
    """With a Rannacher start the model is built with theta = 1: measure the theta = 1/2 matrix."""
    cn_start = cn.condition(biotx.make_model(experiment(theta=0.5, startup_steps=2)))
    cn_plain = cn.condition(biotx.make_model(experiment(theta=0.5)))
    be = cn.condition(biotx.make_model(experiment(theta=1.0)))
    assert cn_start.cond == pytest.approx(cn_plain.cond, rel=1e-8)
    assert abs(cn_plain.cond - be.cond) > 1e-3 * be.cond


def test_jacobi_needs_a_nonzero_diagonal():
    """biot3 with c0 = 0: the pressure block of A is zero, but its Schur diagonal is not."""
    model = biotx.make_model(experiment("biot3", 2, material={**MATERIAL, "c0": 0.0}))
    with pytest.raises(ValueError, match="zero diagonal"):
        cn.condition(model, "jacobi")
    c = cn.condition(model, "schur", dense_check=True)
    assert c.cond == pytest.approx(c.dense, rel=1e-6)


def test_schur_diagonal_by_hand():
    """The p entries of the Schur diagonal against a dense computation; the others are |a_ii|."""
    model = biotx.make_model(experiment("biot3", 2))
    A = model.solver.A
    p = model._dofs["p"]
    s = cn.schur_diagonal(A, p)
    M = A.copy().convert("dense").getDenseArray()
    d = np.diag(M)
    other = np.setdiff1d(np.arange(len(d)), p)
    want = np.abs(d).copy()
    want[p] += np.abs(M[np.ix_(p, other)] * M[np.ix_(other, p)].T / np.abs(d[other])).sum(axis=1)
    assert np.allclose(s, want, rtol=1e-12)


def test_study_sweep_table_and_save(tmp_path):
    exp = experiment().with_({"output": {"dir": str(tmp_path)}})
    res = cn.run(cn.study(exp, sweep={"geometry.n": [[2, 2], [4, 4], [8, 8]]}))
    assert res.h[0] / res.h[1] == pytest.approx(2) and res.h_eff[0] / res.h_eff[1] == pytest.approx(2)
    assert (np.diff(res.cond("raw")) > 0).all() and (np.diff(res.cond("jacobi")) > 0).all()
    assert -2.6 < res.slopes("raw")[-1] < -1.4, res.table()        # cond ~ h^-2 for SIPG
    assert "cond jacobi" in res.table() and "cond schur" in res.table()
    out = cn.load_result(res.save())
    assert out["raw_cond"].shape == (3,) and str(out["key"]) == "geometry.n"
    assert out["raw_share_min_p"].shape == (3,)


@pytest.mark.parametrize("spec, match", [
    ({"sweep": {"time.dt": [0.1], "geometry.h": [0.1]}}, "exactly one"),
    ({"sweep": {"time.dt": 0.1}}, "list of values"),
    ({"sweep": {"time.dt": [0.1]}, "scaling": ["row"]}, "scaling"),
    ({"sweep": {"time.dt": [0.1]}, "dense": True}, "unknown key")])
def test_study_checks(spec, match):
    with pytest.raises(ValueError, match=match):
        cn.study(experiment(), **spec)
