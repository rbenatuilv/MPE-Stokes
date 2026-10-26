"""Convergence studies (biotx.convergence) and manufactured solutions (biotx.mms): observed
orders in space and in time, self-convergence on nested meshes, and the input checks.
The MMS cases exercise every boundary condition kind with data derived from `exact`."""
import numpy as np
import pytest

from biotx import convergence as cv
from biotx import geometry, make_model
from biotx.config import from_dict

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0,
            "rho": 1.0, "rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}
E = "exact"
ALL_KINDS = {"left": {"displacement": E, "pressure": E}, "bottom": {"displacement": E, "flux": E},
             "right": {"traction": E, "flux": E}, "top": {"traction": E, "pressure": E}}
PRESSURE = {k: {**{m: E for m in v if m in ("displacement", "traction")}, "pressure": E}
            for k, v in ALL_KINDS.items()}              # biot3: only zero flux is supported
SMOOTH = {"d": ["(1 + sin(t)) * sin(pi*x) * cos(pi*y)", "(1 + sin(t)) * sin(pi*y) * cos(pi*x)"],
          "w": ["(1 + sin(t)) * (cos(pi*x) * sin(pi*y) + 0.2)", "(1 + sin(t)) * (cos(pi*y) * sin(pi*x) + 0.2)"],
          "p": "(1 + sin(t)) * (cos(pi*x) * cos(pi*y) + 0.5)"}
LINEAR = {"d": ["exp(t) * (x + 2*y)", "exp(t) * (x - y)"], "w": ["cos(t) * (1 + y)", "cos(t) * (2 - x)"],
          "p": "sin(t) * (1 + x + y)"}                  # in P1 in space: only the time error


def experiment(model="biot2", deg=1, n=4, exact=SMOOTH, bnd=None, T=0.1, dt=0.003125, theta=0.5, **extra):
    fields = ("d", "p") if model == "biot2" else ("d", "w", "p")
    raw = {"model": model, "material": MATERIAL,
           "geometry": {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [n, n]},
           "boundaries": bnd or (ALL_KINDS if model == "biot2" else PRESSURE),
           "exact": {k: exact[k] for k in fields}, "discretization": {"deg": deg},
           "time": {"T": T, "dt": dt, "theta": theta}, **extra}
    return from_dict(raw, f"test_{model}")


def test_space_biot2_p1():
    """P1/P1, all four boundary condition kinds: order 2 in L2, 1 in H1, flux order 1."""
    res = cv.run(cv.study(experiment(), study="space", n=[8, 16, 32]))
    r = {k: res.rates(*k)[-1] for k in res.errors}
    assert r[("d", "L2")] > 1.8 and r[("p", "L2")] > 1.8, r
    assert r[("d", "H1")] > 0.9 and r[("p", "H1")] > 0.9 and r[("flux", "L2")] > 0.9, r


@pytest.mark.slow
def test_space_biot3_p2p2p1():
    """P2/P2/P1: d and w order 3 in L2, p order 2."""
    res = cv.run(cv.study(experiment("biot3", deg=2), study="space", n=[4, 8, 16]))
    r = {k: res.rates(*k)[-1] for k in res.errors}
    assert r[("d", "L2")] > 2.7 and r[("w", "L2")] > 2.7 and r[("p", "L2")] > 1.6, r


@pytest.mark.parametrize("model, deg, theta, order", [("biot2", 1, 0.5, 2), ("biot2", 1, 1.0, 1),
                                                      ("biot3", 2, 0.5, 2)])
def test_time_orders(model, deg, theta, order):
    """Crank-Nicolson and Newmark: order 2; backward Euler: order 1."""
    exp = experiment(model, deg, n=2, exact=LINEAR, T=0.5, theta=theta)
    res = cv.run(cv.study(exp, study="time", dt=[0.05, 0.025, 0.0125], fields=["d", "p"]))
    for k in res.errors:
        assert abs(res.rates(*k)[-1] - order) < 0.15, (k, res.rates(*k))


def test_self_convergence_on_nested_meshes():
    """Differences between successive uniform refinements fall at the order of the error."""
    exp = experiment(n=4, T=0.05, dt=0.0125)
    res = cv.run(cv.study(exp, study="self", levels=4, fields=["d", "p"]))
    assert res.h[0] / res.h[1] == pytest.approx(2) and len(res.errors[("d", "L2")]) == 3
    assert res.rates("d", "L2")[-1] > 1.7 and res.rates("p", "L2")[-1] > 1.7, res.table()
    assert res.rates("d", "H1")[-1] > 0.85, res.table()


def test_refine_keeps_the_boundary_names():
    d = geometry.build({"shape": "annulus", "radius": 1.0, "inner": {"center": [0.2, 0], "radius": 0.3},
                        "h": 0.3})
    counts = [np.bincount(d.facet_tags.values)[1:]]
    for _ in range(2):
        d = geometry.refine(d)
        counts.append(np.bincount(d.facet_tags.values)[1:])
    assert all((c == 2 * b).all() for b, c in zip(counts, counts[1:])), counts


def test_at_max_and_save(tmp_path):
    exp = experiment(n=4, T=0.02, dt=0.005, output={"dir": str(tmp_path)})
    final = cv.run(cv.study(exp, study="space", n=[2, 4], fields=["p"], norms=["L2"]))
    worst = cv.run(cv.study(exp, study="space", n=[2, 4], fields=["p"], norms=["L2"], at="max"))
    assert (np.array(worst.errors[("p", "L2")]) >= np.array(final.errors[("p", "L2")]) - 1e-15).all()
    out = cv.load_result(worst.save())
    assert out["p_L2"].shape == (2,) and out["experiment"]["convergence"]["at"] == "max"


@pytest.mark.parametrize("change, match", [
    ({"boundaries": {**ALL_KINDS, "top": {"normal_traction": E, "pressure": E}}}, "normal_traction"),
    ({"sources": {"fluid_source": 1.0}}, "derived"),
    ({"exact": {"d": SMOOTH["d"]}}, "needs exactly"),
])
def test_exact_checks(change, match):
    raw = {**experiment().raw, **change}
    with pytest.raises(ValueError, match=match):
        from_dict(raw)


def test_biot3_exact_flux_needs_zero_normal_w():
    bnd = {**PRESSURE, "right": {"traction": E, "flux": E}}
    with pytest.raises(ValueError, match="w.n"):
        make_model(experiment("biot3", deg=2, bnd=bnd))


@pytest.mark.parametrize("spec, match", [({"study": "space", "n": [4, 8], "h": [0.1, 0.05]}, "either n or h"),
                                         ({"study": "time", "dt": [0.1]}, "two values"),
                                         ({"study": "self", "levels": 3, "dt": [0.1]}, "unknown key"),
                                         ({"study": "space", "n": [4, 8], "norms": ["H2"]}, "norms")])
def test_study_checks(spec, match):
    with pytest.raises(ValueError, match=match):
        cv.study(experiment(), **spec)
