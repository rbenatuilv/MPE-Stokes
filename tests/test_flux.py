"""The H(div) flux reconstruction (biotx.flux): its facet flows are the solver's, its normal
component does not jump, its divergence is the solver's fluid balance, and it converges like the
raw velocity."""
import dolfinx
import numpy as np
import pytest
import ufl

import biotx
from biotx import flux
from biotx.config import from_dict

MATERIAL = {"lambda": 1.0, "mu": 1.0, "alpha": 1.0, "c0": 1.0, "K": 1.0, "muF": 1.0, "rho": 1.0,
            "rho_f": 1.0, "phi": 0.5, "tortuosity": 1.0}
SMOOTH = {"d": ["sin(t)**2 * sin(pi*x) * cos(pi*y)", "sin(t)**2 * sin(pi*y) * cos(pi*x)"],
          "w": ["sin(t)**2 * (cos(pi*x) * sin(pi*y) + 0.2)", "sin(t)**2 * (cos(pi*y) * sin(pi*x) + 0.2)"],
          "p": "sin(t)**2 * (cos(pi*x) * cos(pi*y) + 0.5)"}
E = "exact"


def square(model, deg, theta, n=4):
    """Every boundary-condition kind, with time-dependent data."""
    return from_dict({
        "model": model, "material": MATERIAL,
        "geometry": {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [n, n]},
        "boundaries": {"left": {"displacement": [0, 0], "pressure": "sin(t) * (1 + y)"},
                       "right": {"traction": [0, 0], "flux": 0},
                       "bottom": {"displacement": ["0.1 * sin(t)", 0],
                                  "flux": 0 if model == "biot3" else "0.3 * sin(2*t)"},
                       "top": {"normal_traction": "-sin(t)", "pressure": 0}},
        "discretization": {"deg": deg}, "time": {"T": 0.25, "dt": 0.05, "theta": theta}}, "sq")


@pytest.mark.parametrize("model, deg, theta", [("biot2", 1, 1.0), ("biot2", 2, 0.5), ("biot3", 2, 1.0)])
def test_conservative(model, deg, theta):
    m = biotx.make_model(square(model, deg, theta))
    rec = flux.Reconstruction(m)
    u, old, n, dS = rec.u, rec.old, m.m.n, m.m.dS
    v0 = ufl.TestFunction(dolfinx.fem.functionspace(m.mesh, ("DG", 0)))
    q = ufl.TestFunction(dolfinx.fem.functionspace(m.mesh, ("DG", m.deg_p)))
    interior, bnd = flux.numerical_flux(m, old)
    mis = (ufl.inner(u("+"), n("+")) - interior) * ufl.jump(v0) * dS + sum((ufl.inner(u, n) - b) * v0 * ds for ds, b in bnd)
    if model == "biot3":                                      # flux walls: no flow
        mis += ufl.inner(u, n) * v0 * m.bc.measure("flux")
    forms = {"facets": mis, "through": abs(interior) * (v0("+") + v0("-")) * dS,
             "div": flux.fluid_gained(m, old, q) + ufl.div(u) * q * m.m.dx, "div_scale": abs(ufl.div(u)) * q * m.m.dx}
    forms = {k: dolfinx.fem.form(f) for k, f in forms.items()}
    jump = dolfinx.fem.form(ufl.jump(u, n) ** 2 * dS)
    scale = dolfinx.fem.form(ufl.inner(ufl.avg(u), n("+")) ** 2 * dS)
    for _ in m.steps():
        rec.compute()
        a = {k: dolfinx.fem.assemble_vector(f).array for k, f in forms.items()}
        assert np.abs(a["facets"]).max() < 1e-12 * a["through"].max()               # facet flows = Fhat
        assert dolfinx.fem.assemble_scalar(jump) < 1e-24 * dolfinx.fem.assemble_scalar(scale)
        assert np.abs(a["div"]).max() < 1e-11 * np.abs(a["div_scale"]).max()        # div u = -gained
        old.advance()


def _final_error(model, deg, n, dt, T=0.05, theta=0.5):
    """L2 error at T of the reconstructed flux against the exact step-consistent flux."""
    fields = ("d", "p") if model == "biot2" else ("d", "w", "p")
    if model == "biot2":
        bnd = {"left": {"displacement": E, "pressure": E}, "bottom": {"displacement": E, "flux": E},
               "right": {"traction": E, "flux": E}, "top": {"traction": E, "pressure": E}}
    else:
        bnd = {s: {("displacement" if s in ("left", "bottom") else "traction"): E, "pressure": E}
               for s in ("left", "bottom", "right", "top")}
    m = biotx.make_model(from_dict({
        "model": model, "material": MATERIAL, "geometry": {"shape": "rectangle", "corners": [[0, 0], [1, 1]], "n": [n, n]},
        "boundaries": bnd, "exact": {k: SMOOTH[k] for k in fields}, "discretization": {"deg": deg},
        "time": {"T": T, "dt": dt, "theta": theta}}, "c"))
    rec = flux.Reconstruction(m)
    q_ex = m.mms.exact("flux") * m.exp.scales.factor["velocity"]
    th = theta if model == "biot2" else 0.5                   # biot3: (w^{n+1} - w^n)/dt ~ the midpoint
    target = th * q_ex + (1 - th) * rec.old.at_old_time(q_ex)
    dx = ufl.Measure("dx", domain=m.mesh, metadata={"quadrature_degree": 2 * rec.k + 6})
    form = dolfinx.fem.form(ufl.inner(rec.u - target, rec.u - target) * dx)
    for _ in m.steps():
        rec.compute()
        err = np.sqrt(abs(dolfinx.fem.assemble_scalar(form)))
        rec.old.advance()
    return err


@pytest.mark.parametrize("model, deg, ns, dt, order", [("biot2", 1, [4, 8, 16], 0.003125, 1),
                                                       ("biot2", 2, [4, 8], 0.0015625, 2),
                                                       ("biot3", 2, [4, 8], 0.0015625, 3)])
def test_convergence(model, deg, ns, dt, order):
    """The reconstruction converges at the order of the raw flux: deg_p for biot2, deg + 1 for biot3."""
    e = np.array([_final_error(model, deg, n, dt) for n in ns])
    rate = np.log2(e[-2] / e[-1])
    assert rate > order - 0.35, (e, rate)


# ---- cflux as an output quantity --------------------------------------------------------------
def test_snapshot_window():
    out = lambda snaps: from_dict({**square("biot2", 1, 1.0).raw, "output": {"snapshots": snaps}}, "w")
    exp = out({"from": 0.1, "to": 0.2})
    assert np.allclose(exp.output.times(exp.time), [0.1, 0.15, 0.2])
    exp = out({"from": 0.0, "to": 0.25, "every": 2})
    assert np.allclose(exp.output.times(exp.time), [0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="from, to, every"):
        out({"from": 0.1, "until": 0.2})


def test_cflux_output(tmp_path):
    """cflux in probes, fields (exact DG copy), ParaView and cbflux; the average of the saved fields."""
    exp = from_dict({**square("biot2", 1, 1.0).raw, "output": {
        "dir": str(tmp_path), "series": ["p", "cflux"], "probes": {"c": [0.5, 0.5]},
        "fields": ["p", "cflux"], "snapshots": {"from": 0.05, "to": 0.25}, "paraview": 2}}, "cf", tmp_path)
    model = biotx.make_model(exp)
    path = model.run(biotx.Recorder(model)).save()
    r = biotx.load_results(path)
    assert r["probe_cflux"].shape == (6, 1, 2)
    # the right boundary is impermeable: the conservative flow through it is zero, the raw one is not
    assert np.abs(r["cbflux_right"]).max() < 1e-12 * np.abs(r["cbflux_left"]).max()
    assert np.abs(r["bflux_right"]).max() > 1e-6 * np.abs(r["bflux_left"]).max()
    # the bottom carries the given flux 0.3 sin(2t) (backward Euler: the value at the end of the step)
    assert np.allclose(r["cbflux_bottom"][1:], 0.3 * np.sin(2 * r["t"][1:]), rtol=1e-10)
    # the saved field is exact: no normal jump; the average is the mean of the snapshots
    fl = biotx.load_fields(tmp_path / "cf_fields.npz")
    q = fl.function("cflux", 2)
    n = ufl.FacetNormal(fl.mesh)
    jump = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.jump(q, n) ** 2 * ufl.dS))
    scale = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * ufl.dS))
    assert jump < 1e-24 * scale
    mean = np.mean([fl.function("cflux", i).x.array for i in (1, 2, 3)], axis=0)
    assert np.allclose(fl.average("cflux", 0.05, 0.2).x.array, mean)
    assert (tmp_path / "cf" / "cflux.bp").exists()


def test_cflux_needs_every_step():
    model = biotx.make_model(square("biot2", 1, 1.0))
    next(model.steps())
    with pytest.raises(RuntimeError, match="before the first step"):
        model.quantity("cflux")
