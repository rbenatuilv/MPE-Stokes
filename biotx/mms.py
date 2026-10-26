"""Manufactured solutions. An `exact:` section gives every unknown of the model as a formula
of x and t (SI, the same language as `loads`), and everything else is derived from it,
symbolically and exactly:

    exact:
      d: ["(1 + sin(t)) * sin(pi*x) * cos(pi*y)", "(1 + sin(t)) * sin(pi*y) * cos(pi*x)"]
      p: "(1 + sin(t)) * (cos(pi*x) * cos(pi*y) + 0.5)"
    boundaries:
      left:  {displacement: exact, pressure: exact}     # each condition takes its data from `exact`
      right: {traction: exact, flux: exact}

- sources: the model's strong form applied to the exact fields (`model.strong_form`);
- boundary data: displacement d, total traction (sigma_e(d) - alpha p I) n, pressure p,
  outward flux q.n;
- initial state: the exact fields and their time derivatives at t0 (not rest).

The discrete solution then approximates the exact fields; convergence.py measures the error.
All formulas here are in SI: the mesh is non-dimensional (x = L0 x*), so a derivative with
respect to x is ufl.grad / L0.
"""
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from biotx.expressions import _expand


class Manufactured:
    """The exact solution of one model, and the data derived from it (UFL, SI)."""

    def __init__(self, model):
        self.model, self.L0 = model, model.exp.scales.L0
        self.material = model.exp.material
        self.fields = {k: model.ex(v) for k, v in model.exp.exact.items()}
        self.sources = {k: _expand(v) for k, v in model.strong_form(self.fields, self).items()}

    # -- calculus in SI
    def grad(self, e):
        return ufl.grad(e) / self.L0

    def div(self, e):
        return ufl.div(e) / self.L0

    def dt(self, e, n=1):
        for _ in range(n):
            e = self.model.ex.ddt(e)
        return e

    def sigma_e(self, u):
        m = self.material
        return 2 * m.mu * ufl.sym(self.grad(u)) + m.lam * self.div(u) * ufl.Identity(len(u))

    # -- what the model and the error norms need
    def exact(self, name):
        """Exact value (SI) of an output quantity: d, p, w, flux, velocity."""
        if name == "flux":
            return _expand(self.model.exact_flux(self.fields, self))
        if name == "velocity":
            return self.dt(self.fields["d"])
        return self.fields[name]

    def boundary(self, kind, n):
        """Data of a boundary condition `kind: exact` (SI); n is the facet normal."""
        F, m = self.fields, self.material
        if kind == "displacement":
            return F["d"]
        if kind == "traction":
            I = ufl.Identity(len(F["d"]))
            return _expand((self.sigma_e(F["d"]) - m.alpha * F["p"] * I) * n)
        if kind == "pressure":
            return F["p"]
        if kind == "flux":
            return _expand(ufl.inner(self.exact("flux"), n))
        raise ValueError(f"no exact data for a {kind} condition")

    def initial(self):
        """The exact state at t0, as set_initial keywords."""
        out = {}
        for name, value in self.fields.items():
            out[name] = value
            if name in ("d", "w"):
                out[f"{name}_t"], out[f"{name}_tt"] = self.dt(value), self.dt(value, 2)
        return out

    def require_zero(self, expr, ds, what, samples=5):
        """Raise if the boundary integral of expr^2 is not zero at a few times in [t0, T]."""
        model, tc = self.model, self.model.exp.time
        form = dolfinx.fem.form(expr**2 * ds)
        size = dolfinx.fem.form(dolfinx.fem.Constant(model.mesh, 1.0) * ds)
        area = model.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(size), op=MPI.SUM)
        t_now = model.t.value.copy()
        try:
            for t in np.linspace(tc.t0, tc.T, samples):
                model.t.value = t / model.exp.scales.T0
                v = model.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
                if v > 1e-20 * max(area, 1.0):
                    raise ValueError(f"exact solution: {what} is not zero (at t = {t:.3g} s)")
        finally:
            model.t.value = t_now
