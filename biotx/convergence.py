"""Convergence studies: errors and observed orders under refinement in space or in time.

    python -m biotx.convergence experiments/convergence/mms_biot2_space.yaml [--set k=v]

A study is an experiment YAML with a `convergence:` section. Three kinds:

    convergence:                  # 1. space: manufactured solution (`exact:`), meshes refined
      study: space
      n: [8, 16, 32, 64]          #    cells per side (rectangle, box), or
      h: [4.0e-3, 2.0e-3, ...]    #    target cell sizes (any shape)

    convergence:                  # 2. time: manufactured solution, time step refined on a
      study: time                 #    fixed mesh (take `exact` polynomial of degree <= deg in
      dt: [0.1, 0.05, 0.025]      #    space: then the mesh adds no error)

    base: ../annulus_biot2.yaml   # 3. self: no exact solution; uniform refinement of the base
    convergence:                  #    mesh, differences between successive levels
      study: self
      levels: 4                   #    base mesh + 3 refinements (nested meshes)
      refine_dt: false            #    true: halve dt with h

Common keys (all optional):
      norms: [L2, H1]             # L2 norm and broken H1 seminorm |.|_{H1} = ||grad_h .||
      fields: [d, p, flux]        # default: the model's unknowns and the flux (flux: L2 only)
      at: final                   # space/time: final (t = T) or max (over all time steps);
                                  # self: final or a list of times (maximum over them)

Errors are in SI, integrated with quadrature degree deg + 4. The observed order between two
levels is log(e_i / e_{i+1}) / log(s_i / s_{i+1}), s = h (space, self) or dt (time). For
`self`, e_i = ||u_i - u_{i+1}|| between levels i and i+1, computed on the finer mesh (exact:
the meshes are nested), and its order equals the order of the true error.

    from biotx import convergence as cv
    res = cv.run(cv.load("experiments/convergence/mms_biot2_space.yaml"))
    print(res.table()); res.rates("p", "L2"); res.save()
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import basix
import basix.ufl
import dolfinx
import numpy as np
import ufl
import yaml
from mpi4py import MPI

from biotx import geometry
from biotx.config import FIELDS, _override, from_dict, merge, read_study

STUDIES = ("space", "time", "self")
NORMS = ("L2", "H1")
KEYS = {"space": {"n", "h"}, "time": {"dt"}, "self": {"levels", "refine_dt"}}
COMMON = {"study", "norms", "fields", "at"}


# ---- the study description ---------------------------------------------------------------
@dataclass
class Study:
    """An experiment tree plus its `convergence` section (validated)."""
    raw: dict
    name: str
    root: Path = Path(".")
    spec: dict = field(default_factory=dict)

    def experiment(self, overrides=None):
        return from_dict(_override(self.raw, overrides or {}), self.name, self.root)


def _check(raw):
    spec = dict(raw.get("convergence") or {})
    kind = spec.get("study")
    if kind not in STUDIES:
        raise ValueError(f"convergence.study must be one of {STUDIES}, got {kind!r}")
    unknown = set(spec) - COMMON - KEYS[kind]
    if unknown:
        raise ValueError(f"convergence ({kind}): unknown key(s) {sorted(unknown)}; "
                         f"allowed {sorted(COMMON | KEYS[kind])}")
    model = raw.get("model", "biot2")
    spec.setdefault("norms", list(NORMS))
    spec.setdefault("fields", list(FIELDS[model]) + ["flux"])
    spec.setdefault("at", "final")
    if set(spec["norms"]) - set(NORMS):
        raise ValueError(f"convergence.norms: one or more of {NORMS}")
    allowed = set(FIELDS[model]) | {"flux", "velocity"}
    if set(spec["fields"]) - allowed:
        raise ValueError(f"convergence.fields: {model} has {sorted(allowed)}")
    if kind in ("space", "time"):
        if not raw.get("exact"):
            raise ValueError(f"convergence ({kind}) needs an `exact` solution")
        if spec["at"] not in ("final", "max"):
            raise ValueError("convergence.at: final or max")
    if kind == "space":
        if len([k for k in ("n", "h") if k in spec]) != 1:
            raise ValueError("convergence (space): give either n or h")
        if len(spec.get("n", spec.get("h"))) < 2:
            raise ValueError("convergence (space): at least two levels")
    elif kind == "time":
        if len(spec.get("dt", [])) < 2:
            raise ValueError("convergence (time): dt needs at least two values")
    else:
        if int(spec.get("levels", 0)) < 2:
            raise ValueError("convergence (self): levels >= 2")
        spec.setdefault("refine_dt", False)
        if spec["at"] != "final" and not isinstance(spec["at"], (list, tuple)):
            raise ValueError("convergence.at (self): final or a list of times")
    return spec


def load(path, overrides=None) -> Study:
    """Read a study YAML (with an optional `base:` experiment it extends)."""
    path = Path(path)
    raw = _override(read_study(path), overrides or {})
    return Study(raw, raw.get("output", {}).get("name") or path.stem, path.parent, _check(raw))


def study(exp, **spec) -> Study:
    """A Study from an Experiment and convergence keys, e.g. study(exp, study="time", dt=[...])."""
    raw = merge(exp.raw, {"convergence": spec})
    return Study(raw, exp.name, exp.root, _check(raw))


# ---- error norms ---------------------------------------------------------------------------
class _Norms:
    """L2 and broken H1-seminorm errors (SI) of model quantities against the exact solution."""

    def __init__(self, model, fields, norms):
        s, dim = model.exp.scales, model.mesh.geometry.dim
        dx = ufl.Measure("dx", domain=model.mesh, metadata={"quadrature_degree": model.k + 4})
        self.model, self.forms = model, {}
        for name in fields:
            f, fac = model.quantity(name)
            e = fac * f - model.mms.exact(name)
            for norm in norms:
                if norm == "H1" and name == "flux":
                    continue
                if norm == "L2":
                    form = s.L0**dim * ufl.inner(e, e) * dx
                else:
                    form = s.L0**(dim - 2) * ufl.inner(ufl.grad(e), ufl.grad(e)) * dx
                self.forms[(name, norm)] = dolfinx.fem.form(form)

    def __call__(self):
        comm = self.model.mesh.comm
        for name in {n for n, _ in self.forms}:
            self.model.quantity(name)                    # refresh a post-processed flux
        return {k: float(np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)))
                for k, f in self.forms.items()}


def _mms_errors(model, spec):
    norms = _Norms(model, spec["fields"], spec["norms"])
    worst = {}
    for _ in model.steps():
        if spec["at"] == "max":
            for k, v in norms().items():
                worst[k] = max(worst.get(k, 0.0), v)
    return worst if spec["at"] == "max" else norms()


class _Difference:
    """||u_fine - u_coarse|| (L2, broken H1 seminorm, SI) on the finer of two nested meshes.
    Both are interpolated into DG on the fine mesh with interior (Gauss-Legendre) points, so
    no point sits on a facet; for nested meshes both interpolations are exact."""

    def __init__(self, V_coarse, V_fine, L0, norms, qdeg):
        mesh = V_fine.mesh
        deg = max(V_coarse.ufl_element().degree, V_fine.ufl_element().degree)
        el = basix.ufl.element("DG", mesh.basix_cell(), deg, shape=V_fine.value_shape,
                               lagrange_variant=basix.LagrangeVariant.gl_warped)
        Vt = dolfinx.fem.functionspace(mesh, el)
        self.c, self.f = dolfinx.fem.Function(Vt), dolfinx.fem.Function(Vt)
        nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        self.cells = np.arange(nloc, dtype=np.int32)
        self.data = dolfinx.fem.create_interpolation_data(Vt, V_coarse, self.cells, padding=1e-10)
        dim, e = mesh.geometry.dim, self.f - self.c
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": qdeg})
        self.forms = {}
        for norm in norms:
            form = (L0**dim * ufl.inner(e, e) * dx if norm == "L2"
                    else L0**(dim - 2) * ufl.inner(ufl.grad(e), ufl.grad(e)) * dx)
            self.forms[norm] = dolfinx.fem.form(form)
        self.comm = mesh.comm

    def __call__(self, coarse, fine):
        self.c.interpolate_nonmatching(coarse, self.cells, self.data)
        self.f.interpolate(fine)
        return {k: float(np.sqrt(self.comm.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)))
                for k, f in self.forms.items()}


# ---- results --------------------------------------------------------------------------------
@dataclass
class Result:
    """Errors of one study. errors[(field, norm)]: one value per level (per pair for self)."""
    name: str
    kind: str
    model: str
    label: str                    # what varies: n, h, dt or level
    values: list
    h: np.ndarray                 # largest cell diameter [m] per level
    dt: np.ndarray                # time step [s] per level
    dofs: np.ndarray
    errors: dict
    raw: dict
    root: Path = Path(".")

    @property
    def scale(self):
        """The refinement parameter of each error: dt (time) or h (space, self)."""
        s = self.dt if self.kind == "time" else self.h
        return s[:-1] if self.kind == "self" else s

    def rates(self, name, norm):
        e, s = np.asarray(self.errors[(name, norm)]), self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(e[:-1] / e[1:]) / np.log(s[:-1] / s[1:])

    def table(self):
        keys = list(self.errors)
        first = {"n": "n", "h": "h target", "dt": "dt", "level": "levels"}[self.label]
        head = [first] + ([] if self.kind == "time" else ["h"]) + ["dofs"]
        for name, norm in keys:
            head += [f"{name} {norm}", "rate"]
        rows = [head]
        nrow = len(self.scale)
        for i in range(nrow):
            v = f"{i}-{i + 1}" if self.kind == "self" else str(self.values[i])
            row = [v] + ([] if self.kind == "time" else [f"{self.scale[i]:.3e}"]) + [str(int(self.dofs[i]))]
            for k in keys:
                r = self.rates(*k)
                row += [f"{self.errors[k][i]:.3e}", f"{r[i - 1]:.2f}" if i > 0 else ""]
            rows.append(row)
        w = [max(len(r[j]) for r in rows) for j in range(len(head))]
        title = {"space": "MMS, refinement in space", "time": "MMS, refinement in time",
                 "self": "self-convergence, e_i = ||u_i - u_i+1||"}[self.kind]
        lines = [f"{self.name}: {self.model}, {title}"]
        lines += ["  " + "  ".join(c.rjust(wj) for c, wj in zip(r, w)) for r in rows]
        return "\n".join(lines)

    def save(self, path=None):
        """Write the errors, rates and the study YAML to <root>/<output.dir>/<name>.npz."""
        if MPI.COMM_WORLD.rank != 0:
            return None
        out = self.raw.get("output", {}).get("dir", "results")
        path = Path(path) if path else Path(self.root) / out / f"{self.name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"values": np.array(self.values, dtype=object if self.label == "n" else float),
                "h": self.h, "dt": self.dt, "dofs": self.dofs,
                "experiment": yaml.safe_dump(_plain(self.raw), sort_keys=False)}
        for (name, norm), e in self.errors.items():
            data[f"{name}_{norm}"] = np.asarray(e)
            data[f"rate_{name}_{norm}"] = self.rates(name, norm)
        np.savez(path, **data)
        return path


def _plain(x):
    """The raw tree with Python callables (API-only exact solutions) shown as text."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x if isinstance(x, (str, int, float, bool, type(None))) else repr(x)


def load_result(path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files}
    out["experiment"] = yaml.safe_load(str(out["experiment"]))
    return out


# ---- the three studies -------------------------------------------------------------------
def _make(exp, domain=None):
    from biotx import make_model
    return make_model(exp, domain)


def _dofs(model):
    return model.W.dofmap.index_map.size_global * model.W.dofmap.index_map_bs


def _space(st):
    spec = st.spec
    label = "n" if "n" in spec else "h"
    values, errs, h, dofs = list(spec[label]), {}, [], []
    dim = len(st.raw["geometry"]["corners"][0]) if label == "n" else None
    for v in values:
        ov = {"geometry.n": list(v) if isinstance(v, (list, tuple)) else [int(v)] * dim} if label == "n" \
            else {"geometry.h": float(v)}
        exp = st.experiment(ov)
        model = _make(exp)
        for k, e in _mms_errors(model, spec).items():
            errs.setdefault(k, []).append(e)
        h.append(geometry.cell_size(model.mesh) * exp.scales.L0)
        dofs.append(_dofs(model))
    return Result(st.name, "space", exp.model, label, values, np.array(h),
                  np.full(len(values), exp.time.dt), np.array(dofs), errs, st.raw, st.root)


def _time(st):
    spec = st.spec
    values, errs = [float(v) for v in spec["dt"]], {}
    domain = None
    for dt in values:
        exp = st.experiment({"time.dt": dt})
        model = _make(exp, domain)
        domain = model.domain                            # the same mesh for every dt
        for k, e in _mms_errors(model, spec).items():
            errs.setdefault(k, []).append(e)
    h = geometry.cell_size(domain.mesh) * exp.scales.L0
    return Result(st.name, "time", exp.model, "dt", values, np.full(len(values), h),
                  np.array(values), np.full(len(values), _dofs(model)), errs, st.raw, st.root)


def _self(st):
    spec = st.spec
    levels = int(spec["levels"])
    exp0 = st.experiment()
    times = [exp0.time.T] if spec["at"] == "final" else [float(t) for t in spec["at"]]
    domain = geometry.build(exp0.geometry, exp0.scales.L0, exp0.root)
    states, h, dts, dofs = [], [], [], []
    for lvl in range(levels):
        if lvl > 0:
            domain = geometry.refine(domain)
        dt = exp0.time.dt / 2**lvl if spec["refine_dt"] else exp0.time.dt
        exp = st.experiment({"time.dt": dt})
        model = _make(exp, domain)
        snap = {}
        for t in model.steps():
            hit = [ts for ts in times if abs(t - ts) < dt / 2]
            if hit:
                snap[hit[0]] = {}
                for name in spec["fields"]:
                    f, fac = model.quantity(name)
                    g = dolfinx.fem.Function(f.function_space)
                    g.x.array[:] = fac * f.x.array
                    snap[hit[0]][name] = g
        missing = [t for t in times if t not in snap]
        if missing:
            raise ValueError(f"convergence.at: no time step at {missing}")
        states.append(snap)
        h.append(geometry.cell_size(domain.mesh) * exp.scales.L0)
        dts.append(dt)
        dofs.append(_dofs(model))
    errs = {}
    qdeg = exp0.discretization.deg + 4
    for i in range(levels - 1):
        for name in spec["fields"]:
            norms = [n for n in spec["norms"] if not (n == "H1" and name == "flux")]
            diff = _Difference(states[i][times[0]][name].function_space,
                               states[i + 1][times[0]][name].function_space, exp0.scales.L0, norms, qdeg)
            worst = {}
            for t in times:
                for norm, v in diff(states[i][t][name], states[i + 1][t][name]).items():
                    worst[norm] = max(worst.get(norm, 0.0), v)
            for norm, v in worst.items():
                errs.setdefault((name, norm), []).append(v)
    return Result(st.name, "self", exp0.model, "level", list(range(levels)), np.array(h),
                  np.array(dts), np.array(dofs), errs, st.raw, st.root)


def run(st: Study) -> Result:
    """Run every level of the study and return its errors."""
    return {"space": _space, "time": _time, "self": _self}[st.spec["study"]](st)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convergence study from a YAML file")
    ap.add_argument("yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    res = run(load(args.yaml, overrides))
    path = res.save()
    if MPI.COMM_WORLD.rank == 0:
        print(res.table())
        print(f"saved {path}")
    return res


if __name__ == "__main__":
    main()
