"""Post-processing: what is measured during a run, and how it is saved. Everything stored is
in SI (positions in m, times in s, p in Pa, d in m, q in m/s).

    rec = Recorder(model)                # reads model.exp.output (probes, lines, grid, ...)
    model.run(rec)
    rec.save()                           # -> <yaml dir>/<output.dir>/<name>.npz
    res = load_results(path)             # dict of arrays + the experiment YAML

Quantities (model.quantity(name)): "d", "p", "flux" (Darcy flux q, the raw field), "velocity"
(solid d_t), "cflux" (the conservative Darcy flux of each step, biotx.flux) and "w" for the
three-field model. Asking for cflux in `series` or `fields` makes the Recorder start the
reconstruction, so create the Recorder before the first step.

Recorded every step
    t                         times
    probe_<quantity>          (nsteps, nprobes[, dim]) values at the probes, for `series`
    bflux_<boundary>          integral of q.n over each named boundary [m^2/s per unit depth
                              in 2D, m^3/s in 3D]: positive = fluid leaving the tissue
    cbflux_<boundary>         the same with cflux (when cflux is recorded): exactly the solver's
                              boundary flow over the last step
Recorded at the `snapshots` times (a list, or every step of a window {from, to, every})
    snap_t                    times actually hit (nearest step)
    grid_<quantity>           (nsnap, n*n[, dim]) 2D: on an n x n grid over the bounding box
    slice_<name>_<quantity>   (nsnap, n*n, [dim]) 3D: on an n x n grid in a plane (see slice_points)
    line_<name>_<quantity>    (nsnap, npoints[, dim]) along each line
    jump                      ||[[q.n]]|| / ||{q}.n|| on interior facets (0 = conservative flux)
Points outside the mesh give NaN. With output.fields, the full fields at the snapshots also go
to <name>_fields.npz (see fields.py; compare runs with compare.py).
"""
from pathlib import Path

import dolfinx
import numpy as np
import ufl
import yaml
from mpi4py import MPI

from biotx.fields import DEFAULT as DEFAULT_FIELDS
from biotx.fields import FieldWriter


class Points:
    """Evaluate Functions at fixed points (SI). Works in parallel: each point is evaluated by
    the rank that owns its cell, then the values are combined on every rank."""

    def __init__(self, mesh, pts, L0=1.0):
        pts = np.atleast_2d(np.asarray(pts, dtype=float)) / L0
        self.n = len(pts)
        self.pts = np.zeros((self.n, 3))
        self.pts[:, :pts.shape[1]] = pts
        self.comm = mesh.comm
        tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        cand = dolfinx.geometry.compute_collisions_points(tree, self.pts)
        coll = dolfinx.geometry.compute_colliding_cells(mesh, cand, self.pts)
        nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        idx, cells = [], []
        for i in range(self.n):
            owned = [c for c in coll.links(i) if c < nloc]
            if owned:
                idx.append(i)
                cells.append(owned[0])
        self.idx, self.cells = np.array(idx, dtype=np.int64), np.array(cells, dtype=np.int32)
        count = np.zeros(self.n)
        count[self.idx] = 1
        self.count = np.empty(self.n)
        self.comm.Allreduce(count, self.count, op=MPI.SUM)

    def __call__(self, f, factor=1.0):
        """(npoints, value_size) array; value_size 1 is squeezed. NaN outside the mesh."""
        vs = f.function_space.value_size
        local = np.zeros((self.n, vs))
        if len(self.idx):
            local[self.idx] = f.eval(self.pts[self.idx], self.cells)
        out = np.empty_like(local)
        self.comm.Allreduce(local, out, op=MPI.SUM)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out / self.count[:, None]              # a point on a rank boundary: average
        out[self.count == 0] = np.nan
        out *= factor
        return out[:, 0] if vs == 1 else out


def line_points(spec):
    """n points from spec["from"] to spec["to"] (SI)."""
    a, b = np.asarray(spec["from"], dtype=float), np.asarray(spec["to"], dtype=float)
    s = np.linspace(0.0, 1.0, spec.get("n", 101))[:, None]
    return a + s * (b - a)


def grid_points(mesh, n, L0):
    """n x n cell-centred grid over the bounding box of the (2D) mesh, in SI."""
    x = mesh.geometry.x[:, :2]
    lo, hi = np.empty(2), np.empty(2)
    mesh.comm.Allreduce(x.min(axis=0), lo, op=MPI.MIN)
    mesh.comm.Allreduce(x.max(axis=0), hi, op=MPI.MAX)
    lo, hi = lo * L0, hi * L0
    s = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(lo[0] + s * (hi[0] - lo[0]), lo[1] + s * (hi[1] - lo[1]))
    return np.column_stack([X.ravel(), Y.ravel()])


def slice_points(mesh, spec, L0):
    """n x n cell-centred grid on the plane {x_normal = at} of a 3D mesh, over the bounding box
    of the other two coordinates, in SI. spec: {normal: x|y|z, at: value [m], n: int (81)}.
    Points (n*n, 3), the first in-plane axis varying fastest: reshape(n, n) gives [i_b, i_a]
    with (a, b) = (y, z), (x, z) or (x, y)."""
    axis = "xyz".index(spec["normal"])
    a, b = [i for i in range(3) if i != axis]
    n = spec.get("n", 81)
    lo, hi = np.empty(3), np.empty(3)
    mesh.comm.Allreduce(mesh.geometry.x.min(axis=0), lo, op=MPI.MIN)
    mesh.comm.Allreduce(mesh.geometry.x.max(axis=0), hi, op=MPI.MAX)
    lo, hi = lo * L0, hi * L0
    s = (np.arange(n) + 0.5) / n
    A, B = np.meshgrid(lo[a] + s * (hi[a] - lo[a]), lo[b] + s * (hi[b] - lo[b]))
    pts = np.zeros((n * n, 3))
    pts[:, a], pts[:, b], pts[:, axis] = A.ravel(), B.ravel(), float(spec["at"])
    return pts


def _scalar(mesh, form):
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


class NormalJump:
    """||[[q.n]]|| / ||{q}.n|| over the interior facets: 0 for a mass-conserving (H(div))
    flux. Two-field q = -kappa grad p jumps; three-field q = w_t only weakly."""

    def __init__(self, model):
        q, n, dS = model.flux(), model.m.n, model.m.dS
        self.num = dolfinx.fem.form(ufl.jump(q, n) ** 2 * dS)
        self.den = dolfinx.fem.form(ufl.inner(ufl.avg(q), n("+")) ** 2 * dS)
        self.mesh = model.mesh

    def __call__(self):
        num, den = _scalar(self.mesh, self.num), _scalar(self.mesh, self.den)
        return float(np.sqrt(num / den)) if den > 0 else 0.0


class BoundaryFlux:
    """Integral of q.n over each named boundary, in SI (outflow positive): bflux_<name> with the
    raw flux, and cbflux_<name> with the conservative one (cflux) if asked for."""

    def __init__(self, model, cflux=False):
        s, n = model.exp.scales, model.m.n
        dim = model.mesh.geometry.dim
        self.factor = (s.L0 / s.T0) * s.L0 ** (dim - 1)
        fields = {"bflux": model.flux(), **({"cbflux": model.quantity("cflux")[0]} if cflux else {})}
        self.forms = {f"{key}_{name}": dolfinx.fem.form(ufl.inner(q, n) * model.m.ds(tag))
                      for key, q in fields.items() for name, tag in model.domain.names.items()}
        self.mesh = model.mesh

    def __call__(self):
        return {name: self.factor * _scalar(self.mesh, f) for name, f in self.forms.items()}


def dg_space(V):
    """V itself if it is DG, else the DG space of the same polynomial degree and shape (which
    contains V: an RT field is a polynomial on each cell, so it is copied exactly)."""
    el = V.ufl_element()
    if el.discontinuous:
        return V
    return dolfinx.fem.functionspace(V.mesh, ("DG", el.embedded_superdegree, V.value_shape))


def copy_to(g, f, factor=1.0):
    """g <- factor * f, g in dg_space(f's space)."""
    if g.function_space == f.function_space:
        g.x.array[:] = factor * f.x.array
    else:
        g.interpolate(f)
        g.x.array[:] *= factor
    g.x.scatter_forward()


class Recorder:
    """Records what the experiment's `output` section asks for. Call it with the model after
    every step (model.run(rec) does that); call save() at the end."""

    def __init__(self, model, output=None):
        self.model, self.o = model, output or model.exp.output
        mesh, L0 = model.mesh, model.exp.scales.L0
        self.L0 = L0
        self.probes = Points(mesh, list(self.o.probes.values()), L0) if self.o.probes else None
        self.lines = {k: (line_points(v), Points(mesh, line_points(v), L0)) for k, v in self.o.lines.items()}
        dim = mesh.geometry.dim
        if (self.o.grid and dim != 2) or (self.o.slices and dim != 3):
            raise ValueError("output: `grid` is for 2D meshes, `slices` for 3D meshes")
        self.grid = None
        if self.o.grid:
            pts = grid_points(mesh, self.o.grid, L0)
            self.grid = (pts, Points(mesh, pts, L0))
        self.slices = {k: (pts, Points(mesh, pts, L0))
                       for k, pts in ((k, slice_points(mesh, v, L0)) for k, v in self.o.slices.items())}
        self.snap_times = self.o.times(model.exp.time)
        names = DEFAULT_FIELDS if self.o.fields is True else (self.o.fields or [])
        cflux = "cflux" in self.o.series or "cflux" in names
        if cflux:
            model.quantity("cflux")                      # start the reconstruction (before any step)
        self.jump = NormalJump(model) if self.snap_times else None
        self.bflux = BoundaryFlux(model, cflux) if self.o.boundary_flux else None
        self.data = {}
        self.snap_quantities = sorted(set(self.o.series) | {"flux", "p"})
        self.fields = FieldWriter(model, names) if names else None
        self.writers = self._open_paraview() if self.o.paraview else None

    def _append(self, key, value):
        self.data.setdefault(key, []).append(value)

    def __call__(self, model=None):
        m = model or self.model
        t, dt = m.time, m.exp.time.dt
        self._append("t", t)
        if self.probes is not None:
            for name in self.o.series:
                f, fac = m.quantity(name)
                self._append(f"probe_{name}", self.probes(f, fac))
        if self.bflux is not None:
            for key, v in self.bflux().items():
                self._append(key, v)
        if any(abs(t - ts) < dt / 2 for ts in self.snap_times):
            self._append("snap_t", t)
            for name in self.snap_quantities:
                f, fac = m.quantity(name)
                if self.grid is not None:
                    self._append(f"grid_{name}", self.grid[1](f, fac))
                for sn, (_, pts) in self.slices.items():
                    self._append(f"slice_{sn}_{name}", pts(f, fac))
                for ln, (_, pts) in self.lines.items():
                    self._append(f"line_{ln}_{name}", pts(f, fac))
            self._append("jump", self.jump())
            if self.fields is not None:
                self.fields(m)
        if self.writers is not None and m.step % self.o.paraview == 0:
            self._write_paraview(t)

    # ParaView: one VTX file per quantity (VTX writes DG fields exactly), in SI: the values are
    # copied into SI Functions (cflux, a Raviart-Thomas field, into DG of its polynomial degree,
    # exactly), and the mesh coordinates are scaled by L0 only while writing.
    def _open_paraview(self):
        out = self.path().with_suffix("")
        out.mkdir(parents=True, exist_ok=True)
        ws = {}
        for name in self.snap_quantities + ["d"]:
            f, fac = self.model.quantity(name)
            g = dolfinx.fem.Function(dg_space(f.function_space), name=name)
            ws[name] = (f, fac, g, dolfinx.io.VTXWriter(self.model.mesh.comm, out / f"{name}.bp", [g]))
        return ws

    def _write_paraview(self, t):
        x = self.model.mesh.geometry.x
        x0 = x.copy()
        x *= self.L0
        try:
            for f, fac, g, w in self.writers.values():
                copy_to(g, f, fac)
                w.write(t)
        finally:
            x[:] = x0                                    # exactly the solver's coordinates again

    def path(self):
        exp = self.model.exp
        return Path(exp.root) / self.o.dir / f"{exp.name}.npz"

    def save(self, path=None):
        """Write all records (+ points, mesh size and the experiment YAML) to an .npz."""
        if self.writers is not None:
            for *_, w in self.writers.values():
                w.close()
            self.writers = None
        if self.model.mesh.comm.rank != 0:
            return None
        path = Path(path) if path else self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {"ncells": self.model.mesh.topology.index_map(self.model.mesh.topology.dim).size_global,
                 "experiment": yaml.safe_dump(self.model.exp.raw, sort_keys=False)}
        if self.o.probes:
            extra["probe_names"] = np.array(list(self.o.probes))
            extra["probes"] = np.array(list(self.o.probes.values()), dtype=float)
        for k, (pts, _) in self.lines.items():
            extra[f"line_{k}"] = pts
        if self.grid is not None:
            extra["grid"] = self.grid[0]
        for k, (pts, _) in self.slices.items():
            extra[f"slice_{k}"] = pts
        if self.fields is not None:
            self.fields.save(path.with_name(f"{path.stem}_fields.npz"))
        np.savez(path, **extra, **{k: np.array(v) for k, v in self.data.items()})
        return path


def load_results(path) -> dict:
    """The arrays of a saved run, with 'experiment' parsed back into a dict."""
    with np.load(path) as f:
        out = {k: f[k] for k in f.files}
    out["experiment"] = yaml.safe_load(str(out["experiment"]))
    return out
