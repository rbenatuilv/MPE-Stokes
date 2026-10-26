"""Compare finished runs with a reference run.

    python -m biotx.compare results/ref.npz results/run_a.npz results/run_b.npz [--t-min 2]
    mpirun -n 4 python -m biotx.compare ...          # parallel works too

Every run must have been saved with `output.fields` (its <name>_fields.npz next to <name>.npz).

Fields, at the snapshot times common to both runs:

    e_q(t) = ||q_run - q_ref||_L2 / ||q_ref||_L2        (integrals over the reference mesh)

Both fields are interpolated into DG of degree max(run, ref) on the reference mesh, with its
interpolation points at Gauss-Legendre points, strictly inside the cells: a point on a facet
would take the value of an arbitrary side of the jump. The reference field is exact there,
and so is the run's when the meshes match. The meshes need not match (other h, another gmsh
mesh, another model): the run's field is then sampled at those points, so its jumps add an
error of the size of the jumps themselves, which is the size of its discretisation error.
Near a curved boundary the two polygonal boundaries differ slightly, so the reference points
just outside the run's mesh are evaluated in the nearest cell (padding of one cell size).

Boundary fluxes (from the .npz, every step):

    e_Q = max_t |Q_run - Q_ref| / max_t |Q_ref|         (run interpolated in time, t >= t_min)
"""
import argparse
from pathlib import Path

import basix
import basix.ufl
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from biotx.fields import Fields
from biotx.geometry import cell_size
from biotx.post import load_results


def fields_path(path):
    """<name>.npz -> <name>_fields.npz (a _fields.npz is returned as it is)."""
    p = Path(path)
    return p if p.stem.endswith("_fields") else p.with_name(f"{p.stem}_fields.npz")


class FieldDifference:
    """Relative L2 difference of one quantity of `run` with respect to `ref`, per snapshot."""

    def __init__(self, run: Fields, ref: Fields, name):
        self.run, self.ref, self.name = run, ref, name
        Va, Vr = run.space(name), ref.space(name)
        if Va.value_shape != Vr.value_shape:
            raise ValueError(f"'{name}' has shape {Va.value_shape} in the run, {Vr.value_shape} in the reference")
        deg = max(Va.ufl_element().degree, Vr.ufl_element().degree)
        el = basix.ufl.element("DG", ref.mesh.basix_cell(), deg, shape=Vr.value_shape,
                               lagrange_variant=basix.LagrangeVariant.gl_warped)
        Vt = dolfinx.fem.functionspace(ref.mesh, el)
        self.a, self.r = dolfinx.fem.Function(Vt), dolfinx.fem.Function(Vt)
        nloc = ref.mesh.topology.index_map(ref.mesh.topology.dim).size_local
        self.cells = np.arange(nloc, dtype=np.int32)
        self.data = dolfinx.fem.create_interpolation_data(Vt, Va, self.cells, padding=cell_size(run.mesh))
        dx = ufl.dx(ref.mesh)
        self.num = dolfinx.fem.form(ufl.inner(self.a - self.r, self.a - self.r) * dx)
        self.den = dolfinx.fem.form(ufl.inner(self.r, self.r) * dx)

    def __call__(self, i_run, i_ref):
        self.a.interpolate_nonmatching(self.run.function(self.name, i_run), self.cells, self.data)
        self.r.interpolate(self.ref.function(self.name, i_ref))
        comm = self.ref.mesh.comm
        num = comm.allreduce(dolfinx.fem.assemble_scalar(self.num), op=MPI.SUM)
        den = comm.allreduce(dolfinx.fem.assemble_scalar(self.den), op=MPI.SUM)
        return float(np.sqrt(num / den)) if den > 0 else float(np.sqrt(num))


def _dt(experiment):
    return float(experiment["time"]["dt"])


def field_errors(run: Fields, ref: Fields, names=None, t_min=-np.inf) -> dict:
    """{'t': common snapshot times, q: relative L2 errors at those times} for each quantity
    present in both runs (or the given names)."""
    names = names or [n for n in ref.names if n in run.names]
    tol = 0.5 * min(_dt(run.experiment), _dt(ref.experiment))
    pairs = [(i, j) for j, tr in enumerate(ref.snap_t) if tr >= t_min - tol
             for i, ta in enumerate(run.snap_t) if abs(ta - tr) < tol]
    out = {"t": np.array([ref.snap_t[j] for _, j in pairs])}
    for name in names:
        diff = FieldDifference(run, ref, name)
        out[name] = np.array([diff(i, j) for i, j in pairs])
    return out


def flux_errors(run: dict, ref: dict, t_min=-np.inf) -> dict:
    """{boundary: max-in-time relative error} of the boundary fluxes, from load_results()."""
    out = {}
    keep = (ref["t"] >= t_min) & (ref["t"] >= run["t"][0]) & (ref["t"] <= run["t"][-1])
    for key in ref:
        if key.startswith("bflux_") and key in run:
            Qr = ref[key][keep]
            Qa = np.interp(ref["t"][keep], run["t"], run[key])
            scale = np.abs(Qr).max()
            out[key[6:]] = float(np.abs(Qa - Qr).max() / scale) if scale > 0 else float(np.abs(Qa).max())
    return out


def compare(ref, runs, names=None, t_min=-np.inf, comm=MPI.COMM_WORLD) -> dict:
    """{run stem: {'fields': field_errors, 'flux': flux_errors}} for each run path (.npz)."""
    ref_f = Fields(fields_path(ref), comm)
    ref_r = load_results(ref)
    out = {}
    for path in runs:
        res = {"fields": field_errors(Fields(fields_path(path), comm), ref_f, names, t_min),
               "flux": flux_errors(load_results(path), ref_r, t_min)}
        out[Path(path).stem] = res
    return out


def table(errors) -> str:
    """Max over the snapshots of each field error, and the boundary-flux errors."""
    cols = []
    for res in errors.values():
        cols += [k for k in res["fields"] if k != "t" and k not in cols]
    bcols = []
    for res in errors.values():
        bcols += [k for k in res["flux"] if k not in bcols]
    head = ["run"] + cols + [f"Q_{b}" for b in bcols]
    rows = [head]
    for name, res in errors.items():
        f = res["fields"]
        rows.append([name] + [f"{f[c].max():.3e}" if c in f and len(f[c]) else "-" for c in cols]
                    + [f"{res['flux'][b]:.3e}" if b in res["flux"] else "-" for b in bcols])
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(wi) for c, wi in zip(r, w)) for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Relative errors of runs with respect to a reference")
    ap.add_argument("ref", help="reference <name>.npz")
    ap.add_argument("runs", nargs="+", help="runs to compare (<name>.npz)")
    ap.add_argument("--fields", nargs="*", help="quantities (default: all in both)")
    ap.add_argument("--t-min", type=float, default=-np.inf, help="ignore earlier times")
    a = ap.parse_args(argv)
    errors = compare(a.ref, a.runs, a.fields, a.t_min)
    if MPI.COMM_WORLD.rank == 0:
        print(table(errors))
    return errors


if __name__ == "__main__":
    main()
