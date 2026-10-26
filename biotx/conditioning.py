"""Condition number of the system matrix, cond_2(A) = s_max / s_min, and how it changes with
one parameter of the experiment.

    python -m biotx.conditioning experiments/conditioning/annulus_biot2_h.yaml [--set k=v]

A study is an experiment YAML (usually a `base:` one) with a `conditioning:` section:

    base: ../annulus_biot2.yaml
    conditioning:
      sweep: {geometry.h: [4.0e-3, 2.0e-3, 1.0e-3]}   # ONE dotted key of the YAML and its values
      scaling: [raw, jacobi, schur]   # raw: A;  jacobi: |D|^-1/2 A |D|^-1/2, D = diag(A);
                                      # schur: as jacobi, but p scaled by its Schur diagonal
      dense_check: false              # also the dense numpy SVD (small systems only)

The matrix is the one of a regular time step (biot2 with a Rannacher start: the matrix of the
requested theta, not of the start-up steps). A is not symmetric (the couplings are
antisymmetric), so its eigenvalues are not its singular values, and both ends of the singular
spectrum are computed with SLEPc:

    s_max   largest singular value of A (SVD, Lanczos)
    s_min   1 / sqrt(lambda_max) of (A^T A)^{-1} = A^{-1} A^{-T}: a Hermitian eigenproblem
            whose operator costs one solve with A^T and one with A per iteration, both from one
            MUMPS LU factorisation (raw: the model's own; jacobi, schur: one more)

The singular vectors tell which unknowns carry each end of the spectrum: `share` is, for each
field (d, w, p), the fraction of |v|^2 of the right singular vector v (A v = s u) on its dofs.
For jacobi and schur the vector is in the scaled unknowns |D|^1/2 x.

Why schur. The pressure is often barely controlled by its own diagonal (storage c0, tiny) and
mostly through its couplings to d and w: eliminating them gives the Schur complement
S = C + B diag(A)^-1 B^T for p. `schur` scales every field as jacobi does, except p, which gets
    s_ii = |a_ii| + sum_{j not p} |a_ij a_ji| / |a_jj|
the diagonal of that approximate Schur complement. This is the scaling a block preconditioner
would use; jacobi under-scales p in biot3 (see docs/theory.md, Section 10).

    from biotx import conditioning as cn
    c = cn.condition(biotx.make_model(exp), "jacobi")      # Condition(smax, smin, cond, ...)
    res = cn.run(cn.load("study.yaml")); print(res.table()); res.save()
"""
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import dolfinx
import numpy as np
import ufl
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from biotx import geometry
from biotx.common import make_ksp
from biotx.config import _override, from_dict, merge, read_study

SCALINGS = ("raw", "jacobi", "schur")
KEYS = {"sweep", "scaling", "dense_check"}
DENSE_MAX = 8000                 # dofs: the dense SVD needs ~ 3 n^2 doubles
TOL = 1e-8                       # relative tolerance of the singular values


# ---- one matrix ----------------------------------------------------------------------------
@dataclass
class Condition:
    """cond_2 of one matrix. share_min / share_max: field -> fraction of the singular vector."""
    smax: float
    smin: float
    share_max: dict
    share_min: dict
    its: tuple                   # SLEPc iterations (s_max, s_min)
    seconds: float
    dense: float | None = None   # numpy cond_2, if checked

    @property
    def cond(self):
        return self.smax / self.smin


def step_matrix(model):
    """The matrix of a regular step. In biot2 with a Rannacher start the model is built with
    theta = 1; this switches it to the requested theta (and refactorises at the next solve)."""
    target = getattr(model, "theta_target", None)
    if target is not None and model.theta.value != target:
        model.theta.value = target
        model.solver.update_matrix()
    return model.solver.A


def _check_direct(ksp):
    if ksp.getType() != "preonly" or ksp.getPC().getType() != "lu":
        raise ValueError("conditioning needs the direct LU solver (solver: options for MUMPS only)")


class _InverseNormal:
    """x -> A^{-1} A^{-T} x, with an LU factorisation of A (petsc4py Python matrix context)."""

    def __init__(self, ksp, A):
        self.ksp, self.tmp = ksp, A.createVecLeft()

    def mult(self, mat, x, y):
        self.ksp.solveTranspose(x, self.tmp)
        self.ksp.solve(self.tmp, y)


def _shares(model, vec):
    """Fraction of |vec|^2 on the dofs of each field (owned dofs, summed over the ranks)."""
    x, comm = vec.getArray(readonly=True), model.mesh.comm
    out = {f: float(np.sum(x[idx[idx < len(x)]] ** 2)) for f, idx in model._dofs.items()}
    out = {f: comm.allreduce(v, op=MPI.SUM) for f, v in out.items()}
    total = sum(out.values())
    return {f: v / total for f, v in out.items()}


def _largest_singular(A):
    svd = SLEPc.SVD().create(A.comm)
    svd.setOperators(A)
    svd.setDimensions(1)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setTolerances(TOL, 1000)
    svd.solve()
    if svd.getConverged() < 1:
        raise RuntimeError("SLEPc: the largest singular value did not converge")
    u, v = A.createVecLeft(), A.createVecRight()
    s = svd.getSingularTriplet(0, u, v)
    its = svd.getIterationNumber()
    svd.destroy()
    return s, v, its


def _smallest_singular(A, ksp):
    S = PETSc.Mat().createPython(A.getSizes(), _InverseNormal(ksp, A), comm=A.comm)
    S.setUp()
    eps = SLEPc.EPS().create(A.comm)
    eps.setOperators(S)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setDimensions(1)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setTolerances(TOL, 1000)
    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError("SLEPc: the smallest singular value did not converge")
    v = A.createVecRight()
    lam = eps.getEigenpair(0, v).real
    its = eps.getIterationNumber()
    eps.destroy()
    S.destroy()
    return 1.0 / np.sqrt(lam), v, its


def _scaled(A, diag):
    """S A S with S = |diag|^{-1/2} (a new matrix); diag: the owned entries."""
    zero = A.comm.tompi4py().allreduce(int(np.sum(diag == 0)), op=MPI.SUM)
    if zero:
        raise ValueError(f"scaling: {zero} zero diagonal entries (e.g. biot3 with c0 = 0)")
    d = A.createVecLeft()
    d.setArray(1.0 / np.sqrt(np.abs(diag)))
    B = A.copy()
    B.diagonalScale(d, d)
    return B


def jacobi(A):
    """|D|^{-1/2} A |D|^{-1/2}, D = diag(A) (a new matrix)."""
    return _scaled(A, A.getDiagonal().getArray().copy())


def schur_diagonal(A, p_dofs):
    """diag(A), with the pressure entries replaced by the diagonal of the approximate Schur
    complement |a_ii| + sum_{j not p} |a_ij a_ji| / |a_jj| (serial only)."""
    if A.comm.getSize() > 1:
        raise NotImplementedError("schur scaling: serial runs only")
    diag = A.getDiagonal().getArray().copy()
    is_p = np.zeros(len(diag), dtype=bool)
    is_p[p_dofs] = True
    At = A.transpose(PETSc.Mat())                          # rows of A^T = columns of A
    out = np.abs(diag)
    for i in np.flatnonzero(is_p):
        cols, row = A.getRow(i)
        tcols, col = At.getRow(i)
        a_ji = dict(zip(tcols, col))
        keep = ~is_p[cols]
        out[i] += sum(abs(r * a_ji.get(j, 0.0)) / abs(diag[j]) for j, r in zip(cols[keep], row[keep]))
    At.destroy()
    return out


def dense_cond(A):
    """cond_2 from the dense SVD (numpy, serial, at most DENSE_MAX dofs)."""
    n = A.getSize()[0]
    if A.comm.getSize() > 1 or n > DENSE_MAX:
        raise ValueError(f"dense check: serial runs with at most {DENSE_MAX} dofs only (got {n})")
    s = np.linalg.svd(A.copy().convert("dense").getDenseArray(), compute_uv=False)   # convert: in place
    return float(s[0] / s[-1])


def condition(model, scaling="raw", dense_check=False) -> Condition:
    """cond_2 of the step matrix of the model, raw or Jacobi-scaled."""
    if scaling not in SCALINGS:
        raise ValueError(f"scaling: one of {SCALINGS}")
    _check_direct(model.solver.ksp)
    t0 = time.perf_counter()
    A = step_matrix(model)
    if scaling == "raw":
        M, ksp = A, model.solver.ksp
    else:
        if scaling == "jacobi":
            M = jacobi(A)
        else:
            nown = A.getLocalSize()[0]
            M = _scaled(A, schur_diagonal(A, model._dofs["p"][model._dofs["p"] < nown]))
        ksp = make_ksp(M, model.exp.solver)
    smax, vmax, it_max = _largest_singular(M)
    smin, vmin, it_min = _smallest_singular(M, ksp)
    out = Condition(smax, smin, _shares(model, vmax), _shares(model, vmin), (it_max, it_min),
                    time.perf_counter() - t0)
    if dense_check:
        out.dense = dense_cond(M)
    if scaling != "raw":
        ksp.destroy()
        M.destroy()
    return out


# ---- studies -------------------------------------------------------------------------------
@dataclass
class Study:
    """An experiment tree plus its validated `conditioning` section."""
    raw: dict
    name: str
    root: Path = Path(".")
    spec: dict = field(default_factory=dict)

    @property
    def key(self):
        return next(iter(self.spec["sweep"]))

    @property
    def values(self):
        return list(self.spec["sweep"][self.key])

    def experiment(self, value):
        return from_dict(_override(self.raw, {self.key: value}), self.name, self.root)


def _check(raw):
    spec = dict(raw.get("conditioning") or {})
    unknown = set(spec) - KEYS
    if unknown:
        raise ValueError(f"conditioning: unknown key(s) {sorted(unknown)}; allowed {sorted(KEYS)}")
    sweep = spec.get("sweep")
    if not isinstance(sweep, dict) or len(sweep) != 1:
        raise ValueError("conditioning.sweep: exactly one dotted key and its list of values")
    values = next(iter(sweep.values()))
    if not isinstance(values, (list, tuple)) or len(values) < 1:
        raise ValueError("conditioning.sweep: the key needs a list of values")
    spec.setdefault("scaling", list(SCALINGS))
    if isinstance(spec["scaling"], str):
        spec["scaling"] = [spec["scaling"]]
    if set(spec["scaling"]) - set(SCALINGS):
        raise ValueError(f"conditioning.scaling: one or more of {SCALINGS}")
    spec.setdefault("dense_check", False)
    return spec


def load(path, overrides=None) -> Study:
    """Read a study YAML (with an optional `base:` experiment it extends)."""
    path = Path(path)
    raw = _override(read_study(path), overrides or {})
    return Study(raw, raw.get("output", {}).get("name") or path.stem, path.parent, _check(raw))


def study(exp, **spec) -> Study:
    """A Study from an Experiment and conditioning keys, e.g. study(exp, sweep={"time.dt": [...]})."""
    raw = merge(exp.raw, {"conditioning": spec})
    return Study(raw, exp.name, exp.root, _check(raw))


@dataclass
class Result:
    """Condition numbers of one sweep: conds[scaling] is a list with one Condition per value."""
    name: str
    model: str
    key: str
    values: list
    h: np.ndarray                # largest cell diameter [m] per value
    h_eff: np.ndarray            # (|Omega| / ncells)^(1/dim) [m]: follows the refinement
    dofs: np.ndarray
    conds: dict
    raw: dict
    root: Path = Path(".")

    @property
    def scale(self):
        """What the slope is measured against: h_eff for a mesh sweep (the largest diameter of an
        unstructured mesh hardly follows the refinement), else the swept values."""
        if self.key in ("geometry.h", "geometry.n"):
            return self.h_eff
        try:
            return np.asarray(self.values, dtype=float)
        except (TypeError, ValueError):
            return None                                  # e.g. a sweep over `scales`

    def cond(self, scaling):
        return np.array([c.cond for c in self.conds[scaling]])

    def slopes(self, scaling):
        """log(cond_{i+1} / cond_i) / log(s_{i+1} / s_i): cond ~ s^slope (h_eff: about -2 expected)."""
        c, s = self.cond(scaling), self.scale
        if s is None or len(c) < 2:
            return np.full(max(len(c) - 1, 0), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(c[1:] / c[:-1]) / np.log(s[1:] / s[:-1])

    def table(self):
        fields = list(self.conds[next(iter(self.conds))][0].share_min)
        head = [self.key, "h max", "h eff", "dofs"]
        for sc in self.conds:
            head += [f"cond {sc}", "slope", f"s_min share ({' '.join(fields)})"]
            if self.conds[sc][0].dense is not None:
                head += ["numpy"]
        rows = [head]
        for i, v in enumerate(self.values):
            row = [str(v), f"{self.h[i]:.3e}", f"{self.h_eff[i]:.3e}", str(int(self.dofs[i]))]
            for sc, cs in self.conds.items():
                sl = self.slopes(sc)
                c = cs[i]
                row += [f"{c.cond:.3e}", f"{sl[i - 1]:.2f}" if i > 0 else "",
                        " ".join(f"{c.share_min[f]:.2f}" for f in fields)]
                if c.dense is not None:
                    row += [f"{c.dense:.3e}"]
            rows.append(row)
        w = [max(len(r[j]) for r in rows) for j in range(len(head))]
        lines = [f"{self.name}: {self.model}, condition number cond_2 of the step matrix"]
        lines += ["  " + "  ".join(c.rjust(wj) for c, wj in zip(r, w)) for r in rows]
        return "\n".join(lines)

    def save(self, path=None):
        """Write the singular values, conds, shares and the study YAML to <root>/<output.dir>/<name>.npz."""
        if MPI.COMM_WORLD.rank != 0:
            return None
        out = self.raw.get("output", {}).get("dir", "results")
        path = Path(path) if path else Path(self.root) / out / f"{self.name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"key": self.key, "values": np.array(self.values, dtype=object), "h": self.h,
                "h_eff": self.h_eff, "dofs": self.dofs, "experiment": yaml.safe_dump(self.raw, sort_keys=False)}
        for sc, cs in self.conds.items():
            data[f"{sc}_smax"] = np.array([c.smax for c in cs])
            data[f"{sc}_smin"] = np.array([c.smin for c in cs])
            data[f"{sc}_cond"] = self.cond(sc)
            data[f"{sc}_slope"] = self.slopes(sc)
            data[f"{sc}_seconds"] = np.array([c.seconds for c in cs])
            for f in cs[0].share_min:
                data[f"{sc}_share_min_{f}"] = np.array([c.share_min[f] for c in cs])
                data[f"{sc}_share_max_{f}"] = np.array([c.share_max[f] for c in cs])
            if cs[0].dense is not None:
                data[f"{sc}_dense"] = np.array([c.dense for c in cs])
        np.savez(path, **data)
        return path


def load_result(path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files}
    out["experiment"] = yaml.safe_load(str(out["experiment"]))
    return out


def _mean_size(mesh):
    """(|Omega| / ncells)^(1/dim), in the mesh's own units."""
    tdim = mesh.topology.dim
    one = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx(mesh))
    vol = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(one), op=MPI.SUM)
    return (vol / mesh.topology.index_map(tdim).size_global) ** (1 / tdim)


def run(st: Study, verbose=False) -> Result:
    """Build the model for every value of the sweep and measure its condition numbers."""
    from biotx import make_model
    conds, h, h_eff, dofs = {sc: [] for sc in st.spec["scaling"]}, [], [], []
    for v in st.values:
        exp = st.experiment(v)
        model = make_model(exp)
        for sc in st.spec["scaling"]:
            conds[sc].append(condition(model, sc, st.spec["dense_check"]))
            model.solver.ksp.reset()                     # free the LU: one factorisation at a time
            model.solver.ksp.setOperators(model.solver.A)
            if verbose and MPI.COMM_WORLD.rank == 0:
                c = conds[sc][-1]
                print(f"  {st.key} = {v}: {sc} cond {c.cond:.3e} ({c.seconds:.1f} s)", flush=True)
        h.append(geometry.cell_size(model.mesh) * exp.scales.L0)
        h_eff.append(_mean_size(model.mesh) * exp.scales.L0)
        dofs.append(model.W.dofmap.index_map.size_global * model.W.dofmap.index_map_bs)
    return Result(st.name, exp.model, st.key, st.values, np.array(h), np.array(h_eff),
                  np.array(dofs), conds, st.raw, st.root)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Condition numbers of the step matrix, from a YAML file")
    ap.add_argument("yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    res = run(load(args.yaml, overrides), verbose=True)
    path = res.save()
    if MPI.COMM_WORLD.rank == 0:
        print(res.table())
        print(f"saved {path}")
    return res


if __name__ == "__main__":
    main()
