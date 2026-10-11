"""Unsteady coupled problem. Usage: python -m spx.cases.coupled_time CONFIG space|time"""
import dataclasses
import sys

import dolfinx
import ufl
from petsc4py import PETSc

from spx.blocks import BlockSystem
from spx.cases.coupled_mms import setup
from spx.config import load
from spx.errors import convergence_table, norm_L2
from spx.linalg import BlockSolver
from spx.manufactured import coupled_time_mms, interface_residuals
from spx.newmark import TimeState
from spx.physics.biot import add_biot
from spx.physics.interface import add_interface
from spx.physics.stokes import add_stokes
from spx.spaces import FIELDS
from spx.utils import interpolate


def run(N, cfg, linear_in_space=False) -> dict:
    """Run the unsteady coupled problem from t0 to T on the N-mesh, starting from the exact
    fields, and return the errors of all four fields at the final time."""
    geo, m, ctx, V, R, tr, te, zero_dx = setup(N, cfg)
    mesh, c, n, tc = geo.mesh, ctx.c, ctx.n, cfg.time
    t = dolfinx.fem.Constant(mesh, PETSc.ScalarType(tc.t0))
    exS, exP = coupled_time_mms(mesh, c, t, linear_in_space)

    ts = TimeState(V, mesh, tc)                      # initial state = exact fields at t0
    for f, expr in ((ts.uS_old, exS["u"]), (ts.pP_old, exP["p"]), (ts.d_old, exP["d"]),
                    (ts.z_old, exP["d_t"]), (ts.a_old, exP["d_tt"])):
        interpolate(f, expr)

    data_S = {"f": exS["f"], "u": exS["u"], "traction": exS["sigma"] * n}
    data_P = {"f": exP["f"], "g": exP["g"], "d": exP["d"], "p": exP["p"], "d_t": exP["d_t"],
              "traction": exP["sig_tot"] * n, "flux": ufl.inner(exP["darcy_flux"], n)}
    res = interface_residuals(exS, exP, c, n("+"), exP["d_t"])

    blocks = BlockSystem(mesh, FIELDS, te, zero_dx)
    add_stokes(blocks, tr["uS"], tr["pS"], te["uS"], te["pS"], ctx, data_S, ts)
    add_biot(blocks, tr["d"], tr["pP"], te["d"], te["pP"], ctx, data_P, ts)
    add_interface(blocks, tr["uS"], tr["d"], tr["pP"], te["uS"], te["d"], te["pP"], ctx, res, ts)
    a, L = blocks.finalize()

    sol = {f: dolfinx.fem.Function(V[f]) for f in FIELDS}
    solver = BlockSolver(a, L, [R[f] for f in FIELDS])        # A assembled + factorized once
    nsteps = round((tc.T - tc.t0) / tc.dt)
    assert abs(nsteps * tc.dt - (tc.T - tc.t0)) < 1e-12, "T - t0 must be a multiple of dt"
    for step in range(1, nsteps + 1):
        t.value = tc.t0 + step * tc.dt                        # sources and BCs follow t
        solver.solve_into([sol[f] for f in FIELDS])
        ts.advance(sol)

    q = {"quadrature_degree": ctx.k + 4}
    dxS, dxP = m.dx(m.STOKES, metadata=q), m.dx(m.POROEL, metadata=q)
    return {"N": N, "dt": tc.dt,
            "err_uS_L2": norm_L2(mesh, sol["uS"] - exS["u"], dxS),
            "err_pS_L2": norm_L2(mesh, sol["pS"] - exS["p"], dxS),
            "err_d_L2": norm_L2(mesh, sol["d"] - exP["d"], dxP),
            "err_pP_L2": norm_L2(mesh, sol["pP"] - exP["p"], dxP)}


def space_study(cfg):
    """Refine the mesh at fixed dt: the rates are the spatial rates."""
    rows = [run(N, cfg) for N in cfg.mesh.refinements]
    for r in rows:
        r["h"] = 1.0 / r["N"]
    return convergence_table(rows)


def time_study(cfg, N=4, levels=4, T=1.0, dt0=0.1):
    """Linear-in-space exact fields: the spatial error is zero, so the rate is the time rate.
    (With the trigonometric fields, the spatial error hides the time error.)"""
    rows = []
    for i in range(levels):
        dt = dt0 / 2**i
        r = run(N, dataclasses.replace(cfg, time=dataclasses.replace(cfg.time, T=T, dt=dt)),
                linear_in_space=True)
        r["h"] = dt                                   # the rate column is then the rate in dt
        rows.append(r)
    return convergence_table(rows)


if __name__ == "__main__":
    cfg = load(sys.argv[1])
    study = space_study if sys.argv[2] == "space" else time_study
    print(study(cfg).to_string(float_format="%.3e"))
