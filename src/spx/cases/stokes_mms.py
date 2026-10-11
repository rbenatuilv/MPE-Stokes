"""Milestone A: steady Stokes on Omega_S with a manufactured solution."""
import sys

import multiphenicsx.fem.petsc
import ufl

from spx.blocks import BlockSystem
from spx.config import load
from spx.context import make_context
from spx.errors import convergence_table, norm_L2, seminorm_H1
from spx.manufactured import stokes_mms
from spx.measures import build_measures
from spx.mesh import build_box_geometry
from spx.physics.stokes import add_stokes
from spx.spaces import build_restrictions, build_spaces


MUMPS = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
         "ksp_error_if_not_converged": True}


def solve(N, cfg) -> dict:
    """Solve steady Stokes on Omega_S on the N-mesh and return its size h and the errors."""
    k, kp = cfg.discretization.deg, cfg.discretization.degP
    geo = build_box_geometry(N, cfg.mesh)
    m = build_measures(geo, cfg.mesh, qdeg=2 * max(k, kp) + 1)
    ctx = make_context(geo.mesh, m, cfg)
    V = build_spaces(geo.mesh, k, kp)
    R = build_restrictions(V, geo.cell_marker, m.STOKES, m.POROEL)

    u, p = ufl.TrialFunction(V["uS"]), ufl.TrialFunction(V["pS"])
    v, q = ufl.TestFunction(V["uS"]), ufl.TestFunction(V["pS"])
    ex = stokes_mms(geo.mesh, ctx.c.muS)
    data = {"f": ex["f"], "u": ex["u"], "traction": ex["sigma"] * ctx.n}

    blocks = BlockSystem(geo.mesh, ["uS", "pS"], {"uS": v, "pS": q},
                         {"uS": m.dx(m.STOKES), "pS": m.dx(m.STOKES)})
    add_stokes(blocks, u, p, v, q, ctx, data)
    # Sigma is an outer boundary of Omega_S in this single-physics test: exact traction on it
    blocks.add_L("uS", ufl.inner(ex["sigma"]("+") * ctx.n("+"), v("+")) * m.dS(m.GAMMA))
    a, L = blocks.finalize()

    problem = multiphenicsx.fem.petsc.LinearProblem(
        a, L, bcs=[], kind="mpi", restriction=[R["uS"], R["pS"]],
        petsc_options_prefix="stokes_mms_", petsc_options=MUMPS)
    uh, ph = problem.solve()

    dxe = m.dx(m.STOKES, metadata={"quadrature_degree": k + 4})   # richer quadrature for errors
    return {"N": N, "h": 1.0 / N,
            "err_u_L2": norm_L2(geo.mesh, uh - ex["u"], dxe),
            "err_u_H1": seminorm_H1(geo.mesh, uh - ex["u"], dxe),
            "err_p_L2": norm_L2(geo.mesh, ph - ex["p"], dxe)}


if __name__ == "__main__":
    cfg = load(sys.argv[1])
    df = convergence_table([solve(N, cfg) for N in cfg.mesh.refinements])
    print(df.to_string(float_format="%.3e"))
