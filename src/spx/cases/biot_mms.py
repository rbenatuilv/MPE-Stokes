"""Milestone B: steady Biot on Omega_P with a manufactured solution."""
import sys

import multiphenicsx.fem.petsc
import ufl

from spx.blocks import BlockSystem
from spx.cases.stokes_mms import MUMPS
from spx.config import load
from spx.context import make_context
from spx.errors import convergence_table, norm_L2, seminorm_H1
from spx.manufactured import biot_mms
from spx.measures import build_measures
from spx.mesh import build_box_geometry
from spx.physics.biot import add_biot
from spx.spaces import build_restrictions, build_spaces


def solve(N, cfg) -> dict:
    """Solve steady Biot on Omega_P on the N-mesh and return its size h and the errors."""
    k, kp = cfg.discretization.deg, cfg.discretization.degP
    geo = build_box_geometry(N, cfg.mesh)
    m = build_measures(geo, cfg.mesh, qdeg=2 * max(k, kp) + 1)
    ctx = make_context(geo.mesh, m, cfg)
    V = build_spaces(geo.mesh, k, kp)
    R = build_restrictions(V, geo.cell_marker, m.STOKES, m.POROEL)

    d, p = ufl.TrialFunction(V["d"]), ufl.TrialFunction(V["pP"])
    v, q = ufl.TestFunction(V["d"]), ufl.TestFunction(V["pP"])
    ex = biot_mms(geo.mesh, ctx.c)
    data = {"f": ex["f"], "g": ex["g"], "d": ex["d"], "p": ex["p"],
            "traction": ex["sig_tot"] * ctx.n, "flux": ufl.inner(ex["darcy_flux"], ctx.n)}

    blocks = BlockSystem(geo.mesh, ["d", "pP"], {"d": v, "pP": q},
                         {"d": m.dx(m.POROEL), "pP": m.dx(m.POROEL)})
    add_biot(blocks, d, p, v, q, ctx, data)
    # Sigma is an outer boundary of Omega_P in this single-physics test: give it the exact
    # total traction and Darcy flux (poro side = '-'), otherwise it is a silent zero-Neumann BC
    nP, dG = ctx.n("-"), m.dS(m.GAMMA)
    blocks.add_L("d", ufl.inner(ex["sig_tot"]("-") * nP, v("-")) * dG)
    blocks.add_L("pP", ufl.inner(ex["darcy_flux"]("-"), nP) * q("-") * dG)
    a, L = blocks.finalize()

    problem = multiphenicsx.fem.petsc.LinearProblem(
        a, L, bcs=[], kind="mpi", restriction=[R["d"], R["pP"]],
        petsc_options_prefix="biot_mms_", petsc_options=MUMPS)
    dh, ph = problem.solve()

    dxe = m.dx(m.POROEL, metadata={"quadrature_degree": k + 4})
    return {"N": N, "h": 1.0 / N,
            "err_d_L2": norm_L2(geo.mesh, dh - ex["d"], dxe),
            "err_d_H1": seminorm_H1(geo.mesh, dh - ex["d"], dxe),
            "err_p_L2": norm_L2(geo.mesh, ph - ex["p"], dxe),
            "err_p_H1": seminorm_H1(geo.mesh, ph - ex["p"], dxe)}


if __name__ == "__main__":
    cfg = load(sys.argv[1])
    print(convergence_table([solve(N, cfg) for N in cfg.mesh.refinements]).to_string(float_format="%.3e"))
