"""Milestone C: steady coupled Stokes-Biot with a manufactured solution."""
import sys

import multiphenicsx.fem.petsc
import ufl

from spx.blocks import BlockSystem
from spx.cases.stokes_mms import MUMPS
from spx.config import load
from spx.context import make_context
from spx.errors import convergence_table, norm_L2, seminorm_H1
from spx.manufactured import biot_mms, interface_residuals, stokes_mms
from spx.measures import build_measures
from spx.mesh import build_box_geometry
from spx.physics.biot import add_biot
from spx.physics.interface import add_interface
from spx.physics.stokes import add_stokes
from spx.spaces import FIELDS, build_restrictions, build_spaces
from spx.utils import integrate


def setup(N, cfg):
    """Everything that depends only on the mesh; reused by Step 7 and the tests."""
    k, kp = cfg.discretization.deg, cfg.discretization.degP
    geo = build_box_geometry(N, cfg.mesh)
    m = build_measures(geo, cfg.mesh, qdeg=2 * max(k, kp) + 1)
    ctx = make_context(geo.mesh, m, cfg)
    V = build_spaces(geo.mesh, k, kp)
    R = build_restrictions(V, geo.cell_marker, m.STOKES, m.POROEL)
    tr = {f: ufl.TrialFunction(V[f]) for f in FIELDS}
    te = {f: ufl.TestFunction(V[f]) for f in FIELDS}
    zero_dx = {"uS": m.dx(m.STOKES), "pS": m.dx(m.STOKES), "d": m.dx(m.POROEL), "pP": m.dx(m.POROEL)}
    return geo, m, ctx, V, R, tr, te, zero_dx


def solve(N, cfg) -> dict:
    """Solve the steady coupled problem on the N-mesh and return its size h, the errors of
    all four fields and the interface mass-conservation error."""
    geo, m, ctx, V, R, tr, te, zero_dx = setup(N, cfg)
    mesh, c, n = geo.mesh, ctx.c, ctx.n
    exS, exP = stokes_mms(mesh, c.muS), biot_mms(mesh, c)
    zero_vec = ufl.as_vector([0] * mesh.geometry.dim)               # d_t d = 0 (steady)
    data_S = {"f": exS["f"], "u": exS["u"], "traction": exS["sigma"] * n}
    data_P = {"f": exP["f"], "g": exP["g"], "d": exP["d"], "p": exP["p"],
              "traction": exP["sig_tot"] * n, "flux": ufl.inner(exP["darcy_flux"], n)}
    res = interface_residuals(exS, exP, c, n("+"), zero_vec)

    blocks = BlockSystem(mesh, FIELDS, te, zero_dx)
    add_stokes(blocks, tr["uS"], tr["pS"], te["uS"], te["pS"], ctx, data_S)
    add_biot(blocks, tr["d"], tr["pP"], te["d"], te["pP"], ctx, data_P)
    add_interface(blocks, tr["uS"], tr["d"], tr["pP"], te["uS"], te["d"], te["pP"], ctx, res)
    a, L = blocks.finalize()

    problem = multiphenicsx.fem.petsc.LinearProblem(
        a, L, bcs=[], kind="mpi", restriction=[R[f] for f in FIELDS],
        petsc_options_prefix="coupled_mms_", petsc_options=MUMPS)
    uSh, pSh, dh, pPh = problem.solve()

    q = {"quadrature_degree": ctx.k + 4}
    dxS, dxP = m.dx(m.STOKES, metadata=q), m.dx(m.POROEL, metadata=q)
    nS, nP = n("+"), n("-")
    kappa = c.K / c.muF
    mass = ufl.inner(uSh("+"), nS) - ufl.inner(kappa * ufl.grad(pPh)("-"), nP)   # discrete (I1), steady
    return {"N": N, "h": 1.0 / N,
            "err_uS_L2": norm_L2(mesh, uSh - exS["u"], dxS),
            "err_uS_H1": seminorm_H1(mesh, uSh - exS["u"], dxS),
            "err_pS_L2": norm_L2(mesh, pSh - exS["p"], dxS),
            "err_d_L2": norm_L2(mesh, dh - exP["d"], dxP),
            "err_d_H1": seminorm_H1(mesh, dh - exP["d"], dxP),
            "err_pP_L2": norm_L2(mesh, pPh - exP["p"], dxP),
            "err_pP_H1": seminorm_H1(mesh, pPh - exP["p"], dxP),
            "err_mass": float(integrate(mesh, (mass - res["r_m"])**2 * m.dS(m.GAMMA))**0.5)}


if __name__ == "__main__":
    cfg = load(sys.argv[1])
    print(convergence_table([solve(N, cfg) for N in cfg.mesh.refinements]).to_string(float_format="%.3e"))
