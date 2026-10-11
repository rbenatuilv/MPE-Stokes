"""Steady Stokes on Omega_S, DG-SIPG (theory.md D1, D4, D5). Step 7 adds the mass term."""
import ufl

from spx.dg import h_facet, h_min, sym_bnd, sym_jump


def add_stokes(blocks, u, p, v, q, ctx, data):
    """Add the Stokes blocks (uS-uS, uS-pS, pS-uS, pS-pS) and right-hand sides to `blocks`.
    data: f (source), u (Dirichlet value on dirS), traction (on neuS)."""
    m, n, k, kp, c = ctx.m, ctx.n, ctx.k, ctx.kp, ctx.c
    dx, dI = m.dx(m.STOKES), m.dS(m.INTERIOR_S)
    dD, dN = m.role("dirS"), m.role("neuS")
    hF, hS, h = h_facet(ctx.mesh), h_min(ctx.mesh), ufl.CellDiameter(ctx.mesh)
    eps = lambda w: ufl.sym(ufl.grad(w))
    pen = c.gammaV * c.muS * k**2

    # viscous term a_S(u, v), interior part (D1)
    a = (2 * c.muS * ufl.inner(eps(u), eps(v)) * dx
         - 2 * c.muS * ufl.inner(ufl.avg(eps(u)), sym_jump(v, n)) * dI
         - 2 * c.muS * ufl.inner(ufl.avg(eps(v)), sym_jump(u, n)) * dI
         + pen / hF * ufl.inner(sym_jump(u, n), sym_jump(v, n)) * dI)

    # pressure-velocity coupling b(p, v) (D4): ONE definition, used for B and -B^T
    def b(pp, vv):
        """Pressure-velocity coupling b(pp, vv) in DG form (D4)."""
        form = -pp * ufl.div(vv) * dx + ufl.avg(pp) * ufl.jump(vv, n) * dI  # type:ignore
        if dD is not None:
            form += pp * ufl.inner(vv, n) * dD
        return form

    L_u = ufl.inner(data["f"], v) * dx
    if dD is not None:                      # weak Dirichlet u = g on dirS (Nitsche)
        g = data["u"]
        a += (- 2 * c.muS * ufl.inner(eps(u), sym_bnd(v, n)) * dD
              - 2 * c.muS * ufl.inner(eps(v), sym_bnd(u, n)) * dD
              + pen / h * ufl.inner(sym_bnd(u, n), sym_bnd(v, n)) * dD)
        L_u += (- 2 * c.muS * ufl.inner(eps(v), sym_bnd(g, n)) * dD
                + pen / h * ufl.inner(sym_bnd(g, n), sym_bnd(v, n)) * dD)
        blocks.add_L("pS", -q * ufl.inner(g, n) * dD)
    if dN is not None:                      # traction sigma_S n = data["traction"]
        L_u += ufl.inner(data["traction"], v) * dN

    blocks.add_a("uS", "uS", a)
    blocks.add_a("uS", "pS", b(p, v))
    blocks.add_a("pS", "uS", -b(q, u))
    blocks.add_a("pS", "pS",                # pressure-jump stabilization (D5)
                 c.gammaP * hS / kp * ufl.inner(ufl.jump(p, n), ufl.jump(q, n)) * dI)
    blocks.add_L("uS", L_u)
