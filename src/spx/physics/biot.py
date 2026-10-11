"""Steady two-field Biot on Omega_P, DG-SIPG (theory.md D2, D3, D4). Step 7 adds time terms."""
import ufl

from spx.dg import h_facet, sym_bnd, sym_jump


def b_P(pp, vv, ctx):
    """Pressure-displacement coupling (D4) on Omega_P. Used as +alpha*b_P(p, v) in the
    momentum row and (Step 7) as -alpha*b_P(q, z) in the pressure row: ONE definition."""
    m, n = ctx.m, ctx.n
    form = -pp * ufl.div(vv) * m.dx(m.POROEL) + ufl.avg(pp) * ufl.jump(vv, n) * m.dS(m.INTERIOR_P)
    if m.role("dirP") is not None:          # displacement Dirichlet set -> dirP, never dirE
        form += pp * ufl.inner(vv, n) * m.role("dirP")
    return form


def add_biot(blocks, d, p, v, q, ctx, data):
    """Add the Biot blocks (d-d, d-pP, pP-pP) and right-hand sides to `blocks`.
    data: f, g (sources), d / p (Dirichlet values on dirP / dirE),
    traction (total traction on neuP), flux (Darcy flux on neuE)."""
    m, n, k, c = ctx.m, ctx.n, ctx.k, ctx.c
    dx, dI = m.dx(m.POROEL), m.dS(m.INTERIOR_P)
    dDd, dNd = m.role("dirP"), m.role("neuP")     # displacement BC sets
    dDp, dNp = m.role("dirE"), m.role("neuE")     # pressure BC sets
    hF, h = h_facet(ctx.mesh), ufl.CellDiameter(ctx.mesh)
    kappa = c.K / c.muF                              # Darcy mobility
    eps = lambda w: ufl.sym(ufl.grad(w))
    sig = lambda w: 2 * c.muP * eps(w) + c.lambdaP * ufl.div(w) * ufl.Identity(len(w))
    pen_d = c.eta * (c.lambdaP + 2 * c.muP) * k**2
    pen_p = c.zeta * c.Kpen / c.muF * k**2           # Kpen = max(K, 1), theory.md D3

    # elasticity (D2)
    a_dd = (ufl.inner(sig(d), eps(v)) * dx
            - ufl.inner(ufl.avg(sig(d)), sym_jump(v, n)) * dI
            - ufl.inner(ufl.avg(sig(v)), sym_jump(d, n)) * dI
            + pen_d / hF * ufl.inner(sym_jump(d, n), sym_jump(v, n)) * dI)
    L_d = ufl.inner(data["f"], v) * dx
    if dDd is not None:                              # weak d = g_d on dirP
        gd = data["d"]
        a_dd += (- ufl.inner(sig(d), sym_bnd(v, n)) * dDd
                 - ufl.inner(sig(v), sym_bnd(d, n)) * dDd
                 + pen_d / h * ufl.inner(sym_bnd(d, n), sym_bnd(v, n)) * dDd)
        L_d += (- ufl.inner(sig(v), sym_bnd(gd, n)) * dDd
                + pen_d / h * ufl.inner(sym_bnd(gd, n), sym_bnd(v, n)) * dDd)
    if dNd is not None:                              # total traction on neuP
        L_d += ufl.inner(data["traction"], v) * dNd

    # Darcy (D3)
    a_pp = (kappa * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
            - kappa * ufl.inner(ufl.avg(ufl.grad(p)), ufl.jump(q, n)) * dI
            - kappa * ufl.inner(ufl.avg(ufl.grad(q)), ufl.jump(p, n)) * dI
            + pen_p / hF * ufl.jump(p) * ufl.jump(q) * dI
            + c.beta * p * q * dx)
    L_p = data["g"] * q * dx
    if dDp is not None:                              # weak p = g_p on dirE
        gp = data["p"]
        a_pp += (- kappa * ufl.inner(ufl.grad(p), n) * q * dDp
                 - kappa * ufl.inner(ufl.grad(q), n) * p * dDp
                 + pen_p / h * p * q * dDp)
        L_p += (- kappa * ufl.inner(ufl.grad(q), n) * gp * dDp
                + pen_p / h * gp * q * dDp)
    if dNp is not None:                              # Darcy flux kappa grad p . n on neuE
        L_p += data["flux"] * q * dNp

    blocks.add_a("d", "d", a_dd)
    blocks.add_a("d", "pP", c.alpha * b_P(p, v, ctx))
    blocks.add_a("pP", "pP", a_pp)
    blocks.add_L("d", L_d)
    blocks.add_L("pP", L_p)
