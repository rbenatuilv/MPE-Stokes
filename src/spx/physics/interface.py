"""Interface coupling on Sigma (theory.md I1-I4). '+' = Stokes side, '-' = poro side."""
import ufl

from spx.dg import tangential


def add_interface(blocks, uS, d, pP, vS, vP, qP, ctx, residuals=None):
    """Add the coupling terms on Sigma (normal stress, BJS, mass) to `blocks`.
    residuals: only for manufactured solutions, from interface_residuals()."""
    m, c = ctx.m, ctx.c
    nS, nP = ctx.n("+"), ctx.n("-")
    dG = m.dS(m.GAMMA)
    B = c.gammaBJS * c.muS / ufl.sqrt(c.K)
    tS = lambda w: tangential(w("+"), nS)        # Stokes-side field, tangential part
    tP = lambda w: tangential(w("-"), nS)        # poro-side field, same tangent plane

    # (I2) normal stress -> both momentum rows
    blocks.add_a("uS", "pP", pP("-") * ufl.inner(vS("+"), nS) * dG)
    blocks.add_a("d", "pP", pP("-") * ufl.inner(vP("-"), nP) * dG)
    # (I3) BJS, u_S part (the d_t d part is added in Step 7)
    blocks.add_a("uS", "uS", B * ufl.inner(tS(uS), tS(vS)) * dG)
    blocks.add_a("d", "uS", -B * ufl.inner(tS(uS), tP(vP)) * dG)
    # (I1) mass: the flux leaving Stokes enters the Darcy equation
    blocks.add_a("pP", "uS", -qP("-") * ufl.inner(uS("+"), nS) * dG)

    if residuals is not None:                    # manufactured solutions only
        r = residuals
        blocks.add_L("uS", (-r["r_n"] * ufl.inner(vS("+"), nS)
                            - ufl.inner(r["r_tau"], tS(vS))) * dG)
        blocks.add_L("d", (-r["r_n"] * ufl.inner(vP("-"), nP)
                           + ufl.inner(r["r_tau"], tP(vP))
                           + ufl.inner(r["r_b"], vP("-"))) * dG)
        blocks.add_L("pP", -r["r_m"] * qP("-") * dG)
