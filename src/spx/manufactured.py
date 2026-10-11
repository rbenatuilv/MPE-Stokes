"""Exact solutions in UFL. Sources and boundary data are DERIVED, never typed by hand."""
import numpy as np
import ufl

from spx.dg import tangential


def stokes_mms(mesh, muS):
    """Smooth exact Stokes solution (divergence-free u, pressure p) and everything derived
    from it: the stress sigma and the source f = -div(sigma)."""
    x = ufl.SpatialCoordinate(mesh)
    dim = mesh.geometry.dim
    pi = np.pi
    if dim == 2:
        psi = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])          # stream function
        u = ufl.as_vector((psi.dx(1), -psi.dx(0)))             # div u = 0 exactly
    else:
        # every component nonzero: UFL cannot differentiate a literal 0 component
        A = ufl.as_vector((ufl.sin(pi * x[1]) * ufl.cos(pi * x[2]),
                           ufl.sin(pi * x[2]) * ufl.cos(pi * x[0]),
                           ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])))
        u = ufl.curl(A)                                         # div curl = 0
    p = ufl.cos(pi * x[0]) * ufl.sin(pi * x[dim - 1])
    sigma = 2 * muS * ufl.sym(ufl.grad(u)) - p * ufl.Identity(dim)
    return {"u": u, "p": p, "sigma": sigma, "f": -ufl.div(sigma)}


def biot_mms(mesh, c):
    """Steady Biot exact solution; c = ctx.c (material constants)."""
    x = ufl.SpatialCoordinate(mesh)
    dim = mesh.geometry.dim
    pi = np.pi
    s = [ufl.sin(pi * x[i]) for i in range(dim)]
    co = [ufl.cos(pi * x[i]) for i in range(dim)]
    d = ufl.as_vector([s[i] * co[(i + 1) % dim] for i in range(dim)])
    p = co[0] * co[dim - 1] + 0.5                       # nonzero on the boundary
    kappa = c.K / c.muF
    sig_e = 2 * c.muP * ufl.sym(ufl.grad(d)) + c.lambdaP * ufl.div(d) * ufl.Identity(dim)
    sig_tot = sig_e - c.alpha * p * ufl.Identity(dim)
    return {"d": d, "p": p, "sig_tot": sig_tot, "darcy_flux": kappa * ufl.grad(p),
            "f": -ufl.div(sig_tot),                                     # (B1), steady
            "g": -ufl.div(kappa * ufl.grad(p)) + c.beta * p}            # (B2), steady


def interface_residuals(exS, exP, c, nS, dt_d):
    """How much the exact fields violate (I1)-(I4) on Sigma. Stokes fields are taken
    from the '+' side, poro fields from the '-' side; nS = n('+')."""
    S = lambda e: e("+")
    P = lambda e: e("-")
    nP = -nS
    B = c.gammaBJS * c.muS / ufl.sqrt(c.K)
    traction_S = S(exS["sigma"]) * nS                       # sigma_S n_S
    return {
        "r_n": -ufl.inner(traction_S, nS) - P(exP["p"]),                            # (I2)
        "r_tau": tangential(-traction_S - B * (S(exS["u"]) - P(dt_d)), nS),           # (I3)
        "r_b": traction_S + P(exP["sig_tot"]) * nP,                                   # (I4)
        "r_m": ufl.inner(S(exS["u"]), nS) + ufl.inner(P(dt_d) - P(exP["darcy_flux"]), nP),  # (I1)
    }


def linear_mms(mesh, c):
    """Exact fields that are LINEAR in space (any dim): DG P1 reproduces them exactly,
    so a time study with them shows the pure temporal error."""
    x = ufl.SpatialCoordinate(mesh)
    dim = mesh.geometry.dim
    I = ufl.Identity(dim)
    # divergence-free linear velocity: u_i = 1 + sum_j A_ij x_j with trace(A) = 0
    A = [[(i + 2 * j + 1) % 3 - 1.0 for j in range(dim)] for i in range(dim)]
    A[dim - 1][dim - 1] = -sum(A[i][i] for i in range(dim - 1))
    u = ufl.as_vector([1.0 + sum(A[i][j] * x[j] for j in range(dim)) for i in range(dim)])
    pS = 0.3 + x[0] - 0.5 * x[dim - 1]
    d = ufl.as_vector([0.2 * (i + 1) + x[i] - 0.5 * x[(i + 1) % dim] for i in range(dim)])
    pP = 1.0 + 0.7 * x[0] + 0.4 * x[dim - 1]
    kappa = c.K / c.muF
    sigma = 2 * c.muS * ufl.sym(ufl.grad(u)) - pS * I
    sig_tot = 2 * c.muP * ufl.sym(ufl.grad(d)) + c.lambdaP * ufl.div(d) * I - c.alpha * pP * I
    S = {"u": u, "p": pS, "sigma": sigma, "f": -ufl.div(sigma)}
    P = {"d": d, "p": pP, "sig_tot": sig_tot, "darcy_flux": kappa * ufl.grad(pP),
         "f": -ufl.div(sig_tot), "g": -ufl.div(kappa * ufl.grad(pP)) + c.beta * pP}
    return S, P


def coupled_time_mms(mesh, c, t, linear_in_space=False):
    """Exact fields = (1 + sin t) * steady fields. t is a dolfinx Constant.
    ufl.variable(t) lets UFL take exact time derivatives with ufl.diff.
    linear_in_space=True: use linear_mms (zero spatial error, for time studies)."""
    tv = ufl.variable(t)
    T = 1 + ufl.sin(tv)
    dim = mesh.geometry.dim
    I = ufl.Identity(dim)
    sS, sP = linear_mms(mesh, c) if linear_in_space else (stokes_mms(mesh, c.muS), biot_mms(mesh, c))
    uS, pS, d, pP = T * sS["u"], T * sS["p"], T * sP["d"], T * sP["p"]
    d_t = ufl.diff(d, tv)
    d_tt = ufl.diff(d_t, tv)
    kappa = c.K / c.muF
    sigma = 2 * c.muS * ufl.sym(ufl.grad(uS)) - pS * I
    sig_tot = 2 * c.muP * ufl.sym(ufl.grad(d)) + c.lambdaP * ufl.div(d) * I - c.alpha * pP * I
    S = {"u": uS, "p": pS, "sigma": sigma,
         "f": c.rhoS * ufl.diff(uS, tv) - ufl.div(sigma)}                                   # (S1)
    P = {"d": d, "p": pP, "d_t": d_t, "d_tt": d_tt, "sig_tot": sig_tot,
         "darcy_flux": kappa * ufl.grad(pP),
         "f": c.rhoP * d_tt - ufl.div(sig_tot),                                             # (B1)
         "g": c.c0 * ufl.diff(pP, tv) + c.alpha * ufl.div(d_t)
              - ufl.div(kappa * ufl.grad(pP)) + c.beta * pP}                               # (B2)
    return S, P
