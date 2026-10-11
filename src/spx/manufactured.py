"""Exact solutions in UFL. Sources and boundary data are DERIVED, never typed by hand."""
import numpy as np
import ufl


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
