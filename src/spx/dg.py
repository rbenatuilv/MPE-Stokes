"""DG building blocks shared by all physics modules and the error norms."""
import ufl


def h_facet(mesh):
    """Harmonic mean of the two cell diameters on an interior facet."""
    h = ufl.CellDiameter(mesh)
    return 2 * h("+") * h("-") / (h("+") + h("-"))  # type: ignore


def h_min(mesh):
    """Smaller of the two cell diameters (Stokes pressure stabilization)."""
    h = ufl.CellDiameter(mesh)
    return ufl.min_value(h("+"), h("-"))


def sym_jump(u, n):
    """Symmetrized jump [[u (x) n]] of a vector field on an interior facet."""
    return ufl.sym(ufl.outer(u("+"), n("+")) + ufl.outer(u("-"), n("-")))


def sym_bnd(u, n):
    """Same on a boundary facet (one side only)."""
    return ufl.sym(ufl.outer(u, n))


def tangential(w, n):
    """w - (w.n) n ; w and n must already be restricted to a side."""
    return w - ufl.inner(w, n) * n