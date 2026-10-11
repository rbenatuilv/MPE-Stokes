"""Everything a physics module needs about one mesh, built once per mesh."""
from dataclasses import dataclass
from types import SimpleNamespace

import dolfinx
import ufl
from petsc4py import PETSc


@dataclass(frozen=True)
class Context:
    """Bundle passed to every physics function: mesh, measures, normal, degrees, constants."""
    mesh: object
    m: object                 # spx.measures.Measures
    n: object                 # ufl.FacetNormal
    k: int
    kp: int
    c: SimpleNamespace        # constants: c.muS, c.K, c.gammaV, ...


def make_context(mesh, measures, cfg) -> Context:
    """Turn the config's plain floats into dolfinx Constants on this mesh and bundle them
    with the measures, the facet normal and the polynomial degrees."""

    def const(value):
        """A dolfinx Constant holding `value` (can be changed later without recompiling)."""
        return dolfinx.fem.Constant(mesh, PETSc.ScalarType(value))  # type: ignore

    mat, disc = cfg.material, cfg.discretization
    c = {name: const(getattr(mat, name)) for name in mat.__dataclass_fields__}
    c["Kpen"] = const(max(mat.K, 1.0))
    for name in ("gammaV", "gammaP", "eta", "zeta"):
        c[name] = const(getattr(disc, name))
    return Context(mesh, measures, ufl.FacetNormal(mesh), disc.deg, disc.degP, SimpleNamespace(**c))
