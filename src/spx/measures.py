"""Integration measures. On the interface, '+' is ALWAYS the Stokes cell."""
from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl


@dataclass(frozen=True)
class Measures:
    dx: ufl.Measure
    ds: ufl.Measure
    dS: ufl.Measure
    STOKES: int
    POROEL: int
    roles: dict
    INTERIOR_S: int = 1
    INTERIOR_P: int = 2
    GAMMA: int = 3

    def role(self, name):
        return self.roles[name]


def _interior_entities(mesh, facets):
    ent = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet, mesh.topology, facets.astype(np.int32)
    )
    return np.asarray(ent, dtype=np.int32).reshape(-1, 4)


def _sum(ds, ids):
    m = None
    for i in ids:
        m = ds(i) if m is None else m + ds(i)
    return m


def build_measures(geo, mesh_cfg, qdeg: int) -> Measures:
    mesh = geo.mesh
    stokes, poroel = mesh_cfg.cells["stokes"], mesh_cfg.cells["poroel"]
    md = {"quadrature_degree": qdeg}

    gamma = _interior_entities(mesh, geo.interior["interface"])
    swap = geo.cell_marker[gamma[:, 0]] != stokes
    gamma[swap] = gamma[swap][:, [2, 3, 0, 1]]

    dS = ufl.Measure("dS", domain=mesh, metadata=md, subdomain_data=[
        (1, _interior_entities(mesh, geo.interior["interior_S"]).flatten()),
        (2, _interior_entities(mesh, geo.interior["interior_P"]).flatten()),
        (3, gamma.flatten())
    ])

    cell_tags = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim, np.arange(geo.cell_marker.size, dtype=np.int32), geo.cell_marker
    )

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, metadata=md)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=geo.facet_tags, metadata=md)
    roles = {name: _sum(ds, ids) for name, ids in mesh_cfg.facets.items()}
    return Measures(dx, ds, dS, stokes, poroel, roles)
