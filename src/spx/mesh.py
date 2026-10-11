"""Mesh, cell tags and facet tags. Interface = plane x[dim-1] = 0, Stokes below."""
from dataclasses import dataclass

import dolfinx
import numpy as np
from mpi4py import MPI


S_BOTTOM, S_SIDES, P_TOP, P_SIDES = 10, 11, 12, 13


@dataclass(frozen=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    cell_marker: np.ndarray
    facet_tags: dolfinx.mesh.MeshTags
    interior: dict


def box_mesh(N: int, dim: int, comm=MPI.COMM_WORLD) -> dolfinx.mesh.Mesh:
    ghost = dolfinx.mesh.GhostMode.shared_facet
    if dim == 2:
        return dolfinx.mesh.create_rectangle(
            comm, [[0.0, -1.0], [1.0, 1.0]], [N, 2 * N],
            dolfinx.mesh.CellType.triangle, ghost_mode=ghost
        )
    return dolfinx.mesh.create_box(
        comm, [[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]], [N, N, 2 * N],
        dolfinx.mesh.CellType.tetrahedron, ghost_mode=ghost
    )


def tag_cells(mesh, dim, stokes, poroel) -> np.ndarray:
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    marker = np.full(imap.size_local + imap.num_ghosts, poroel, dtype=np.int32)
    below = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[dim - 1] <= 1e-12)
    marker[below] = stokes
    return marker


def tag_box_boundary(mesh, dim) -> dolfinx.mesh.MeshTags:
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    z = dolfinx.mesh.compute_midpoints(mesh, fdim, facets)[:, dim - 1]
    values = np.where(np.isclose(z, -1.0), S_BOTTOM,
             np.where(np.isclose(z, 1.0), P_TOP,
             np.where(z < 0.0, S_SIDES, P_SIDES))).astype(np.int32)
    order = np.argsort(facets)
    return dolfinx.mesh.meshtags(mesh, fdim, facets[order], values[order])


def classify_interior_facets(mesh, cell_marker, stokes, poroel) -> dict:
    tdim, fdim = mesh.topology.dim, mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    f2c = mesh.topology.connectivity(fdim, tdim)
    offs = f2c.offsets
    interior = np.flatnonzero(np.diff(offs) == 2)
    t0 = cell_marker[f2c.array[offs[interior]]]
    t1 = cell_marker[f2c.array[offs[interior] + 1]]
    return {
        "interior_S": interior[(t0 == stokes) & (t1 == stokes)].astype(np.int32),
        "interior_P": interior[(t0 == poroel) & (t1 == poroel)].astype(np.int32),
        "interface": interior[t0 != t1].astype(np.int32)
    }


def build_box_geometry(N: int, mesh_cfg) -> Geometry:
    stokes, poroel = mesh_cfg.cells["stokes"], mesh_cfg.cells["poroel"]
    mesh = box_mesh(N, mesh_cfg.dim)
    marker = tag_cells(mesh, mesh_cfg.dim, stokes, poroel)
    return Geometry(
        mesh, marker, tag_box_boundary(mesh, mesh_cfg.dim),
        classify_interior_facets(mesh, marker, stokes, poroel)
    )
