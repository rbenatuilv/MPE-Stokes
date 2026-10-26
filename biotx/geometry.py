"""Meshes with named boundaries. Dimensions in the YAML are in SI; the mesh returned is
already non-dimensional (coordinates divided by L0).

    geometry:
      shape: rectangle          # boundaries: left, right, bottom, top
      corners: [[0, 0], [1, 1]]
      n: [16, 16]               # cells per side (or h: target cell size)

      shape: box                # 3D; boundaries: left, right (x), front, back (y), bottom, top (z)
      corners: [[0, 0, 0], [1, 1, 1]]
      n: [8, 8, 8]              # (or h)

      shape: disc               # boundaries: outer
      center: [0, 0]
      radius: 1.0
      h: 0.05

      shape: annulus            # boundaries: outer, inner (inner circle may be off-centre)
      center: [0, 0]
      radius: 1.0
      inner: {center: [0.3, 0], radius: 0.25}
      h: 0.05

      shape: ball               # 3D; boundaries: outer
      center: [0, 0, 0]
      radius: 1.0
      h: 0.1

      shape: shell              # 3D annulus: boundaries outer, inner (inner sphere may be off-centre)
      center: [0, 0, 0]
      radius: 1.0
      inner: {center: [0.3, 0, 0], radius: 0.25}
      h: 0.1

      shape: gmsh               # any .msh file; boundaries = its named physical groups
      file: brain.msh           # (relative to the YAML file)
      dim: 2                    # 3 for a volume mesh
      tags: {ventricle: 1}      # optional: names for unnamed physical groups

      names: {inner: ventricle, outer: pia}   # optional, any shape: rename boundaries
"""
from dataclasses import dataclass
from pathlib import Path

import dolfinx
import dolfinx.io.gmsh
import numpy as np
from mpi4py import MPI


# the keys each shape accepts (besides `shape` and `names`)
SHAPE_KEYS = {
    "rectangle": {"corners", "n", "h", "diagonal"},
    "box": {"corners", "n", "h"},
    "disc": {"center", "radius", "h"}, "ball": {"center", "radius", "h"},
    "annulus": {"center", "radius", "inner", "h"}, "shell": {"center", "radius", "inner", "h"},
    "gmsh": {"file", "dim", "tags"},
}


def check(g: dict):
    """Unknown shape or keys, and missing sizes, raise ValueError (before any meshing)."""
    shape = g.get("shape")
    if shape not in SHAPE_KEYS:
        raise ValueError(f"unknown geometry shape '{shape}'; one of {sorted(SHAPE_KEYS)}")
    unknown = set(g) - SHAPE_KEYS[shape] - {"shape", "names"}
    if unknown:
        raise ValueError(f"geometry ({shape}): unknown key(s) {sorted(unknown)}; "
                         f"allowed {sorted(SHAPE_KEYS[shape] | {'names'})}")
    if shape in ("rectangle", "box") and not ({"n", "h"} & set(g)):
        raise ValueError(f"geometry ({shape}): give `n` or `h`")
    if "inner" in g and set(g["inner"]) - {"center", "radius"}:
        raise ValueError(f"geometry ({shape}): inner takes `center` and `radius`")


@dataclass(frozen=True)
class Domain:
    """Non-dimensional mesh, boundary facet tags and the boundary names -> tag values."""
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    names: dict


def _partitioner():
    # DG needs the cells on both sides of every facet: ghost the shared facets
    return dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet, 2)


def _structured(g, L0):
    """rectangle (triangles) or box (tetrahedra), each side its own boundary."""
    lo, hi = (np.asarray(c, dtype=float) / L0 for c in g["corners"])
    dim = len(lo)
    if "n" in g:
        n = list(g["n"])
    else:
        n = [max(1, round((b - a) / (g["h"] / L0))) for a, b in zip(lo, hi)]
    if dim == 2:
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, [lo, hi], n, dolfinx.mesh.CellType.triangle,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            diagonal=dolfinx.mesh.DiagonalType[g.get("diagonal", "right")])
        sides = {"left": (0, lo[0]), "right": (0, hi[0]), "bottom": (1, lo[1]), "top": (1, hi[1])}
    else:
        mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [lo, hi], n, dolfinx.mesh.CellType.tetrahedron,
                                       ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
        sides = {"left": (0, lo[0]), "right": (0, hi[0]), "front": (1, lo[1]), "back": (1, hi[1]),
                 "bottom": (2, lo[2]), "top": (2, hi[2])}
    fdim = mesh.topology.dim - 1
    facets, values, names = [], [], {}
    for tag, (name, (axis, value)) in enumerate(sides.items(), start=1):
        f = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[axis], value))
        facets.append(f)
        values.append(np.full(len(f), tag, dtype=np.int32))
        names[name] = tag
    facets, values = np.concatenate(facets), np.concatenate(values)
    facets, idx = np.unique(facets, return_index=True)     # (2D corners: first side wins)
    return mesh, dolfinx.mesh.meshtags(mesh, fdim, facets, values[idx]), names


def _gmsh_round(g, L0):
    """disc / annulus (2D) and ball / shell (3D) with gmsh OCC: cells in one physical group,
    each circle / sphere its own boundary group."""
    import gmsh
    dim = 3 if g["shape"] in ("ball", "shell") else 2
    c = np.zeros(3)
    c[:dim] = np.asarray(g.get("center", [0.0] * dim), dtype=float) / L0
    R = g["radius"] / L0
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    add = (lambda c, r: occ.addDisk(*c, r, r)) if dim == 2 else (lambda c, r: occ.addSphere(*c, r))
    body = [(dim, add(c, R))]
    if g["shape"] in ("annulus", "shell"):
        ci = np.zeros(3)
        ci[:dim] = np.asarray(g["inner"]["center"], dtype=float) / L0
        body, _ = occ.cut(body, [(dim, add(ci, g["inner"]["radius"] / L0))])
    occ.synchronize()
    gmsh.model.addPhysicalGroup(dim, [b[1] for b in body], 1)
    bnd = [e for _, e in gmsh.model.getBoundary(body, oriented=False)]
    size = {e: gmsh.model.getBoundingBox(dim - 1, e)[3] - gmsh.model.getBoundingBox(dim - 1, e)[0]
            for e in bnd}
    names = {}
    for tag, e in enumerate(sorted(bnd, key=lambda e: -size[e]), start=1):
        names["outer" if tag == 1 else "inner"] = tag    # the larger circle / sphere is the outer one
        gmsh.model.addPhysicalGroup(dim - 1, [e], tag)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", g["h"] / L0)
    gmsh.model.mesh.generate(dim)
    md = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=dim, partitioner=_partitioner())
    gmsh.finalize()
    return md.mesh, md.facet_tags, names


def _gmsh_file(g, L0, root):
    dim = g.get("dim", 2)
    md = dolfinx.io.gmsh.read_from_msh(Path(root) / g["file"], MPI.COMM_WORLD, 0, gdim=dim,
                                       partitioner=_partitioner())
    md.mesh.geometry.x[:] /= L0
    names = {n: pg.tag for n, pg in md.physical_groups.items() if pg.dim == dim - 1}
    names.update(g.get("tags", {}))
    return md.mesh, md.facet_tags, names


def build(g: dict, L0: float = 1.0, root=".") -> Domain:
    """Mesh and named boundaries of a geometry section (dimensions in SI)."""
    check(g)
    shape = g["shape"]
    if shape in ("rectangle", "box"):
        mesh, tags, names = _structured(g, L0)
    elif shape in ("disc", "annulus", "ball", "shell"):
        mesh, tags, names = _gmsh_round(g, L0)
    elif shape == "gmsh":
        mesh, tags, names = _gmsh_file(g, L0, root)
    rename = g.get("names", {})
    names = {rename.get(n, n): t for n, t in names.items()}
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    return Domain(mesh, tags, names)
