import dolfinx
import multiphenicsx.fem
import numpy as np


FIELDS = ["uS", "pS", "d", "pP"]


def build_spaces(mesh, k: int, kp: int) -> dict:
    gdim = mesh.geometry.dim
    VS = dolfinx.fem.functionspace(mesh, ("DG", k, (gdim,)))
    return {"uS": VS,
            "pS": dolfinx.fem.functionspace(mesh, ("DG", kp)),
            "d":  VS.clone(),
            "pP": dolfinx.fem.functionspace(mesh, ("DG", k))}


def build_restrictions(spaces: dict, cell_marker, stokes, poroel) -> dict:
    mesh = next(iter(spaces.values())).mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim)
    lives_on = {"uS": stokes, "pS": stokes, "d": poroel, "pP": poroel}
    out = {}
    for name, V in spaces.items():
        cells = np.flatnonzero(cell_marker == lives_on[name]).astype(np.int32)
        dofs = dolfinx.fem.locate_dofs_topological(V, tdim, cells)
        out[name] = multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs)  # type: ignore
    return out
