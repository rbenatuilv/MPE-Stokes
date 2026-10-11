import dolfinx
import numpy as np
import pytest
import ufl

from spx.config import load
from spx.measures import build_measures
from spx.mesh import build_box_geometry
from spx.spaces import build_restrictions, build_spaces
from spx.utils import integrate

CFG = load("configs/mms_stokes_2d.yaml")


@pytest.fixture
def geo_and_measures():
    """A small 2D box (N = 4) and its measures, shared by the tests below."""
    geo = build_box_geometry(4, CFG.mesh)
    return geo, build_measures(geo, CFG.mesh, qdeg=2)


def test_volumes_and_interface_length(geo_and_measures):
    """Both subdomains have area 1 and the interface has length 1."""
    geo, m = geo_and_measures
    one = dolfinx.fem.Constant(geo.mesh, 1.0)
    assert np.isclose(integrate(geo.mesh, one * m.dx(m.STOKES)), 1.0)
    assert np.isclose(integrate(geo.mesh, one * m.dx(m.POROEL)), 1.0)
    assert np.isclose(integrate(geo.mesh, one * m.dS(m.GAMMA)), 1.0)


def test_plus_side_is_stokes(geo_and_measures):
    """On the interface, n('+') is the outward normal of the Stokes side (points up)."""
    geo, m = geo_and_measures
    n = ufl.FacetNormal(geo.mesh)
    # Stokes lies below the interface, so its outward normal points to +y
    assert np.isclose(integrate(geo.mesh, n("+")[1] * m.dS(m.GAMMA)), 1.0)  # type: ignore


def test_restriction_sizes(geo_and_measures):
    """The restrictions keep exactly the DOFs of the Stokes cells, no more, no fewer."""
    geo, _ = geo_and_measures
    r = build_restrictions(build_spaces(geo.mesh, 1, 1), geo.cell_marker, 2, 1)
    n_owned = geo.mesh.topology.index_map(2).size_local
    n_stokes = geo.mesh.comm.allreduce(int(np.sum(geo.cell_marker[:n_owned] == 2)))
    # DG1 on triangles: 3 dofs per cell; velocity has 2 components
    assert r["uS"].index_map.size_global * r["uS"].index_map_bs == 3 * 2 * n_stokes
    assert r["pS"].index_map.size_global == 3 * n_stokes
