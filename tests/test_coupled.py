import dolfinx
import multiphenicsx.fem.petsc as mpx

from spx.blocks import BlockSystem
from spx.cases.coupled_mms import setup, solve
from spx.config import load
from spx.errors import convergence_table
from spx.physics.interface import add_interface
from spx.spaces import FIELDS

CFG = load("configs/mms_coupled_2d.yaml")


def test_every_interface_block_is_nonzero():
    """Each interface block assembled alone is nonzero (catches wrong '+'/'-' restrictions)."""
    geo, m, ctx, V, R, tr, te, zero_dx = setup(4, CFG)
    blocks = BlockSystem(geo.mesh, FIELDS, te, zero_dx)
    add_interface(blocks, tr["uS"], tr["d"], tr["pP"], te["uS"], te["d"], te["pP"], ctx)
    for i, row in enumerate(FIELDS):
        for j, col in enumerate(FIELDS):
            if blocks.a[i][j] is not None:
                A = mpx.assemble_matrix(dolfinx.fem.form(blocks.a[i][j]), bcs=[],
                                        restriction=(R[row], R[col]))
                A.assemble()
                assert A.norm() > 0, f"interface block {row}-{col} is zero: wrong '+'/'-'?"


def test_coupled_rates():
    """On three meshes, all coupled errors fall at the expected rates."""
    last = convergence_table([solve(N, CFG) for N in (4, 8, 16)]).iloc[-1]
    assert last["rate_uS_L2"] > 1.8 and last["rate_d_L2"] > 1.8 and last["rate_pP_L2"] > 1.8
    assert last["rate_pS_L2"] > 0.85
    assert last["rate_mass"] > 0.4
