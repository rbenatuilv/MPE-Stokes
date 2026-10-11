import dolfinx, multiphenicsx.fem, numpy as np, ufl
from mpi4py import MPI
from petsc4py import PETSc
from importlib.metadata import version   # multiphenicsx has no __version__


if MPI.COMM_WORLD.rank == 0:
    print("dolfinx", dolfinx.__version__, "| multiphenicsx", version("multiphenicsx"))
    print("PETSc scalar:", PETSc.ScalarType)                        # expect float64
    print("MUMPS available:", PETSc.Sys.hasExternalPackage("mumps"))
    print("ADIOS2 (VTX output):", hasattr(dolfinx.io, "VTXWriter"))

# a real multiphenicsx call: fails with TypeError if the ABI does not match
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
V = dolfinx.fem.functionspace(mesh, ("DG", 1, (2,)))
multiphenicsx.fem.DofMapRestriction(V.dofmap, np.arange(3, dtype=np.int32))
print("rank", MPI.COMM_WORLD.rank, "of", MPI.COMM_WORLD.size, "| DofMapRestriction: OK")