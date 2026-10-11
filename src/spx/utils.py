import dolfinx
from mpi4py import MPI


def integrate(mesh, integrand_times_measure) -> float:
    """Assemble a scalar form on every rank and add up the parts."""
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(integrand_times_measure))  # type: ignore
    return mesh.comm.allreduce(local, op=MPI.SUM)
