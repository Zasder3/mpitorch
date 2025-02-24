from typing import Tuple

from mpi4py import MPI

class Mesh:
    def __init__(self, comm: MPI.Comm, mesh_shape: Tuple[int, ...]):
        self.comm = comm
        self.mesh_shape = mesh_shape

    def __len__(self):
        return self.comm.size
