from functools import cache
from math import prod
from typing import Tuple

import numpy as np
from mpi4py import MPI


class Mesh:
    def __init__(self, comm: MPI.Comm, mesh_shape: Tuple[int, ...]):
        self.comm = comm
        self.mesh_shape = mesh_shape
        self.loc = np.unravel_index(self.comm.rank, self.mesh_shape)
        assert prod(mesh_shape) == comm.size, "Mesh shape must match communicator size"

    def __len__(self):
        return self.comm.size

    def get_axis_size(self, axis: int) -> int:
        return self.mesh_shape[axis]

    def get_axis_rank(self, axis: int) -> int:
        return self.loc[axis]

    @cache
    def get_comm_subgroup(self, axis: int) -> MPI.Comm:
        """
        Get a subcommunicator of the communicator along a given axis.

        Args:
            axis: The axis to get the subgroup along.

        Returns:
            A subcommunicator of the communicator along the given axis.
        """
        mesh_shape_excluding_axis = self.mesh_shape[:axis] + self.mesh_shape[axis + 1 :]
        color = np.arange(prod(mesh_shape_excluding_axis)).reshape(
            mesh_shape_excluding_axis
        )
        color = color[self.loc[:axis] + self.loc[axis + 1 :]]
        return self.comm.Split(color)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    mesh = Mesh(comm, (2, 2, 3))

    subgroup = mesh.get_comm_subgroup(0)
    subgroup_ranks = subgroup.allgather(comm.rank)
    if subgroup.rank == 0:
        # print global comm ranks in the subgroup
        print(subgroup_ranks)
    assert subgroup is mesh.get_comm_subgroup(0)
