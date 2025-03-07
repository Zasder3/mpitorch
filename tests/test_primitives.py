import pytest
import torch
from mpi4py import MPI

from mpitorch.utils.primitives import all_gather, all_reduce, broadcast, reduce_scatter


@pytest.mark.mpi
def test_all_gather_forward():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a 2x1 matrix unique to each rank
    matrix_tensor = torch.tensor(
        [rank, rank], dtype=torch.float32, requires_grad=True
    ).view(2, 1)

    # Perform all gather operation
    gathered_tensor = all_gather(matrix_tensor, comm, 1)
    assert gathered_tensor.shape == (2, size)
    assert torch.allclose(
        gathered_tensor,
        torch.arange(size, dtype=torch.float32).view(1, size).repeat(2, 1),
    )


@pytest.mark.mpi
def test_all_gather_backward():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a 2x1 matrix unique to each rank
    matrix_tensor = torch.tensor(
        [rank, rank], dtype=torch.float32, requires_grad=True
    ).view(2, 1)
    matrix_tensor.retain_grad()

    # Perform all gather operation
    gathered_tensor = all_gather(matrix_tensor, comm, 1)

    # Perform backward pass
    gathered_tensor.backward(torch.ones_like(gathered_tensor))

    # Check that the gradient is correct
    assert matrix_tensor.grad is not None
    assert torch.allclose(matrix_tensor.grad, torch.ones_like(matrix_tensor) * size)


@pytest.mark.mpi
def test_reduce_scatter_forward():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a size x 1 matrix identical on each rank
    matrix_tensor = torch.arange(size, dtype=torch.float32).view(size, 1)

    # Perform reduce scatter operation
    reduced_tensor = reduce_scatter(matrix_tensor, comm, 0)

    # Check that the reduced tensor is correct
    assert reduced_tensor.shape == (1, 1)
    assert torch.allclose(
        reduced_tensor, torch.tensor([[rank * size]], dtype=torch.float32)
    )


@pytest.mark.mpi
def test_reduce_scatter_backward():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # Create a size x 1 matrix identical on each rank
    matrix_tensor = torch.arange(size, dtype=torch.float32, requires_grad=True).view(
        size, 1
    )
    matrix_tensor.retain_grad()

    # Perform reduce scatter operation
    reduced_tensor = reduce_scatter(matrix_tensor, comm, 0)

    # Perform backward pass
    reduced_tensor.backward(torch.ones_like(reduced_tensor))

    # Check that the gradient is correct
    assert matrix_tensor.grad is not None
    assert torch.allclose(matrix_tensor.grad, torch.ones_like(matrix_tensor))


@pytest.mark.mpi
def test_all_reduce_forward():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # Create a size x 1 matrix identical on each rank
    matrix_tensor = torch.arange(size, dtype=torch.float32).view(size, 1)

    # Perform all reduce operation
    reduced_tensor = all_reduce(matrix_tensor, comm, 0)

    # Check that the reduced tensor is correct
    assert reduced_tensor.shape == (size, 1)
    assert torch.allclose(
        reduced_tensor, torch.arange(size, dtype=torch.float32).view(size, 1) * size
    )


@pytest.mark.mpi
def test_all_reduce_backward():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # Create a size x 1 matrix identical on each rank
    matrix_tensor = torch.arange(size, dtype=torch.float32, requires_grad=True).view(
        size, 1
    )
    matrix_tensor.retain_grad()

    # Perform all reduce operation
    reduced_tensor = all_reduce(matrix_tensor, comm, 0)

    # Perform backward pass
    reduced_tensor.backward(torch.ones_like(reduced_tensor))

    # Check that the gradient is correct
    assert matrix_tensor.grad is not None
    assert torch.allclose(matrix_tensor.grad, torch.ones_like(matrix_tensor) * size)


@pytest.mark.mpi
def test_broadcast_forward():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a 1x1 matrix unique to each rank
    matrix_tensor = torch.tensor(rank, dtype=torch.float32).view(1, 1)

    # Perform broadcast operation
    broadcasted_tensor = broadcast(matrix_tensor, comm, size - 1)

    # Check that the broadcasted tensor is correct
    assert broadcasted_tensor.shape == (1, 1)
    assert torch.allclose(
        broadcasted_tensor, torch.tensor([[size - 1]], dtype=torch.float32)
    )


@pytest.mark.mpi
def test_broadcast_backward():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a 1x1 matrix unique to each rank
    matrix_tensor = torch.tensor(rank, dtype=torch.float32, requires_grad=True).view(
        1, 1
    )
    matrix_tensor.retain_grad()

    # Perform broadcast operation
    broadcasted_tensor = broadcast(matrix_tensor, comm, size - 1)

    # Perform backward pass
    broadcasted_tensor.backward(torch.ones_like(broadcasted_tensor))

    # Check that the gradient is correct
    assert matrix_tensor.grad is not None
    if rank == size - 1:
        assert torch.allclose(matrix_tensor.grad, torch.ones_like(matrix_tensor) * size)
    else:
        assert torch.allclose(matrix_tensor.grad, torch.zeros_like(matrix_tensor))
