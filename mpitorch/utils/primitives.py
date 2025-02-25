import torch
from mpi4py import MPI
from torch.autograd import Function


class _AllGather(Function):
    @staticmethod
    def forward(ctx, i: torch.Tensor, comm: MPI.Comm, gather_dim: int):
        i = i.transpose(gather_dim, 0).detach()  # (gather_dim, ...)
        ctx.comm = comm
        ctx.gather_dim = gather_dim  # Save gather_dim for backward pass
        output = torch.empty(
            i.shape[0] * comm.size, *i.shape[1:], device=i.device, dtype=i.dtype
        )  # (gather_dim, ...)
        comm.Allgather(i, output)
        output = output.transpose(
            gather_dim, 0
        ).clone()  # Clone the output to avoid view issues
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output is the gradient of the output of the allgather
        # we need to scatter the gradient back to the original tensors
        # we do this by transposing the gradient and using reduce scatter
        grad_output = grad_output.detach().transpose(
            ctx.gather_dim, 0
        )  # (gather_dim, N, ...)
        grad_input = torch.empty(
            grad_output.shape[0] // ctx.comm.size,
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )  # (gather_dim, N, ...)
        comm = ctx.comm
        comm.Reduce_scatter(grad_output, grad_input, op=MPI.SUM)
        grad_input = grad_input.transpose(ctx.gather_dim, 0)  # (..., gather_dim, ...)
        return grad_input, None, None


class _ReduceScatter(Function):
    @staticmethod
    def forward(ctx, i: torch.Tensor, comm: MPI.Comm, reduce_dim: int):
        i = i.transpose(reduce_dim, 0).detach()  # (reduce_dim, ...)
        ctx.comm = comm
        ctx.reduce_dim = reduce_dim  # Save reduce_dim for backward pass
        output = torch.empty(
            i.shape[0] // comm.size, *i.shape[1:], device=i.device, dtype=i.dtype
        )  # (reduce_dim, ...)
        comm.Reduce_scatter(i, output)
        output = output.transpose(
            reduce_dim, 0
        ).clone()  # Clone the output to avoid view issues
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output is the gradient of the output of the reduce_scatter
        # we need to scatter the gradient back to the original tensors
        # we do this by transposing the gradient and using all gather
        grad_output = grad_output.detach().transpose(
            ctx.reduce_dim, 0
        )  # (reduce_dim, N, ...)
        comm = ctx.comm
        grad_input = torch.empty(
            grad_output.shape[0] * comm.size,
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )  # (reduce_dim, N, ...)
        comm.Allgather(grad_output, grad_input)
        grad_input = grad_input.transpose(ctx.reduce_dim, 0)  # (..., reduce_dim, ...)
        return grad_input, None, None


class _Broadcast(Function):
    @staticmethod
    def forward(ctx, i: torch.Tensor, comm: MPI.Comm, broadcast_dim: int):
        ctx.comm = comm
        output = torch.empty_like(i)
        comm.Bcast(i, root=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.detach()
        comm = ctx.comm
        grad_input = torch.empty_like(grad_output)
        comm.Reduce(grad_output, grad_input, root=0, op=MPI.SUM)
        if comm.rank != 0:
            grad_input.zero_()
        return grad_input, None, None


def all_gather(i: torch.Tensor, comm: MPI.Comm, gather_dim: int):
    return _AllGather.apply(i, comm, gather_dim)


def reduce_scatter(i: torch.Tensor, comm: MPI.Comm, reduce_dim: int):
    return _ReduceScatter.apply(i, comm, reduce_dim)


def all_reduce(i: torch.Tensor, comm: MPI.Comm, reduce_dim: int):
    return _AllGather.apply(_ReduceScatter.apply(i, comm, reduce_dim), comm, reduce_dim)


def broadcast(i: torch.Tensor, comm: MPI.Comm, broadcast_dim: int):
    return _Broadcast.apply(i, comm, broadcast_dim)
