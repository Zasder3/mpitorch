import numpy as np
import torch
from torch import nn

from mpitorch.utils.mesh import Mesh
from mpitorch.utils.primitives import all_gather, broadcast, reduce_scatter


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        dp_axis: int,
        fsdp_axis: int,
        tp_axis: int,
        mesh: Mesh,
    ):
        super().__init__()
        self.dp_comm = mesh.get_comm_subgroup(dp_axis)
        self.fsdp_comm = mesh.get_comm_subgroup(fsdp_axis)
        self.tp_comm = mesh.get_comm_subgroup(tp_axis)

        self.w_in = nn.Parameter(
            torch.randn(
                hidden_features // mesh.get_axis_size(tp_axis),
                in_features // mesh.get_axis_size(fsdp_axis),
            )
            * np.sqrt(2 / in_features)
        )  # (H // TP, D // FSDP)

        self.w_out = nn.Parameter(
            torch.randn(
                out_features // mesh.get_axis_size(fsdp_axis),
                hidden_features // mesh.get_axis_size(tp_axis),
            )
            * np.sqrt(2 / hidden_features)
        )  # (O // FSDP, H // TP)

        # broadcast parameters across the DP axis
        self.w_in.data = broadcast(self.w_in.data, self.dp_comm, 0)
        self.w_out.data = broadcast(self.w_out.data, self.dp_comm, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = all_gather(x, self.tp_comm, 1)  # (N, D)
        w_in = all_gather(self.w_in, self.fsdp_comm, 1)  # (H, D)
        z = x @ w_in.T  # (N, H // TP)
        a = nn.functional.gelu(z)  # (N, H // TP)
        w_out = all_gather(self.w_out, self.fsdp_comm, 0)  # (O, H // TP)
        out = a @ w_out.T  # (N, O) {TP}
        return reduce_scatter(out, self.tp_comm, 1)  # (N, O)


class MLPModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_blocks: int,
        dp_axis: int,
        tp_axis: int,
        fsdp_axis: int,
        mesh: Mesh,
    ):
        super().__init__()
        self.mesh = mesh
        self.tp_size = mesh.get_axis_size(tp_axis)
        self.tp_rank = mesh.get_axis_rank(tp_axis)

        self.blocks = nn.ModuleList(
            [
                MLPBlock(
                    in_features,
                    hidden_features,
                    hidden_features,
                    dp_axis,
                    fsdp_axis,
                    tp_axis,
                    mesh,
                ),
                *[
                    MLPBlock(
                        hidden_features,
                        hidden_features,
                        hidden_features,
                        dp_axis,
                        fsdp_axis,
                        tp_axis,
                        mesh,
                    )
                    for _ in range(hidden_blocks)
                ],
                MLPBlock(
                    hidden_features,
                    out_features,
                    hidden_features,
                    dp_axis,
                    fsdp_axis,
                    tp_axis,
                    mesh,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.blocks[0].w_in.data.shape[1] * self.tp_size:
            # reshape x to accommodate the TP sharding
            d = x.shape[-1] // self.tp_size
            x = x[..., self.tp_rank * d : (self.tp_rank + 1) * d].clone()

        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    from mpi4py import MPI

    mesh = Mesh(MPI.COMM_WORLD, (2, 2, 2))
    model = MLPModel(768, 10, 128, 1, 0, 1, 2, mesh)

    if mesh.comm.rank == 0:
        for block in model.blocks:
            print(block.w_in.data.T.shape)
            print(block.w_out.data.T.shape)

    x = torch.randn(16, 768)
    out = model(x)
    print(out.shape)
