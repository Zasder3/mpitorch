from typing import Tuple, Union

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
        seed: int,
    ):
        super().__init__()
        self.dp_comm = mesh.get_comm_subgroup(dp_axis)
        self.fsdp_comm = mesh.get_comm_subgroup(fsdp_axis)
        self.tp_comm = mesh.get_comm_subgroup(tp_axis)
        self.rank = mesh.comm.rank

        gen = torch.Generator()
        gen.manual_seed(seed)
        fsdp_rank = mesh.get_axis_rank(fsdp_axis)
        tp_rank = mesh.get_axis_rank(tp_axis)

        h_tp = hidden_features // mesh.get_axis_size(tp_axis)
        d_fsdp = in_features // mesh.get_axis_size(fsdp_axis)
        o_fsdp = out_features // mesh.get_axis_size(fsdp_axis)

        # Generate full tensors first
        w_in_full = torch.randn(hidden_features, in_features, generator=gen) * np.sqrt(
            2 / in_features
        )
        w_out_full = torch.randn(
            out_features, hidden_features, generator=gen
        ) * np.sqrt(2 / hidden_features)

        # Then shard them - first by TP, then by FSDP
        w_in_tensor = w_in_full[
            tp_rank * h_tp : (tp_rank + 1) * h_tp, :
        ]  # Shard by TP first
        w_in_tensor = w_in_tensor[
            :, fsdp_rank * d_fsdp : (fsdp_rank + 1) * d_fsdp
        ]  # Then by FSDP

        w_out_tensor = w_out_full[
            :, tp_rank * h_tp : (tp_rank + 1) * h_tp
        ]  # Shard by TP first
        w_out_tensor = w_out_tensor[
            fsdp_rank * o_fsdp : (fsdp_rank + 1) * o_fsdp, :
        ]  # Then by FSDP

        self.w_in = nn.Parameter(w_in_tensor)
        self.w_out = nn.Parameter(w_out_tensor)

        # broadcast parameters across the DP axis
        self.w_in.data = broadcast(self.w_in.data, self.dp_comm, 0)
        self.w_out.data = broadcast(self.w_out.data, self.dp_comm, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = all_gather(x, self.tp_comm, 1)  # (N, D)
        w_in = all_gather(self.w_in, self.fsdp_comm, 1)  # (H // TP, D)
        z = x @ w_in.T  # (N, H // TP)
        a = nn.functional.gelu(z)  # (N, H // TP)
        w_out = all_gather(self.w_out, self.fsdp_comm, 0)  # (O, H // TP)
        out = a @ w_out.T  # (N, O) {TP}
        out = reduce_scatter(out, self.tp_comm, 1)  # (N, O // TP)
        return out

    def gather_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # First gather weights across the TP axis
        w_in = all_gather(self.w_in.data, self.tp_comm, 0)  # (H, D//FSDP)
        w_out = all_gather(self.w_out.data, self.tp_comm, 1)  # (O//FSDP, H)

        # Then gather weights across the FSDP axis
        w_in = all_gather(w_in, self.fsdp_comm, 1)  # (H, D)
        w_out = all_gather(w_out, self.fsdp_comm, 0)  # (O, H)

        if self.rank != 0:
            return None

        return w_in, w_out


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
        seed: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_blocks = hidden_blocks
        self.seed = seed

        self.mesh = mesh
        self.dp_size = mesh.get_axis_size(dp_axis)
        self.dp_rank = mesh.get_axis_rank(dp_axis)
        self.tp_size = mesh.get_axis_size(tp_axis)
        self.tp_rank = mesh.get_axis_rank(tp_axis)
        self.fsdp_size = mesh.get_axis_size(fsdp_axis)
        self.fsdp_rank = mesh.get_axis_rank(fsdp_axis)

        self.dp_axis = dp_axis
        self.tp_axis = tp_axis
        self.fsdp_axis = fsdp_axis

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
                    seed,
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
                        seed + i + 1,
                    )
                    for i in range(hidden_blocks)
                ],
                MLPBlock(
                    hidden_features,
                    out_features,
                    hidden_features,
                    dp_axis,
                    fsdp_axis,
                    tp_axis,
                    mesh,
                    seed + hidden_blocks + 1,
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

    def gather_model(self) -> Union["MLPModel", None]:
        """
        Gather the model across the mesh onto rank 0. Only do work on data parallel
        rank 0.

        Returns None if the current rank is not the data parallel rank 0.
        """
        gathered_blocks = []
        if self.mesh.get_axis_rank(self.dp_axis) != 0:
            return None

        for block in self.blocks:
            gathered_blocks.append(block.gather_weights())

        if self.mesh.comm.rank != 0:
            return None

        # Create a new communicator with just rank 0
        rank_0_comm = self.mesh.comm.Create_group(self.mesh.comm.group.Incl([0]))
        rank_0_mesh = Mesh(rank_0_comm, (1, 1, 1))

        model = MLPModel(
            self.in_features,
            self.out_features,
            self.hidden_features,
            self.hidden_blocks,
            self.dp_axis,
            self.tp_axis,
            self.fsdp_axis,
            rank_0_mesh,
            self.seed + 1,  # adds test robustness
        )

        for block, gathered_block in zip(model.blocks, gathered_blocks):
            block.w_in.data = gathered_block[0]
            block.w_out.data = gathered_block[1]

        return model


if __name__ == "__main__":
    from mpi4py import MPI

    in_dim = 8
    out_dim = 10
    hidden_dim = 4
    hidden_blocks = 1
    dp_axis = 0
    tp_axis = 1
    fsdp_axis = 2

    mesh = Mesh(MPI.COMM_WORLD, (2, 2, 2))
    model = MLPModel(
        in_dim,
        out_dim,
        hidden_dim,
        hidden_blocks,
        dp_axis,
        tp_axis,
        fsdp_axis,
        mesh,
        42,
    )

    gen = torch.Generator()
    gen.manual_seed(0)
    x = torch.randn(16, in_dim, generator=gen)
    out = model(x)
    print(out.shape)
    # gather across TP axis
    # out = all_gather(out, mesh.get_comm_subgroup(1), 1)
    # if mesh.comm.rank == 0:
    #     print(out.shape)
    #     print(out)

    # gathered_model = model.gather_model()
    # if gathered_model is not None:
    #     out = gathered_model(x)
    #     print(out.shape)
    #     print(out)
