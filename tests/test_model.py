import torch
from mpi4py import MPI

from mpitorch.model import MLPModel
from mpitorch.utils.mesh import Mesh
from mpitorch.utils.primitives import all_gather
from mpitorch.utils.testing import exact_mpi_size


@exact_mpi_size(2)
def test_dp_init():
    """Ensure that the models across the DP axis are identical"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mesh = Mesh(
        comm,
        (2, 1, 1),
    )

    model = MLPModel(
        in_features=10,
        out_features=10,
        hidden_features=10,
        hidden_blocks=1,
        dp_axis=0,
        tp_axis=1,
        fsdp_axis=2,
        mesh=mesh,
        seed=42,
    )

    # Exchange state dicts between ranks using sendrecv
    state_dict = model.state_dict()
    other_state_dict = comm.sendrecv(
        state_dict,
        dest=(rank + 1) % size,
        source=(rank - 1) % size,
    )
    # Compare state dicts
    for key in state_dict:
        assert torch.allclose(state_dict[key], other_state_dict[key]), (
            f"Mismatch in {key}"
        )


@exact_mpi_size(4)
def test_fsdp_tp_init():
    """Ensure that the models across the FSDP and TP axis are identical"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mesh = Mesh(
        comm,
        (1, 2, 2),
    )

    sharded_model = MLPModel(
        in_features=10,
        out_features=10,
        hidden_features=10,
        hidden_blocks=1,
        dp_axis=0,
        tp_axis=1,
        fsdp_axis=2,
        mesh=mesh,
        seed=42,
    )

    if rank == 0:
        singleton_mesh = Mesh(
            comm.Create_group(mesh.comm.group.Incl([0])),
            (1, 1, 1),
        )
        singleton_model = MLPModel(
            in_features=10,
            out_features=10,
            hidden_features=10,
            hidden_blocks=1,
            dp_axis=0,
            tp_axis=1,
            fsdp_axis=2,
            mesh=singleton_mesh,
            seed=42,
        )

    gather_model = sharded_model.gather_model()
    if rank == 0:
        for key in gather_model.state_dict():
            assert not torch.allclose(
                gather_model.state_dict()[key],
                torch.zeros_like(gather_model.state_dict()[key]),
            )
            torch.set_printoptions(precision=2, linewidth=80)
            print(
                torch.abs(
                    gather_model.state_dict()[key] - singleton_model.state_dict()[key]
                )
            )
            assert torch.allclose(
                gather_model.state_dict()[key], singleton_model.state_dict()[key]
            )
    else:
        assert gather_model is None


@exact_mpi_size(4)
def test_fsdp_tp_forward():
    """Ensure that the models across the FSDP and TP axis are identical"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    mesh = Mesh(comm, (1, 2, 2))

    sharded_model = MLPModel(
        in_features=10,
        out_features=10,
        hidden_features=10,
        hidden_blocks=0,
        dp_axis=0,
        tp_axis=1,
        fsdp_axis=2,
        mesh=mesh,
        seed=42,
    )

    if rank == 0:
        singleton_mesh = Mesh(
            comm.Create_group(mesh.comm.group.Incl([0])),
            (1, 1, 1),
        )
        singleton_model = MLPModel(
            in_features=10,
            out_features=10,
            hidden_features=10,
            hidden_blocks=0,
            dp_axis=0,
            tp_axis=1,
            fsdp_axis=2,
            mesh=singleton_mesh,
            seed=42,
        )

    gen = torch.Generator()
    gen.manual_seed(42)
    data = torch.randn(32, 10, generator=gen)
    sharded_data = data[
        mesh.get_axis_rank(2) * 16 : (mesh.get_axis_rank(2) + 1) * 16, :
    ]

    # Let the model handle the sharding
    tp_fsdp_sharded_output = sharded_model(data)

    # Gather across TP and FSDP
    fsdp_sharded_output = all_gather(
        tp_fsdp_sharded_output, sharded_model.blocks[-1].tp_comm, 1
    )
    output = fsdp_sharded_output
    # output = all_gather(fsdp_sharded_output, sharded_model.blocks[-1].fsdp_comm, 0)

    if rank == 0:
        singleton_output = singleton_model(data)

        assert singleton_output.shape == output.shape

        assert not torch.allclose(singleton_output, torch.zeros_like(singleton_output))
        print("Max diff: ", torch.abs(output - singleton_output).max())
        print("Mean diff: ", torch.abs(output - singleton_output).mean())
        print("Min diff: ", torch.abs(output - singleton_output).min())
        assert torch.allclose(output, singleton_output, atol=1e-5)


@exact_mpi_size(4)
def test_fsdp_tp_backward():
    """Ensure that the models across the FSDP and TP axis are identical"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mesh = Mesh(
        comm,
        (1, 2, 2),
    )

    sharded_model = MLPModel(
        in_features=10,
        out_features=10,
        hidden_features=10,
        hidden_blocks=10,
        dp_axis=0,
        tp_axis=1,
        fsdp_axis=2,
        mesh=mesh,
        seed=42,
    )

    if rank == 0:
        singleton_mesh = Mesh(
            comm.Create_group(mesh.comm.group.Incl([0])),
            (1, 1, 1),
        )
        singleton_model = MLPModel(
            in_features=10,
            out_features=10,
            hidden_features=10,
            hidden_blocks=10,
            dp_axis=0,
            tp_axis=1,
            fsdp_axis=2,
            mesh=singleton_mesh,
            seed=42,
        )
        singleton_optim = torch.optim.SGD(singleton_model.parameters(), lr=0.01)

    # generate data and take a step
    gen = torch.Generator()
    gen.manual_seed(42)
    data = torch.randn(32, 10, generator=gen)

    sharded_optim = torch.optim.SGD(sharded_model.parameters(), lr=0.01)

    sharded_output = sharded_model(data)
    output = all_gather(sharded_output, sharded_model.blocks[-1].tp_comm, 1)
    output.backward(torch.ones_like(output))
    sharded_optim.step()

    gather_model = sharded_model.gather_model()

    if rank == 0:
        # take a step
        singleton_output = singleton_model(data)
        print(singleton_output.shape)
        singleton_output.backward(torch.ones_like(singleton_output) / 4)
        singleton_optim.step()

        # check that the models are identical
        for key in gather_model.state_dict():
            assert torch.allclose(
                gather_model.state_dict()[key], singleton_model.state_dict()[key]
            )
