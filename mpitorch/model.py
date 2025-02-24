from mpitorch.utils.mesh import Mesh

def MLPBlock(in_features: int, out_features: int, hidden_features: int, fsdp_axis: int, tp_axis: int, mesh: Mesh):
    ... 