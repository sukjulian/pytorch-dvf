import torch

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

# Compilation performance
torch.set_float32_matmul_precision("high")


@torch.compile
def pool(
    layer: MessagePassing,
    x: torch.Tensor,
    pos: torch.Tensor,
    sampling_idcs: torch.Tensor,
    pool_source: torch.Tensor,
    pool_target: torch.Tensor,
    **kwargs,
) -> tuple:

    edge_index = torch.cat((pool_source[None, :], pool_target[None, :]), dim=0)
    kwargs = {key: (value, value[sampling_idcs]) for key, value in kwargs.items()}

    return (
        layer(
            (x, x[sampling_idcs]),
            (pos, pos[sampling_idcs]),
            edge_index.long(),
            **kwargs,
        ),
        pos[sampling_idcs],
    )


@torch.compile
def interp(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    x_skip: torch.Tensor,
    pos_source: torch.Tensor,
    pos_target: torch.Tensor,
    interp_source: torch.Tensor,
    interp_target: torch.Tensor,
) -> torch.Tensor:

    pos_diff = pos_source[interp_source] - pos_target[interp_target]
    squared_pos_dist = torch.clamp(
        torch.sum(pos_diff**2, dim=-1, keepdim=True), min=1e-16
    )

    x = scatter(
        x[interp_source] / squared_pos_dist,
        interp_target.long(),
        dim=0,
        reduce="sum",
    ) / scatter(
        1.0 / squared_pos_dist,
        interp_target.long(),
        dim=0,
        reduce="sum",
    )

    return mlp(torch.cat((x, x_skip), dim=-1))


class PointCloudPooling(MessagePassing):
    def __init__(self, mlp: torch.nn.Module, **kwargs):
        kwargs.setdefault("aggr", "mean")

        super().__init__(**kwargs)

        self.mlp = mlp

    def message(
        self, x_j: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat((x_j, pos_j - pos_i), dim=1))

    def forward(
        self, x: tuple, pos: tuple, edge_index: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, pos=pos, **kwargs)
