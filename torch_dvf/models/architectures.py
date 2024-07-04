import torch

from torch_dvf.data import Data
from torch_dvf.nn.gnn import interp, pool


class PointNet(torch.nn.Module):

    def __init__(
        self,
        point_cloud_pooling_layers: torch.nn.ModuleList,
        mlp_layers: torch.nn.ModuleList,
    ):
        super().__init__()

        self.point_cloud_pooling_layers = point_cloud_pooling_layers
        self.mlp_layers = mlp_layers

        self.num_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        print(f"{self.__class__.__name__} ({self.num_parameters} parameters)")

    def forward(self, data: Data) -> torch.Tensor:
        x, pos = data.x, data.pos

        x_cache = []
        pos_cache = []

        for i, layer in enumerate(self.point_cloud_pooling_layers):
            x_cache.append(x)
            pos_cache.append(pos)

            x, pos = pool(layer, x, pos, data, scale_id=i)

        for layer in self.mlp_layers:
            x = interp(
                layer, x, x_cache.pop(), pos, pos := pos_cache.pop(), data, scale_id=i
            )
            i -= 1

        return x
