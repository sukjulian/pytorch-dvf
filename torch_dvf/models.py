import torch
from .nn.mlp import MLP
from .nn.gnn import PointCloudPooling, pool, interp
from .data import Data


class PointNet(torch.nn.Module):

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        num_hierarchies: int,
        num_latent_channels: int
    ):
        super().__init__()

        self.point_cloud_pooling_layers = torch.nn.ModuleList()
        self.mlp_layers = torch.nn.ModuleList()

        self.point_cloud_pooling_layers.append(PointCloudPooling(MLP(
            (num_input_channels + 3, num_latent_channels, num_latent_channels),
            plain_last=False,
            use_norm_in_first=False
        )))
        self.mlp_layers.insert(0, MLP(
            (num_latent_channels + num_input_channels, *[num_latent_channels] * 2, num_output_channels),
            use_norm_in_first=False
        ))

        for _ in range(num_hierarchies - 1):
            self.point_cloud_pooling_layers.append(PointCloudPooling(MLP(
                (num_latent_channels + 3, num_latent_channels, num_latent_channels),
                plain_last=False
            )))
            self.mlp_layers.insert(0, MLP((2 * num_latent_channels, *[num_latent_channels] * 2, num_latent_channels)))

        print(f"PointNet++ ({sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)} parameters)")

    def forward(self, data: Data) -> torch.Tensor:
        x, pos = data.x, data.pos

        x_cache = []
        pos_cache = []

        for i, layer in enumerate(self.point_cloud_pooling_layers):
            x_cache.append(x)
            pos_cache.append(pos)

            x, pos = pool(layer, x, pos, data, scale_id=i)

        for layer in self.mlp_layers:
            x = interp(layer, x, x_cache.pop(), pos, pos := pos_cache.pop(), data, scale_id=i)
            i -= 1

        return x
