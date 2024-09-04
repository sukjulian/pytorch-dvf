import torch

from torch_dvf.models import architectures
from torch_dvf.nn.gnn import PointCloudPooling
from torch_dvf.nn.mlp.vanilla import MLP


class PointNet(architectures.PointNetBase):
    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        num_hierarchies: int,
        num_latent_channels: int,
        use_running_stats_in_norm: bool = True,
    ):

        point_cloud_pooling_layers = torch.nn.ModuleList()
        mlp_layers = torch.nn.ModuleList()

        point_cloud_pooling_layers.append(
            PointCloudPooling(
                MLP(
                    (num_input_channels + 3, num_latent_channels, num_latent_channels),
                    plain_last=False,
                    use_norm_in_first=False,
                    use_running_stats_in_norm=use_running_stats_in_norm,
                )
            )
        )
        mlp_layers.insert(
            0,
            MLP(
                (
                    num_latent_channels + num_input_channels,
                    *[num_latent_channels] * 2,
                    num_output_channels,
                ),
                use_norm_in_first=False,
                use_running_stats_in_norm=use_running_stats_in_norm,
            ),
        )

        for _ in range(num_hierarchies - 1):
            point_cloud_pooling_layers.append(
                PointCloudPooling(
                    MLP(
                        (
                            num_latent_channels + 3,
                            num_latent_channels,
                            num_latent_channels,
                        ),
                        plain_last=False,
                        use_running_stats_in_norm=use_running_stats_in_norm,
                    )
                )
            )
            mlp_layers.insert(
                0,
                MLP(
                    (
                        2 * num_latent_channels,
                        *[num_latent_channels] * 2,
                        num_latent_channels,
                    ),
                    use_running_stats_in_norm=use_running_stats_in_norm,
                ),
            )

        super().__init__(point_cloud_pooling_layers, mlp_layers)
