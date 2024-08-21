import lab_gatr.models.lab_gatr
import torch

from lab_gatr.models.lab_gatr import PointCloudPooling, interp
from lab_gatr.nn.mlp.geometric_algebra import MLP

from torch_dvf.data import Data
from torch_dvf.models import architectures
from torch_dvf.nn.gnn import pool


class GAPointNet(architectures.PointNetBase):
    def __init__(
        self,
        geometric_algebra_interface: object,
        num_hierarchies: int,
        num_latent_channels: int,
    ):
        self.geometric_algebra_interface = geometric_algebra_interface()

        num_input_channels = self.geometric_algebra_interface.num_input_channels
        num_output_channels = self.geometric_algebra_interface.num_output_channels

        num_input_scalars = self.geometric_algebra_interface.num_input_scalars
        num_output_scalars = self.geometric_algebra_interface.num_output_scalars

        point_cloud_pooling_layers = torch.nn.ModuleList()
        mlp_layers = torch.nn.ModuleList()

        point_cloud_pooling_layers.append(
            PointCloudPooling(
                MLP(
                    (num_input_channels + 1, num_latent_channels, num_latent_channels),
                    num_input_scalars,
                    num_output_scalars=4 * num_latent_channels,
                    plain_last=False,
                    use_norm_in_first=False,
                ),
                node_dim=0,
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
                num_input_scalars=4 * num_latent_channels + num_input_scalars,
                num_output_scalars=num_output_scalars,
                use_norm_in_first=False,
            ),
        )

        for _ in range(num_hierarchies - 1):
            point_cloud_pooling_layers.append(
                PointCloudPooling(
                    MLP(
                        (
                            num_latent_channels + 1,
                            num_latent_channels,
                            num_latent_channels,
                        ),
                        num_input_scalars=4 * num_latent_channels,
                        num_output_scalars=4 * num_latent_channels,
                        plain_last=False,
                    ),
                    node_dim=0,
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
                    num_input_scalars=(4 + 4) * num_latent_channels,
                    num_output_scalars=4 * num_latent_channels,
                ),
            )

        super().__init__(point_cloud_pooling_layers, mlp_layers)

    def forward(self, data: Data) -> torch.Tensor:
        multivectors, scalars = self.geometric_algebra_interface.embed(data)
        pos = data.pos

        multivectors_cache, scalars_cache = [], []
        pos_cache = []

        # Join reference for reflection equivariance
        reference_multivector = (
            lab_gatr.models.lab_gatr.Tokeniser.construct_reference_multivector(
                multivectors, data.batch
            )
        )

        reference_multivector_cache = []

        for i, layer in enumerate(self.point_cloud_pooling_layers):
            multivectors_cache.append(multivectors)
            scalars_cache.append(scalars)
            pos_cache.append(pos)

            reference_multivector_cache.append(reference_multivector)

            (multivectors, scalars), pos = pool(
                layer,
                multivectors,
                pos,
                data,
                scale_id=i,
                scalars=scalars,
                reference_multivector=reference_multivector,
            )
            reference_multivector = reference_multivector[
                data[f"scale{i}_sampling_index"]
            ]

        for layer in self.mlp_layers:
            multivectors, scalars = interp(
                layer,
                multivectors,
                multivectors_cache.pop(),
                scalars,
                scalars_cache.pop(),
                pos,
                pos := pos_cache.pop(),
                data,
                scale_id=i,
                reference_multivector=reference_multivector_cache.pop(),
            )
            i -= 1

        return self.geometric_algebra_interface.dislodge(multivectors, scalars)
