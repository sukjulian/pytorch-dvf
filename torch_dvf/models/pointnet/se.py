import torch

from e3nn.o3 import Irreps

from torch_dvf.models import architectures
from torch_dvf.nn.gnn import PointCloudPooling
from torch_dvf.nn.mlp.se import SEMLP


"""
The following class (BalancedIrreps) is copied from "E(3) Steerable GNN" on GitHub (https://github.com/RobDHess/Steerable-E3-GNN).
"""

def BalancedIrreps(lmax, vec_dim, sh_type=True):
    """Allocates irreps equally along channel budget, resulting
        in unequal numbers of irreps in ratios of 2l_i + 1 to 2l_j + 1.
    Parameters
    ----------
    lmax : int
        Maximum order of irreps.
    vec_dim : int
        Dim of feature vector.
    sh_type : bool
        if true, use spherical harmonics. Else the full set of irreps (with redundance).
    Returns
    -------
    Irreps
        Resulting irreps for feature vectors.
    """
    irrep_spec = "0e"
    for l in range(1, lmax + 1):
        if sh_type:
            irrep_spec += " + {0}".format(l) + ("e" if (l % 2) == 0 else "o")
        else:
            irrep_spec += " + {0}e + {0}o".format(l)
    irrep_spec_split = irrep_spec.split(" + ")
    dims = [int(irrep[0]) * 2 + 1 for irrep in irrep_spec_split]
    # Compute ratios
    ratios = [1 / dim for dim in dims]
    # Determine how many copies per irrep
    irrep_copies = [int(vec_dim * r / len(ratios)) for r in ratios]
    # Determine the current effective irrep sizes
    irrep_dims = [n * dim for (n, dim) in zip(irrep_copies, dims)]
    # Add trivial irreps until the desired size is reached
    irrep_copies[0] += vec_dim - sum(irrep_dims)

    # Convert to string
    str_out = ""
    for spec, dim in zip(irrep_spec_split, irrep_copies):
        str_out += str(dim) + "x" + spec
        str_out += " + "
    str_out = str_out[:-3]
    # Generate the irrep
    return Irreps(str_out)


class SEPointNet(architectures.PointNet):
    def __init__(
        self,
        input_irreps: Irreps,
        output_irreps: Irreps,
        num_hierarchies: int,
        num_latent_channels: int,
        max_order_latent_irreps: int = 1,
    ):

        point_cloud_pooling_layers = torch.nn.ModuleList()
        mlp_layers = torch.nn.ModuleList()

        latent_irreps = BalancedIrreps(
            lmax=max_order_latent_irreps, vec_dim=num_latent_channels
        )
        vector_irrep = Irreps("1x1o")

        point_cloud_pooling_layers.append(
            PointCloudPooling(
                SEMLP(
                    (input_irreps + vector_irrep, latent_irreps, latent_irreps),
                    plain_last=False,
                    use_norm_in_first=False,
                )
            )
        )
        mlp_layers.insert(
            0,
            SEMLP(
                (
                    latent_irreps + input_irreps,
                    *[latent_irreps] * 2,
                    output_irreps,
                ),
                use_norm_in_first=False,
            ),
        )

        for _ in range(num_hierarchies - 1):
            point_cloud_pooling_layers.append(
                PointCloudPooling(
                    SEMLP(
                        (
                            latent_irreps + vector_irrep,
                            latent_irreps,
                            latent_irreps,
                        ),
                        plain_last=False,
                    )
                )
            )
            mlp_layers.insert(
                0,
                SEMLP(
                    (
                        2 * latent_irreps,
                        *[latent_irreps] * 2,
                        latent_irreps,
                    )
                ),
            )

        super().__init__(point_cloud_pooling_layers, mlp_layers)
