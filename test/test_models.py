import torch
import torch_geometric as pyg

from torch_dvf.models import PointNet
from torch_dvf.transforms import RadiusPointCloudHierarchy


def test_pointnet_forward():
    n = 1000

    data = pyg.data.Data(x=torch.rand(n, 4), pos=torch.rand(n, 3))

    transform = RadiusPointCloudHierarchy(
        rel_sampling_ratios=(0.333, 0.333, 0.333),
        cluster_radii=(0.10, 0.16, 0.25),
        interp_simplex="tetrahedron",
    )

    c_out = 2

    model = PointNet(
        num_input_channels=data.x.size(1),
        num_output_channels=c_out,
        num_hierarchies=len(transform.rel_sampling_ratios),
        num_latent_channels=32,
    )

    assert model(transform(data)).shape == torch.Size((n, c_out))
