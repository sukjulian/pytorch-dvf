import torch
import torch_geometric as pyg

from e3nn import o3

from torch_dvf.models import PointNet, SEPointNet
from torch_dvf.transforms import PointCloudHierarchy


def test_pointnet_forward():

    n = 1000

    data = pyg.data.Data(x=torch.rand(n, 4), pos=torch.rand(n, 3))

    transform = PointCloudHierarchy(
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


def test_se_pointnet_forward():

    n = 1000

    data = pyg.data.Data(x=torch.rand(n, 4), pos=torch.rand(n, 3))

    transform = PointCloudHierarchy(
        rel_sampling_ratios=(0.333, 0.333, 0.333),
        cluster_radii=(0.10, 0.16, 0.25),
        interp_simplex="tetrahedron",
    )

    input_irreps = o3.Irreps("1x1o+1x0e")
    output_irreps = o3.Irreps("1x1o")

    model = SEPointNet(
        input_irreps,
        output_irreps,
        num_hierarchies=len(transform.rel_sampling_ratios),
        num_latent_channels=32,
    )

    random_rotation_matrix = o3.rand_matrix()

    def apply_rotation(data):
        data.x = data.x @ input_irreps.D_from_matrix(random_rotation_matrix).T
        data.pos = data.pos @ o3.Irreps("1x1o").D_from_matrix(random_rotation_matrix).T

        return data

    data = transform(data)

    # O(3)-equivariance: f(Rx) = Rf(x)
    f_Rx = model(apply_rotation(data.clone()))
    Rf_x = model(data) @ output_irreps.D_from_matrix(random_rotation_matrix).T

    assert (f_Rx - Rf_x).abs().mean() < 1e-5, "O(3)-equivariance seems to be broken."
