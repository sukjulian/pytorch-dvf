import numpy as np
import torch
import torch_geometric as pyg

from torch_dvf.transforms import RadiusPointCloudHierarchy, SkeletonPointCloudHierarchy


def test_radius_hierarchy():
    n = 1000
    data = pyg.data.Data(x=torch.rand(n, 4), pos=torch.rand(n, 3))

    data_trans = RadiusPointCloudHierarchy(
        rel_sampling_ratios=(0.333, 0.333, 0.333),
        cluster_radii=(0.10, 0.16, 0.25),
        interp_simplex="tetrahedron",
    )(data)

    assert np.all([f"scale{i}_sampling_index" in data_trans.keys() for i in range(3)])
    assert np.all([f"scale{i}_pool_target" in data_trans.keys() for i in range(3)])
    assert np.all([f"scale{i}_pool_source" in data_trans.keys() for i in range(3)])


def test_skeleton_hierarchy():
    n = 1000
    k = 100
    k_edge_index = torch.tensor([[i, i + 1] for i in range(k - 1)]).T

    data = pyg.data.Data(
        x=torch.rand(n, 4),
        pos=torch.rand(n, 3),
        skeleton_pos=torch.rand(k, 3),
        skeleton_edge_index=k_edge_index,
    )

    data_trans = SkeletonPointCloudHierarchy(
        rel_sampling_ratios=(0.333, 0.333, 0.333),
        cluster_dists=(0.10, 0.16, 0.25),
        interp_simplex="tetrahedron",
    )(data)

    assert np.all([f"scale{i}_sampling_index" in data_trans.keys() for i in range(3)])
    assert np.all([f"scale{i}_pool_target" in data_trans.keys() for i in range(3)])
    assert np.all([f"scale{i}_pool_source" in data_trans.keys() for i in range(3)])
