import torch_geometric as pyg
from .data import Data
import torch
from torch_cluster import fps, radius, knn


class PointCloudHierarchy():
    """Nested hierarchy of sub-sampled point clouds. Graph edges connect each coarse-scale point to a cluster of fine-scale points.
    Barycentric interpolation from the coarse to the fine scales. For correct batching, "torch_geometric.data.Data.__inc__()" has to be
    overridden.

    Args:
        rel_sampling_ratios (tuple): relative ratios for successive farthest point sampling
        cluster_radii (tuple): radii for spherical clusters
        interp_simplex (str): reference simplex for barycentric interpolation ('triangle' or 'tetrahedron')
    """

    def __init__(self, rel_sampling_ratios: tuple, cluster_radii: tuple, interp_simplex: str):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.cluster_radii = cluster_radii
        self.interp_simplex = interp_simplex

        self.dim_interp_simplex = {'triangle': 2, 'tetrahedron': 3}[interp_simplex]

    def __call__(self, data: pyg.data.Data) -> Data:

        pos = data.pos
        batch = data.surface_id.long() if hasattr(data, 'surface_id') else torch.zeros(pos.size(0), dtype=torch.long)

        for i, (sampling_ratio, cluster_radius) in enumerate(zip(self.rel_sampling_ratios, self.cluster_radii)):

            sampling_idcs = fps(pos, batch, ratio=sampling_ratio)  # takes some time but is worth it

            pool_target, pool_source = radius(pos, pos[sampling_idcs], cluster_radius, batch, batch[sampling_idcs])
            interp_target, interp_source = knn(pos[sampling_idcs], pos, self.dim_interp_simplex + 1, batch[sampling_idcs], batch)

            data[f'scale{i}_pool_target'], data[f'scale{i}_pool_source'] = pool_target.int(), pool_source.int()
            data[f'scale{i}_interp_target'], data[f'scale{i}_interp_source'] = interp_target.int(), interp_source.int()
            data[f'scale{i}_sampling_index'] = sampling_idcs.int()

            pos = pos[sampling_idcs]
            batch = batch[sampling_idcs]

        return Data(**data)

    def __repr__(self) -> str:

        repr_str = "{self.__class__.__name__}(rel_sampling_ratios={}, cluster_radii={}, interp_simplex={})".format(
            self.rel_sampling_ratios,
            self.cluster_radii,
            self.interp_simplex
        )

        return repr_str
