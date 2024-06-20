from typing import Tuple, Optional
import torch_geometric as pyg
from .data import Data
import torch
from torch_cluster import fps, radius, knn


class PointCloudHierarchy():
    """Nested hierarchy of sub-sampled point clouds. Graph edges connect each coarse-scale point to a cluster of fine-scale points.
    Interpolation from the coarse to the fine scales. For correct batching, "torch_geometric.data.Data.__inc__()" has to be overridden.

    Args:
        rel_sampling_ratios (Tuple[float]): relative ratios for successive farthest point sampling
        interp_simplex (str): reference simplex for interpolation ('triangle' or 'tetrahedron')
        cluster_radii (Tuple[float], optional): radii for spherical clusters, estimated from first seen data if None (default: None)
    """

    def __init__(self, rel_sampling_ratios: Tuple[float], interp_simplex: str, cluster_radii: Optional[Tuple[float]] = None):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.interp_simplex = interp_simplex

        self.cluster_radii_arg = cluster_radii  # for "__repr__()"
        self.cluster_radii = cluster_radii if cluster_radii else [None] * len(rel_sampling_ratios)

        self.dim_interp_simplex = {'triangle': 2, 'tetrahedron': 3}[interp_simplex]

    def __call__(self, data: pyg.data.Data) -> Data:

        pos = data.pos
        batch = data.surface_id.long() if hasattr(data, 'surface_id') else torch.zeros(pos.size(0), dtype=torch.long)

        for i, (sampling_ratio, cluster_radius) in enumerate(zip(self.rel_sampling_ratios, self.cluster_radii)):

            sampling_idcs = fps(pos, batch, ratio=sampling_ratio)  # takes some time but is worth it

            if cluster_radius is None:
                cluster_radius = self.estimate_cluster_radius(pos, pos[sampling_idcs], batch, batch[sampling_idcs])
                self.cluster_radii[i] = cluster_radius

            pool_target, pool_source = radius(pos, pos[sampling_idcs], cluster_radius, batch, batch[sampling_idcs])
            interp_target, interp_source = knn(pos[sampling_idcs], pos, self.dim_interp_simplex + 1, batch[sampling_idcs], batch)

            data[f'scale{i}_pool_target'], data[f'scale{i}_pool_source'] = pool_target.int(), pool_source.int()
            data[f'scale{i}_interp_target'], data[f'scale{i}_interp_source'] = interp_target.int(), interp_source.int()
            data[f'scale{i}_sampling_index'] = sampling_idcs.int()

            pos = pos[sampling_idcs]
            batch = batch[sampling_idcs]

        return Data(**data)

    def estimate_cluster_radius(self, pos_source, pos_target, batch_source, batch_target):
        k = {'triangle': 7, 'tetrahedron': 14}[self.interp_simplex]
        target_idcs, source_idcs = knn(pos_source, pos_target, k, batch_source, batch_target)

        return (pos_source[source_idcs] - pos_target[target_idcs]).norm(dim=1).quantile(0.75).item()

    def __repr__(self) -> str:

        repr_str = "{}(rel_sampling_ratios={}, interp_simplex={}, cluster_radii={})".format(
            self.__class__.__name__,
            self.rel_sampling_ratios,
            self.interp_simplex,
            self.cluster_radii_arg
        )

        return repr_str
