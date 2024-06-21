import torch_geometric as pyg
from .data import Data
import torch
from torch_cluster import fps, radius, knn
import networkx as nx


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
    
class SkeletonPointCloudHierarchy():
    """Nested hierarchy of sub-sampled point clouds. Graph edges connect each coarse-scale point to a cluster of fine-scale points.
    Barycentric interpolation from the coarse to the fine scales. For correct batching, "torch_geometric.data.Data.__inc__()" has to be
    overridden.

    Points are clustered based on the provided unerlying skeleton graph according to:
    https://link.springer.com/chapter/10.1007/978-3-031-43990-2_73

    Args:
        rel_sampling_ratios (tuple): relative ratios for successive farthest point sampling
        cluster_dists (tuple): centerline distances for clusters
        interp_simplex (str): reference simplex for barycentric interpolation ('triangle' or 'tetrahedron')
    """

    def __init__(self, rel_sampling_ratios: tuple, cluster_dists: tuple, interp_simplex: str):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.cluster_dists = cluster_dists
        self.interp_simplex = interp_simplex

        self.dim_interp_simplex = {'triangle': 2, 'tetrahedron': 3}[interp_simplex]

    def _construct_nx_graph(self, centerline_edge_index: torch.tensor, edge_lengths: torch.tensor):
            graph = nx.DiGraph()
            graph.add_edges_from(centerline_edge_index.T.numpy())
            nx.set_edge_attributes(graph, edge_lengths, "edge_length")

            return graph

    def _centerline_distance_LUT(
            self, 
            pos: torch.tensor, 
            centerline_pos: torch.tensor,
            centerline_edge_index: torch.tensor,
        ):
        _, pool_source = knn(pos, centerline_pos)
        edge_lengths = torch.linalg.norm(centerline_pos[centerline_edge_index[0]] - centerline_pos[centerline_edge_index[1]], dim=1)
        dist_mat = torch.tensor(nx.floyd_warshall_numpy(
            self._construct_nx_graph(centerline_edge_index, edge_lengths).to_undirected(),
            weight="edge_length"
        ))

        return dist_mat, pool_source
    
    def _get_pooling_scale(self, dist_LUT: torch.tensor, centerline_pool_source: torch.tensor, sampling_idcs: torch.tensor, cluster_dist: float):
        adj_mat = dist_LUT <= cluster_dist
        pool_source = centerline_pool_source[sampling_idcs]
        edge_index = torch.argwhere(adj_mat[pool_source])


    def __call__(self, data: pyg.data.Data) -> Data:

        pos = data.pos
        centerline_pos = data.centerline_pos
        centerline_edge_index = data.centerline_edge_index

        dist_LUT, centerline_pool_source = self._centerline_distance_LUT(pos, centerline_pos, centerline_edge_index)
        batch = data.surface_id.long() if hasattr(data, 'surface_id') else torch.zeros(pos.size(0), dtype=torch.long)

        for i, (sampling_ratio, cluster_dist) in enumerate(zip(self.rel_sampling_ratios, self.cluster_dists)):

            sampling_idcs = fps(pos, batch, ratio=sampling_ratio)  # takes some time but is worth it

            pool_target, pool_source = self._get_pooling_scale(pos, batch, sampling_idcs, cluster_dist)
            interp_target, interp_source = knn(pos[sampling_idcs], pos, self.dim_interp_simplex + 1, batch[sampling_idcs], batch)

            data[f'scale{i}_pool_target'], data[f'scale{i}_pool_source'] = pool_target.int(), pool_source.int()
            data[f'scale{i}_interp_target'], data[f'scale{i}_interp_source'] = interp_target.int(), interp_source.int()
            data[f'scale{i}_sampling_index'] = sampling_idcs.int()

            pos = pos[sampling_idcs]
            batch = batch[sampling_idcs]

        return Data(**data)

    def __repr__(self) -> str:

        repr_str = "{self.__class__.__name__}(rel_sampling_ratios={}, cluster_dists={}, interp_simplex={})".format(
            self.rel_sampling_ratios,
            self.cluster_dists,
            self.interp_simplex
        )

        return repr_str
