from typing import Optional

import networkx as nx
import torch
import torch_geometric as pyg

from torch_cluster import fps, knn, radius

from .data import Data


class RadiusPointCloudHierarchy:
    """Nested hierarchy of sub-sampled point clouds. Graph edges connect each coarse-scale point to a cluster of fine-scale points.
    Barycentric interpolation from the coarse to the fine scales. For correct batching, "torch_geometric.data.Data.__inc__()" has to be
    overridden.

    Args:
        rel_sampling_ratios (tuple): relative ratios for successive farthest point sampling
        cluster_radii (tuple): radii for spherical clusters
        interp_simplex (str): reference simplex for barycentric interpolation ('triangle' or 'tetrahedron')
    """

    def __init__(
        self,
        rel_sampling_ratios: tuple,
        cluster_radii: tuple,
        interp_simplex: str,
        max_num_neighbors: int = 32,
    ):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.cluster_radii = cluster_radii
        self.interp_simplex = interp_simplex
        self.max_num_neighbors = max_num_neighbors

        self.dim_interp_simplex = {"triangle": 2, "tetrahedron": 3}[interp_simplex]

    def __call__(self, data: pyg.data.Data) -> Data:

        pos = data.pos
        batch = (
            data.surface_id.long()
            if hasattr(data, "surface_id")
            else torch.zeros(pos.size(0), dtype=torch.long)
        )

        for i, (sampling_ratio, cluster_radius) in enumerate(
            zip(self.rel_sampling_ratios, self.cluster_radii)
        ):

            sampling_idcs = fps(pos, batch, ratio=sampling_ratio)

            pool_target, pool_source = radius(
                pos,
                pos[sampling_idcs],
                cluster_radius,
                batch,
                batch[sampling_idcs],
                max_num_neighbors=self.max_num_neighbors,
            )
            interp_target, interp_source = knn(
                pos[sampling_idcs],
                pos,
                self.dim_interp_simplex + 1,
                batch[sampling_idcs],
                batch,
            )

            data[f"scale{i}_pool_target"], data[f"scale{i}_pool_source"] = (
                pool_target.int(),
                pool_source.int(),
            )
            data[f"scale{i}_interp_target"], data[f"scale{i}_interp_source"] = (
                interp_target.int(),
                interp_source.int(),
            )
            data[f"scale{i}_sampling_index"] = sampling_idcs.int()

            pos = pos[sampling_idcs]
            batch = batch[sampling_idcs]

        return Data(**data)

    def __repr__(self) -> str:

        repr_str = (
            "{}(rel_sampling_ratios={}, cluster_radii={}, interp_simplex={})".format(
                self.__class__.__name__,
                self.rel_sampling_ratios,
                self.cluster_radii,
                self.interp_simplex,
            )
        )

        return repr_str


class SkeletonPointCloudHierarchy:
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

    def __init__(
        self,
        rel_sampling_ratios: tuple,
        cluster_dists: tuple,
        interp_simplex: str,
        max_num_neighbors: Optional[int] = None,
    ):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.cluster_dists = cluster_dists
        self.interp_simplex = interp_simplex
        self.max_num_neighbors = max_num_neighbors

        self.dim_interp_simplex = {"triangle": 2, "tetrahedron": 3}[interp_simplex]

    def _construct_nx_graph(
        self, centerline_edge_index: torch.tensor, edge_lengths: torch.tensor
    ):
        graph = nx.DiGraph()
        graph.add_edges_from(centerline_edge_index.T.numpy())

        edge_lengths_dict = {
            tuple(edge_index): edge_length.item()
            for edge_index, edge_length in zip(
                centerline_edge_index.T.numpy(), edge_lengths
            )
        }
        nx.set_edge_attributes(graph, edge_lengths_dict, "edge_length")

        return graph.to_undirected()

    def _skeleton_distance_LUT(
        self,
        pos: torch.tensor,
        skeleton_pos: torch.tensor,
        skeleton_edge_index: torch.tensor,
    ):

        # Compute mapping from surface onto skeleton
        _, skeleton_map = knn(skeleton_pos, pos, 1)

        # Precompute all-pair skeleton distance LUT
        edge_lengths = torch.linalg.norm(
            skeleton_pos[skeleton_edge_index[0]] - skeleton_pos[skeleton_edge_index[1]],
            dim=1,
        )
        nx_graph = self._construct_nx_graph(skeleton_edge_index, edge_lengths)
        dist_LUT = torch.tensor(nx.floyd_warshall_numpy(nx_graph, weight="edge_length"))

        return dist_LUT, skeleton_map

    def _get_pooling_scale(
        self,
        dist_LUT: torch.tensor,
        skeleton_map: torch.tensor,
        sampling_idcs: torch.tensor,
        cluster_dist: float,
        batch: torch.tensor,
    ):
        skeleton_adj = dist_LUT <= cluster_dist

        pool_edge_index = []

        for i, sampling_idc in enumerate(sampling_idcs):
            skeleton_sampling_idc = skeleton_map[sampling_idc]

            sampling_idc_ego = torch.argwhere(skeleton_adj[skeleton_sampling_idc])
            pool_source = torch.argwhere(torch.isin(skeleton_map, sampling_idc_ego))
            pool_source = pool_source[batch[pool_source] == batch[sampling_idc]]

            if (
                self.max_num_neighbors is not None
                and pool_source.size(0) > self.max_num_neighbors
            ):
                pool_source = pool_source[
                    torch.randperm(pool_source.size(0))[: self.max_num_neighbors]
                ]

            pool_target = torch.ones_like(pool_source) * i
            pool_edge_index.append(torch.stack([pool_target, pool_source]))

        return torch.cat(pool_edge_index, -1)

    def __call__(self, data: pyg.data.Data) -> Data:
        pos = data.pos
        skeleton_pos = data.skeleton_pos
        skeleton_edge_index = data.skeleton_edge_index

        dist_LUT, skeleton_map = self._skeleton_distance_LUT(
            pos, skeleton_pos, skeleton_edge_index
        )
        batch = (
            data.surface_id.long()
            if hasattr(data, "surface_id")
            else torch.zeros(pos.size(0), dtype=torch.long)
        )

        for i, (sampling_ratio, cluster_dist) in enumerate(
            zip(self.rel_sampling_ratios, self.cluster_dists)
        ):
            sampling_idcs = fps(pos, batch, ratio=sampling_ratio)

            pool_target, pool_source = self._get_pooling_scale(
                dist_LUT, skeleton_map, sampling_idcs, cluster_dist, batch
            )
            interp_target, interp_source = knn(
                pos[sampling_idcs],
                pos,
                self.dim_interp_simplex + 1,
                batch[sampling_idcs],
                batch,
            )

            data[f"scale{i}_pool_target"], data[f"scale{i}_pool_source"] = (
                pool_target.int(),
                pool_source.int(),
            )
            data[f"scale{i}_interp_target"], data[f"scale{i}_interp_source"] = (
                interp_target.int(),
                interp_source.int(),
            )
            data[f"scale{i}_sampling_index"] = sampling_idcs.int()

            pos = pos[sampling_idcs]
            batch = batch[sampling_idcs]
            skeleton_map = skeleton_map[sampling_idcs]

        return Data(**data)

    def __repr__(self) -> str:

        repr_str = (
            "{}(rel_sampling_ratios={}, cluster_dists={}, interp_simplex={})".format(
                self.__class__.__name__,
                self.rel_sampling_ratios,
                self.cluster_dists,
                self.interp_simplex,
            )
        )

        return repr_str
