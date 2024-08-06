from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import networkx as nx
import torch
import torch_geometric as pyg

from torch_cluster import fps, knn, radius

from .data import Data


class PointCloudHierarchy(ABC):
    """Abstract class for nested hierarchy of sub-sampled point clouds. Graph edges connect each coarse-scale point to a cluster of fine-scale points.
    Interpolation from the coarse to the fine scales. For correct batching, "torch_geometric.data.Data.__inc__()" has to be overridden.

    Args:
        rel_sampling_ratios (Tuple[float]): relative ratios for successive farthest point sampling
        interp_simplex (str): reference simplex for interpolation ('triangle' or 'tetrahedron')
    """

    def __init__(self, rel_sampling_ratios: Tuple[float], interp_simplex: str):
        self.rel_sampling_ratios = rel_sampling_ratios
        self.interp_simplex = interp_simplex
        self.dim_interp_simplex = {"triangle": 2, "tetrahedron": 3}[interp_simplex]
        self.subsample_keys = ["pos", "batch"]

    def _init_state(self, data: pyg.data.Data) -> Dict[str, torch.Tensor]:
        pos = data.pos
        batch = (
            data.surface_id.long()
            if hasattr(data, "surface_id")
            else torch.zeros(pos.size(0), dtype=torch.long)
        )

        return {"pos": pos, "batch": batch}

    def _update_state(
        self, state: Dict[str, torch.Tensor], sampling_idcs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        for key in self.subsample_keys:
            state[key] = state[key][sampling_idcs]
        return state

    @abstractmethod
    def _get_cluster(
        self, scale_id: int, sampling_idcs: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor]:
        pass

    def __call__(self, data: pyg.data.Data) -> Data:
        state_dict = self._init_state(data)

        for scale_id, rel_sampling_ratio in enumerate(self.rel_sampling_ratios):
            pos, batch = state_dict["pos"], state_dict["batch"]
            sampling_idcs = fps(pos, batch, ratio=rel_sampling_ratio)

            pool_target, pool_source = self._get_cluster(
                scale_id, sampling_idcs, **state_dict
            )
            interp_target, interp_source = knn(
                pos[sampling_idcs],
                pos,
                self.dim_interp_simplex + 1,
                batch[sampling_idcs],
                batch,
            )

            (
                data[f"scale{scale_id}_pool_target"],
                data[f"scale{scale_id}_pool_source"],
            ) = (
                pool_target.int(),
                pool_source.int(),
            )
            (
                data[f"scale{scale_id}_interp_target"],
                data[f"scale{scale_id}_interp_source"],
            ) = (
                interp_target.int(),
                interp_source.int(),
            )
            data[f"scale{scale_id}_sampling_index"] = sampling_idcs.int()
            state_dict = self._update_state(state_dict, sampling_idcs)

        return Data(**data)


class RadiusPointCloudHierarchy(PointCloudHierarchy):
    """Clustering point cloud according to ball radius.

    Args:
        rel_sampling_ratios (Tuple[float]): relative ratios for successive farthest point sampling
        interp_simplex (str): reference simplex for interpolation ('triangle' or 'tetrahedron')
        cluster_radii (Tuple[float], optional): radii for spherical clusters, estimated from first seen data if None (default: None)
        max_num_neighbors (int): The maximum number of neighbors to return for each sampled element
    """

    def __init__(
        self,
        rel_sampling_ratios: Tuple[float],
        interp_simplex: str,
        cluster_radii: Optional[Tuple[float]] = None,
        max_num_neighbors: int = 32,
    ):
        super().__init__(rel_sampling_ratios, interp_simplex)
        self.max_num_neighbors = max_num_neighbors

        self.cluster_radii_arg = cluster_radii  # for "__repr__()"
        self.cluster_radii = (
            cluster_radii if cluster_radii else [None] * len(rel_sampling_ratios)
        )

    def _get_cluster(
        self,
        scale_id: int,
        sampling_idcs: torch.Tensor,
        pos: torch.tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:

        cluster_radius = self.cluster_radii[scale_id]

        if cluster_radius is None:
            cluster_radius = self._estimate_cluster_radius(
                pos, pos[sampling_idcs], batch, batch[sampling_idcs]
            )
            self.cluster_radii[scale_id] = cluster_radius

        pool_target, pool_source = radius(
            pos,
            pos[sampling_idcs],
            cluster_radius,
            batch,
            batch[sampling_idcs],
            max_num_neighbors=self.max_num_neighbors,
        )

        return pool_target, pool_source

    def _estimate_cluster_radius(
        self, pos_source, pos_target, batch_source, batch_target
    ) -> float:
        k = {"triangle": 7, "tetrahedron": 14}[self.interp_simplex]
        target_idcs, source_idcs = knn(
            pos_source, pos_target, k, batch_source, batch_target
        )

        return (
            (pos_source[source_idcs] - pos_target[target_idcs])
            .norm(dim=1)
            .quantile(0.75)
            .item()
        )

    def __repr__(self) -> str:
        repr_str = "{}(rel_sampling_ratios={}, interp_simplex={}, cluster_radii={}, max_num_neighbors={})".format(
            self.__class__.__name__,
            self.rel_sampling_ratios,
            self.interp_simplex,
            self.cluster_radii_arg,
            self.max_num_neighbors,
        )

        return repr_str


class SkeletonPointCloudHierarchy(PointCloudHierarchy):
    """Clustering point cloud according to underlying skeleton graph, based on:
    https://link.springer.com/chapter/10.1007/978-3-031-43990-2_73

    Args:
        rel_sampling_ratios (tuple): relative ratios for successive farthest point sampling
        cluster_dists (tuple): centerline distances for clusters
        interp_simplex (str): reference simplex for barycentric interpolation ('triangle' or 'tetrahedron')
        max_num_neighbors (int, optional): The maximum number of neighbors to return for each sampled element, return all if None (default: None)
    """

    def __init__(
        self,
        rel_sampling_ratios: tuple,
        cluster_dists: tuple,
        interp_simplex: str,
        max_num_neighbors: Optional[int] = None,
    ):
        super().__init__(rel_sampling_ratios, interp_simplex)

        self.cluster_dists = cluster_dists
        self.max_num_neighbors = max_num_neighbors
        self.subsample_keys += ["skeleton_map"]

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
        """Precomputing skeleton graph all-pair distance lookup table (LUT)"""

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

    def _init_state(
        self, data: pyg.data.Data
    ) -> Union[Dict[str, torch.Tensor], List[str]]:
        state = super()._init_state(data)
        dist_LUT, skeleton_map = self._skeleton_distance_LUT(
            data.pos, data.skeleton_pos, data.skeleton_edge_index
        )

        state["dist_LUT"] = dist_LUT
        state["skeleton_map"] = skeleton_map

        return state

    def _get_cluster(
        self,
        scale_id: int,
        sampling_idcs: torch.Tensor,
        batch: torch.Tensor,
        dist_LUT: torch.Tensor,
        skeleton_map: torch.Tensor,
        **kwargs,
    ) -> Union[torch.Tensor]:

        cluster_dist = self.cluster_dists[scale_id]
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

    def __repr__(self) -> str:
        repr_str = "{}(rel_sampling_ratios={}, cluster_dists={}, interp_simplex={}, max_num_neighbors={})".format(
            self.__class__.__name__,
            self.rel_sampling_ratios,
            self.cluster_dists,
            self.interp_simplex,
            self.max_num_neighbors,
        )

        return repr_str
