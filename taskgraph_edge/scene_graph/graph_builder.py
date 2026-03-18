"""
Scene Graph Builder: constructs a graph from detected objects.
Nodes = objects, Edges = spatial relationships.
Uses KNN connectivity for sparse, efficient graphs.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from taskgraph_edge.detection.detector import Detection
from taskgraph_edge.scene_graph.spatial_relations import (
    compute_edge_features, get_relation_names
)


@dataclass
class SceneNode:
    """A node in the scene graph representing a detected object."""
    idx: int
    detection: Detection
    features: np.ndarray = field(default=None, repr=False)  # Combined node features
    affordance_vector: Optional[np.ndarray] = field(default=None, repr=False)
    task_score: float = 0.0  # Filled in later by ranking


@dataclass
class SceneEdge:
    """An edge in the scene graph representing a spatial relationship."""
    src_idx: int
    dst_idx: int
    features: np.ndarray = field(default=None, repr=False)
    relation_names: List[str] = field(default_factory=list)


@dataclass
class SceneGraph:
    """
    Complete scene graph with nodes and edges.
    Provides adjacency and feature matrices for GNN processing.
    """
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    image_size: Tuple[int, int] = (640, 640)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_node_features(self) -> np.ndarray:
        """Return (N, D) matrix of node features."""
        if not self.nodes:
            return np.zeros((0, 0), dtype=np.float32)
        features = [n.features for n in self.nodes]
        return np.stack(features, axis=0)

    def get_edge_index(self) -> np.ndarray:
        """Return (2, E) edge index array [src, dst]."""
        if not self.edges:
            return np.zeros((2, 0), dtype=np.int64)
        src = [e.src_idx for e in self.edges]
        dst = [e.dst_idx for e in self.edges]
        return np.array([src, dst], dtype=np.int64)

    def get_edge_features(self) -> np.ndarray:
        """Return (E, D_edge) matrix of edge features."""
        if not self.edges:
            return np.zeros((0, 0), dtype=np.float32)
        features = [e.features for e in self.edges]
        return np.stack(features, axis=0)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return (N, N) adjacency matrix."""
        n = self.num_nodes
        adj = np.zeros((n, n), dtype=np.float32)
        for e in self.edges:
            adj[e.src_idx, e.dst_idx] = 1.0
            adj[e.dst_idx, e.src_idx] = 1.0  # undirected
        return adj

    def describe(self) -> str:
        """Return a human-readable description of the scene graph."""
        lines = [f"SceneGraph: {self.num_nodes} nodes, {self.num_edges} edges"]
        for node in self.nodes:
            lines.append(f"  [{node.idx}] {node.detection.class_name} "
                        f"(conf={node.detection.confidence:.2f})")
        for edge in self.edges:
            src_name = self.nodes[edge.src_idx].detection.class_name
            dst_name = self.nodes[edge.dst_idx].detection.class_name
            rels = ", ".join(edge.relation_names) if edge.relation_names else "connected"
            lines.append(f"  {src_name} --[{rels}]--> {dst_name}")
        return "\n".join(lines)


class SceneGraphBuilder:
    """
    Builds a scene graph from object detections.
    Uses KNN connectivity for sparse graph structure.
    """

    def __init__(self, config=None):
        from taskgraph_edge.config import SceneGraphConfig
        self.config = config or SceneGraphConfig()

    def build(
        self,
        detections: List[Detection],
        image_size: Tuple[int, int] = (640, 640),
        affordance_vectors: Optional[List[np.ndarray]] = None,
    ) -> SceneGraph:
        """
        Build a scene graph from detections.
        
        Args:
            detections: List of Detection objects
            image_size: (H, W) of the original image
            affordance_vectors: Optional pre-computed affordance vectors per object
            
        Returns:
            SceneGraph object
        """
        if not detections:
            return SceneGraph(nodes=[], edges=[], image_size=image_size)

        # 1. Create nodes
        nodes = self._create_nodes(detections, affordance_vectors)

        # 2. Create edges using KNN
        edges = self._create_edges(nodes, image_size)

        return SceneGraph(nodes=nodes, edges=edges, image_size=image_size)

    def _create_nodes(
        self,
        detections: List[Detection],
        affordance_vectors: Optional[List[np.ndarray]] = None,
    ) -> List[SceneNode]:
        """Create scene graph nodes from detections."""
        nodes = []
        for i, det in enumerate(detections):
            # Build node feature vector
            features = self._build_node_features(det)

            aff_vec = None
            if affordance_vectors is not None and i < len(affordance_vectors):
                aff_vec = affordance_vectors[i]

            node = SceneNode(
                idx=i,
                detection=det,
                features=features,
                affordance_vector=aff_vec,
            )
            nodes.append(node)
        return nodes

    def _build_node_features(self, detection: Detection) -> np.ndarray:
        """
        Build feature vector for a node combining:
        - Visual features from detector (128-d)
        - Spatial features (8-d)
        - Class one-hot (compressed to 16-d via hashing)
        """
        parts = []

        # Visual features
        if detection.visual_features is not None:
            parts.append(detection.visual_features)
        else:
            parts.append(np.zeros(128, dtype=np.float32))

        # Spatial features (normalized bbox)
        bbox = detection.bbox
        spatial = np.array([
            bbox[0] / 640.0, bbox[1] / 640.0,  # normalized x1, y1
            bbox[2] / 640.0, bbox[3] / 640.0,  # normalized x2, y2
            (bbox[2] - bbox[0]) / 640.0,         # width
            (bbox[3] - bbox[1]) / 640.0,         # height
            detection.center[0] / 640.0,          # center x
            detection.center[1] / 640.0,          # center y
        ], dtype=np.float32)
        parts.append(spatial)

        # Class embedding (hash-based, 16-d)
        class_hash = hash(detection.class_name) % (2 ** 31)
        np.random.seed(class_hash)
        class_emb = np.random.randn(16).astype(np.float32)
        class_emb /= np.linalg.norm(class_emb) + 1e-8
        parts.append(class_emb)

        # Confidence as feature
        parts.append(np.array([detection.confidence], dtype=np.float32))

        return np.concatenate(parts)  # 128 + 8 + 16 + 1 = 153 dims

    def _create_edges(
        self,
        nodes: List[SceneNode],
        image_size: Tuple[int, int],
    ) -> List[SceneEdge]:
        """Create edges using KNN connectivity based on spatial distance."""
        n = len(nodes)
        if n <= 1:
            return []

        # Compute pairwise distances between node centers
        centers = np.array([
            list(node.detection.center) for node in nodes
        ], dtype=np.float32)

        # Normalize by image size
        h, w = image_size
        centers[:, 0] /= max(w, 1)
        centers[:, 1] /= max(h, 1)

        # Compute distance matrix
        diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))    # (N, N)

        # KNN: for each node, connect to k nearest neighbors
        k = min(self.config.k_neighbors, n - 1)
        edges = []
        added_edges = set()

        for i in range(n):
            # Sort by distance, exclude self
            distances = dist_matrix[i]
            neighbors = np.argsort(distances)

            count = 0
            for j in neighbors:
                if j == i:
                    continue
                if count >= k:
                    break

                # Skip if distance too large
                if distances[j] > self.config.max_distance:
                    continue

                # Avoid duplicate edges
                edge_key = (min(i, j), max(i, j))
                if edge_key in added_edges:
                    count += 1
                    continue

                # Compute edge features
                edge_feat = compute_edge_features(
                    nodes[i].detection.bbox,
                    nodes[j].detection.bbox,
                    image_size,
                )
                rel_names = get_relation_names(edge_feat[:9])

                edge = SceneEdge(
                    src_idx=i,
                    dst_idx=j,
                    features=edge_feat,
                    relation_names=rel_names,
                )
                edges.append(edge)

                # Also add reverse edge for undirected graph
                edge_rev = SceneEdge(
                    src_idx=j,
                    dst_idx=i,
                    features=edge_feat,  # same features for undirected
                    relation_names=rel_names,
                )
                edges.append(edge_rev)

                added_edges.add(edge_key)
                count += 1

        return edges
