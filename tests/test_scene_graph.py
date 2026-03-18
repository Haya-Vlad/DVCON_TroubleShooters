"""Unit tests for scene graph construction."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskgraph_edge.detection.detector import Detection
from taskgraph_edge.scene_graph.graph_builder import SceneGraphBuilder
from taskgraph_edge.scene_graph.spatial_relations import (
    compute_spatial_relation, compute_edge_features, get_relation_names
)


class TestSpatialRelations:
    def test_above_relation(self):
        bbox_a = (100, 50, 150, 100)   # higher (smaller y)
        bbox_b = (100, 200, 150, 250)  # lower
        rel = compute_spatial_relation(bbox_a, bbox_b, (480, 640))
        names = get_relation_names(rel)
        assert "above" in names

    def test_below_relation(self):
        bbox_a = (100, 300, 150, 350)
        bbox_b = (100, 50, 150, 100)
        rel = compute_spatial_relation(bbox_a, bbox_b, (480, 640))
        names = get_relation_names(rel)
        assert "below" in names

    def test_near_relation(self):
        bbox_a = (100, 100, 150, 150)
        bbox_b = (120, 120, 170, 170)
        rel = compute_spatial_relation(bbox_a, bbox_b, (480, 640))
        names = get_relation_names(rel)
        assert "near" in names

    def test_edge_features_shape(self):
        feat = compute_edge_features(
            (10, 20, 50, 80), (100, 100, 200, 200), (480, 640)
        )
        assert feat.shape == (16,)


class TestSceneGraphBuilder:
    @pytest.fixture
    def detections(self):
        return [
            Detection(class_id=39, class_name="bottle", confidence=0.9,
                     bbox=(100, 100, 150, 200),
                     visual_features=np.random.randn(128).astype(np.float32)),
            Detection(class_id=41, class_name="cup", confidence=0.85,
                     bbox=(200, 150, 250, 220),
                     visual_features=np.random.randn(128).astype(np.float32)),
            Detection(class_id=52, class_name="chair", confidence=0.8,
                     bbox=(400, 300, 500, 450),
                     visual_features=np.random.randn(128).astype(np.float32)),
        ]

    def test_build_graph(self, detections):
        builder = SceneGraphBuilder()
        graph = builder.build(detections, (480, 640))
        assert graph.num_nodes == 3
        assert graph.num_edges > 0

    def test_node_features_shape(self, detections):
        builder = SceneGraphBuilder()
        graph = builder.build(detections, (480, 640))
        features = graph.get_node_features()
        assert features.shape[0] == 3
        assert features.shape[1] == 153  # 128+8+16+1

    def test_edge_index_shape(self, detections):
        builder = SceneGraphBuilder()
        graph = builder.build(detections, (480, 640))
        edge_idx = graph.get_edge_index()
        assert edge_idx.shape[0] == 2  # [src, dst]

    def test_adjacency_matrix(self, detections):
        builder = SceneGraphBuilder()
        graph = builder.build(detections, (480, 640))
        adj = graph.get_adjacency_matrix()
        assert adj.shape == (3, 3)
        # Should be symmetric
        np.testing.assert_array_equal(adj, adj.T)

    def test_empty_detections(self):
        builder = SceneGraphBuilder()
        graph = builder.build([], (480, 640))
        assert graph.num_nodes == 0
        assert graph.num_edges == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
