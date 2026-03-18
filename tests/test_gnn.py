"""Unit tests for GNN module."""
import numpy as np
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskgraph_edge.gnn.gnn_layers import LightweightGATLayer, NodeScoringHead
from taskgraph_edge.gnn.task_gnn import TaskConditionedGNN
from taskgraph_edge.config import GNNConfig


class TestGATLayer:
    def test_forward_shape(self):
        layer = LightweightGATLayer(
            in_dim=64, out_dim=64, edge_dim=16, num_heads=4
        )
        x = torch.randn(5, 64)
        edge_idx = torch.tensor([[0,1,2,3], [1,2,3,4]])
        edge_feat = torch.randn(4, 16)
        out = layer(x, edge_idx, edge_feat)
        assert out.shape == (5, 64)

    def test_no_edges(self):
        layer = LightweightGATLayer(
            in_dim=32, out_dim=32, edge_dim=16, num_heads=4
        )
        x = torch.randn(3, 32)
        edge_idx = torch.zeros(2, 0, dtype=torch.long)
        out = layer(x, edge_idx)
        assert out.shape == (3, 32)


class TestNodeScoringHead:
    def test_output_range(self):
        head = NodeScoringHead(in_dim=64, hidden_dim=32)
        x = torch.randn(5, 64)
        scores = head(x)
        assert scores.shape == (5,)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestTaskConditionedGNN:
    @pytest.fixture
    def gnn(self):
        config = GNNConfig(
            hidden_dim=64, output_dim=32,
            num_layers=2, num_heads=4
        )
        return TaskConditionedGNN(config)

    def test_forward(self, gnn):
        N = 5
        node_feat = torch.randn(N, 153)  # node raw features
        edge_idx = torch.tensor([[0,1,2,3], [1,2,3,4]])
        edge_feat = torch.randn(4, 16)
        task_emb = torch.randn(384)

        output = gnn(node_feat, edge_idx, edge_feat, task_emb)
        assert output['scores'].shape == (N,)
        assert output['embeddings'].shape == (N, 32)

    def test_predict_numpy(self, gnn):
        N = 3
        node_feat = np.random.randn(N, 153).astype(np.float32)
        edge_idx = np.array([[0, 1], [1, 2]])
        edge_feat = np.random.randn(2, 16).astype(np.float32)
        task_emb = np.random.randn(384).astype(np.float32)

        output = gnn.predict(node_feat, edge_idx, edge_feat, task_emb)
        assert output['scores'].shape == (N,)

    def test_parameter_count(self, gnn):
        count = gnn.count_parameters()
        assert count > 0
        assert count < 200_000  # Should be lightweight (<200K params)
        print(f"GNN params: {count:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
