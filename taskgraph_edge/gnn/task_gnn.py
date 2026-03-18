"""
Task-Conditioned Graph Neural Network.
Fuses task semantics with scene graph node features for context-aware reasoning.
Architecture: [NodeInit → GAT(→128) → ReLU → GAT(→64) → NodeScore]
Total parameters: ~50K (extremely lightweight).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

from taskgraph_edge.gnn.gnn_layers import LightweightGATLayer, NodeScoringHead


class TaskConditionedGNN(nn.Module):
    """
    Task-Conditioned Graph Neural Network.
    
    Pipeline:
    1. Project node features to hidden dimension
    2. Concatenate task embedding to each node
    3. Run 2-layer GAT with edge features
    4. Score each node for task relevance
    """

    def __init__(self, config=None):
        super().__init__()
        from taskgraph_edge.config import GNNConfig
        cfg = config or GNNConfig()

        self.hidden_dim = cfg.hidden_dim
        self.output_dim = cfg.output_dim
        self.task_condition = cfg.task_condition

        # Node feature projection (153 raw dims → hidden)
        node_raw_dim = 153  # visual(128) + spatial(8) + class(16) + conf(1)
        task_dim = 384       # MiniLM embedding dim

        # Input projection
        if cfg.task_condition:
            self.input_proj = nn.Linear(node_raw_dim + task_dim, cfg.hidden_dim)
        else:
            self.input_proj = nn.Linear(node_raw_dim, cfg.hidden_dim)

        # GAT layers
        self.gat1 = LightweightGATLayer(
            in_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            edge_dim=16,  # edge feature dim from spatial_relations
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_edge_features=True,
        )

        self.gat2 = LightweightGATLayer(
            in_dim=cfg.hidden_dim,
            out_dim=cfg.output_dim,
            edge_dim=16,
            num_heads=min(cfg.num_heads, cfg.output_dim),  # ensure divisible
            dropout=cfg.dropout,
            use_edge_features=True,
        )

        # Node scoring head
        self.scorer = NodeScoringHead(cfg.output_dim, hidden_dim=32)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,      # (N, node_dim)
        edge_index: torch.Tensor,          # (2, E)
        edge_features: torch.Tensor,       # (E, 16)
        task_embedding: Optional[torch.Tensor] = None,  # (384,)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: (N, 153) raw node features
            edge_index: (2, E) edge indices
            edge_features: (E, 16) edge features
            task_embedding: (384,) task vector, broadcast to all nodes
            
        Returns:
            dict with:
              'scores': (N,) per-node relevance scores
              'embeddings': (N, output_dim) context-enriched node embeddings
        """
        N = node_features.size(0)

        # Concatenate task embedding to each node
        if self.task_condition and task_embedding is not None:
            task_expanded = task_embedding.unsqueeze(0).expand(N, -1)  # (N, 384)
            x = torch.cat([node_features, task_expanded], dim=-1)     # (N, 153+384)
        else:
            x = node_features

        # Input projection
        x = F.relu(self.input_proj(x))  # (N, hidden_dim)

        # GAT layer 1
        x = F.relu(self.gat1(x, edge_index, edge_features))  # (N, hidden_dim)

        # GAT layer 2
        x = self.gat2(x, edge_index, edge_features)  # (N, output_dim)

        # Score each node
        scores = self.scorer(x)  # (N,)

        return {
            'scores': scores,
            'embeddings': x,
        }

    @torch.no_grad()
    def predict(
        self,
        node_features_np: np.ndarray,
        edge_index_np: np.ndarray,
        edge_features_np: np.ndarray,
        task_embedding_np: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Numpy-friendly prediction (no gradient computation).
        Converts numpy arrays to tensors, runs forward, converts back.
        """
        self.eval()

        # Convert to tensors
        device = next(self.parameters()).device
        node_feat = torch.from_numpy(node_features_np).float().to(device)
        edge_idx = torch.from_numpy(edge_index_np).long().to(device)
        edge_feat = torch.from_numpy(edge_features_np).float().to(device)

        task_emb = None
        if task_embedding_np is not None:
            task_emb = torch.from_numpy(task_embedding_np).float().to(device)

        # Forward
        output = self.forward(node_feat, edge_idx, edge_feat, task_emb)

        return {
            'scores': output['scores'].cpu().numpy(),
            'embeddings': output['embeddings'].cpu().numpy(),
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, any]:
        """Return model architecture info."""
        return {
            "total_parameters": self.count_parameters(),
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "task_conditioned": self.task_condition,
            "layers": 2,
        }
