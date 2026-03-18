"""
Lightweight Graph Neural Network layers — pure PyTorch, no PyG dependency.
Implements Graph Attention Network (GAT) with edge feature integration.
Designed for ~50K total parameters for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class LightweightGATLayer(nn.Module):
    """
    Graph Attention Layer with edge feature integration.
    Supports multi-head attention for richer representations.
    
    Pure PyTorch implementation — no PyTorch Geometric required.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_edge_features = use_edge_features

        assert out_dim % num_heads == 0, \
            f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # Edge feature projection
        if use_edge_features:
            self.W_edge = nn.Linear(edge_dim, num_heads, bias=False)

        # Attention parameters
        self.attn_scale = self.head_dim ** -0.5

        # Output projection
        self.W_out = nn.Linear(out_dim, out_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual connection (with projection if dims differ)
        self.residual_proj = None
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,      # (N, in_dim)
        edge_index: torch.Tensor,          # (2, E) - [src, dst]
        edge_features: Optional[torch.Tensor] = None,  # (E, edge_dim)
    ) -> torch.Tensor:
        """
        Forward pass of GAT layer.
        
        Args:
            node_features: (N, in_dim)
            edge_index: (2, E) source and destination indices
            edge_features: (E, edge_dim) optional edge features
            
        Returns:
            (N, out_dim) updated node features
        """
        N = node_features.size(0)
        H = self.num_heads
        D = self.head_dim

        # Linear projections
        Q = self.W_q(node_features).view(N, H, D)  # (N, H, D)
        K = self.W_k(node_features).view(N, H, D)  # (N, H, D)
        V = self.W_v(node_features).view(N, H, D)  # (N, H, D)

        if edge_index.size(1) == 0:
            # No edges — return projected features
            out = V.reshape(N, -1)
            out = self.W_out(out)
            return self.layer_norm(out)

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Compute attention scores
        q_dst = Q[dst]      # (E, H, D) - query from destination
        k_src = K[src]      # (E, H, D) - key from source
        attn = (q_dst * k_src).sum(dim=-1) * self.attn_scale  # (E, H)

        # Add edge feature bias
        if self.use_edge_features and edge_features is not None:
            edge_bias = self.W_edge(edge_features)  # (E, H)
            attn = attn + edge_bias

        # Softmax over neighbors (sparse)
        attn = self._sparse_softmax(attn, dst, N)  # (E, H)
        attn = self.dropout(attn)

        # Weighted aggregation
        v_src = V[src]  # (E, H, D)
        weighted = v_src * attn.unsqueeze(-1)  # (E, H, D)

        # Scatter-add to destination nodes
        out = torch.zeros(N, H, D, device=node_features.device, dtype=node_features.dtype)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted)
        out.scatter_add_(0, dst_expanded, weighted)

        # Reshape: (N, H, D) -> (N, out_dim)
        out = out.reshape(N, -1)

        # Output projection
        out = self.W_out(out)
        out = self.dropout(out)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(node_features)
        else:
            residual = node_features

        out = self.layer_norm(out + residual)
        return out

    def _sparse_softmax(
        self,
        attn: torch.Tensor,   # (E, H)
        indices: torch.Tensor, # (E,) destination indices
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax over sparse neighborhoods."""
        # Max trick for numerical stability
        max_vals = torch.zeros(num_nodes, attn.size(1),
                              device=attn.device, dtype=attn.dtype)
        max_vals.scatter_reduce_(
            0,
            indices.unsqueeze(-1).expand_as(attn),
            attn,
            reduce='amax',
            include_self=True,
        )
        attn = attn - max_vals[indices]

        # Exp
        attn_exp = torch.exp(attn)

        # Sum per node
        sum_vals = torch.zeros(num_nodes, attn.size(1),
                              device=attn.device, dtype=attn.dtype)
        sum_vals.scatter_add_(
            0,
            indices.unsqueeze(-1).expand_as(attn_exp),
            attn_exp,
        )

        # Normalize
        return attn_exp / (sum_vals[indices] + 1e-8)


class NodeScoringHead(nn.Module):
    """
    MLP head that produces per-node relevance scores.
    Takes context-enriched node embeddings and outputs a scalar score.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, in_dim)
        Returns:
            (N,) scores in [0, 1]
        """
        return self.mlp(node_features).squeeze(-1)
