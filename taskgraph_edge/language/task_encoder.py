"""
Lightweight Task Language Encoder using all-MiniLM-L6-v2.
Encodes natural language task descriptions into 384-d embeddings.
Optimized for edge with caching and optional ONNX export.
"""

import numpy as np
from typing import Dict, List, Optional, Union


class TaskEncoder:
    """
    Encodes task descriptions into dense vector embeddings using
    all-MiniLM-L6-v2 (22.7M params, 384-d output, ~5x faster than BERT-base).
    """

    def __init__(self, config=None):
        """
        Args:
            config: LanguageConfig dataclass or None for defaults
        """
        from taskgraph_edge.config import LanguageConfig
        self.config = config or LanguageConfig()
        self.model = None
        self._cache: Dict[str, np.ndarray] = {}
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model)
            print(f"[TaskEncoder] Loaded {self.config.model} "
                  f"(dim={self.config.embedding_dim})")
        except ImportError:
            print("[TaskEncoder] WARNING: sentence-transformers not installed. "
                  "Using fallback word-hash embeddings.")
            self.model = None

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode one or more task descriptions into embeddings.
        
        Args:
            text: Single string or list of strings
            
        Returns:
            np.ndarray of shape (384,) for single input or (N, 384) for batch
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache
        for i, t in enumerate(text):
            t_lower = t.lower().strip()
            if self.config.cache_embeddings and t_lower in self._cache:
                embeddings.append(self._cache[t_lower])
            else:
                embeddings.append(None)
                uncached_texts.append(t)
                uncached_indices.append(i)

        # Encode uncached texts
        if uncached_texts:
            if self.model is not None:
                new_embeddings = self.model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if new_embeddings.ndim == 1:
                    new_embeddings = new_embeddings.reshape(1, -1)
            else:
                # Fallback: deterministic hash-based embeddings
                new_embeddings = np.array([
                    self._fallback_encode(t) for t in uncached_texts
                ])

            # Fill in results and cache
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                if self.config.cache_embeddings:
                    self._cache[text[idx].lower().strip()] = emb

        result = np.array(embeddings, dtype=np.float32)

        if single:
            return result[0]
        return result

    def encode_object_class(self, class_name: str) -> np.ndarray:
        """
        Encode a COCO object class name into a descriptive embedding.
        Uses a descriptive phrase for richer semantics.
        """
        # Create a descriptive phrase for better embeddings
        descriptive = f"a {class_name} object that can be used for tasks"
        return self.encode(descriptive)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def batch_similarity(
        self,
        query: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query and multiple candidates.
        
        Args:
            query: (D,) embedding
            candidates: (N, D) embeddings
            
        Returns:
            (N,) similarity scores
        """
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        cand_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
        return cand_norms @ query_norm

    def _fallback_encode(self, text: str) -> np.ndarray:
        """
        Deterministic fallback encoder when sentence-transformers is unavailable.
        Uses word hashing to create consistent embeddings.
        """
        dim = self.config.embedding_dim
        embedding = np.zeros(dim, dtype=np.float32)

        words = text.lower().split()
        for word in words:
            # Hash each word to a position + value
            h = hash(word) % (2 ** 31)
            np.random.seed(h)
            word_vec = np.random.randn(dim).astype(np.float32)
            embedding += word_vec

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding

    def get_cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
