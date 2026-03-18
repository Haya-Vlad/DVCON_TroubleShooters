"""
Affordance Scorer: computes compatibility between task requirements
and object affordances using both hard and soft matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from taskgraph_edge.affordance.affordance_kb import (
    AffordanceKnowledgeBase, AFFORDANCE_TYPES, AFFORDANCE_TO_IDX
)
from taskgraph_edge.language.task_definitions import (
    TASK_DEFINITIONS, get_task_affordances
)


class AffordanceScorer:
    """
    Scores how well an object's affordances match a task's requirements.
    Uses two strategies:
    
    1. Hard matching: exact overlap between task-required and object affordances
    2. Soft matching: semantic similarity between affordance embeddings
    """

    def __init__(self, config=None, task_encoder=None):
        """
        Args:
            config: AffordanceConfig or None
            task_encoder: TaskEncoder instance for soft matching (optional)
        """
        from taskgraph_edge.config import AffordanceConfig
        self.config = config or AffordanceConfig()
        self.kb = AffordanceKnowledgeBase()
        self.task_encoder = task_encoder

        # Pre-compute affordance embeddings for soft matching
        self._affordance_embeddings = None
        if self.task_encoder is not None:
            self._precompute_affordance_embeddings()

    def _precompute_affordance_embeddings(self):
        """Pre-compute semantic embeddings for all affordance types."""
        if self.task_encoder is None:
            return

        # Create descriptive phrases for each affordance
        descriptions = []
        for aff in AFFORDANCE_TYPES:
            desc = aff.replace("_", " ")
            descriptions.append(f"an object that can {desc}")

        self._affordance_embeddings = self.task_encoder.encode(descriptions)

    def score_object_for_task(
        self,
        object_class: str,
        task_name: str,
        task_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Score how well an object matches a task's requirements.
        
        Args:
            object_class: COCO class name
            task_name: Task identifier or free-text
            task_embedding: Pre-computed task embedding (optional)
            
        Returns:
            Dict with 'hard_score', 'soft_score', 'combined_score', and details
        """
        # Get task affordance requirements
        task_affordances = get_task_affordances(task_name)

        # Get object affordances
        object_affs = self.kb.get_affordances(object_class)
        object_aff_names = set(a[0] for a in object_affs)
        object_aff_dict = {a[0]: a[1] for a in object_affs}

        # --- Hard Matching ---
        hard_score = self._hard_match(task_affordances, object_aff_names, object_aff_dict)

        # --- Soft Matching ---
        soft_score = self._soft_match(object_class, task_embedding)

        # --- Anti-Object Penalty ---
        penalty = self._anti_object_penalty(object_class, task_name)

        # --- Combined Score ---
        combined = (
            self.config.hard_match_weight * hard_score +
            self.config.soft_match_weight * soft_score
        ) * (1.0 - penalty)

        return {
            "hard_score": hard_score,
            "soft_score": soft_score,
            "combined_score": float(np.clip(combined, 0.0, 1.0)),
            "penalty": penalty,
            "matched_affordances": list(object_aff_names & set(task_affordances)),
            "missing_affordances": list(set(task_affordances) - object_aff_names),
        }

    def _hard_match(
        self,
        required_affordances: List[str],
        object_affordances: set,
        object_aff_confidences: Dict[str, float],
    ) -> float:
        """
        Compute hard matching score based on exact affordance overlap.
        Score = weighted_overlap / total_required
        """
        if not required_affordances:
            return 0.5  # neutral score if no requirements

        total_weight = 0.0
        matched_weight = 0.0

        for i, aff in enumerate(required_affordances):
            # Required affordances have higher weight
            weight = 1.0 if i < len(required_affordances) // 2 else 0.5
            total_weight += weight

            if aff in object_affordances:
                matched_weight += weight * object_aff_confidences.get(aff, 0.5)

        return matched_weight / max(total_weight, 1e-8)

    def _soft_match(
        self,
        object_class: str,
        task_embedding: Optional[np.ndarray],
    ) -> float:
        """
        Compute soft matching using semantic similarity between
        object affordance embedding and task embedding.
        """
        if task_embedding is None or self._affordance_embeddings is None:
            return 0.5  # neutral if no embeddings

        # Get object's affordance vector
        aff_vector = self.kb.get_affordance_vector(object_class)

        if aff_vector.sum() < 1e-8:
            return 0.1  # very low score for objects with no affordances

        # Weighted combination of affordance embeddings
        # Weight by object's affordance confidences
        aff_weights = aff_vector / (aff_vector.sum() + 1e-8)
        object_aff_embedding = (
            self._affordance_embeddings.T @ aff_weights
        )

        # Cosine similarity with task embedding
        norm_obj = np.linalg.norm(object_aff_embedding)
        norm_task = np.linalg.norm(task_embedding)

        if norm_obj < 1e-8 or norm_task < 1e-8:
            return 0.1

        similarity = float(np.dot(object_aff_embedding, task_embedding) / (norm_obj * norm_task))

        # Map from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    def _anti_object_penalty(self, object_class: str, task_name: str) -> float:
        """
        Apply penalty for objects that are explicitly marked as unsuitable.
        """
        task_def = TASK_DEFINITIONS.get(task_name)
        if task_def and object_class.lower() in task_def.get("anti_objects", []):
            return 0.5  # 50% penalty
        return 0.0

    def batch_score(
        self,
        object_classes: List[str],
        task_name: str,
        task_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, float]]:
        """
        Score multiple objects for the same task.
        
        Returns:
            List of score dicts, one per object
        """
        return [
            self.score_object_for_task(cls, task_name, task_embedding)
            for cls in object_classes
        ]
