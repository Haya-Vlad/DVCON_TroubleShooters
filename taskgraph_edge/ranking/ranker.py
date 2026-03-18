"""
Object Ranker: multi-factor scoring with explainability.
Combines detection confidence, task similarity, affordance matching,
and scene context (GNN) into a final ranking.
Supports early-exit for high-confidence cases.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from taskgraph_edge.detection.detector import Detection


@dataclass
class RankedObject:
    """A ranked object with per-factor score breakdown and explanation."""
    rank: int
    detection: Detection
    total_score: float
    detection_score: float
    task_similarity_score: float
    affordance_score: float
    scene_context_score: float
    matched_affordances: List[str] = field(default_factory=list)
    explanation: str = ""


class ObjectRanker:
    """
    Multi-factor object ranker with explainability and early-exit.
    
    Score = α·detection_conf + β·task_similarity + γ·affordance_match + δ·scene_context
    """

    def __init__(self, config=None):
        from taskgraph_edge.config import RankingConfig
        self.config = config or RankingConfig()
        self.weights = self.config.weights

    def rank(
        self,
        detections: List[Detection],
        task_similarity_scores: np.ndarray,
        affordance_scores: List[Dict[str, float]],
        scene_context_scores: Optional[np.ndarray] = None,
        task_name: str = "",
    ) -> List[RankedObject]:
        """
        Rank objects by combined multi-factor score.
        
        Args:
            detections: List of Detection objects
            task_similarity_scores: (N,) cosine similarity between task and each object
            affordance_scores: List of dicts from AffordanceScorer
            scene_context_scores: (N,) GNN-produced context scores (optional)
            task_name: Task name for explanation generation
            
        Returns:
            List of RankedObject sorted by total_score descending
        """
        if not detections:
            return []

        n = len(detections)

        # Normalize all scores to [0, 1]
        det_scores = np.array([d.confidence for d in detections])
        det_scores = self._normalize(det_scores)

        task_sim = self._normalize(task_similarity_scores[:n])

        aff_scores = np.array([
            a.get("combined_score", 0.0) for a in affordance_scores[:n]
        ])

        if scene_context_scores is not None and len(scene_context_scores) >= n:
            ctx_scores = self._normalize(scene_context_scores[:n])
        else:
            ctx_scores = np.full(n, 0.5)

        # Compute weighted sum
        total = (
            self.weights.detection_confidence * det_scores +
            self.weights.task_similarity * task_sim +
            self.weights.affordance_match * aff_scores +
            self.weights.scene_context * ctx_scores
        )

        # Build ranked list
        ranked = []
        for i in range(n):
            matched_affs = affordance_scores[i].get("matched_affordances", [])
            explanation = self._generate_explanation(
                detections[i], total[i], det_scores[i], task_sim[i],
                aff_scores[i], ctx_scores[i], matched_affs, task_name,
            )

            obj = RankedObject(
                rank=0,  # set after sorting
                detection=detections[i],
                total_score=float(total[i]),
                detection_score=float(det_scores[i]),
                task_similarity_score=float(task_sim[i]),
                affordance_score=float(aff_scores[i]),
                scene_context_score=float(ctx_scores[i]),
                matched_affordances=matched_affs,
                explanation=explanation,
            )
            ranked.append(obj)

        # Sort by total score descending
        ranked.sort(key=lambda r: r.total_score, reverse=True)

        # Assign ranks
        for i, obj in enumerate(ranked):
            obj.rank = i + 1

        return ranked[:self.config.top_k]

    def check_early_exit(
        self,
        detections: List[Detection],
        task_similarity_scores: np.ndarray,
        affordance_scores: List[Dict[str, float]],
        task_name: str = "",
    ) -> Optional[RankedObject]:
        """
        Check if any object is so clearly the best that we can skip GNN reasoning.
        Uses only detection confidence, task similarity, and affordance scores.
        
        Returns:
            RankedObject if early exit triggered, None otherwise
        """
        if not detections:
            return None

        n = len(detections)

        for i in range(n):
            # Compute quick score without GNN
            det_sc = detections[i].confidence
            task_sc = float(task_similarity_scores[i]) if i < len(task_similarity_scores) else 0.0
            aff_sc = affordance_scores[i].get("combined_score", 0.0) if i < len(affordance_scores) else 0.0

            quick_score = (
                self.weights.detection_confidence * det_sc +
                self.weights.task_similarity * task_sc +
                self.weights.affordance_match * aff_sc
            ) / (1.0 - self.weights.scene_context)  # Normalize without context weight

            if quick_score > self.config.early_exit_threshold:
                matched_affs = affordance_scores[i].get("matched_affordances", [])
                return RankedObject(
                    rank=1,
                    detection=detections[i],
                    total_score=float(quick_score),
                    detection_score=float(det_sc),
                    task_similarity_score=float(task_sc),
                    affordance_score=float(aff_sc),
                    scene_context_score=0.0,  # skipped
                    matched_affordances=matched_affs,
                    explanation=self._generate_explanation(
                        detections[i], quick_score, det_sc, task_sc,
                        aff_sc, 0.0, matched_affs, task_name,
                        early_exit=True,
                    ),
                )

        return None

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        if len(scores) == 0:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.full_like(scores, 0.5)
        return (scores - min_val) / (max_val - min_val)

    def _generate_explanation(
        self,
        detection: Detection,
        total: float,
        det_sc: float,
        task_sc: float,
        aff_sc: float,
        ctx_sc: float,
        matched_affs: List[str],
        task_name: str,
        early_exit: bool = False,
    ) -> str:
        """Generate a human-readable explanation for the ranking."""
        parts = []
        parts.append(f"'{detection.class_name}' scored {total:.3f}")

        # Find dominant factor
        factors = {
            "high detection confidence": det_sc,
            "strong task relevance": task_sc,
            "matching affordances": aff_sc,
            "favorable scene context": ctx_sc,
        }

        # Sort by contribution
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        top_reason = sorted_factors[0][0]

        parts.append(f"primarily due to {top_reason}")

        if matched_affs:
            aff_str = ", ".join(matched_affs[:3])
            parts.append(f"(affordances: {aff_str})")

        if early_exit:
            parts.append("[early-exit: high confidence]")

        return " — ".join(parts)

    def get_ranking_summary(self, ranked: List[RankedObject]) -> str:
        """Generate a summary of the ranking results."""
        if not ranked:
            return "No objects detected."

        lines = ["=" * 50]
        lines.append("OBJECT RANKING RESULTS")
        lines.append("=" * 50)

        for obj in ranked:
            lines.append(f"\n#{obj.rank}: {obj.detection.class_name.upper()}")
            lines.append(f"  Total Score:      {obj.total_score:.4f}")
            lines.append(f"  Detection Conf:   {obj.detection_score:.4f}")
            lines.append(f"  Task Similarity:  {obj.task_similarity_score:.4f}")
            lines.append(f"  Affordance Match: {obj.affordance_score:.4f}")
            lines.append(f"  Scene Context:    {obj.scene_context_score:.4f}")
            if obj.matched_affordances:
                lines.append(f"  Affordances:      {', '.join(obj.matched_affordances)}")
            lines.append(f"  Reason: {obj.explanation}")

        lines.append("\n" + "=" * 50)
        best = ranked[0]
        lines.append(f"BEST OBJECT: {best.detection.class_name}")
        lines.append(f"CONFIDENCE:  {best.total_score:.4f}")
        lines.append("=" * 50)

        return "\n".join(lines)
