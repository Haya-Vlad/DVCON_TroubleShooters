"""Unit tests for object ranker."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskgraph_edge.detection.detector import Detection
from taskgraph_edge.ranking.ranker import ObjectRanker, RankedObject
from taskgraph_edge.config import RankingConfig


class TestObjectRanker:
    @pytest.fixture
    def ranker(self):
        return ObjectRanker()

    @pytest.fixture
    def sample_data(self):
        detections = [
            Detection(class_id=39, class_name="bottle", confidence=0.9,
                     bbox=(100, 100, 150, 200)),
            Detection(class_id=41, class_name="cup", confidence=0.85,
                     bbox=(200, 150, 250, 220)),
            Detection(class_id=43, class_name="knife", confidence=0.8,
                     bbox=(300, 200, 400, 220)),
        ]
        task_sim = np.array([0.8, 0.7, 0.3])
        aff_scores = [
            {"combined_score": 0.9, "matched_affordances": ["hold_liquid", "pour_liquid"]},
            {"combined_score": 0.8, "matched_affordances": ["hold_liquid"]},
            {"combined_score": 0.2, "matched_affordances": ["cut"]},
        ]
        context_scores = np.array([0.7, 0.6, 0.4])
        return detections, task_sim, aff_scores, context_scores

    def test_ranking_order(self, ranker, sample_data):
        dets, sim, aff, ctx = sample_data
        ranked = ranker.rank(dets, sim, aff, ctx, "pour_water")
        assert len(ranked) > 0
        assert ranked[0].rank == 1
        # Bottle should rank highest for pour_water
        assert ranked[0].detection.class_name == "bottle"

    def test_score_range(self, ranker, sample_data):
        dets, sim, aff, ctx = sample_data
        ranked = ranker.rank(dets, sim, aff, ctx)
        for obj in ranked:
            assert 0.0 <= obj.total_score <= 1.0

    def test_explainability(self, ranker, sample_data):
        dets, sim, aff, ctx = sample_data
        ranked = ranker.rank(dets, sim, aff, ctx, "pour_water")
        assert ranked[0].explanation != ""
        assert ranked[0].matched_affordances

    def test_empty_input(self, ranker):
        ranked = ranker.rank([], np.array([]), [], None)
        assert ranked == []

    def test_summary(self, ranker, sample_data):
        dets, sim, aff, ctx = sample_data
        ranked = ranker.rank(dets, sim, aff, ctx, "pour_water")
        summary = ranker.get_ranking_summary(ranked)
        assert "BEST OBJECT" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
