"""Unit tests for affordance knowledge base and scorer."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskgraph_edge.affordance.affordance_kb import (
    AffordanceKnowledgeBase, AFFORDANCE_TYPES, OBJECT_AFFORDANCES
)
from taskgraph_edge.affordance.affordance_scorer import AffordanceScorer
from taskgraph_edge.config import AffordanceConfig


class TestAffordanceKB:
    @pytest.fixture
    def kb(self):
        return AffordanceKnowledgeBase()

    def test_coverage_all_coco(self, kb):
        """KB should cover most COCO classes."""
        stats = kb.get_coverage_stats()
        assert stats["total_objects"] >= 75  # at least 75 of 80

    def test_affordance_vector_shape(self, kb):
        vec = kb.get_affordance_vector("bottle")
        assert vec.shape == (len(AFFORDANCE_TYPES),)

    def test_bottle_affordances(self, kb):
        affs = kb.get_affordance_names("bottle")
        assert "hold_liquid" in affs
        assert "pour_liquid" in affs
        assert "graspable" in affs

    def test_chair_affordances(self, kb):
        affs = kb.get_affordance_names("chair")
        assert "sit_on" in affs
        assert "support_weight" in affs

    def test_has_affordance(self, kb):
        has, conf = kb.has_affordance("knife", "cut")
        assert has is True
        assert conf > 0.8

    def test_find_objects_with_affordance(self, kb):
        objects = kb.find_objects_with_affordance("hold_liquid")
        names = [o[0] for o in objects]
        assert "bottle" in names
        assert "cup" in names

    def test_unknown_object(self, kb):
        affs = kb.get_affordances("nonexistent_object")
        assert affs == []


class TestAffordanceScorer:
    @pytest.fixture
    def scorer(self):
        return AffordanceScorer(AffordanceConfig())

    def test_score_range(self, scorer):
        result = scorer.score_object_for_task("bottle", "pour_water")
        assert 0.0 <= result["combined_score"] <= 1.0

    def test_bottle_good_for_pour(self, scorer):
        result = scorer.score_object_for_task("bottle", "pour_water")
        assert result["combined_score"] > 0.3

    def test_chair_bad_for_pour(self, scorer):
        result = scorer.score_object_for_task("chair", "pour_water")
        result_bottle = scorer.score_object_for_task("bottle", "pour_water")
        assert result["combined_score"] < result_bottle["combined_score"]

    def test_batch_scoring(self, scorer):
        results = scorer.batch_score(
            ["bottle", "cup", "chair", "knife"],
            "pour_water"
        )
        assert len(results) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
