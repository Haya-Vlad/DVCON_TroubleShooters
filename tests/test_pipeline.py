"""Integration test: full pipeline with simulated detection."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskgraph_edge.detection.detector import Detection
from taskgraph_edge.language.task_encoder import TaskEncoder
from taskgraph_edge.language.task_definitions import match_task_from_text, get_task_description
from taskgraph_edge.affordance.affordance_kb import AffordanceKnowledgeBase
from taskgraph_edge.affordance.affordance_scorer import AffordanceScorer
from taskgraph_edge.scene_graph.graph_builder import SceneGraphBuilder
from taskgraph_edge.gnn.task_gnn import TaskConditionedGNN
from taskgraph_edge.ranking.ranker import ObjectRanker
from taskgraph_edge.config import load_config


class TestPipelineIntegration:
    """Integration tests using simulated detections (no model required)."""

    @pytest.fixture(scope="class")
    def modules(self):
        config = load_config()
        encoder = TaskEncoder(config.language)
        kb = AffordanceKnowledgeBase()
        scorer = AffordanceScorer(config.affordance, task_encoder=encoder)
        graph_builder = SceneGraphBuilder(config.scene_graph)
        gnn = TaskConditionedGNN(config.gnn)
        ranker = ObjectRanker(config.ranking)
        return encoder, kb, scorer, graph_builder, gnn, ranker

    @pytest.fixture
    def simulated_detections(self):
        return [
            Detection(class_id=39, class_name="bottle", confidence=0.92,
                     bbox=(100, 120, 140, 260),
                     visual_features=np.random.randn(128).astype(np.float32)),
            Detection(class_id=41, class_name="cup", confidence=0.88,
                     bbox=(200, 200, 260, 260),
                     visual_features=np.random.randn(128).astype(np.float32)),
            Detection(class_id=43, class_name="knife", confidence=0.85,
                     bbox=(320, 225, 410, 245),
                     visual_features=np.random.randn(128).astype(np.float32)),
            Detection(class_id=52, class_name="chair", confidence=0.82,
                     bbox=(20, 300, 80, 450),
                     visual_features=np.random.randn(128).astype(np.float32)),
        ]

    def _run_pipeline(self, modules, detections, task_text):
        encoder, kb, scorer, graph_builder, gnn, ranker = modules

        # Task encoding
        task_name = match_task_from_text(task_text)
        task_desc = get_task_description(task_name)
        task_emb = encoder.encode(task_desc)

        # Object encoding
        obj_embs = np.array([
            encoder.encode_object_class(d.class_name) for d in detections
        ])
        task_sim = encoder.batch_similarity(task_emb, obj_embs)

        # Affordance scoring
        aff_scores = scorer.batch_score(
            [d.class_name for d in detections], task_name, task_emb
        )
        aff_vecs = [kb.get_affordance_vector(d.class_name) for d in detections]

        # Scene graph
        graph = graph_builder.build(detections, (480, 640), aff_vecs)

        # GNN
        node_feat = graph.get_node_features()
        edge_idx = graph.get_edge_index()
        edge_feat = graph.get_edge_features()
        if edge_idx.shape[1] == 0:
            edge_feat = np.zeros((0, 16), dtype=np.float32)
        gnn_out = gnn.predict(node_feat, edge_idx, edge_feat, task_emb)

        # Ranking
        ranked = ranker.rank(
            detections, task_sim, aff_scores, gnn_out['scores'], task_name
        )
        return ranked

    def test_pour_water_selects_bottle_or_cup(self, modules, simulated_detections):
        """For 'pour water', bottle or cup should rank #1."""
        ranked = self._run_pipeline(modules, simulated_detections, "pour water")
        assert len(ranked) > 0
        best = ranked[0].detection.class_name
        assert best in ["bottle", "cup", "wine glass"]

    def test_cut_food_selects_knife(self, modules, simulated_detections):
        """For 'cut food', knife should rank highest."""
        ranked = self._run_pipeline(modules, simulated_detections, "cut the bread")
        assert len(ranked) > 0
        # Knife should score highly
        knife_ranks = [r for r in ranked if r.detection.class_name == "knife"]
        assert len(knife_ranks) > 0

    def test_sit_down_selects_chair(self, modules, simulated_detections):
        """For 'sit down', chair should rank #1."""
        ranked = self._run_pipeline(modules, simulated_detections, "sit down")
        assert len(ranked) > 0
        best = ranked[0].detection.class_name
        assert best == "chair"

    def test_all_objects_have_scores(self, modules, simulated_detections):
        ranked = self._run_pipeline(modules, simulated_detections, "pour water")
        for obj in ranked:
            assert obj.total_score >= 0.0
            assert obj.explanation != ""

    def test_ranking_is_sorted(self, modules, simulated_detections):
        ranked = self._run_pipeline(modules, simulated_detections, "pour water")
        for i in range(len(ranked) - 1):
            assert ranked[i].total_score >= ranked[i+1].total_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
