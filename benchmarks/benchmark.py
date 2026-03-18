"""
TaskGraph-Edge Performance Benchmark
Measures per-stage latency, throughput, and memory usage.
"""

import sys
import os
import time
import numpy as np

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


def create_simulated_detections(n=10):
    """Create N simulated detections."""
    classes = ["bottle", "cup", "knife", "chair", "potted plant",
               "book", "laptop", "bowl", "fork", "spoon"]
    dets = []
    for i in range(n):
        x1 = np.random.randint(0, 500)
        y1 = np.random.randint(0, 400)
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        dets.append(Detection(
            class_id=i, class_name=classes[i % len(classes)],
            confidence=np.random.uniform(0.5, 0.99),
            bbox=(x1, y1, x1+w, y1+h),
            visual_features=np.random.randn(128).astype(np.float32),
        ))
    return dets


def benchmark():
    config = load_config()

    print("=" * 60)
    print("  TaskGraph-Edge Performance Benchmark")
    print("=" * 60)

    # Initialize modules
    print("\n[Init] Loading modules...")
    encoder = TaskEncoder(config.language)
    kb = AffordanceKnowledgeBase()
    scorer = AffordanceScorer(config.affordance, task_encoder=encoder)
    graph_builder = SceneGraphBuilder(config.scene_graph)
    gnn = TaskConditionedGNN(config.gnn)
    ranker = ObjectRanker(config.ranking)

    print(f"[Init] GNN parameters: {gnn.count_parameters():,}")

    # Test configs
    num_objects_list = [3, 5, 10, 15, 20]
    num_runs = 20
    warmup = 5

    tasks = ["pour water", "cut food", "sit down"]

    results = {}

    for num_objects in num_objects_list:
        print(f"\n--- {num_objects} objects ---")
        timings = {
            "encoding": [], "similarity": [], "affordance": [],
            "scene_graph": [], "gnn": [], "ranking": [], "total": []
        }

        for run in range(num_runs + warmup):
            dets = create_simulated_detections(num_objects)
            task_text = tasks[run % len(tasks)]
            task_name = match_task_from_text(task_text)
            task_desc = get_task_description(task_name)

            t_total = time.perf_counter()

            # Encoding
            t0 = time.perf_counter()
            task_emb = encoder.encode(task_desc)
            obj_embs = np.array([encoder.encode_object_class(d.class_name) for d in dets])
            t_enc = (time.perf_counter() - t0) * 1000

            # Similarity
            t0 = time.perf_counter()
            task_sim = encoder.batch_similarity(task_emb, obj_embs)
            t_sim = (time.perf_counter() - t0) * 1000

            # Affordance
            t0 = time.perf_counter()
            aff_scores = scorer.batch_score([d.class_name for d in dets], task_name, task_emb)
            aff_vecs = [kb.get_affordance_vector(d.class_name) for d in dets]
            t_aff = (time.perf_counter() - t0) * 1000

            # Scene graph
            t0 = time.perf_counter()
            graph = graph_builder.build(dets, (480, 640), aff_vecs)
            t_graph = (time.perf_counter() - t0) * 1000

            # GNN
            t0 = time.perf_counter()
            node_feat = graph.get_node_features()
            edge_idx = graph.get_edge_index()
            edge_feat = graph.get_edge_features()
            if edge_idx.shape[1] == 0:
                edge_feat = np.zeros((0, 16), dtype=np.float32)
            gnn_out = gnn.predict(node_feat, edge_idx, edge_feat, task_emb)
            t_gnn = (time.perf_counter() - t0) * 1000

            # Ranking
            t0 = time.perf_counter()
            ranked = ranker.rank(dets, task_sim, aff_scores, gnn_out['scores'], task_name)
            t_rank = (time.perf_counter() - t0) * 1000

            t_tot = (time.perf_counter() - t_total) * 1000

            if run >= warmup:
                timings["encoding"].append(t_enc)
                timings["similarity"].append(t_sim)
                timings["affordance"].append(t_aff)
                timings["scene_graph"].append(t_graph)
                timings["gnn"].append(t_gnn)
                timings["ranking"].append(t_rank)
                timings["total"].append(t_tot)

        # Print results
        for stage, times in timings.items():
            avg = np.mean(times)
            std = np.std(times)
            print(f"  {stage:<15s}: {avg:7.2f} ± {std:5.2f} ms")

        results[num_objects] = {k: np.mean(v) for k, v in timings.items()}

    # Summary table
    print("\n" + "=" * 60)
    print("  Summary (avg ms)")
    print("=" * 60)
    print(f"{'Objects':<10s} {'Encode':<10s} {'Afford':<10s} {'Graph':<10s} {'GNN':<10s} {'Rank':<10s} {'TOTAL':<10s}")
    for n, t in results.items():
        print(f"{n:<10d} {t['encoding']:<10.2f} {t['affordance']:<10.2f} "
              f"{t['scene_graph']:<10.2f} {t['gnn']:<10.2f} {t['ranking']:<10.2f} "
              f"{t['total']:<10.2f}")


if __name__ == "__main__":
    benchmark()
