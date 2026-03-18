"""
TaskGraph-Edge Interactive Demo
Demonstrates the full pipeline with visualization.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_image(save_path: str) -> np.ndarray:
    """Create a synthetic sample image with objects for demo."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # light gray bg

    # Draw a table (brown rectangle)
    cv2.rectangle(img, (50, 250), (590, 400), (42, 82, 139), -1)
    cv2.rectangle(img, (50, 250), (590, 400), (30, 60, 100), 2)

    # Draw a bottle on the table (green)
    cv2.rectangle(img, (100, 150), (140, 260), (0, 150, 50), -1)
    cv2.rectangle(img, (110, 120), (130, 150), (0, 150, 50), -1)
    cv2.putText(img, "bottle", (85, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Draw a cup on the table (blue)
    cv2.rectangle(img, (200, 200), (260, 260), (180, 100, 30), -1)
    cv2.ellipse(img, (230, 200), (30, 8), 0, 0, 360, (180, 100, 30), -1)
    cv2.putText(img, "cup", (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Draw a knife (gray)
    cv2.rectangle(img, (320, 230), (410, 240), (150, 150, 150), -1)
    cv2.putText(img, "knife", (340, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Draw a potted plant (green pot + leaves)
    cv2.rectangle(img, (480, 190), (540, 260), (40, 80, 120), -1)
    cv2.circle(img, (510, 160), 40, (30, 170, 50), -1)
    cv2.putText(img, "plant", (485, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Draw a chair (outline)
    cv2.rectangle(img, (20, 300), (80, 450), (80, 60, 40), 2)
    cv2.rectangle(img, (20, 300), (80, 340), (100, 80, 50), -1)
    cv2.putText(img, "chair", (25, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    # Add title
    cv2.putText(img, "TaskGraph-Edge Demo Scene", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    return img


def visualize_result(image, result, save_path=None):
    """Visualize pipeline results on the image."""
    from taskgraph_edge.ranking.ranker import RankedObject

    vis = image.copy()
    h, w = vis.shape[:2]

    # Color scheme for rankings
    colors = [
        (0, 255, 0),   # #1 Green (best)
        (0, 200, 255), # #2 Cyan
        (0, 150, 255), # #3 Orange
        (100, 100, 255), # #4 Red-ish
        (150, 150, 150), # #5 Gray
    ]

    # Draw detections with ranking info
    for ranked_obj in result.ranked_objects:
        det = ranked_obj.detection
        rank = ranked_obj.rank
        color = colors[min(rank - 1, len(colors) - 1)]

        x1, y1, x2, y2 = det.bbox
        thickness = 3 if rank == 1 else 2

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label with rank and score
        label = f"#{rank} {det.class_name} ({ranked_obj.total_score:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Label background
        cv2.rectangle(vis, (x1, y1 - label_size[1] - 8),
                     (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Affordance text for top object
        if rank == 1 and ranked_obj.matched_affordances:
            aff_text = f"Affordances: {', '.join(ranked_obj.matched_affordances[:3])}"
            cv2.putText(vis, aff_text, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Task info panel
    panel_y = 10
    cv2.rectangle(vis, (w - 280, 5), (w - 5, 85), (0, 0, 0), -1)
    cv2.rectangle(vis, (w - 280, 5), (w - 5, 85), (0, 255, 0), 1)

    cv2.putText(vis, "TaskGraph-Edge", (w - 270, panel_y + 18),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, f"Task: {result.task_description[:25]}", (w - 270, panel_y + 38),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if result.best_object:
        cv2.putText(vis, f"Best: {result.best_object.detection.class_name}",
                   (w - 270, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    total_time = sum(result.timing.values())
    cv2.putText(vis, f"Latency: {total_time:.1f}ms", (w - 270, panel_y + 72),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"[Demo] Visualization saved: {save_path}")

    return vis


def run_demo_without_model():
    """
    Demo mode that simulates detection results when YOLOv8 model is not available.
    This is useful for demonstrating the pipeline logic without needing the model downloaded.
    """
    from taskgraph_edge.detection.detector import Detection
    from taskgraph_edge.language.task_encoder import TaskEncoder
    from taskgraph_edge.language.task_definitions import match_task_from_text, get_task_description, TASK_DEFINITIONS
    from taskgraph_edge.affordance.affordance_kb import AffordanceKnowledgeBase
    from taskgraph_edge.affordance.affordance_scorer import AffordanceScorer
    from taskgraph_edge.scene_graph.graph_builder import SceneGraphBuilder
    from taskgraph_edge.gnn.task_gnn import TaskConditionedGNN
    from taskgraph_edge.ranking.ranker import ObjectRanker, RankedObject
    from taskgraph_edge.config import load_config

    config = load_config()

    print("\n" + "=" * 60)
    print("  TaskGraph-Edge — Simulation Demo")
    print("  (Using simulated detections)")
    print("=" * 60)

    # Simulated detections (as if YOLOv8 detected these)
    simulated_detections = [
        Detection(class_id=39, class_name="bottle", confidence=0.92,
                  bbox=(100, 120, 140, 260),
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=41, class_name="cup", confidence=0.88,
                  bbox=(200, 200, 260, 260),
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=43, class_name="knife", confidence=0.85,
                  bbox=(320, 225, 410, 245),
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=54, class_name="potted plant", confidence=0.78,
                  bbox=(480, 130, 540, 260),
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=52, class_name="chair", confidence=0.82,
                  bbox=(20, 300, 80, 450),
                  visual_features=np.random.randn(128).astype(np.float32)),
    ]

    # Demo tasks
    demo_tasks = [
        "water the plant",
        "cut some food",
        "pour a drink",
        "sit down and rest",
        "reach the top shelf",
    ]

    print("\n[Init] Loading task encoder...")
    task_encoder = TaskEncoder(config.language)

    print("[Init] Loading affordance engine...")
    affordance_kb = AffordanceKnowledgeBase()
    affordance_scorer = AffordanceScorer(config.affordance, task_encoder=task_encoder)

    print("[Init] Loading scene graph builder...")
    graph_builder = SceneGraphBuilder(config.scene_graph)

    print("[Init] Loading GNN...")
    gnn = TaskConditionedGNN(config.gnn)
    print(f"[Init] GNN parameters: {gnn.count_parameters():,}")

    print("[Init] Loading ranker...")
    ranker = ObjectRanker(config.ranking)

    print("\n[Init] System ready!\n")

    for task_text in demo_tasks:
        print(f"\n{'─' * 60}")
        print(f"  TASK: \"{task_text}\"")
        print(f"{'─' * 60}")

        t_total_start = time.perf_counter()

        # Step 1: Match task
        task_name = match_task_from_text(task_text)
        task_desc = get_task_description(task_name)
        print(f"  Matched task: {task_name}")

        # Step 2: Encode task
        t0 = time.perf_counter()
        task_embedding = task_encoder.encode(task_desc)
        t_encode = (time.perf_counter() - t0) * 1000

        # Step 3: Encode objects and compute similarity
        t0 = time.perf_counter()
        object_embeddings = np.array([
            task_encoder.encode_object_class(d.class_name) for d in simulated_detections
        ])
        task_sim = task_encoder.batch_similarity(task_embedding, object_embeddings)
        t_sim = (time.perf_counter() - t0) * 1000

        # Step 4: Affordance scoring
        t0 = time.perf_counter()
        aff_scores = affordance_scorer.batch_score(
            [d.class_name for d in simulated_detections],
            task_name, task_embedding,
        )
        aff_vectors = [affordance_kb.get_affordance_vector(d.class_name) for d in simulated_detections]
        t_aff = (time.perf_counter() - t0) * 1000

        # Step 5: Scene graph
        t0 = time.perf_counter()
        scene_graph = graph_builder.build(simulated_detections, (480, 640), aff_vectors)
        t_graph = (time.perf_counter() - t0) * 1000

        # Step 6: GNN reasoning
        t0 = time.perf_counter()
        node_features = scene_graph.get_node_features()
        edge_index = scene_graph.get_edge_index()
        edge_features = scene_graph.get_edge_features()
        if edge_index.shape[1] == 0:
            edge_features = np.zeros((0, 16), dtype=np.float32)

        gnn_output = gnn.predict(node_features, edge_index, edge_features, task_embedding)
        context_scores = gnn_output['scores']
        t_gnn = (time.perf_counter() - t0) * 1000

        # Step 7: Ranking
        t0 = time.perf_counter()
        ranked = ranker.rank(
            simulated_detections, task_sim, aff_scores,
            context_scores, task_name,
        )
        t_rank = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        # Print results
        print(f"\n  Objects detected: {len(simulated_detections)}")
        print(f"  Scene graph: {scene_graph.num_nodes} nodes, {scene_graph.num_edges} edges")
        print()
        for obj in ranked:
            marker = "  ★" if obj.rank == 1 else "   "
            print(f"{marker} #{obj.rank}: {obj.detection.class_name:<15s} "
                  f"score={obj.total_score:.4f}  "
                  f"[det={obj.detection_score:.2f} task={obj.task_similarity_score:.2f} "
                  f"aff={obj.affordance_score:.2f} ctx={obj.scene_context_score:.2f}]")
            if obj.matched_affordances:
                print(f"       affordances: {', '.join(obj.matched_affordances)}")

        print(f"\n  Timing breakdown:")
        print(f"    Encoding:     {t_encode:6.1f} ms")
        print(f"    Similarity:   {t_sim:6.1f} ms")
        print(f"    Affordance:   {t_aff:6.1f} ms")
        print(f"    Scene Graph:  {t_graph:6.1f} ms")
        print(f"    GNN:          {t_gnn:6.1f} ms")
        print(f"    Ranking:      {t_rank:6.1f} ms")
        print(f"    TOTAL:        {t_total:6.1f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="TaskGraph-Edge Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --task "water plant"
  python demo.py --image photo.jpg --task "pour water"
  python demo.py --simulate
        """
    )
    parser.add_argument("--image", type=str, default=None,
                       help="Path to input image")
    parser.add_argument("--task", type=str, default="water the plant",
                       help="Task description")
    parser.add_argument("--simulate", action="store_true",
                       help="Run simulation demo without model")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config.yaml")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save visualization")

    args = parser.parse_args()

    if args.simulate:
        run_demo_without_model()
        return

    # Full pipeline demo
    try:
        from taskgraph_edge.pipeline import TaskGraphPipeline

        print("\n" + "=" * 60)
        print("  TaskGraph-Edge — Full Pipeline Demo")
        print("=" * 60 + "\n")

        pipeline = TaskGraphPipeline(config_path=args.config)

        # Load or create image
        if args.image and os.path.exists(args.image):
            image = cv2.imread(args.image)
            print(f"[Demo] Loaded image: {args.image}")
        else:
            sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
            sample_path = os.path.join(sample_dir, "demo_scene.jpg")
            image = create_sample_image(sample_path)
            print(f"[Demo] Created sample image: {sample_path}")

        # Run pipeline
        print(f"[Demo] Task: \"{args.task}\"")
        print("[Demo] Running pipeline...\n")

        result = pipeline.run(image, args.task)

        # Print results
        print(result.summary())

        # Print detailed ranking
        print(pipeline.ranker.get_ranking_summary(result.ranked_objects))

        # Visualize
        if result.ranked_objects:
            save_path = args.save or os.path.join(
                os.path.dirname(__file__), "output_result.jpg"
            )
            visualize_result(image, result, save_path)

        # Print system info
        print("\n[System Info]")
        info = pipeline.get_system_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"\n[Error] {e}")
        print("\n[Fallback] Running simulation demo instead...\n")
        run_demo_without_model()


if __name__ == "__main__":
    main()
