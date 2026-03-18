"""
TaskGraph-Edge Pipeline: End-to-end orchestrator.
Coordinates all modules: Detection → Encoding → Affordance → SceneGraph → GNN → Ranking.
Includes performance profiling and FPGA offloading support.
"""

import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from taskgraph_edge.config import load_config, TaskGraphConfig
from taskgraph_edge.detection.detector import ObjectDetector, Detection
from taskgraph_edge.language.task_encoder import TaskEncoder
from taskgraph_edge.language.task_definitions import (
    match_task_from_text, get_task_description, TASK_DEFINITIONS
)
from taskgraph_edge.affordance.affordance_kb import AffordanceKnowledgeBase
from taskgraph_edge.affordance.affordance_scorer import AffordanceScorer
from taskgraph_edge.scene_graph.graph_builder import SceneGraphBuilder, SceneGraph
from taskgraph_edge.gnn.task_gnn import TaskConditionedGNN
from taskgraph_edge.ranking.ranker import ObjectRanker, RankedObject


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""
    ranked_objects: List[RankedObject]
    best_object: Optional[RankedObject]
    scene_graph: Optional[SceneGraph]
    detections: List[Detection]
    task_name: str
    task_description: str
    timing: Dict[str, float] = field(default_factory=dict)
    early_exit: bool = False

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  TaskGraph-Edge Result")
        lines.append(f"{'='*60}")
        lines.append(f"  Task: {self.task_description}")
        lines.append(f"  Objects detected: {len(self.detections)}")
        if self.scene_graph:
            lines.append(f"  Scene graph: {self.scene_graph.num_nodes} nodes, "
                        f"{self.scene_graph.num_edges} edges")

        if self.best_object:
            lines.append(f"\n  ★ BEST OBJECT: {self.best_object.detection.class_name.upper()}")
            lines.append(f"    Score: {self.best_object.total_score:.4f}")
            lines.append(f"    Reason: {self.best_object.explanation}")
            if self.best_object.matched_affordances:
                lines.append(f"    Affordances: {', '.join(self.best_object.matched_affordances)}")
        else:
            lines.append("\n  ⚠ No suitable object found.")

        if self.early_exit:
            lines.append("\n  ⚡ Early exit triggered (high confidence)")

        if self.timing:
            lines.append(f"\n  Timing:")
            total = sum(self.timing.values())
            for stage, ms in self.timing.items():
                lines.append(f"    {stage:.<30s} {ms:.1f} ms")
            lines.append(f"    {'TOTAL':.<30s} {total:.1f} ms")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class TaskGraphPipeline:
    """
    End-to-end TaskGraph-Edge pipeline.
    
    Pipeline stages:
    1. Object Detection (YOLOv8n)
    2. Task Encoding (MiniLM)
    3. Affordance Scoring
    4. Scene Graph Construction
    5. GNN Reasoning (optional, skipped on early-exit)
    6. Object Ranking
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[TaskGraphConfig] = None):
        """
        Initialize all pipeline modules.
        
        Args:
            config_path: Path to config.yaml
            config: Pre-loaded config (takes precedence over config_path)
        """
        self.config = config or load_config(config_path)
        self._profiling = self.config.performance.profiling
        self._fpga_bridge = None

        print("[Pipeline] Initializing TaskGraph-Edge...")
        self._init_modules()
        print("[Pipeline] Ready.")

    def _init_modules(self):
        """Initialize all sub-modules."""
        # Module 1: Object Detector
        self.detector = ObjectDetector(self.config.detection)

        # Module 2: Task Encoder
        self.task_encoder = TaskEncoder(self.config.language)

        # Module 3: Affordance Engine
        self.affordance_kb = AffordanceKnowledgeBase()
        self.affordance_scorer = AffordanceScorer(
            self.config.affordance,
            task_encoder=self.task_encoder,
        )

        # Module 4: Scene Graph Builder
        self.graph_builder = SceneGraphBuilder(self.config.scene_graph)

        # Module 5: GNN
        self.gnn = TaskConditionedGNN(self.config.gnn)
        print(f"[Pipeline] GNN parameters: {self.gnn.count_parameters():,}")

        # Module 6: Ranker
        self.ranker = ObjectRanker(self.config.ranking)

        # Module 7: FPGA bridge (optional)
        if self.config.fpga.enabled:
            self._init_fpga()

    def _init_fpga(self):
        """Initialize FPGA communication bridge."""
        try:
            from taskgraph_edge.fpga_bridge import FPGABridge
            self._fpga_bridge = FPGABridge(self.config.fpga)
            print("[Pipeline] FPGA bridge initialized")
        except Exception as e:
            print(f"[Pipeline] FPGA bridge unavailable: {e}")
            self._fpga_bridge = None

    def run(
        self,
        image: np.ndarray,
        task_text: str,
        use_gnn: bool = True,
    ) -> PipelineResult:
        """
        Run the complete pipeline.
        
        Args:
            image: BGR image (H, W, 3) numpy array
            task_text: Free-form task description (e.g., "pour water")
            use_gnn: Whether to use GNN reasoning (can be disabled for speed)
            
        Returns:
            PipelineResult with ranked objects, explanations, and timing
        """
        timing = {}
        h, w = image.shape[:2]

        # ─── Stage 1: Object Detection ───
        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        timing["detection"] = (time.perf_counter() - t0) * 1000

        if not detections:
            return PipelineResult(
                ranked_objects=[], best_object=None, scene_graph=None,
                detections=[], task_name=task_text, task_description=task_text,
                timing=timing, early_exit=False,
            )

        # ─── Stage 2: Task Encoding ───
        t0 = time.perf_counter()
        # Match to predefined task if possible
        task_name = match_task_from_text(task_text)
        task_desc = get_task_description(task_name) if task_name in TASK_DEFINITIONS else task_text

        # Encode task
        task_embedding = self.task_encoder.encode(task_desc)

        # Encode object classes for similarity
        object_embeddings = np.array([
            self.task_encoder.encode_object_class(d.class_name)
            for d in detections
        ])
        timing["task_encoding"] = (time.perf_counter() - t0) * 1000

        # ─── Stage 3: Task-Object Similarity ───
        t0 = time.perf_counter()
        task_similarity_scores = self.task_encoder.batch_similarity(
            task_embedding, object_embeddings
        )
        timing["similarity"] = (time.perf_counter() - t0) * 1000

        # ─── Stage 4: Affordance Scoring ───
        t0 = time.perf_counter()
        affordance_scores = self.affordance_scorer.batch_score(
            [d.class_name for d in detections],
            task_name,
            task_embedding,
        )

        # Get affordance vectors for graph nodes
        affordance_vectors = [
            self.affordance_kb.get_affordance_vector(d.class_name)
            for d in detections
        ]
        timing["affordance"] = (time.perf_counter() - t0) * 1000

        # ─── Stage 4.5: Early Exit Check ───
        early_exit_result = self.ranker.check_early_exit(
            detections, task_similarity_scores, affordance_scores, task_name,
        )
        if early_exit_result is not None:
            return PipelineResult(
                ranked_objects=[early_exit_result],
                best_object=early_exit_result,
                scene_graph=None,
                detections=detections,
                task_name=task_name,
                task_description=task_desc,
                timing=timing,
                early_exit=True,
            )

        # ─── Stage 5: Scene Graph Construction ───
        t0 = time.perf_counter()
        scene_graph = self.graph_builder.build(
            detections, (h, w), affordance_vectors
        )
        timing["scene_graph"] = (time.perf_counter() - t0) * 1000

        # ─── Stage 6: GNN Reasoning ───
        scene_context_scores = None
        if use_gnn and scene_graph.num_nodes > 0:
            t0 = time.perf_counter()

            node_features = scene_graph.get_node_features()
            edge_index = scene_graph.get_edge_index()
            edge_features = scene_graph.get_edge_features()

            # Handle empty edge case
            if edge_index.shape[1] == 0:
                edge_features = np.zeros((0, 16), dtype=np.float32)

            gnn_output = self.gnn.predict(
                node_features, edge_index, edge_features, task_embedding,
            )
            scene_context_scores = gnn_output['scores']
            timing["gnn_reasoning"] = (time.perf_counter() - t0) * 1000

        # ─── Stage 7: Final Ranking ───
        t0 = time.perf_counter()
        ranked_objects = self.ranker.rank(
            detections, task_similarity_scores, affordance_scores,
            scene_context_scores, task_name,
        )
        timing["ranking"] = (time.perf_counter() - t0) * 1000

        best_object = ranked_objects[0] if ranked_objects else None

        return PipelineResult(
            ranked_objects=ranked_objects,
            best_object=best_object,
            scene_graph=scene_graph,
            detections=detections,
            task_name=task_name,
            task_description=task_desc,
            timing=timing,
            early_exit=False,
        )

    def run_from_file(self, image_path: str, task_text: str, **kwargs) -> PipelineResult:
        """Run pipeline from an image file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.run(image, task_text, **kwargs)

    def run_benchmark(
        self,
        image: np.ndarray,
        task_text: str,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Run the pipeline multiple times and return average timing.
        """
        timings = []
        for i in range(num_runs + self.config.performance.warmup_runs):
            result = self.run(image, task_text)
            if i >= self.config.performance.warmup_runs:
                timings.append(result.timing)

        # Average timings
        avg = {}
        if timings:
            for key in timings[0].keys():
                avg[key] = np.mean([t.get(key, 0) for t in timings])
            avg["total"] = sum(avg.values())

        return avg

    def get_system_info(self) -> Dict[str, any]:
        """Return system information for reporting."""
        return {
            "project": "TaskGraph-Edge",
            "version": "1.0.0",
            "detection_model": self.config.detection.model,
            "language_model": self.config.language.model,
            "gnn_params": self.gnn.count_parameters(),
            "gnn_info": self.gnn.get_model_info(),
            "affordance_coverage": self.affordance_kb.get_coverage_stats(),
            "fpga_enabled": self.config.fpga.enabled,
            "fpga_port": self.config.fpga.port if self.config.fpga.enabled else "N/A",
        }
