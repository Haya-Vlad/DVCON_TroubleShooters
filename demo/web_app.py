"""
TaskGraph-Edge Web UI
Interactive demo with image upload and task-aware object selection.
"""

import os
import sys
import json
import base64
import io
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload


# Recursively convert numpy types to native Python for JSON
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ─── Global pipeline (lazy loaded) ───
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("[WebApp] Loading TaskGraph-Edge pipeline...")
        from taskgraph_edge.language.task_encoder import TaskEncoder
        from taskgraph_edge.affordance.affordance_kb import AffordanceKnowledgeBase
        from taskgraph_edge.affordance.affordance_scorer import AffordanceScorer
        from taskgraph_edge.scene_graph.graph_builder import SceneGraphBuilder
        from taskgraph_edge.gnn.task_gnn import TaskConditionedGNN
        from taskgraph_edge.ranking.ranker import ObjectRanker
        from taskgraph_edge.config import load_config

        config = load_config()
        encoder = TaskEncoder(config.language)
        affordance_kb = AffordanceKnowledgeBase()
        affordance_scorer = AffordanceScorer(config.affordance, encoder)
        graph_builder = SceneGraphBuilder(config.scene_graph)
        gnn = TaskConditionedGNN(config.gnn)
        ranker = ObjectRanker(config.ranking)

        pipeline = {
            'encoder': encoder,
            'affordance_kb': affordance_kb,
            'affordance_scorer': affordance_scorer,
            'graph_builder': graph_builder,
            'gnn': gnn,
            'ranker': ranker,
            'config': config,
        }

        # Try to load detector
        try:
            from taskgraph_edge.detection.detector import ObjectDetector
            pipeline['detector'] = ObjectDetector(config.detection)
            print("[WebApp] YOLOv8 detector loaded")
        except Exception as e:
            print(f"[WebApp] Detector not available ({e}), using simulated detections")
            pipeline['detector'] = None

        # Try to connect FPGA
        pipeline['fpga'] = None
        if config.fpga.enabled:
            try:
                from taskgraph_edge.fpga_bridge import FPGABridge
                bridge = FPGABridge(config.fpga)
                if bridge.is_connected():
                    pipeline['fpga'] = bridge
                    print("[WebApp] ✓ FPGA accelerator connected")
                else:
                    print("[WebApp] FPGA enabled but not connected — using CPU fallback")
            except Exception as e:
                print(f"[WebApp] FPGA unavailable: {e} — using CPU fallback")

        print("[WebApp] Pipeline ready!")
    return pipeline


def simulate_detections(image_shape):
    """Generate simulated detections for demo."""
    from taskgraph_edge.detection.detector import Detection
    h, w = image_shape[:2]

    detections = [
        Detection(class_id=39, class_name="bottle", confidence=0.92,
                  bbox=[int(w*0.15), int(h*0.20), int(w*0.30), int(h*0.75)],
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=41, class_name="cup", confidence=0.85,
                  bbox=[int(w*0.55), int(h*0.40), int(w*0.72), int(h*0.70)],
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=42, class_name="fork", confidence=0.78,
                  bbox=[int(w*0.35), int(h*0.55), int(w*0.50), int(h*0.85)],
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=43, class_name="knife", confidence=0.81,
                  bbox=[int(w*0.72), int(h*0.30), int(w*0.88), int(h*0.80)],
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=56, class_name="chair", confidence=0.74,
                  bbox=[int(w*0.05), int(h*0.10), int(w*0.45), int(h*0.95)],
                  visual_features=np.random.randn(128).astype(np.float32)),
        Detection(class_id=58, class_name="potted plant", confidence=0.69,
                  bbox=[int(w*0.80), int(h*0.05), int(w*0.95), int(h*0.45)],
                  visual_features=np.random.randn(128).astype(np.float32)),
    ]
    return detections


def run_pipeline(image_array, task_text):
    """Run the full TaskGraph-Edge pipeline."""
    pipe = get_pipeline()
    timings = {}
    t0 = time.time()

    # ─── Detection ───
    t = time.time()
    if pipe['detector'] is not None:
        try:
            print(f"[Pipeline] Running YOLOv8 on image {image_array.shape}...")
            detections = pipe['detector'].detect(image_array)
            print(f"[Pipeline] YOLOv8 found {len(detections)} objects: "
                  f"{[d.class_name for d in detections]}")
        except Exception as e:
            print(f"[Pipeline] Detection failed: {e}, using simulated")
            import traceback
            traceback.print_exc()
            detections = simulate_detections(image_array.shape)
    else:
        print("[Pipeline] No detector loaded, using simulated detections")
        detections = simulate_detections(image_array.shape)
    timings['detection'] = (time.time() - t) * 1000

    if not detections:
        return {'error': 'No objects detected', 'detections': [], 'ranked': []}

    # ─── FPGA Feature Extraction ───
    fpga_active = False
    fpga_bridge = pipe.get('fpga')
    if fpga_bridge is not None and fpga_bridge.is_connected():
        t = time.time()
        import cv2
        # Use fixed-seed weights for consistent CNN features
        np.random.seed(42)
        cnn_weights = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1

        fpga_count = 0
        for det in detections:
            x1, y1, x2, y2 = det.bbox[:4]
            h, w = image_array.shape[:2]
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(w, int(x2)), min(h, int(y2))
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            # Crop and resize to 32×32×3 (what the FPGA expects)
            crop = image_array[cy1:cy2, cx1:cx2]
            patch = cv2.resize(crop, (32, 32)).astype(np.float32) / 255.0

            # Send to FPGA
            result = fpga_bridge.accelerate_cnn(patch, cnn_weights, mode=2)
            if result is not None:
                # FPGA output: flatten and pad/truncate to 128 dims
                feat = result.flatten()
                if len(feat) < 128:
                    feat = np.pad(feat, (0, 128 - len(feat)))
                else:
                    feat = feat[:128]
                det.visual_features = feat.astype(np.float32)
                fpga_count += 1

        timings['fpga_features'] = (time.time() - t) * 1000
        if fpga_count > 0:
            fpga_active = True
            print(f"[Pipeline] FPGA extracted features for {fpga_count}/{len(detections)} objects "
                  f"({timings['fpga_features']:.1f}ms)")
        else:
            print("[Pipeline] FPGA connected but feature extraction failed — using CPU features")
    else:
        if fpga_bridge is not None:
            print("[Pipeline] FPGA disconnected — using CPU features")
        timings['fpga_features'] = 0

    # ─── Task Matching & Encoding ───
    t = time.time()
    from taskgraph_edge.language.task_definitions import match_task_from_text, TASK_DEFINITIONS
    matched_task_name = match_task_from_text(task_text)
    matched_task_def = TASK_DEFINITIONS.get(matched_task_name, {})
    task_embedding = pipe['encoder'].encode(task_text)
    timings['encoding'] = (time.time() - t) * 1000

    # ─── Similarity ───
    t = time.time()
    similarities = []
    for det in detections:
        # Dual encoding: combine direct name similarity + descriptive similarity
        # Direct: captures word-level associations like "tea" → "cup"
        direct_emb = pipe['encoder'].encode(det.class_name)
        direct_sim = float(np.dot(task_embedding, direct_emb) /
                     (np.linalg.norm(task_embedding) * np.linalg.norm(direct_emb) + 1e-8))

        # Descriptive: captures functional associations like "a cup object that can be used for tasks"
        desc_emb = pipe['encoder'].encode_object_class(det.class_name)
        desc_sim = float(np.dot(task_embedding, desc_emb) /
                     (np.linalg.norm(task_embedding) * np.linalg.norm(desc_emb) + 1e-8))

        # Take the higher of the two — whichever captures the association better
        sim = max(direct_sim, desc_sim)
        similarities.append(max(0, sim))
    timings['similarity'] = (time.time() - t) * 1000

    # ─── Affordance ───
    t = time.time()
    affordance_results = []
    affordance_name_lists = []
    for det in detections:
        result = pipe['affordance_scorer'].score_object_for_task(
            det.class_name, matched_task_name, task_embedding
        )
        affordance_results.append(result)
        # Get affordance names from the KB
        affs = pipe['affordance_kb'].get_affordances(det.class_name)
        # affs is a list of (name, confidence) tuples
        if affs:
            affordance_name_lists.append([a[0] for a in affs[:5]])
        else:
            affordance_name_lists.append([])
    timings['affordance'] = (time.time() - t) * 1000

    # ─── Scene Graph ───
    t = time.time()
    scene_graph = pipe['graph_builder'].build(detections, image_size=image_array.shape[:2])
    timings['scene_graph'] = (time.time() - t) * 1000

    # ─── GNN ───
    t = time.time()
    node_features = scene_graph.get_node_features()   # (N, 153)
    edge_index = scene_graph.get_edge_index()          # (2, E)
    edge_features = scene_graph.get_edge_features()    # (E, 16)

    gnn_output = pipe['gnn'].predict(node_features, edge_index, edge_features, task_embedding)
    gnn_scores = gnn_output['scores']
    if gnn_scores.ndim == 0:
        gnn_scores = np.array([float(gnn_scores)])
    # Normalize
    gnn_min, gnn_max = gnn_scores.min(), gnn_scores.max()
    if gnn_max > gnn_min:
        gnn_scores = (gnn_scores - gnn_min) / (gnn_max - gnn_min)
    timings['gnn'] = (time.time() - t) * 1000

    # ─── Ranking ───
    t = time.time()
    w = pipe['config'].ranking.weights
    ranked = []
    max_conf = max(d.confidence for d in detections) or 1.0
    max_sim = max(similarities) if similarities else 1.0

    for i, det in enumerate(detections):
        det_score = det.confidence / max_conf
        task_score = similarities[i] / max_sim if max_sim > 0 else 0
        aff_score = affordance_results[i].get('combined_score', 0.0)
        ctx_score = float(gnn_scores[i]) if i < len(gnn_scores) else 0

        total = (w.detection_confidence * det_score +
                 w.task_similarity * task_score +
                 w.affordance_match * aff_score +
                 w.scene_context * ctx_score)

        ranked.append({
            'class_name': det.class_name,
            'class_id': det.class_id,
            'bbox': det.bbox,
            'confidence': float(det.confidence),
            'total_score': float(total),
            'det_score': float(det_score),
            'task_score': float(task_score),
            'aff_score': float(aff_score),
            'ctx_score': float(ctx_score),
            'affordances': affordance_name_lists[i],
        })
    timings['ranking'] = (time.time() - t) * 1000

    ranked.sort(key=lambda x: x['total_score'], reverse=True)
    timings['total'] = (time.time() - t0) * 1000

    return {
        'task': task_text,
        'matched_task': matched_task_name,
        'matched_description': matched_task_def.get('description', ''),
        'num_detections': len(detections),
        'num_edges': scene_graph.num_edges,
        'ranked': ranked,
        'timings': {k: round(v, 1) for k, v in timings.items()},
        'fpga_active': fpga_active,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get task text
        task_text = request.form.get('task', 'pour water')

        # Get image
        if 'image' in request.files and request.files['image'].filename:
            from PIL import Image
            img_file = request.files['image']
            img = Image.open(img_file.stream).convert('RGB')
            img_array = np.array(img)
        else:
            # Generate a placeholder image
            img_array = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

        # Run pipeline
        results = run_pipeline(img_array, task_text)

        # Encode image as base64 for display
        from PIL import Image
        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        results['image_b64'] = img_b64
        results['image_width'] = img_array.shape[1]
        results['image_height'] = img_array.shape[0]

        return jsonify(convert_numpy(results))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/tasks', methods=['GET'])
def get_tasks():
    """Return available task definitions."""
    from taskgraph_edge.language.task_definitions import TASK_DEFINITIONS
    tasks = [{'name': name, 'description': info['description']}
             for name, info in TASK_DEFINITIONS.items()]
    return jsonify(convert_numpy(tasks))


@app.route('/fpga_status', methods=['GET'])
def fpga_status():
    """Check real-time FPGA connection status."""
    pipe = get_pipeline()
    bridge = pipe.get('fpga')
    if bridge is not None and bridge.is_connected():
        stats = bridge.get_performance_stats()
        return jsonify({'connected': True, 'port': bridge.config.port, 'stats': convert_numpy(stats)})
    return jsonify({'connected': False, 'port': None, 'stats': {}})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  TaskSight AI — Scene Intelligence")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
