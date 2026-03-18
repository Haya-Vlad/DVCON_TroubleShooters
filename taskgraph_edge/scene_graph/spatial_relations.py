"""
Spatial relationship extraction between detected objects.
Computes geometric predicates for scene graph edges.
"""

import numpy as np
from typing import List, Tuple, Dict


# Spatial relation types
SPATIAL_RELATIONS = [
    "above",        # Object A is above B
    "below",        # Object A is below B
    "left_of",      # Object A is left of B
    "right_of",     # Object A is right of B
    "near",         # Objects are close together
    "far",          # Objects are far apart
    "overlapping",  # Bounding boxes overlap
    "inside",       # Object A is inside B
    "on_top_of",    # Object A is on top of B (above + overlapping)
]

RELATION_TO_IDX = {r: i for i, r in enumerate(SPATIAL_RELATIONS)}
NUM_RELATIONS = len(SPATIAL_RELATIONS)


def compute_spatial_relation(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
    image_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """
    Compute spatial relation vector between two bounding boxes.
    
    Args:
        bbox_a: (x1, y1, x2, y2) for object A
        bbox_b: (x1, y1, x2, y2) for object B
        image_size: (height, width) for normalization
        
    Returns:
        (9,) binary relation vector
    """
    h, w = image_size
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    # Centers
    cx_a, cy_a = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    cx_b, cy_b = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

    # Areas
    area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))

    # Normalized center distance
    dist = np.sqrt(((cx_a - cx_b) / w) ** 2 + ((cy_a - cy_b) / h) ** 2)

    # IoU overlap
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    iou = inter_area / (area_a + area_b - inter_area + 1e-8)

    # Containment ratio (how much of A is inside B)
    containment = inter_area / (area_a + 1e-8)

    # Compute relations
    relation = np.zeros(NUM_RELATIONS, dtype=np.float32)

    # Vertical relations (in image coords: y increases downward)
    if cy_a < cy_b - 0.02 * h:
        relation[RELATION_TO_IDX["above"]] = 1.0
    if cy_a > cy_b + 0.02 * h:
        relation[RELATION_TO_IDX["below"]] = 1.0

    # Horizontal relations
    if cx_a < cx_b - 0.02 * w:
        relation[RELATION_TO_IDX["left_of"]] = 1.0
    if cx_a > cx_b + 0.02 * w:
        relation[RELATION_TO_IDX["right_of"]] = 1.0

    # Distance relations
    if dist < 0.15:
        relation[RELATION_TO_IDX["near"]] = 1.0
    if dist > 0.4:
        relation[RELATION_TO_IDX["far"]] = 1.0

    # Overlap
    if iou > 0.1:
        relation[RELATION_TO_IDX["overlapping"]] = 1.0

    # Inside (A mostly contained in B)
    if containment > 0.7:
        relation[RELATION_TO_IDX["inside"]] = 1.0

    # On top of (above + overlapping)
    if cy_a < cy_b and iou > 0.05:
        relation[RELATION_TO_IDX["on_top_of"]] = 1.0

    return relation


def compute_edge_features(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
    image_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    """
    Compute comprehensive edge features between two objects.
    
    Returns:
        (16,) feature vector containing:
        - Spatial relations (9)
        - Normalized distance (1)
        - IoU overlap (1)
        - Relative position vector (2)
        - Size ratio (1)
        - Angle between centers (1)
        - Containment ratio (1)
    """
    h, w = image_size
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    # Spatial relations
    relations = compute_spatial_relation(bbox_a, bbox_b, image_size)

    # Centers
    cx_a, cy_a = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    cx_b, cy_b = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

    # Areas
    area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))

    # Normalized distance
    dist = np.sqrt(((cx_a - cx_b) / w) ** 2 + ((cy_a - cy_b) / h) ** 2)

    # IoU
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    iou = inter_area / (area_a + area_b - inter_area + 1e-8)

    # Relative position (normalized)
    rel_pos = np.array([(cx_b - cx_a) / w, (cy_b - cy_a) / h], dtype=np.float32)

    # Size ratio
    size_ratio = np.log(area_a / (area_b + 1e-8) + 1e-8)
    size_ratio = np.clip(size_ratio, -3, 3) / 3.0  # normalize

    # Angle between centers
    angle = np.arctan2(cy_b - cy_a, cx_b - cx_a) / np.pi  # [-1, 1]

    # Containment ratio
    containment = inter_area / (area_a + 1e-8)

    # Combine all features
    edge_feat = np.concatenate([
        relations,                           # 9 dims
        np.array([dist], dtype=np.float32),              # 1 dim
        np.array([iou], dtype=np.float32),               # 1 dim
        rel_pos,                             # 2 dims
        np.array([size_ratio], dtype=np.float32),        # 1 dim
        np.array([angle], dtype=np.float32),             # 1 dim
        np.array([containment], dtype=np.float32),       # 1 dim
    ])  # Total: 16 dims

    return edge_feat


def get_relation_names(relation_vector: np.ndarray) -> List[str]:
    """Convert a binary relation vector to human-readable names."""
    names = []
    for i, val in enumerate(relation_vector[:NUM_RELATIONS]):
        if val > 0.5:
            names.append(SPATIAL_RELATIONS[i])
    return names
