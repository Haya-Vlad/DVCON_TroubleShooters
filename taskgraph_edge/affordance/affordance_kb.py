"""
Affordance Knowledge Base for all 80 COCO object classes.
Maps each object to its functional affordances with confidence scores.
This is the core innovation — affordance-based task matching.
"""

from typing import Dict, List, Tuple, Set


# ============================================================
# AFFORDANCE TYPES
# ============================================================
AFFORDANCE_TYPES = [
    "graspable",       # Can be picked up and held
    "pour_liquid",     # Can pour liquid from it
    "hold_liquid",     # Can contain liquid
    "support_weight",  # Can support weight of person/objects
    "contain",         # Can contain objects inside
    "cut",             # Has sharp edge for cutting
    "illuminate",      # Can produce light
    "transport",       # Can be used for transportation
    "protect",         # Can provide protection/shelter
    "sit_on",          # Can be sat on
    "step_on",         # Can be stepped on to gain height
    "lie_on",          # Can lie down on
    "open_close",      # Can be opened/closed
    "write_on",        # Can write on it
    "wear",            # Can be worn
    "electronic",      # Is an electronic device
    "heat",            # Can heat things
    "cool",            # Can cool things
    "clean",           # Can be used for cleaning
    "display_info",    # Can display information
    "throwable",       # Can be thrown
    "ride_on",         # Can be ridden
    "scoop",           # Can scoop up material
    "input_text",      # Can be used to input text
    "readable",        # Contains readable information
    "sharp_edge",      # Has a sharp edge
    "absorbent",       # Can absorb liquid
    "stable",          # Is stable and won't tip
    "comfortable",     # Provides comfort
    "portable",        # Can be easily carried
    "large_surface",   # Has a large flat surface
    "round",           # Has a round shape
    "lightweight",     # Is lightweight
    "wheeled",         # Has wheels
]

AFFORDANCE_TO_IDX = {a: i for i, a in enumerate(AFFORDANCE_TYPES)}


# ============================================================
# OBJECT → AFFORDANCE MAPPING (all 80 COCO classes)
# Format: {class_name: [(affordance, confidence), ...]}
# ============================================================
OBJECT_AFFORDANCES: Dict[str, List[Tuple[str, float]]] = {
    # --- Persons ---
    "person": [
        ("graspable", 0.1), ("support_weight", 0.3), ("transport", 0.4),
    ],

    # --- Vehicles ---
    "bicycle": [
        ("ride_on", 0.95), ("transport", 0.95), ("wheeled", 0.95),
        ("graspable", 0.4), ("step_on", 0.3),
    ],
    "car": [
        ("ride_on", 0.95), ("transport", 0.95), ("wheeled", 0.95),
        ("protect", 0.7), ("contain", 0.7), ("sit_on", 0.6),
    ],
    "motorcycle": [
        ("ride_on", 0.95), ("transport", 0.95), ("wheeled", 0.95),
        ("sit_on", 0.5),
    ],
    "airplane": [
        ("ride_on", 0.9), ("transport", 0.95), ("protect", 0.8),
        ("contain", 0.8),
    ],
    "bus": [
        ("ride_on", 0.95), ("transport", 0.95), ("wheeled", 0.95),
        ("protect", 0.8), ("contain", 0.9), ("sit_on", 0.7),
    ],
    "train": [
        ("ride_on", 0.95), ("transport", 0.95), ("wheeled", 0.9),
        ("protect", 0.8), ("contain", 0.9), ("sit_on", 0.7),
    ],
    "truck": [
        ("ride_on", 0.8), ("transport", 0.95), ("wheeled", 0.95),
        ("contain", 0.8), ("protect", 0.6),
    ],
    "boat": [
        ("ride_on", 0.9), ("transport", 0.9), ("contain", 0.6),
        ("sit_on", 0.5),
    ],

    # --- Outdoor objects ---
    "traffic light": [
        ("illuminate", 0.9), ("display_info", 0.7), ("electronic", 0.8),
    ],
    "fire hydrant": [
        ("pour_liquid", 0.7), ("hold_liquid", 0.5), ("stable", 0.8),
    ],
    "stop sign": [
        ("display_info", 0.9), ("readable", 0.8), ("stable", 0.7),
    ],
    "parking meter": [
        ("electronic", 0.6), ("display_info", 0.5), ("stable", 0.8),
    ],
    "bench": [
        ("sit_on", 0.95), ("support_weight", 0.95), ("step_on", 0.6),
        ("stable", 0.9), ("comfortable", 0.5), ("large_surface", 0.6),
    ],

    # --- Animals ---
    "bird": [("lightweight", 0.8), ("throwable", 0.1)],
    "cat": [("graspable", 0.3), ("portable", 0.3)],
    "dog": [("transport", 0.1)],
    "horse": [("ride_on", 0.9), ("transport", 0.8), ("support_weight", 0.8)],
    "sheep": [("wear", 0.2)],
    "cow": [("ride_on", 0.3), ("support_weight", 0.4)],
    "elephant": [("ride_on", 0.6), ("transport", 0.6), ("support_weight", 0.9)],
    "bear": [],
    "zebra": [("ride_on", 0.3)],
    "giraffe": [],

    # --- Accessories ---
    "backpack": [
        ("contain", 0.95), ("graspable", 0.8), ("portable", 0.95),
        ("wear", 0.9), ("protect", 0.3),
    ],
    "umbrella": [
        ("protect", 0.9), ("graspable", 0.9), ("portable", 0.8),
        ("support_weight", 0.1),
    ],
    "handbag": [
        ("contain", 0.9), ("graspable", 0.9), ("portable", 0.9),
        ("wear", 0.5),
    ],
    "tie": [
        ("wear", 0.9), ("graspable", 0.8), ("portable", 0.9),
    ],
    "suitcase": [
        ("contain", 0.95), ("graspable", 0.7), ("portable", 0.7),
        ("wheeled", 0.5), ("support_weight", 0.4), ("step_on", 0.3),
    ],

    # --- Sports ---
    "frisbee": [
        ("throwable", 0.95), ("graspable", 0.9), ("round", 0.9),
        ("lightweight", 0.9), ("portable", 0.9),
    ],
    "skis": [
        ("ride_on", 0.8), ("transport", 0.5), ("graspable", 0.5),
        ("step_on", 0.3),
    ],
    "snowboard": [
        ("ride_on", 0.85), ("transport", 0.4), ("graspable", 0.4),
        ("step_on", 0.3),
    ],
    "sports ball": [
        ("throwable", 0.95), ("graspable", 0.9), ("round", 0.95),
        ("lightweight", 0.7), ("portable", 0.9),
    ],
    "kite": [
        ("throwable", 0.5), ("graspable", 0.6), ("lightweight", 0.9),
        ("portable", 0.7),
    ],
    "baseball bat": [
        ("graspable", 0.9), ("support_weight", 0.3), ("cut", 0.1),
    ],
    "baseball glove": [
        ("graspable", 0.5), ("wear", 0.9), ("contain", 0.4),
        ("protect", 0.5),
    ],
    "skateboard": [
        ("ride_on", 0.9), ("transport", 0.7), ("wheeled", 0.9),
        ("step_on", 0.5), ("graspable", 0.5),
    ],
    "surfboard": [
        ("ride_on", 0.8), ("step_on", 0.5), ("large_surface", 0.6),
        ("graspable", 0.3),
    ],
    "tennis racket": [
        ("graspable", 0.9), ("throwable", 0.3), ("portable", 0.7),
    ],

    # --- Kitchen items ---
    "bottle": [
        ("hold_liquid", 0.95), ("pour_liquid", 0.95), ("graspable", 0.95),
        ("contain", 0.9), ("portable", 0.9), ("open_close", 0.7),
    ],
    "wine glass": [
        ("hold_liquid", 0.95), ("pour_liquid", 0.8), ("graspable", 0.9),
        ("contain", 0.7), ("portable", 0.8),
    ],
    "cup": [
        ("hold_liquid", 0.95), ("pour_liquid", 0.85), ("graspable", 0.95),
        ("contain", 0.8), ("portable", 0.9), ("scoop", 0.3),
    ],
    "fork": [
        ("graspable", 0.95), ("scoop", 0.6), ("cut", 0.3),
        ("portable", 0.9), ("sharp_edge", 0.3),
    ],
    "knife": [
        ("cut", 0.95), ("graspable", 0.9), ("sharp_edge", 0.95),
        ("portable", 0.8),
    ],
    "spoon": [
        ("graspable", 0.95), ("scoop", 0.95), ("portable", 0.9),
        ("hold_liquid", 0.3), ("pour_liquid", 0.2),
    ],
    "bowl": [
        ("contain", 0.95), ("hold_liquid", 0.9), ("graspable", 0.8),
        ("pour_liquid", 0.6), ("scoop", 0.2), ("portable", 0.7),
    ],

    # --- Food ---
    "banana": [
        ("graspable", 0.9), ("portable", 0.9), ("throwable", 0.5),
        ("lightweight", 0.8),
    ],
    "sandwich": [
        ("graspable", 0.8), ("portable", 0.7),
    ],
    "hot dog": [
        ("graspable", 0.7), ("portable", 0.7),
    ],
    "pizza": [
        ("graspable", 0.5), ("large_surface", 0.4), ("round", 0.6),
    ],
    "donut": [
        ("graspable", 0.8), ("portable", 0.8), ("round", 0.8),
        ("lightweight", 0.7),
    ],
    "cake": [
        ("graspable", 0.4), ("cut", 0.1), ("large_surface", 0.3),
    ],

    # --- Furniture ---
    "chair": [
        ("sit_on", 0.95), ("support_weight", 0.95), ("step_on", 0.7),
        ("stable", 0.8), ("comfortable", 0.6), ("graspable", 0.3),
    ],
    "couch": [
        ("sit_on", 0.95), ("lie_on", 0.85), ("support_weight", 0.95),
        ("comfortable", 0.9), ("stable", 0.9), ("large_surface", 0.7),
    ],
    "potted plant": [
        ("graspable", 0.5), ("portable", 0.5), ("contain", 0.3),
    ],
    "bed": [
        ("lie_on", 0.95), ("sit_on", 0.8), ("support_weight", 0.95),
        ("comfortable", 0.95), ("stable", 0.9), ("large_surface", 0.9),
    ],
    "dining table": [
        ("support_weight", 0.95), ("large_surface", 0.95), ("stable", 0.95),
        ("step_on", 0.5), ("write_on", 0.6),
    ],
    "toilet": [
        ("sit_on", 0.9), ("hold_liquid", 0.6), ("contain", 0.5),
        ("stable", 0.9), ("clean", 0.2),
    ],

    # --- Electronics ---
    "tv": [
        ("display_info", 0.95), ("electronic", 0.95), ("readable", 0.7),
        ("illuminate", 0.5),
    ],
    "laptop": [
        ("display_info", 0.95), ("electronic", 0.95), ("input_text", 0.95),
        ("readable", 0.8), ("portable", 0.8), ("graspable", 0.5),
    ],
    "mouse": [
        ("graspable", 0.9), ("electronic", 0.8), ("input_text", 0.5),
        ("portable", 0.9), ("lightweight", 0.8),
    ],
    "remote": [
        ("graspable", 0.95), ("electronic", 0.9), ("input_text", 0.3),
        ("portable", 0.95), ("lightweight", 0.8),
    ],
    "keyboard": [
        ("input_text", 0.95), ("electronic", 0.9), ("graspable", 0.4),
        ("portable", 0.5),
    ],
    "cell phone": [
        ("electronic", 0.95), ("graspable", 0.95), ("portable", 0.95),
        ("display_info", 0.8), ("input_text", 0.7), ("readable", 0.6),
        ("lightweight", 0.8),
    ],

    # --- Appliances ---
    "microwave": [
        ("heat", 0.95), ("electronic", 0.9), ("contain", 0.8),
        ("open_close", 0.9), ("stable", 0.8),
    ],
    "oven": [
        ("heat", 0.95), ("contain", 0.8), ("open_close", 0.9),
        ("stable", 0.9), ("electronic", 0.7),
    ],
    "toaster": [
        ("heat", 0.9), ("electronic", 0.9), ("graspable", 0.5),
        ("portable", 0.5),
    ],
    "sink": [
        ("hold_liquid", 0.8), ("pour_liquid", 0.7), ("clean", 0.9),
        ("contain", 0.6), ("stable", 0.9),
    ],
    "refrigerator": [
        ("cool", 0.95), ("contain", 0.95), ("open_close", 0.95),
        ("stable", 0.95), ("electronic", 0.7),
    ],

    # --- Indoor objects ---
    "book": [
        ("readable", 0.95), ("display_info", 0.8), ("graspable", 0.9),
        ("portable", 0.9), ("write_on", 0.3), ("lightweight", 0.6),
    ],
    "clock": [
        ("display_info", 0.9), ("readable", 0.8), ("portable", 0.5),
        ("round", 0.7),
    ],
    "vase": [
        ("hold_liquid", 0.9), ("contain", 0.9), ("pour_liquid", 0.7),
        ("graspable", 0.7), ("portable", 0.6),
    ],
    "scissors": [
        ("cut", 0.95), ("graspable", 0.9), ("sharp_edge", 0.9),
        ("portable", 0.8), ("open_close", 0.5),
    ],
    "teddy bear": [
        ("graspable", 0.9), ("portable", 0.8), ("comfortable", 0.5),
        ("throwable", 0.4), ("lightweight", 0.7),
    ],
    "hair drier": [
        ("heat", 0.8), ("electronic", 0.9), ("graspable", 0.9),
        ("portable", 0.7), ("clean", 0.3),
    ],
    "toothbrush": [
        ("clean", 0.95), ("graspable", 0.95), ("portable", 0.95),
        ("absorbent", 0.3), ("lightweight", 0.9),
    ],
}


class AffordanceKnowledgeBase:
    """
    Knowledge base mapping COCO objects to their functional affordances.
    Supports querying affordances and converting to vector representations.
    """

    def __init__(self):
        self.affordances = OBJECT_AFFORDANCES
        self.affordance_types = AFFORDANCE_TYPES
        self.affordance_dim = len(AFFORDANCE_TYPES)

    def get_affordances(self, class_name: str) -> List[Tuple[str, float]]:
        """Get all affordances for an object class."""
        return self.affordances.get(class_name.lower(), [])

    def get_affordance_names(self, class_name: str) -> List[str]:
        """Get just the affordance names for an object."""
        return [a[0] for a in self.get_affordances(class_name)]

    def get_affordance_vector(self, class_name: str) -> 'np.ndarray':
        """
        Get a binary-weighted affordance vector for an object.
        Returns: (num_affordances,) numpy array
        """
        import numpy as np
        vec = np.zeros(self.affordance_dim, dtype=np.float32)
        for aff_name, confidence in self.get_affordances(class_name):
            if aff_name in AFFORDANCE_TO_IDX:
                vec[AFFORDANCE_TO_IDX[aff_name]] = confidence
        return vec

    def has_affordance(self, class_name: str, affordance: str) -> Tuple[bool, float]:
        """
        Check if an object has a specific affordance.
        Returns: (has_it, confidence)
        """
        for aff_name, conf in self.get_affordances(class_name):
            if aff_name == affordance:
                return True, conf
        return False, 0.0

    def find_objects_with_affordance(self, affordance: str) -> List[Tuple[str, float]]:
        """Find all objects that have a given affordance, sorted by confidence."""
        results = []
        for obj_name, affs in self.affordances.items():
            for aff_name, conf in affs:
                if aff_name == affordance:
                    results.append((obj_name, conf))
                    break
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_all_classes(self) -> List[str]:
        """Return all object class names in the KB."""
        return list(self.affordances.keys())

    def get_coverage_stats(self) -> Dict[str, int]:
        """Return stats about the knowledge base."""
        total_objects = len(self.affordances)
        total_mappings = sum(len(v) for v in self.affordances.values())
        unique_affordances = set()
        for affs in self.affordances.values():
            for aff_name, _ in affs:
                unique_affordances.add(aff_name)
        return {
            "total_objects": total_objects,
            "total_mappings": total_mappings,
            "unique_affordances": len(unique_affordances),
            "avg_affordances_per_object": total_mappings / max(1, total_objects),
        }
