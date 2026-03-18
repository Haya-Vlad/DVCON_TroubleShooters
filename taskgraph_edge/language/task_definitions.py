"""
14 Task Definitions for DVCon India 2026.
Each task includes a natural language description, required affordances,
and example objects to enable task-conditioned reasoning.
"""

from typing import Dict, List, Any


# ============================================================
# 14 TASK DEFINITIONS
# ============================================================
TASK_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # --- Household / Kitchen Tasks ---
    "pour_water": {
        "description": "Pour water into a container or onto something",
        "detailed_prompt": "I need to pour water from one container to another",
        "required_affordances": ["pour_liquid", "hold_liquid", "graspable"],
        "preferred_affordances": ["contain"],
        "example_objects": ["bottle", "cup", "wine glass", "bowl"],
        "anti_objects": ["chair", "tv", "book"],  # objects that should NOT be selected
        "category": "kitchen",
    },

    "cut_food": {
        "description": "Cut or slice food items",
        "detailed_prompt": "I need to cut or slice food into smaller pieces",
        "required_affordances": ["cut", "graspable"],
        "preferred_affordances": ["sharp_edge"],
        "example_objects": ["knife", "scissors", "fork"],
        "anti_objects": ["cup", "tv", "chair"],
        "category": "kitchen",
    },

    "eat_meal": {
        "description": "Eat a meal or consume food",
        "detailed_prompt": "I need utensils or items to eat a meal",
        "required_affordances": ["graspable", "scoop"],
        "preferred_affordances": ["contain"],
        "example_objects": ["spoon", "fork", "bowl", "cup"],
        "anti_objects": ["car", "tv", "bed"],
        "category": "kitchen",
    },

    "serve_drink": {
        "description": "Serve a drink to someone",
        "detailed_prompt": "I need a container to serve a drink",
        "required_affordances": ["hold_liquid", "graspable"],
        "preferred_affordances": ["pour_liquid"],
        "example_objects": ["wine glass", "cup", "bottle", "bowl"],
        "anti_objects": ["knife", "scissors", "tv"],
        "category": "kitchen",
    },

    "heat_food": {
        "description": "Heat or warm up food",
        "detailed_prompt": "I need to heat food using an appliance",
        "required_affordances": ["heat", "contain"],
        "preferred_affordances": ["electric"],
        "example_objects": ["microwave", "oven", "toaster"],
        "anti_objects": ["cup", "scissors", "book"],
        "category": "kitchen",
    },

    # --- Furniture / Mobility Tasks ---
    "reach_high_shelf": {
        "description": "Reach something on a high shelf",
        "detailed_prompt": "I need to reach an object that is placed on a high shelf",
        "required_affordances": ["step_on", "support_weight"],
        "preferred_affordances": ["stable"],
        "example_objects": ["chair", "bench", "dining table"],
        "anti_objects": ["cup", "fork", "knife"],
        "category": "mobility",
    },

    "sit_down": {
        "description": "Sit down and rest",
        "detailed_prompt": "I need something to sit on comfortably",
        "required_affordances": ["sit_on", "support_weight"],
        "preferred_affordances": ["comfortable"],
        "example_objects": ["chair", "couch", "bench", "bed"],
        "anti_objects": ["cup", "bottle", "tv"],
        "category": "furniture",
    },

    "sleep_rest": {
        "description": "Sleep or take a rest",
        "detailed_prompt": "I need a surface to lie down and sleep on",
        "required_affordances": ["support_weight", "lie_on"],
        "preferred_affordances": ["comfortable", "large_surface"],
        "example_objects": ["bed", "couch"],
        "anti_objects": ["chair", "cup", "knife"],
        "category": "furniture",
    },

    # --- Work / Study Tasks ---
    "read_document": {
        "description": "Read a document or book",
        "detailed_prompt": "I need something to read information from",
        "required_affordances": ["display_info", "readable"],
        "preferred_affordances": ["portable"],
        "example_objects": ["book", "laptop", "tv"],
        "anti_objects": ["cup", "fork", "chair"],
        "category": "study",
    },

    "type_text": {
        "description": "Type text or enter information",
        "detailed_prompt": "I need a device with a keyboard to type text",
        "required_affordances": ["input_text", "electronic"],
        "preferred_affordances": ["portable"],
        "example_objects": ["laptop", "keyboard", "remote"],
        "anti_objects": ["cup", "chair", "bed"],
        "category": "work",
    },

    # --- Outdoor / Sport Tasks ---
    "play_catch": {
        "description": "Play catch or throw something",
        "detailed_prompt": "I need a ball or throwable object to play catch",
        "required_affordances": ["throwable", "graspable"],
        "preferred_affordances": ["round", "lightweight"],
        "example_objects": ["sports ball", "frisbee"],
        "anti_objects": ["tv", "bed", "car"],
        "category": "sports",
    },

    "ride_transport": {
        "description": "Ride or use a mode of transport",
        "detailed_prompt": "I need a vehicle or device for transportation",
        "required_affordances": ["ride_on", "transport"],
        "preferred_affordances": ["wheeled"],
        "example_objects": ["bicycle", "skateboard", "car", "motorcycle"],
        "anti_objects": ["cup", "fork", "book"],
        "category": "transport",
    },

    # --- Care / Hygiene Tasks ---
    "water_plant": {
        "description": "Water a plant or garden",
        "detailed_prompt": "I need something that can hold and pour water onto a plant",
        "required_affordances": ["hold_liquid", "pour_liquid"],
        "preferred_affordances": ["graspable"],
        "example_objects": ["bottle", "cup", "vase", "bowl"],
        "anti_objects": ["knife", "tv", "chair"],
        "category": "care",
    },

    "clean_surface": {
        "description": "Clean a surface or object",
        "detailed_prompt": "I need something to clean dirty surfaces",
        "required_affordances": ["clean", "graspable"],
        "preferred_affordances": ["absorbent"],
        "example_objects": ["toothbrush", "sink", "bottle"],
        "anti_objects": ["tv", "bed", "car"],
        "category": "hygiene",
    },
}


def get_task_names() -> List[str]:
    """Return all 14 task names."""
    return list(TASK_DEFINITIONS.keys())


def get_task_description(task_name: str) -> str:
    """Get the detailed prompt for a task."""
    task = TASK_DEFINITIONS.get(task_name)
    if task:
        return task["detailed_prompt"]
    return task_name  # fallback: use the name itself


def get_task_affordances(task_name: str) -> List[str]:
    """Get required affordances for a task."""
    task = TASK_DEFINITIONS.get(task_name)
    if task:
        return task["required_affordances"] + task.get("preferred_affordances", [])
    return []


def get_task_examples(task_name: str) -> List[str]:
    """Get example objects for a task."""
    task = TASK_DEFINITIONS.get(task_name)
    if task:
        return task["example_objects"]
    return []


def get_anti_objects(task_name: str) -> List[str]:
    """Get objects that should NOT be selected for a task."""
    task = TASK_DEFINITIONS.get(task_name)
    if task:
        return task.get("anti_objects", [])
    return []


def match_task_from_text(text: str) -> str:
    """
    Find the best matching predefined task from free-form text.
    Uses keyword matching as a lightweight fallback.
    """
    text_lower = text.lower()

    # Keyword to task mapping
    keyword_map = {
        "pour": "pour_water", "water": "pour_water",
        "cut": "cut_food", "slice": "cut_food", "chop": "cut_food",
        "eat": "eat_meal", "meal": "eat_meal", "dine": "eat_meal",
        "serve": "serve_drink", "drink": "serve_drink", "beverage": "serve_drink",
        "heat": "heat_food", "warm": "heat_food", "cook": "heat_food", "microwave": "heat_food",
        "reach": "reach_high_shelf", "shelf": "reach_high_shelf", "high": "reach_high_shelf",
        "sit": "sit_down", "seat": "sit_down",
        "sleep": "sleep_rest", "rest": "sleep_rest", "nap": "sleep_rest", "lie": "sleep_rest",
        "read": "read_document", "book": "read_document", "document": "read_document",
        "type": "type_text", "keyboard": "type_text", "write": "type_text",
        "play": "play_catch", "catch": "play_catch", "throw": "play_catch",
        "ride": "ride_transport", "drive": "ride_transport", "transport": "ride_transport",
        "plant": "water_plant", "garden": "water_plant",
        "clean": "clean_surface", "wash": "clean_surface", "scrub": "clean_surface",
    }

    for keyword, task in keyword_map.items():
        if keyword in text_lower:
            return task

    return "pour_water"  # default fallback
