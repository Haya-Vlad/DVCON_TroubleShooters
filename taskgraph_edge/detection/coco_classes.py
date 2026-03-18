"""
COCO 80-class object definitions with semantic embeddings.
Each class includes category info and a brief description for affordance linking.
"""

# COCO 80 class names (index = class_id, matching YOLOv8 output mapping)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",          # 0-4
    "bus", "train", "truck", "boat", "traffic light",              # 5-9
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", # 10-14
    "cat", "dog", "horse", "sheep", "cow",                         # 15-19
    "elephant", "bear", "zebra", "giraffe", "backpack",            # 20-24
    "umbrella", "handbag", "tie", "suitcase", "frisbee",           # 25-29
    "skis", "snowboard", "sports ball", "kite", "baseball bat",    # 30-34
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",  # 35-39
    "wine glass", "cup", "fork", "knife", "spoon",                 # 40-44
    "bowl", "banana", "sandwich", "hot dog", "pizza",              # 45-49
    "donut", "cake", "chair", "couch", "potted plant",             # 50-54
    "bed", "dining table", "toilet", "tv", "laptop",               # 55-59
    "mouse", "remote", "cell phone", "microwave", "oven",          # 60-64
    "toaster", "sink", "refrigerator", "book", "clock",            # 65-69
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",  # 70-74
    "hat", "backpack_2", "shoe", "eye glasses", "handbag_2",       # 75-79
]

# Class ID to name mapping
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(COCO_CLASSES)}
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(COCO_CLASSES)}

# Supercategory groupings (useful for reasoning)
COCO_SUPERCATEGORIES = {
    "person": ["person"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase",
                   "hat", "backpack_2", "shoe", "eye glasses", "handbag_2"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": ["banana", "sandwich", "hot dog", "pizza", "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronic": ["tv", "laptop", "mouse", "remote", "cell phone"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}

# Reverse mapping: class → supercategory
CLASS_TO_SUPERCATEGORY = {}
for supercat, classes in COCO_SUPERCATEGORIES.items():
    for cls in classes:
        CLASS_TO_SUPERCATEGORY[cls] = supercat


def get_class_name(class_id: int) -> str:
    """Get class name from class ID."""
    return CLASS_ID_TO_NAME.get(class_id, f"unknown_{class_id}")


def get_class_id(class_name: str) -> int:
    """Get class ID from class name."""
    return CLASS_NAME_TO_ID.get(class_name.lower(), -1)


def get_supercategory(class_name: str) -> str:
    """Get supercategory for a class."""
    return CLASS_TO_SUPERCATEGORY.get(class_name.lower(), "unknown")
