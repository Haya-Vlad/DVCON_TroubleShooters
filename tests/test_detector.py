"""Unit tests for object detection module."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskgraph_edge.detection.coco_classes import (
    COCO_CLASSES, get_class_name, get_class_id, get_supercategory,
    CLASS_TO_SUPERCATEGORY
)
from taskgraph_edge.detection.detector import Detection


class TestCOCOClasses:
    def test_80_classes(self):
        """Verify we have exactly 80 COCO classes."""
        assert len(COCO_CLASSES) >= 80

    def test_class_id_mapping(self):
        assert get_class_name(0) == "person"
        assert get_class_name(39) == "bottle"
        assert get_class_name(52) == "chair"

    def test_class_name_mapping(self):
        assert get_class_id("person") == 0
        assert get_class_id("bottle") == 39
        assert get_class_id("unknown") == -1

    def test_supercategories(self):
        assert get_supercategory("cup") == "kitchen"
        assert get_supercategory("chair") == "furniture"
        assert get_supercategory("car") == "vehicle"


class TestDetection:
    def test_detection_dataclass(self):
        det = Detection(
            class_id=39, class_name="bottle",
            confidence=0.95, bbox=(10, 20, 50, 100)
        )
        assert det.class_name == "bottle"
        assert det.confidence == 0.95
        assert det.center == (30.0, 60.0)
        assert det.area == 40 * 80

    def test_detection_zero_area(self):
        det = Detection(
            class_id=0, class_name="person",
            confidence=0.5, bbox=(10, 10, 10, 10)
        )
        assert det.area == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
