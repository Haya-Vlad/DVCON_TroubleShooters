"""
TaskGraph-Edge: Hybrid Vision-Language Graph Reasoning Engine
============================================================

A task-aware object selection system combining lightweight object detection,
language-based task encoding, graph neural network scene reasoning, and
affordance matching — optimized for VEGA RISC-V + FPGA edge deployment.
"""

__version__ = "1.0.0"
__project__ = "TaskGraph-Edge"

from taskgraph_edge.pipeline import TaskGraphPipeline

__all__ = ["TaskGraphPipeline"]
