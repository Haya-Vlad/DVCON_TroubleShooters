"""
Configuration management for TaskGraph-Edge.
Loads config.yaml and provides typed access to all parameters.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DetectionConfig:
    model: str = "yolov8n"
    input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 20
    device: str = "cpu"
    feature_dim: int = 128


@dataclass
class LanguageConfig:
    model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_length: int = 128
    cache_embeddings: bool = True


@dataclass
class AffordanceConfig:
    embedding_dim: int = 64
    soft_match_weight: float = 0.4
    hard_match_weight: float = 0.6


@dataclass
class SceneGraphConfig:
    k_neighbors: int = 5
    max_distance: float = 0.5
    spatial_feature_dim: int = 9
    edge_feature_dim: int = 16


@dataclass
class GNNConfig:
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    task_condition: bool = True


@dataclass
class RankingWeights:
    detection_confidence: float = 0.15
    task_similarity: float = 0.30
    affordance_match: float = 0.35
    scene_context: float = 0.20


@dataclass
class RankingConfig:
    weights: RankingWeights = field(default_factory=RankingWeights)
    early_exit_threshold: float = 0.92
    top_k: int = 5


@dataclass
class QuantizationConfig:
    enabled: bool = True
    bits: int = 8
    scale_factor: float = 127.0


@dataclass
class FPGAConfig:
    enabled: bool = False
    port: str = "COM4"
    baud_rate: int = 115200
    timeout: float = 5.0
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass
class PerformanceConfig:
    profiling: bool = True
    warmup_runs: int = 3
    benchmark_runs: int = 100


@dataclass
class TaskGraphConfig:
    """Root configuration for the entire TaskGraph-Edge system."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    affordance: AffordanceConfig = field(default_factory=AffordanceConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    fpga: FPGAConfig = field(default_factory=FPGAConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a nested dataclass."""
    if data is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            # Check if the field type is itself a dataclass
            if isinstance(ft, type) and hasattr(ft, '__dataclass_fields__') and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ft, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> TaskGraphConfig:
    """
    Load configuration from YAML file.
    Falls back to defaults if file not found.
    """
    if config_path is None:
        # Look for config.yaml in project root
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)

        config = TaskGraphConfig()

        if 'detection' in raw:
            config.detection = _dict_to_dataclass(DetectionConfig, raw['detection'])
        if 'language' in raw:
            config.language = _dict_to_dataclass(LanguageConfig, raw['language'])
        if 'affordance' in raw:
            config.affordance = _dict_to_dataclass(AffordanceConfig, raw['affordance'])
        if 'scene_graph' in raw:
            config.scene_graph = _dict_to_dataclass(SceneGraphConfig, raw['scene_graph'])
        if 'gnn' in raw:
            config.gnn = _dict_to_dataclass(GNNConfig, raw['gnn'])
        if 'ranking' in raw:
            rk = raw['ranking']
            weights = RankingWeights()
            if 'weights' in rk:
                weights = _dict_to_dataclass(RankingWeights, rk['weights'])
            config.ranking = RankingConfig(
                weights=weights,
                early_exit_threshold=rk.get('early_exit_threshold', 0.92),
                top_k=rk.get('top_k', 5),
            )
        if 'fpga' in raw:
            fp = raw['fpga']
            quant = QuantizationConfig()
            if 'quantization' in fp:
                quant = _dict_to_dataclass(QuantizationConfig, fp['quantization'])
            config.fpga = FPGAConfig(
                enabled=fp.get('enabled', False),
                port=fp.get('port', 'COM4'),
                baud_rate=fp.get('baud_rate', 115200),
                timeout=fp.get('timeout', 5.0),
                quantization=quant,
            )
        if 'performance' in raw:
            config.performance = _dict_to_dataclass(PerformanceConfig, raw['performance'])

        return config
    else:
        print(f"[Config] No config file found at {config_path}, using defaults.")
        return TaskGraphConfig()
