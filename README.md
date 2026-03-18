# TaskGraph-Edge: Hybrid Vision-Language Graph Reasoning Engine

> **DVCon India 2026 Design Contest Entry**
> Task-Aware Object Selection System with FPGA Acceleration

## 🔑 Key Innovation

Instead of basic YOLO → BERT → classifier pipelines, TaskGraph-Edge uses **scene graph reasoning** and **affordance prediction** to understand *why* an object is suitable for a task:

```
Image + Task → YOLOv8n → MiniLM → Affordance KB → Scene Graph → GNN → Ranked Objects
```

## 🏗 Architecture

```
Image + Task prompt
        │
        ▼
┌─────────────────────────┐
│  YOLOv8-nano Detector   │ ──→ N detected objects with features
│  (ONNX / FPGA accel.)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  MiniLM Task Encoder    │ ──→ 384-d task embedding
│  (all-MiniLM-L6-v2)     │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Affordance Engine      │ ──→ Object-task compatibility scores
│  (80 COCO × 34 types)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Scene Graph + GNN      │ ──→ Context-aware node scores
│  (2-layer GAT, ~50K)    │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Object Ranker          │ ──→ Ranked objects with explanations
│  (multi-factor + early  │
│   exit optimization)    │
└─────────────────────────┘
```

## 📦 Project Structure

```
DVCON/
├── taskgraph_edge/          # Python ML pipeline
│   ├── detection/           # YOLOv8n detector
│   ├── language/            # MiniLM task encoder + 14 task defs
│   ├── affordance/          # 80-class affordance KB + scorer
│   ├── scene_graph/         # Spatial graph construction
│   ├── gnn/                 # Pure PyTorch GAT (no PyG needed)
│   ├── ranking/             # Multi-factor ranker
│   ├── pipeline.py          # End-to-end orchestrator
│   └── fpga_bridge.py       # USB/UART Zynq-7000 bridge
├── fpga/                    # FPGA RTL (SystemVerilog)
│   ├── rtl/                 # Conv2D→ReLU→MaxPool + AXI-Lite
│   ├── tb/                  # Self-checking testbench
│   └── constraints/         # Zynq-7000 XDC constraints
├── demo/                    # Interactive demo
├── tests/                   # Unit + integration tests
└── benchmarks/              # Performance benchmarks
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd DVCON
pip install -r requirements.txt
```

### 2. Run simulation demo (no model download needed)
```bash
python demo/demo.py --simulate
```

### 3. Run full pipeline demo (downloads ~96 MB on first run)
```bash
python demo/demo.py --task "water the plant"
python demo/demo.py --image path/to/image.jpg --task "pour a drink"
```

### 4. Run tests
```bash
python -m pytest tests/ -v
```

### 5. Run benchmark
```bash
python benchmarks/benchmark.py
```

## ⚡ FPGA Acceleration (Zynq-7000)

The CNN feature extraction backbone is accelerated on the Zynq-7000 PL fabric:
- **INT8 systolic array** for Conv2D multiply-accumulate
- **Pipelined ReLU + MaxPool** for streaming processing
- **AXI-Lite** interface for ARM Cortex-A9 control
- **USB/UART** data transfer bridge

### Simulate FPGA design
```bash
# Using Vivado
vivado -mode batch -source fpga/sim/run_sim.tcl

# Using Icarus Verilog
iverilog -g2012 -o tb_cnn fpga/rtl/*.sv fpga/tb/tb_cnn_accelerator.sv
vvp tb_cnn
```

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Detection latency | 3-5 ms (FPGA) / 15-25 ms (CPU) |
| Task encoding | ~5 ms (cached: <0.1 ms) |
| Scene graph + GNN | ~2-5 ms |
| Total pipeline | <20 ms (FPGA) / <50 ms (CPU) |
| GNN parameters | ~50K |
| Affordance coverage | 80 COCO classes × 34 types |

## 🎯 14 Supported Tasks

| Task | Key Affordances | Example Best Object |
|------|----------------|-------------------|
| pour_water | hold_liquid, pour_liquid | bottle, cup |
| cut_food | cut, sharp_edge | knife, scissors |
| eat_meal | graspable, scoop | spoon, fork |
| serve_drink | hold_liquid | wine glass, cup |
| heat_food | heat, contain | microwave, oven |
| reach_high_shelf | step_on, support_weight | chair, bench |
| sit_down | sit_on | chair, couch |
| sleep_rest | lie_on, comfortable | bed, couch |
| read_document | display_info, readable | book, laptop |
| type_text | input_text, electronic | laptop, keyboard |
| play_catch | throwable, round | sports ball, frisbee |
| ride_transport | ride_on, wheeled | bicycle, skateboard |
| water_plant | hold_liquid, pour_liquid | bottle, vase |
| clean_surface | clean, graspable | toothbrush, sink |

## 🏆 What Makes This Unique

| Feature | Basic Systems | TaskGraph-Edge |
|---------|--------------|----------------|
| Object selection | Class matching | Affordance reasoning |
| Scene understanding | None | Graph Neural Network |
| Task encoding | Keyword match | Semantic embeddings |
| Explainability | None | Per-factor breakdown |
| Edge optimization | Basic pruning | FPGA acceleration + INT8 |
| Dependencies | PyTorch Geometric | Pure PyTorch |
