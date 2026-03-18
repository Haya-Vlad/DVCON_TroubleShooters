from setuptools import setup, find_packages

setup(
    name="taskgraph-edge",
    version="1.0.0",
    description="TaskGraph-Edge: Hybrid Vision-Language Graph Reasoning Engine",
    author="DVCon India 2026 Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.0.0",
        "onnxruntime>=1.15.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "pyserial>=3.5",
    ],
)
