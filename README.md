# LiteRT CLI

A Python-based command-line toolkit for managing, converting, quantizing, and
executing LiteRT (TFLite) models.

## 🚀 Installation

Install `litert-cli` with pip from the local source or Test PyPI.

```bash
# Installation from local source (recommended for development)
pip install -e .

# Or install from PyPI
pip install litert-cli
```

## 🎯 Quick Start

### Common Commands

**1. Download a model from HuggingFace Hub:**
```bash
# Download only .tflite files
litert download litert-community/MobileNet-v3-large \
  --file "*.tflite" \
  --output mobilenet

# Download full repository
litert download litert-community/MobileNet-v3-large \
  --output mobilenet_full
```

**2. Visualize a model architecture:**
```bash
# Open in Model Explorer graph
litert visualize model.tflite

# Clean up and stop visualizer background servers
litert visualize --stop_all
```

