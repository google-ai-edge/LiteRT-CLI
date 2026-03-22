# LiteRT CLI

A Python-based command-line toolkit for managing, converting, quantizing, running
and benchmarking LiteRT (TFLite) models.

Please follow the demo scripts to try:
https://github.com/google-ai-edge/LiteRT-CLI-staging/blob/main/test_scripts/litert_cli_demo.sh

Verified in linux and macbook, with Python 3.13

## 🚀 Installation

Install `litert-cli` with pip from the local source or Test PyPI.

```bash
# Installation from local source (recommended for development)
pip install -e .

# Or install from Test PyPI (verified in linux and macbook, with Python 3.13)
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple litert-cli==0.1.1.dev12
```

## 🎯 Quick Start

Run the end-to-end demo script to see download and quantization in action:
```bash
# Verify it by running e2e demo
bash test_scripts/run_tests_portable.sh
```

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

**2. Convert a PyTorch model into a LiteRT model:**
```bash
# Automated HF Conversion
litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen

# Generic Script Injection
litert convert my_model.py --output /tmp/mymodel

# Known Script Conversion (e.g., ResNet18)
litert convert resnet18.py --output /tmp/resnet18
```

**3. Quantize a LiteRT model:**
```bash
# Dynamic INT8 Quantization (Default)
litert quantize model.tflite \
  --type int8_dynamic \
  --output dynamic.tflite

# Weight-Only Quantization
litert quantize model.tflite \
  --type int8_weight_only \
  --output weight_only.tflite

# Static Range Quantization (requires calibration data)
litert quantize model.tflite \
  --type static \
  --calibration-data calib_data.py \
  --output static.tflite

# Recipe-based Custom Quantization
litert quantize model.tflite \
  --recipe recipe.json \
  --output recipe.tflite
```

**4. Run a LiteRT model on Desktop or Android:**
```bash
# Run locally on desktop (CPU)
litert run model.tflite --desktop --cpu

# Run with GPU acceleration
litert run model.tflite --gpu

# Run on connected Android device
litert run model.tflite --android

# Run multiple iterations and print output tensors
litert run model.tflite \
  --iterations 5 \
  --print_tensors

# Run with custom input formats (supports image, raw binary, numpy array)
litert run model.tflite \
  --input "image.png" \
  --print_tensors
```

**5. Benchmark a model performance:**
```bash
# Benchmark on Android (CPU side)
litert benchmark model.tflite --android --cpu

# Benchmark on Android GPU
litert benchmark model.tflite --android --gpu
```

**6. Visualize a model architecture:**
```bash
# Open in Model Explorer graph
litert visualize model.tflite

# Clean up and stop visualizer background servers
litert visualize --stop_all
```
