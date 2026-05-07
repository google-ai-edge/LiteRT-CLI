# LiteRT CLI (Fishfood)

A convenient command-line toolkit to streamline LiteRT development workflow,
including converting, quantizing, compiling, manging, running and benchmarking
LiteRT (TFLite) model on various hardware (CPU / GPU / NPU) across platform
(desktop, mobile or cloud)

## Use in Coding Agent

Add LiteRT CLI skills .agents/skills/litert_cli/SKILL.md into your AI coding agent like
Antigravity or Gemini CLI, then you can use prompts like:
- `Download LiteRT model litert-community/efficientnet_b1 and run it on CPU`
- `Benchmark LiteRT model litert-community/efficientnet_b1 on my Android GPU`
- `Compile LiteRT model litert-community/efficientnet_b1 for NPU target sm8750`
- `Visualize LiteRT model litert-community/efficientnet_b1`

It will install related tools automatically, including python virtual environment, LiteRT-CLI and related dependencies.

## 🚀 Installation

Install `litert-cli` with pip from the local source or PyPI in a Python virtual
environment.

```bash
# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Installation from local source (recommended for development)
pip install -e .

# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple litert-cli==0.1.1.dev22
```

Tested platforms:
- Host machines: Linux (Ubuntu) and Macbook (with Apple Silicon chip), with Python 3.13.
- Android devices: Xiaomi 15 Pro (Qualcomm Snapdragon 8750).

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

# Download models from huggingface id with model reference as hf id too.
litert download litert-community/MobileNet-v3-large

# Download models from huggingface id with my own model reference
litert download litert-community/MobileNet-v3-large --model_ref my_model_ref
```

**2. Convert a PyTorch model into a LiteRT model:**
```bash
# Automated HF Conversion
# support 
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

**4. Compile a LiteRT model for NPU AOT:**
**NOTE**: Only work on linux host for now, and only support qualcomm NPU for now.

```bash
# Basic Compilation for specific NPU, like qualcomm sm8750, used in Xiaomi 15 pro.
litert compile model.tflite --target sm8750

# Compile for multiple targets and export AI Pack for Android
litert compile model.tflite --target sm8750 --target mt6989 --export-aipack my_npu_models
```

**5. Run a LiteRT model on Desktop or Android:**
```bash
# Run locally on desktop (CPU)
litert run model.tflite --desktop --cpu

# Run locally on desktop (CPU)
litert run my_model_ref --desktop --cpu

# Run with GPU acceleration
litert run model.tflite --gpu

# Run on connected Android device
litert run model.tflite --android

# Run on connected Android device with NPU acceleration, with JIT mode.
litert run model.tflite --android --npu

# Run on connected Android device with NPU acceleration, with NPU AOT compiled model
litert run model_sm8450.tflite --android --npu

# Run multiple iterations and print output tensors
litert run model.tflite \
  --iterations 5 \
  --print_tensors

# Run with custom input formats (supports image, raw binary, numpy array)
litert run model.tflite \
  --input "image.png" \
  --print_tensors
```

**6. Benchmark a model performance:**
```bash
# Benchmark on Android (CPU side)
litert benchmark my_model_ref --android --cpu
litert benchmark model.tflite --android --cpu

# Benchmark on Android NPU with JIT mode.
litert benchmark model.tflite --android --npu

# Benchmark AOT compiled model on Android NPU
litert benchmark model_sm8450.tflite --android --npu

# Benchmark on Android GPU
litert benchmark model.tflite --android --gpu

# Benchmark on Macbook (CPU)
litert benchmark my_model_ref --desktop --cpu
```

**7. Visualize a model architecture:**
```bash
# Open in Model Explorer graph
litert visualize model.tflite

# Clean up and stop visualizer background servers
litert visualize --stop_all
```

**8. Import a local model:**
```bash
# Import a local file into the centralized cache
litert import my_model.tflite --model_ref my_model

# Import a directory and associate with Hugging Face ID.
litert import ./my_model_dir --model_ref my_model --hf_id my_org_name/my_model
```

**9. List managed models:**
```bash
# List all managed models
litert list

# Show detailed contents of a specific model
litert list my_model
```

**10. Delete a managed model:**
```bash
# Delete a model from cache
litert delete my_model
```

**11. Run a generative LLM model using LiteRT-LM CLI:**
```bash
# Run a generative LLM model by profixing to LiteRT-LM CLI.
litert lm run gemma-4-E2B-it.litertlm

# Example with custom prompt
litert lm run gemma-4-E2B-it.litertlm --prompt "Hello, how are you?"
```

**12. Clean up all caches:**
```bash
# Clean up model cache, etc.
litert clean
```