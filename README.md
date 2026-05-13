# LiteRT CLI (Preview)

A convenient command-line toolkit to streamline LiteRT development workflow,
including converting, quantizing, compiling, managing, running, and benchmarking
LiteRT (TFLite) models on various hardware (CPU / GPU / NPU) across platforms
(desktop, mobile, or cloud).

## 🤖 Use in Coding Agent

Add the LiteRT CLI skill
[`SKILL.md`](file:///.agents/skills/litert_cli/SKILL.md) into your AI coding
agent (like Antigravity or Gemini CLI) to enable prompts such as:

*   "Download LiteRT model `litert-community/efficientnet_b1` and run it on CPU"
*   "Benchmark LiteRT model `litert-community/efficientnet_b1` on my Android
    GPU"
*   "Compile LiteRT model `litert-community/efficientnet_b1` for NPU target
    `sm8750`"
*   "Visualize LiteRT model `litert-community/efficientnet_b1`"

The agent will automatically install the necessary tools, including Python
virtual environments, `litert-cli`, and all required dependencies.

--------------------------------------------------------------------------------

## 🚀 Installation

We support installation using either **`uv`** (recommended for ultra-fast
dependency resolution) or standard **`pip`** within a virtual environment.

### Option 1: Use UV (Recommended)

`uv` is an extremely fast Python package manager written in Rust.

#### 1. Create and Activate Virtual Environment

```bash
# Create a virtual environment with Python 3.13 in the current directory
uv venv --clear --python=3.13
source .venv/bin/activate
```

#### 2. Install `litert-cli`

##### 2a. Install from Test PyPI

```bash
# Install the package into the active virtual environment
uv pip install -q -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple litert-cli==0.1.1.dev24
```

##### 2b. Or Install from Local Clone (Recommended for Development)

```bash
# Clone the repository via SSH
git clone git@github.com:google-ai-edge/LiteRT-CLI.git
# Or clone using your Personal Access Token (PAT)
git clone https://<your-access-token>@github.com/google-ai-edge/LiteRT-CLI.git
cd LiteRT-CLI

# Install in editable mode inside the active virtual environment
uv pip install -e .
```

#### 3. Run Commands

Check more comprehensive usage examples under the `test_scripts/` directory
(e.g., `test_scripts/models/efficientnet.sh`). You can run the CLI via `uv run`:

```bash
# Run help command
uv run litert --help

# Download a LiteRT model
uv run litert download litert-community/MobileNet-v3-large --file "*.tflite" --output mobilenet
```

--------------------------------------------------------------------------------

### Option 2: Use Standard Pip

#### 1. Create and Activate Virtual Environment

```bash
# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install `litert-cli`

##### 2a. Install from Test PyPI

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple litert-cli==0.1.1.dev24
```

##### 2b. Or Install from Local Clone

```bash
# Clone the repository
git clone git@github.com:google-ai-edge/LiteRT-CLI.git
cd LiteRT-CLI

# Install in editable mode
pip install -e .
```

#### 3. Run Commands

```bash
# Run help command
litert --help

# Download a LiteRT model
litert download litert-community/MobileNet-v3-large --file "*.tflite" --output mobilenet
```

### Tested Platforms

*   **Host Machines**:
    *   Linux (Ubuntu) with Python 3.13
    *   macOS (Apple Silicon) with Python 3.13
*   **Android Devices**:
    *   Xiaomi 15 Pro (Qualcomm Snapdragon 8750)

--------------------------------------------------------------------------------

## 💡 Common Commands

### 1. Download a model from HuggingFace Hub

```bash
# Download only .tflite files
litert download litert-community/MobileNet-v3-large \
  --file "*.tflite" \
  --output mobilenet

# Download full repository
litert download litert-community/MobileNet-v3-large \
  --output mobilenet_full

# Download models using Hugging Face ID (uses HF ID as model reference too)
litert download litert-community/MobileNet-v3-large

# Download models with custom model reference
litert download litert-community/MobileNet-v3-large --model-ref my_model_ref
```

### 2. Convert a PyTorch model into a LiteRT model

```bash
# Automated HF Conversion
litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen

# Generic Script Injection
litert convert my_model.py --output /tmp/mymodel

# Known Script Conversion (e.g., ResNet18)
litert convert resnet18.py --output /tmp/resnet18
```

### 3. Quantize a LiteRT model

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

### 4. Compile a LiteRT model for NPU AOT

> [!NOTE] Currently only supported on Linux hosts and Qualcomm NPUs.

```bash
# Basic compilation for specific Qualcomm NPU (e.g., sm8750 in Xiaomi 15 Pro)
litert compile model.tflite --target sm8750

# Compile for multiple targets and export an AI Pack for Android
litert compile model.tflite --target sm8750 --target mt6989 --export-aipack my_npu_models
```

### 5. Run a LiteRT model on Desktop or Android

```bash
# Run locally on desktop (CPU)
litert run model.tflite --desktop --cpu
litert run my_model_ref --desktop --cpu

# Run with GPU acceleration and CPU fallback (multi-accelerator)
litert run model.tflite --gpu --cpu
litert run model.tflite --accelerator gpu,cpu

# Run on connected Android device
litert run model.tflite --android

# Run on connected Android device with NPU acceleration and CPU fallback
litert run model.tflite --android --npu --cpu
litert run model.tflite --android --accelerator npu,cpu

# Run on connected Android device with NPU AOT-compiled model
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

### 6. Benchmark a model's performance

```bash
# Benchmark on Android (CPU side)
litert benchmark my_model_ref --android --cpu
litert benchmark model.tflite --android --cpu

# Benchmark on Android NPU (JIT mode)
litert benchmark model.tflite --android --npu

# Benchmark AOT compiled model on Android NPU
litert benchmark model_sm8450.tflite --android --npu

# Benchmark on Android GPU
litert benchmark model.tflite --android --gpu

# Benchmark on macOS (CPU)
litert benchmark my_model_ref --desktop --cpu
```

### 7. Visualize a model's architecture

```bash
# Open in Model Explorer graph
litert visualize model.tflite

# Clean up and stop visualizer background servers
litert visualize --stop-all
```

### 8. Import a local model

```bash
# Import a local file into the centralized cache
litert import my_model.tflite --model-ref my_model

# Import a directory and associate with a Hugging Face ID
litert import ./my_model_dir --model-ref my_model --hf-id my_org_name/my_model
```

### 9. List managed models

```bash
# List all managed models
litert list

# Show detailed contents of a specific model
litert list my_model
```

### 10. Delete a managed model

```bash
# Delete a model from cache
litert delete my_model
```

### 11. Run a generative LLM model using LiteRT-LM CLI

```bash
# Run a generative LLM model
litert lm run gemma-4-E2B-it.litertlm

# Example with a custom prompt
litert lm run gemma-4-E2B-it.litertlm --prompt "Hello, how are you?"
```

### 12. Clean up all caches

```bash
# Clean up model cache, etc.
litert clean
```