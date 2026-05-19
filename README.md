# LiteRT CLI (Preview)

A convenient command-line toolkit to streamline
[LiteRT](https://ai.google.dev/edge/litert) related development workflow,
including converting, quantizing, compiling, managing, running, and benchmarking
LiteRT (TFLite) models on various hardware (CPU / GPU / NPU) across platforms
(desktop, mobile, or cloud).

> [!NOTE]
> It's a still early preview release under active development, thus has
> limited platform and feature support, plus possible bugs. We appreciate your
> patience and feedback to help us improve it.

--------------------------------------------------------------------------------

## 🚀 Installation

You can install `litert-cli-nightly` from PyPI or from local clone. LiteRT CLI
will install the dependencies on-demands, based on which commands to run, to
speed up initial installation.

We support installation using either
**[uv](https://docs.astral.sh/uv/getting-started/installation/)** (recommended
for ultra-fast dependency resolution) or standard
**[pip](https://pip.pypa.io/)** within a Python virtual environment.

#### Option 1: Use UV (Recommended)

`uv` is an extremely fast Python package manager written in Rust.

```bash
# 1. Create a virtual environment with Python 3.13.
# TIP: When meeting dependency resolution error, try to set environment variable:
#    export UV_INDEX_URL=https://pypi.org/simple
uv venv --clear --python=3.13 --seed
source .venv/bin/activate

# 2. Install the package into the active virtual environment
uv pip install litert-cli-nightly

# 3. Run help command
litert --help
```

### Option 2: Use Standard Pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -q litert-cli-nightly
litert --help
```

#### Option 3. Install from Local Clone (for development)

```bash
uv venv --clear --python=3.13 --seed
source .venv/bin/activate
git clone git@github.com:google-ai-edge/LiteRT-CLI.git
cd LiteRT-CLI
uv pip install -e .
```

--------------------------------------------------------------------------------

## Quick Start

### Try colab

Try
[LiteRT CLI Colab](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/examples/litert_cli.ipynb)
to explore different features quickly.

### Follow command help

You can always follow `litert --help` or `litert {command} --help` to find how
to use the CLI tool. Check detailed instructions for each command below.

```bash
# Run help command
litert --help

# Download a LiteRT model
litert download --help
litert download litert-community/efficientnet_b1 --file "*.tflite" --output efficientnet

# Run and benchmark a LiteRT model on your devices
litert run --help
litert run efficientnet/efficientnet_b1.tflite --desktop --cpu
litert benchmark --help
litert benchmark efficientnet/efficientnet_b1.tflite --android --gpu
```

### Quick Demos

Check comprehensive usage examples under the `examples/` directory, which
contains per-command demos and model-specific demos.

If you have cloned the repo, you can run the following commands to see the
demos:

```bash
# Run all command demos
./examples/run_commands.sh

# Run all model demos
./examples/run_models.sh

# Run a specific model demo
./examples/run_models.sh efficientnet
```

### 🤖 Use in Coding Agent

Add the LiteRT CLI skill
[`SKILL.md`]([file:///.agents/skills/litert_cli/SKILL.md]\(https://github.com/google-ai-edge/LiteRT-CLI/blob/main/.agents/skills/litert_cli/SKILL.md\))
into your AI coding agent (like Google Antigravity) and try prompts such as:

*   "Download LiteRT model `litert-community/efficientnet_b1` and run it on CPU"
*   "Benchmark LiteRT model `litert-community/efficientnet_b1` on my Android
    GPU"
*   "Compile LiteRT model `litert-community/efficientnet_b1` for NPU target
    `sm8750`"
*   "Visualize LiteRT model `litert-community/efficientnet_b1`"
*   "Download the FP32 EfficientNet model `litert-community/efficientnet_b1` from
    HuggingFace. Quantize it to INT8 dynamic range (`--recipe dynamic_wi8_afp32`),
    then benchmark both the original FP32 model and the newly quantized INT8 model
    on the GPU of my connected Android device. Compare the average latency and
    report the throughput speedup."
*   "convert the model `Qwen/Qwen1.5-0.5B-Chat` from HuggingFace Hub to LiteRT format, 
    and run it locally using the prompt 'Explain edge machine learning in one sentence'."
*   "Download EfficientNet from huggingface repo `litert-community/efficientnet_b1`
    . Offline compile (AOT) the model for the `sm8750` target NPU, and output 
    the compiled model into `./models/compiled`. Then, run an on-device inference 
    and benchmark using this newly compiled AOT model on the connected Android 
    device's NPU (`--npu`). Confirm that the graph loads directly without 
    dynamic JIT compilation warmup latency."

The agent will automatically install the necessary tools, including Python
virtual environments, `litert-cli-nightly`, and all required dependencies.

--------------------------------------------------------------------------------

### Verified Platforms

Verified in Python 3.13.

*   **Host Machines**:
    *   Linux (Ubuntu)
    *   macOS (Apple Silicon): don't support `litert compile`
    *   Windows: partially supported
*   **Android**:
    *   CPU, GPU
    *   NPU: Qualcomm, MediaTek (soon), Google Tensor (soon)

--------------------------------------------------------------------------------

### Troubleshooting & Tips

* Always active the virtual environment before running `litert` command, to avoid conflicts.
* When `uv` fails to resolve dependencies, try to set environment variable:
  `export UV_INDEX_URL=https://pypi.org/simple` before running `uv` command.
* `litert compile` only supports running on Linux now, and it requires newer
  Clang has version `18.x.x` or above. Try
  `sudo apt install clang libc++-dev libc++abi-dev`
* When run or benchmark failed on GPU using `--gpu` flag, try to add both `--cpu --gpu` flags
  in the command, then the CLI will try CPU first, and fall back to GPU when CPU fails.
* When running `litert run` on Android device, if the device is not detected, try to
  run `adb kill-server && adb start-server` first. You can also forward your device USB port
  to host machine using `adb forward tcp:50000 localabstract:adb-hub`.
* When benchmark using `--gcp` flag, you need to
  1) [Join the EAP program in Google AI Edge Portal](https://ai.google.dev/edge/ai-edge-portal);
  2) Login to GCP using `gcloud auth login`; 
  3) Set your GCP project using `--gcp=<Your-GCP-Project>`;
* When `litert visualize` fails to launch Model Explorer, try to run `litert visualize --stop-all` first.

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

# Automated HF Conversion with INT4 Weight-Only Quantization
litert convert Qwen/Qwen1.5-0.5B-Chat --quantize-recipe weight_only_wi4_afp32 --output /tmp/qwen_w4

# Generic Script Injection with INT8 Dynamic Quantization
litert convert my_model.py --quantize-recipe dynamic_wi8_afp32 --output /tmp/mymodel
```

### 3. Quantize a LiteRT model

```bash
# Dynamic INT8 Quantization (Default)
litert quantize model.tflite \
  --recipe dynamic_wi8_afp32 \
  --output dynamic.tflite

# Weight-Only Quantization
litert quantize model.tflite \
  --recipe weight_only_wi8_afp32 \
  --output weight_only.tflite

# Static Range Quantization (requires calibration data)
litert quantize model.tflite \
  --recipe static_wi8_ai8 \
  --calibration-data calib_data.py \
  --output static.tflite

# Custom JSON Recipe
litert quantize model.tflite \
  --custom-recipe recipe.json \
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

# Benchmark on Google AI Edge Portal in Google Cloud. Prerequisites:
# - Set up your Google AI Edge Portal account by following up the instructions at:
#   https://ai.google.dev/edge/ai-edge-portal
# - Set up authentication by running: gcloud auth login
# - You can set the default GCP project by setting the environment variable LITERT_GCP_PROJECT, or by providing the --gcp-project option.
# - You can specific your GCP bucket by --gcp-bucket, otherwise, it will create default
#   one.
litert benchmark model.tflite --gcp --device "pixel 7" --gcp-project "your-gcp-project-id" --gcp-bucket "your-gcp-bucket"
litert benchmark model.tflite --gcp --devices "pixel 7, sm-s931u1" --gpu
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
