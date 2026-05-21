# LiteRT CLI (Preview)

A convenient command-line toolkit to streamline
[LiteRT](https://ai.google.dev/edge/litert) related development workflows,
including converting, quantizing, compiling, running, benchmarking and
visualizing LiteRT (TFLite) models on various hardware (CPU / GPU / NPU) across
platforms (desktop, mobile, or cloud).

🚀 [Installation](#-installation) | ⚡ [Quick start](#-quick-start) | 💡
[Common commands](#-common-commands) ｜ 📓 [Try Colab](#-try-colab) | 🌟
[Quick demos](#-quick-demos) | 🤖 [Use in coding agent](#-use-in-coding-agent)

> [!NOTE]
>
> It's still an early preview under active development, thus has limited
> platform and feature support, plus possible bugs. We appreciate your patience
> and feedback to help us improve it. Welcome issues and PRs!

LiteRT CLI is built on top of [Google AI Edge](https://ai.google.dev/edge)
stacks, including [LiteRT](https://github.com/google-ai-edge/LiteRT),
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM),
[LiteRT Torch](https://github.com/google-ai-edge/LiteRT-Torch),
[AI Edge Quantizer](https://github.com/google-ai-edge/ai-edge-quantizer),
[AI Edge Portal](https://ai.google.dev/edge/ai-edge-portal), and
[Model Explorer](https://ai.google.dev/edge/model-explorer).

--------------------------------------------------------------------------------

## 🚀 Installation

Please install `litert-cli-nightly` from PyPI or from local clone. LiteRT CLI
will install the dependencies on-demand, based on which commands to run, to
speed up initial installation.

We support installation using either
**[uv](https://docs.astral.sh/uv/getting-started/installation/)** (recommended
for ultra-fast dependency resolution) or standard
**[pip](https://pip.pypa.io/)** within a Python virtual environment.

### Option 1: Use uv (recommended)

`uv` is an extremely fast Python package manager written in Rust.

```bash
# 1. Create a virtual environment with Python 3.13.
# TIP: Sometimes setting env var `UV_INDEX_URL=https://pypi.org/simple` helps
# resolve dependency resolution errors.
uv venv --clear --python=3.13 --seed
source .venv/bin/activate

# 2. Install the package into the active virtual environment
uv pip install litert-cli-nightly

# 3. Run help command
litert --help
```

### Option 2: Use standard pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -q litert-cli-nightly
litert --help
```

### Option 3: Install from local clone (for development)

```bash
uv venv --clear --python=3.13 --seed
source .venv/bin/activate
git clone git@github.com:google-ai-edge/LiteRT-CLI.git
cd LiteRT-CLI
uv pip install -e .
```

--------------------------------------------------------------------------------

## ⚡ Quick start

### 📓 Try Colab

Try
[LiteRT CLI Colab](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/examples/litert_cli.ipynb)
to explore different features quickly.

### Follow command help

You can always follow `litert --help` or `litert {command} --help` to find how
to use the CLI tool. Check
[detailed instructions for each command](#-common-commands) below.

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

### 🌟 Quick demos

Check comprehensive usage examples under the
[examples/](https://github.com/google-ai-edge/LiteRT-CLI/tree/main/examples)
directory, which contains per-command demos and model-specific demos.

If you have cloned the repo, you can run the following commands to see the
demos. Note: running all demos will take time and disk space.

```bash
# Run all command demos
./examples/run_commands.sh
# Run specific command demos
./examples/run_commands.sh download,benchmark

# Run all model demos
./examples/run_models.sh

# Run a specific model demo
./examples/run_models.sh efficientnet
```

### 🤖 Use in coding agent

Add the LiteRT CLI skill
[`SKILL.md`](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/.agents/skills/litert_cli/SKILL.md)
into your coding agent (like [Google Antigravity](https://antigravity.google/))
and try prompts such as:

*   *Download LiteRT model `litert-community/efficientnet_b1` and run it on CPU*
*   *Benchmark LiteRT model `litert-community/efficientnet_b1` on my Android
    GPU*
*   *Compile LiteRT model `litert-community/efficientnet_b1` for NPU target
    `sm8750`*
*   *Visualize LiteRT model `litert-community/efficientnet_b1`*
*   *Download the FP32 model `litert-community/efficientnet_b1` , quantize it to
    INT8 dynamic range (`--recipe dynamic_wi8_afp32`), then benchmark both the
    original FP32 model and the newly quantized INT8 model on the GPU of my
    connected Android device. Compare the average latency and report the
    throughput speedup.*
*   *Convert the model `Qwen/Qwen1.5-0.5B-Chat` from HuggingFace, and run it
    locally using the prompt 'Explain edge machine learning one sentence'*
*   *Download EfficientNet from huggingface repo
    `litert-community/efficientnet_b1`, offline compile (AOT) the model for the
    `sm8750` target NPU, and output the compiled model into `./models/compiled`.
    Then, run an on-device inference and benchmark using this newly compiled AOT
    model on the connected Android device's NPU (`--npu`). Confirm that the
    graph loads directly without dynamic JIT compilation warmup latency.*

The agent will automatically install the necessary tools, including Python
virtual environments, `litert-cli-nightly`, and all required dependencies.

--------------------------------------------------------------------------------

### Verified platforms

Verified in Python 3.13.

*   **Host Machines**:
    *   Linux (Ubuntu)
    *   macOS (Apple Silicon): don't support `litert compile` yet.
    *   Windows: `litert compile` and `litert convert` not supported yet.
*   **Android**:
    *   CPU, GPU
    *   NPU: Qualcomm, MediaTek (soon), Google Tensor (soon)

--------------------------------------------------------------------------------

### Troubleshooting and tips

*   Always activate python virtual environment before running `litert` command,
    to avoid conflicts.
*   When `uv` fails to resolve dependencies, try to set below environment
    variable first: `export UV_INDEX_URL=https://pypi.org/simple`.
*   When run fails on GPU using `--gpu` flag, try to add both `--cpu --gpu`
    flags in the command, then the CLI will try CPU first, and fall back to GPU
    when CPU fails.
*   When `litert run` fails on Android device, if the device is not detected,
    try to run `adb kill-server` first.
*   When convert a traditional PyTorch model, you need to write a script to wrap
    it with required functions `get_model` and `get_args`. Check the script
    format in
    [resnet18.py](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/litert_cli/test_data/resnet18.py).
*   LLM conversion only supports HuggingFace models with type
    AutoModelForCausalLM and
*   Gemma family now.
*   For large models like LLMs, `litert convert` will take large memories and
    disks, and spend multiple minutes. Please make sure you have enough memory
    and disks, and be patient.
*   `litert compile` only supports running on Linux now, and it requires newer
    Clang has version `18.x.x` or above. Try `sudo apt install clang libc++-dev
    libc++abi-dev`.
*   When benchmark using `--gcp` flag, you need to 1) Join the EAP program of
    [Google AI Edge Portal](https://ai.google.dev/edge/ai-edge-portal); 2) Login
    to GCP using `gcloud auth login`; 3) Set your GCP project using
    `--gcp=<Your-GCP-Project>`.
*   When `litert visualize` fails to launch Model Explorer, try to run `litert
    visualize --stop-all` first.
*   Exporting environment variable `LITERT_VERBOSE=1` can enable verbose
    logging.
*   `litert clean` will clean all local caches, like model files and binaries,
    which will free your disk space, and further, it will be very helpful for
    fixing complicated issues, like issues caused by NPU library version
    mismatch.

--------------------------------------------------------------------------------

## 💡 Common commands

### 1. Download a model from Hugging Face Hub

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

# Static W8A8 Quantization (with calibration data)
litert quantize model.tflite \
  --recipe static_wi8_ai8 \
  --calibration-data calib_data.py \
  --output static.tflite

# Custom Recipe
litert quantize model.tflite \
  --custom-recipe quantize_recipe.json \
  --output custom_quant.tflite
```

### 4. AOT Compile a LiteRT model for NPU

> [!NOTE]
>
> Currently only support on Linux hosts and Qualcomm NPUs, and other NPU
> supports are coming soon!

```bash
# Basic compilation for specific Qualcomm NPU (e.g., sm8750)
litert compile model.tflite --target sm8750

# Compile for multiple targets and export an AI Pack for Android
litert compile model.tflite --target sm8750 --target mt6989 --export-aipack my_npu_models
```

### 5. Run a LiteRT model on desktop or Android

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
  --print-tensors

# Run with custom input formats (supports image, raw binary, numpy array)
litert run model.tflite \
  --input "image.png" \
  --print-tensors
```

### 6. Benchmark a LiteRT model

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

### 7. Run and benchmark a generative LLM model using LiteRT-LM CLI

`litert lm` command will utlitize `litert-lm`, and you can use the same command
with `litert-lm`, for example, both `litert lm run` and `litert-lm run` or
`litert lm benchmark` and `litert-lm benchmark` achieve the same results.

Please follow the
[LiteRT-LM CLI guide](https://ai.google.dev/edge/litert-lm/cli) for detailed
instructions.

```bash
# Run a generative LLM model, and load from hugging face
litert lm run  \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="What is the capital of France?"

# Or load from a local LLM model file
litert lm run ./my_model.litertlm

# Example with a custom prompt
litert lm run ./my_model.litertlm --prompt "Hello, how are you?"

# Benchmark a generative LLM model
litert lm benchmark ./my_model.litertlm
```

### 8. Visualize a model's architecture

```bash
# Open in Model Explorer graph
litert visualize model.tflite

# Clean up and stop visualizer background servers
litert visualize --stop-all
```

### 9. Import a local model

```bash
# Import a local file into the centralized cache
litert import my_model.tflite --model-ref my_model

# Import a directory and associate with a Hugging Face ID
litert import ./my_model_dir --model-ref my_model --hf-id my_org_name/my_model
```

### 10. List managed models

```bash
# List all managed models
litert list

# Show detailed contents of a specific model using model reference.
litert list my_model
```

### 11. Delete a managed model

```bash
# Delete a model from cache
litert delete my_model
```
### 12. Clean up all caches

```bash
# Clean up local cache, like model files and binaries.
litert clean
```
