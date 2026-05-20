---
name: litert-cli
description: LiteRT CLI tool to download, convert, quantize, run, benchmark, and visualize LiteRT models.
---

# LiteRT-CLI Agent Skill

This skill allows the agent to download, convert, quantize, run, benchmark, and
visualize LiteRT models using the `litert` command on desktop, device, or Google
Cloud.

## Setup & Prerequisites (Agent Auto-Setup Guide)

Before running any `litert` commands, you must ensure a Python virtual environment is active and the `litert-cli` package is correctly installed.

### Option 1: Install from Local Clone (Editable Mode)
Use this method if you are developing inside the cloned repository clone:

*   **Using `uv` (Recommended - Super Fast)**:
    ```bash
    # Create a venv with seed packages (critical for dynamic deps.py auto-installers)
    uv venv --clear --python=3.13 --seed
    source .venv/bin/activate

    # Install local clone
    uv pip install -e .
    ```
*   **Using standard venv & pip**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e .
    ```

### Option 2: Install Standalone from PyPI (Nightly)
Use this method if you are installing the published package:

*   **Using `uv`**:
    ```bash
    uv venv --clear --python=3.13 --seed
    source .venv/bin/activate
    uv pip install litert-cli-nightly
    ```
*   **Using standard venv & pip**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install litert-cli-nightly
    ```

> [!TIP]
> If you encounter package resolution or network errors with `uv`, set the standard PyPI index URL first: `export UV_INDEX_URL=https://pypi.org/simple`


## Core Commands

### 💡 The Model Reference (`model-ref`) System

To avoid handling complex and fragile absolute filesystem paths, the LiteRT CLI uses a centralized **Model Reference (`model-ref`)** catalog.

When you download or import a model to the centralized cache, you can assign it
a reference alias (and optional sub-references): * **Format**: `<alias_name>` or
`<alias_name>:<sub_reference>` (e.g., `mobilenet`, `resnet18:gpu`,
`efficientnet:int8`). * **Default alias**: For HuggingFace downloads, if
`--model-ref` is omitted, the CLI automatically assigns a flattened repository
ID (e.g., `litert-community__MobileNet-v3-large`) as the default alias.

Once a model is registered, **all CLI commands** (including `run`, `benchmark`,
`compile`, `delete`, `list`) accept this `<model_ref>` directly instead of a
file path! The CLI will automatically resolve it to the correct absolute cache
file path on the fly.

**Examples:**
```bash
# Run inference using the central alias directly
litert run mobilenet --android --cpu

# Benchmark using a specific sub-reference GPU file

litert benchmark resnet18:gpu --android --gpu

# Compile for NPU directly using the reference alias
litert compile efficientnet --target sm8750

# Delete from the central cache
litert delete mobilenet
```

### 1. Download

Download public LiteRT models from HuggingFace Hub or a direct URL.

```bash
# Download public models using repo path
litert download repo_id_or_url --output ./models

# Download only .tflite files from HuggingFace
litert download litert-community/MobileNet-v3-large --file "*.tflite" --output ./models

# Download models with a custom model-ref alias
litert download litert-community/MobileNet-v3-large --model-ref my_model_ref
```
> [!NOTE]
> If `--output` is omitted during HuggingFace downloads, the model is downloaded to `~/.cache/litert-cli/models/` and cataloged automatically via `metadata.json` (associating it with the repo ID as the `model-ref`). If `--output` is provided, it is treated as a standalone folder and is **not** cataloged.

### 2. Convert (PyTorch to LiteRT)

Convert a PyTorch or HuggingFace model into a TFLite model.

```bash
# Automated HF model conversion
litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen

# Conversion with INT8 Weight-Only Quantization
litert convert Qwen/Qwen1.5-0.5B-Chat --quantize-recipe weight_only_wi8_afp32 --output /tmp/qwen_w8

# From a custom local PyTorch script
litert convert my_model.py --quantize-recipe dynamic_wi8_afp32 --output /tmp/mymodel
```
> [!NOTE]
**Custom Python Script Interface (`my_model.py`):**
To convert from a custom Python script, the file must expose functions to return the instantiated PyTorch model and generate sample inputs for tracer graph execution:
*   `--model-func`: Function name returning the model (`torch.nn.Module`). Default: `get_model`.
*   `--input-func`: Function name returning sample trace inputs (tuple/dict). Default: `get_args`.

**Minimal Script Example:**
```python
import torch

def get_model() -> torch.nn.Module:
    return MyPyTorchModel()

def get_args() -> tuple:
    return (torch.randn(1, 3, 224, 224),)
```

### 3. Quantize

Quantize a TFLite model using optimized recipes.

```bash
# Dynamic Range Quantization (dynamic_wi8_afp32) (Default)
litert quantize model.tflite --recipe dynamic_wi8_afp32 --output dynamic.tflite

# Weight-Only Quantization
litert quantize model.tflite --recipe weight_only_wi8_afp32 --output weight_only.tflite

# Static Range Quantization (requires calibration data)
litert quantize model.tflite --recipe static_wi8_ai8 --calibration-data calib_data.py --output static.tflite

# Custom JSON Recipe
litert quantize model.tflite --custom-recipe recipe.json --output recipe.tflite
```

### 4. Compile (NPU Offline AOT Compilation)

Apply Ahead-of-Time (AOT) offline compilation to a standard TFLite model for edge SoC target NPUs.

> [!NOTE]
> Currently only supported on Linux hosts for Qualcomm targets. Other targets are coming soon!

```bash
# Basic offline compile for Qualcomm sm8750 NPU
litert compile model.tflite --target sm8750

# Compile for multiple NPUs and export as Android AI Pack
litert compile model.tflite --target sm8750 --target mt6989 --export-aipack my_npu_models

# Download the latest SoC metadata target maps from GitHub
litert compile --update-targets main
```

### 5. Run (Inference)

Run a TFLite model locally on desktop or on an adb-connected Android device.

**Desktop Execution:**
```bash
# Run locally on desktop (CPU)
litert run model.tflite --desktop --cpu

# Run using registered model-ref
litert run my_model_ref --desktop --cpu

# Enable GPU acceleration with CPU fallback (multi-accelerator)
litert run model.tflite --desktop --accelerator gpu,cpu
```
*   Output logs are **clean by default**. To enable C++ verbose debug logs, set the environment variable: `export LITERT_VERBOSE=1`.

**Android Execution (CPU, GPU, or NPU):**
```bash
# Run on connected Android device (CPU side)
litert run model.tflite --android --cpu

# Enable GPU acceleration with CPU fallback
litert run model.tflite --android --accelerator gpu,cpu

# Enable NPU acceleration with CPU fallback (JIT compilation mode)
litert run standard_model.tflite --android --accelerator npu,cpu
```
*   **NPU Ahead-Of-Time (AOT) execution mode**: Pass an already NPU-compiled TFLite model (compiled offline via `litert compile`). The on-device runtime loads the compiled binary block directly, avoiding graph-compilation warmup overhead:
```bash
litert run resnet18_compiled_sm8750.tflite --android --npu
```

**Custom Inputs and Formats:**
```bash
# Run multiple iterations and print output tensors
litert run model.tflite --iterations 5 --print-tensors

# Multi-Input Formats using literals or numpy arrays
litert run model.tflite --desktop --input inputs="[0.5, 0.5, 0.5]"

# Multi-Input Formats using files (numpy arrays, raw binaries, or images)
litert run model.tflite --desktop --input "image.png" --print-tensors
```

### 6. Benchmark

Benchmark LiteRT models on different platforms (Android, Google Cloud, or Desktop).

```bash
# Benchmark on connected Android device via CPU
litert benchmark model.tflite --android --cpu

# Benchmark on Android GPU using OpenCL/OpenGL delegates
litert benchmark model.tflite --android --gpu

# Benchmark on Android NPU (JIT compilation mode)
litert benchmark model.tflite --android --npu

# Benchmark compiled AOT model on Android NPU
litert benchmark model_compiled.tflite --android --npu

# Benchmark on macOS CPU
litert benchmark my_model_ref --desktop --cpu

# Benchmark on Google AI Edge Portal (Google Cloud)
# Note: You must join EAP program first, authenticate using 'gcloud auth login', and configure
# --gcp-project and --gcp-bucket.
litert benchmark model.tflite --gcp --device "pixel 7" --gcp-project "your-gcp-project-id" --gcp-bucket "your-gcp-bucket"
```

### 7. Large Language Models (LM)

Interact with LLM generative models (like Qwen 1.5 or Gemma 4) using native `litert-lm` utilities.

> [!TIP]
> **Non-interactive / Background Execution:** When running LLM inferences in scripts or background tasks, the process will block waiting for chat prompts on `stdin`. To prevent hanging, **always redirect stdin from `/dev/null`** (i.e. append `< /dev/null` to the end of command).

```bash
# Run a generative model loading from Hugging Face
litert lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm --prompt="What is the capital of France?" < /dev/null

# Run using a local compiled LLM (.litertlm) file
litert lm run gemma-4-E2B-it.litertlm --prompt "Hello, how are you?" < /dev/null

# Benchmark a generative LLM model
litert lm benchmark gemma-4-E2B-it.litertlm
```

### 8. Visualize

Launch Model Explorer to visualize the model structure.

```bash
# Open model structure graph in web browser
litert visualize model.tflite

# Clean up and stop all background visualization servers
litert visualize --stop-all
```

### 9. Import

Import a local model file or folder into the centralized cache.

```bash
# Import a local file into the cache
litert import my_model.tflite --model-ref my_model

# Import a directory and associate with a Hugging Face repository ID
litert import ./my_model_dir --model-ref my_model --hf-id my_org/my_model
```

### 10. List

List managed model references or view details of a specific model.

```bash
# List all models in the catalog
litert list

# Show detailed configurations of a specific model
litert list my_model
```

### 11. Delete

Delete a managed model reference from the centralized catalog.

```bash
# Delete a model reference from the cache
litert delete my_model
```

### 12. Clean

Clean up local caches, downloads, and temporary directories.

```bash
# Clean up local caches, downloaded files, and temporary on-device directories
litert clean
```

## 🧪 Testing

Agents should run tests after modifying code to ensure no regressions.

To run unit tests locally:
```bash
python litert_cli/litert_test.py
python litert_cli/litert_help_test.py
```

To run comprehensive end-to-end regression tests:
```bash
./examples/run_smoke_tests.sh
./examples/run_commands.sh
./examples/run_models.sh
```

## Best Practices for Agents

*   Pipe outputs to text files or grep them if you are looking for specific tensor shapes or runtime metrics.
*   **Avoid hanging background processes**: When executing the `litert lm run` command in a script or in the background, **always** append `< /dev/null` to redirect standard input. Otherwise, the process will block indefinitely waiting on stdin.
*   **Explore Demos**: Refer to the `examples/` directory to explore comprehensive per-command demos (under `examples/commands/`) and model-specific demos (under `examples/models/`) for complete automation patterns.
*   **Read Troubleshooting Guides**: Refer to the main project `README.md` file's **Troubleshooting & Tips** section for platform-specific environmental setup guides, adb port recoveries, and NPU offline compiler clang version requirements.

## 🤖 Example Agent Prompts

These prompts demonstrate how developers can leverage this skill. You can copy and use them directly in your agent queries:

*   "Download LiteRT model `litert-community/efficientnet_b1` and run it on CPU"
*   "Benchmark LiteRT model `litert-community/efficientnet_b1` on my Android GPU"
*   "Compile LiteRT model `litert-community/efficientnet_b1` for NPU target `sm8750`"
*   "Visualize LiteRT model `litert-community/efficientnet_b1`"
*   "Download the FP32 EfficientNet model `litert-community/efficientnet_b1` from HuggingFace. Quantize it to INT8 dynamic range (`--recipe dynamic_wi8_afp32`), then benchmark both the FP32 and INT8 models on my Android GPU, comparing the throughput speedup."
*   "Convert the model `Qwen/Qwen1.5-0.5B-Chat` from HuggingFace Hub to LiteRT format, and run it locally with the prompt 'Explain edge machine learning in one sentence'."
*   "Download `litert-community/efficientnet_b1`, offline compile (AOT) it for the `sm8750` target NPU into `./models/compiled`, then run on-device inference and benchmark on the NPU, confirming zero JIT warmup latency."

