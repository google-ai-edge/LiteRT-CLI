---
name: litert-cli
description: LiteRT CLI tool to convert, download, quantize, run, benchmark, and visualize LiteRT models.
---

# LiteRT-CLI Agent Skill

This skill allows the agent to download, convert, quantize, run, benchmark, and
visualize LiteRT models using the `litert` command on desktop, device, or Google
Cloud

## Setup & Prerequisites

Before running any `litert` commands, an agent should ensure it is in a Python virtual environment and `litert-cli` is installed.

### 1. Check/Create Virtual Environment

We highly recommend using **`uv`** (written in Rust) for extremely fast environment management and package installs.

**Option A: Use UV (Recommended - Super Fast):**
```bash
# Create a virtual environment with Python 3.13.
# We use --seed to pre-install pip, setuptools, and wheel inside the venv.
# This is critical to allow our CLI dynamic dependency auto-installers (deps.py) to function.
uv venv --clear --python=3.13 --seed
source .venv/bin/activate
```

**Option B: Use Standard Pip/Venv:**
```bash
# Create and activate a standard Python virtual environment.
python3 -m venv litert-cli-venv
source litert-cli-venv/bin/activate

# Upgrade pip, setuptools, and wheel immediately.
# This is crucial to meet the project's PEP 517 requirements (setuptools>=61.0).
pip install --upgrade pip setuptools wheel
```

### 2. Check/Install LiteRT CLI

Ensure `litert-cli` and any required optional extensions (extras) are installed:

**Using UV:**
```bash
# Install in editable mode from local source
uv pip install -e .

# Or install from local source with extras (e.g., convert, lm, compile)
uv pip install -e ".[convert,lm,compile]"
```

**Using standard Pip:**
```bash
# Install in editable mode
pip install -e .

# Or install with extras
pip install -e ".[convert,lm,compile]"
```

## Core Commands

### 1. Run (Inference)

Run a tflite model locally on desktop or on a adb connected Android device.

**Desktop Execution (CPU and Local GPU):** `litert run <path_to_model>
--desktop --cpu --quiet` * `--quiet`: Silences verbose C++ setup logs. Highly
recommended for agents to parse clean output. * `--gpu`: Use desktop GPU if
available.

**Android Execution:** `litert run <path_to_model> --android --cpu`

**Multi-Input Formats (Literals or Arrays):** `litert run model.tflite
--desktop --input inputs="[0.5, 0.5, 0.5]" --print_tensors`

**Multi-Input Formats (Files - .npy, .raw, .png):** `litert run
model.tflite --desktop --input inputs="test_input.npy" --print_tensors`

### 2. Quantize

**Standard Selection:** `litert quantize <path_to_model> --output
<output_path>`

**Dynamic Quantization (int8_dynamic):** `litert quantize model.tflite
--type int8_dynamic --output dynamic.tflite`

**Static Quantization with Calibration Data:** `litert quantize
model.tflite --type static --calibration-data "calib_data.py" --output
static.tflite`

**Recipe-based Quantization:** `litert quantize model.tflite --recipe
"recipe.json" --output recipe.tflite`

### 3. Visualize

Launch the Model Explorer to visualize the model structure.

```bash
litert visualize <path_to_model>
```

**Advanced Visualisation Options:** * `--no_reuse_server`: Force creation of a
NEW server port. * `--stop_all`: Kill all existing visualization background
servers.

```bash
litert visualize --stop_all
```

### 4. Download

Download public LiteRT models from HuggingFace Hub or direct URL.

```bash
litert download <repo_id_or_url> --output <output_dir>
```

**Key Behavioral Nuance (Custom Output vs Centralized Cache):**
*   **Direct URL Downloads**: Metadata is **never** saved.
*   **HuggingFace Downloads (Default Central Cache)**: If `--output` is **omitted**, it downloads to `~/.cache/litert-cli/models/` and **automatically** creates `metadata.json` to catalog the model for CLI commands (like `litert list`).
*   **HuggingFace Downloads (Custom Folder)**: If `--output` is **provided**, it acts as a pure, clean download of only the model files. It **does not** generate a `metadata.json` file in the output folder.

**Filter by File Type:**
```bash
litert download litert-community/MobileNet-v3-large --file "*.tflite" --output ./models
```

**With Custom Model Reference:**
```bash
litert download litert-community/MobileNet-v3-large --model_ref my_model_ref
```

### 5. Import

Import a local file or directory into the centralized cache.

```bash
litert import my_model.tflite --model_ref my_model
```

### 6. List

List all managed models or detailed contents of a specific model.

```bash
litert list
litert list my_model
```

### 7. Convert (PyTorch to LiteRT)

Convert a PyTorch or HuggingFace model into a LiteRT model.

**From HuggingFace Model Hub:**
```bash
litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen
```

**From Generic Python Script:**
```bash
litert convert my_model.py --output /tmp/mymodel
```
*   `--model-func`: Name of function that returns the model
    (`torch.nn.Module`). Default: `get_model`.
*   `--input-func`: Name of function that returns sample inputs. Default: `get_args`.

### 8. Large Language Models (LM)

Interact with LLM generative models (like Qwen 1.5 or Gemma 2) using native `litert-lm-cli` or python fallback.

```bash
litert lm run <model_path_or_repo_id>
```

**Run with prompt and specific model file path:**
```bash
# Generative LLM models require the path to the compiled .litertlm model file or directory
litert lm run <model_dir>/model.litertlm --prompt "What is edge AI?"
```

### 9. Benchmark

Benchmark LiteRT models on different platforms (Android, Google Cloud, or Desktop).

**On connected Android device via ADB (CPU, GPU, or NPU):**
```bash
# Benchmark on CPU (Default)
litert benchmark model.tflite --android --cpu

# Benchmark on NPU (Requires compiling for NPU first)
litert benchmark model.tflite --android --npu

# Benchmark on GPU (using OpenCL/OpenGL delegates)
litert benchmark model.tflite --android --gpu
```

**On Macbook (CPU):**
```bash
litert benchmark my_model_ref --desktop --cpu
```

**On Google Cloud AI Edge Portal devices (e.g., Pixel 7):**
```bash
# Benchmark on GCP Pixel 7 CPU
litert benchmark model.tflite --gcp --device "pixel 7"

# Benchmark on multiple devices at once on GPU
litert benchmark model.tflite --gcp --devices "pixel 7, sm-s931u1" --gpu
```

### 10. Compile (NPU Offline AOT Compilation)

Apply Ahead-of-Time (AOT) offline compilation to a standard LiteRT (.tflite) model for specific edge SoC target NPUs (e.g., Qualcomm sm8550, MediaTek mt6989).

**Basic target NPU compilation:**
```bash
litert compile my_model.tflite --target sm8750
```

**Compile for multiple NPU targets and export an Android AI Pack (for PODAI deployment):**
```bash
litert compile my_model.tflite --target sm8550 --target mt6989 --export-aipack my_npu_models
```

**Compile and specify a custom output directory:**
```bash
litert compile my_model.tflite --target sm8750 --output-dir ./compiled
```

**Update target SoC metadata configurations from GitHub repository:**
```bash
# Pass 'main' for latest targets, or a version tag like 'v2.1.4'
litert compile --update-targets main
```

### 11. Delete

Delete a managed model from the centralized cache.

```bash
litert delete my_model
```

### 12. Clean

Clean up model cache, etc.

```bash
litert clean
```

## 🧪 Testing

Agents should run tests after modifying code to ensure no regressions.

To run unit tests locally:
```bash
python litert_test.py
python litert_help_test.py
```

To run tests in Google3:
```bash
blaze test //third_party/py/litert_cli:litert_test
blaze test //third_party/py/litert_cli:litert_help_test
```

## Best Practices for Agents

*   Pipe outputs to text files or grep them if you are looking for specific tensor shapes or runtime metrics.
*   Always use `--seed` when setting up virtual environments with `uv venv` if you plan to rely on the CLI's dynamic on-the-fly installation of extras.

## 🌟 High-Tier Developer Scenario Prompts

These complex prompts showcase how to combine and leverage this skill. You can use them directly in agent queries:

### Prompt 1: Dynamic Quantization & Android GPU Benchmarking
> "Download the FP32 EfficientNet model `litert-community/efficientnet_b1` from HuggingFace Hub. Quantize it to INT8 dynamic range (`--type int8_dynamic`), then benchmark both the original FP32 model and the newly quantized INT8 model on the GPU of my connected Android device. Compare the average latency and report the throughput speedup."

### Prompt 2: End-to-End Qwen LLM Conversion & Local Prompt Execution
> "Initialize a Python virtual environment using `uv` with Python 3.13, and do a local editable install of `litert-cli` along with `convert` and `lm` optional dependencies. Then, convert the generative model `Qwen/Qwen1.5-0.5B-Chat` from HuggingFace Hub to LiteRT format. Run a local inference on the compiled `model.litertlm` file using the prompt 'Explain edge machine learning in one sentence' and save the output."

### Prompt 3: ResNet Compilation & Custom NPU Compilation Target
> "Create a clean local sandbox folder and install the `litert-cli` tool. Download the `resnet18` model, compile it natively for the Qualcomm `sm8750` target NPU, and export the compiled `.tflite` model inside `./models/compiled`."
