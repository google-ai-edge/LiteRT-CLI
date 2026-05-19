---
name: litert-cli
description: LiteRT CLI tool to convert, download, quantize, run, benchmark, and visualize LiteRT models.
---

# LiteRT-CLI Agent Skill

This skill allows the agent to download, convert, quantize, run, benchmark, and
visualize LiteRT models using the `litert` command on desktop, device, or Google
Cloud.

## Setup & Prerequisites

Before running any `litert` commands, an agent should ensure it is in a Python
virtual environment and `litert-cli` is installed.

### 1. Check/Create Virtual Environment

We highly recommend using **`uv`** (written in Rust) for extremely fast environment management and package installs.

**Option A: Use UV (Recommended - Super Fast):**
```bash

# Create a virtual environment with Python 3.13.

# We use --seed to pre-install pip, setuptools, and wheel inside the venv.

# This is critical to allow our CLI dynamic dependency auto-installers (deps.py) to function.

# When meeting dependency resolution error, try to set environment variable:

# UV_INDEX_URL=https://pypi.org/simple

uv venv --clear --python=3.13 --seed source .venv/bin/activate ```

**Option B: Use Standard Pip/Venv:** ```bash

# Create and activate a standard Python virtual environment.

python3 -m venv litert-cli-venv source litert-cli-venv/bin/activate

# Upgrade pip, setuptools, and wheel immediately.

# This is crucial to meet the project's PEP 517 requirements (setuptools>=61.0).

pip install --upgrade pip setuptools wheel ```

### 2. Check/Install LiteRT CLI

Ensure `litert-cli` and any required optional extensions (extras) are installed:

**Using UV:**
```bash
# Install in editable mode from local source
uv pip install -e .

# Or install from local source with extras (e.g., convert, lm, compile)

uv pip install -e ".[convert,lm,compile]"
```

**Using standard Pip:** ```bash

# Install in editable mode

pip install -e .

# Or install with extras

pip install -e ".[convert,lm,compile]"
```

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

### 1. Run (Inference)

Run a tflite model locally on desktop or on a adb connected Android device.

**Desktop Execution (CPU and Local GPU):** `litert run <path_to_model> --desktop --cpu`
* Output logs are **clean by default**.
* To enable C++ verbose debug setup logs, set the environment variable: `export LITERT_VERBOSE=1`.
* `--gpu`: Use desktop GPU if available.
* **Accelerator Fallback**: If running on GPU (`--gpu`) fails, you can pass both **`--gpu --cpu`** (or `--accelerator gpu,cpu`). The CLI will attempt GPU first and gracefully fall back to CPU on failure.

**Android Execution (CPU, GPU, or NPU):** `litert run <path_to_model> --android --cpu`
* `--gpu`: Run on Android GPU using OpenCL/WebGPU.
* **Accelerator Fallback**: Similarly, pass both **`--gpu --cpu`** (or `--accelerator gpu,cpu`) on Android to use CPU as a fallback if GPU delegate creation fails.
* `--npu`: Run on Android device NPU. Supports **two execution paradigms** based on the input model:

**1. JIT (Just-In-Time) compilation mode:** Pass a standard, non-compiled
`.tflite` model. The on-device LiteRT runtime will automatically download/invoke
the vendor-specific compiler plugin to compile operators dynamically at graph
initialization time. `bash litert run standard_model.tflite --android --npu`

**2. AOT (Ahead-Of-Time) execution mode:** Pass an already NPU-compiled
`.tflite` model (compiled offline via `litert compile`). The on-device runtime
loads the compiled binary block directly on the NPU. This avoids
graph-compilation warmup overhead, leading to **sub-millisecond initialization
latency**. `bash litert run resnet18_compiled_sm8750.tflite --android --npu`

**Multi-Input Formats (Literals or Arrays):** `bash litert run model.tflite
--desktop --input inputs="[0.5, 0.5, 0.5]" --print-tensors`

**Multi-Input Formats (Files - .npy, .raw, .png):** `bash litert run
model.tflite --desktop --input inputs="test_input.npy" --print-tensors`

### 2. Quantize

**Standard Selection:** `bash litert quantize <path_to_model> --output <output_path>`

**Dynamic Quantization (dynamic_wi8_afp32):** `bash litert quantize model.tflite --recipe dynamic_wi8_afp32 --output dynamic.tflite`

**Static Quantization with Calibration Data:** `bash litert quantize
model.tflite --recipe static_wi8_ai8 --calibration-data "calib_data.py" --output
static.tflite`

**Custom JSON Recipe:** `bash litert quantize model.tflite --custom-recipe
"recipe.json" --output recipe.tflite`

### 3. Visualize

Launch the Model Explorer to visualize the model structure.

```bash
litert visualize <path_to_model>
```

**Advanced Visualization Options:**
*   `--no-reuse-server`: Force creation of a NEW server port.
*   `--stop-all`: Kill all existing visualization background servers.

```bash
litert visualize --stop-all
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

**With Custom Model Reference:** `bash litert download
litert-community/MobileNet-v3-large --model-ref my_model_ref`

### 5. Import

Import a local file or directory into the centralized cache.

```bash
litert import my_model.tflite --model-ref my_model
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

# With INT4 Weight-Only Quantization (Recommended for LLM)
litert convert Qwen/Qwen1.5-0.5B-Chat --quantize-recipe weight_only_wi4_afp32 --output /tmp/qwen_w4
```

**From Generic Python Script:** ```bash litert convert my_model.py --output
/tmp/mymodel

# With INT8 Dynamic Quantization

litert convert my_model.py --quantize-recipe dynamic_wi8_afp32 --output
/tmp/mymodel_quant ```*`--model-func`: Name of function that returns the model
(`torch.nn.Module`). Default:`get_model`. *`--input-func`: Name of function that
returns sample inputs. Default:`get_args`.
*`--quantize-recipe`(Alias`--quantize`): Quantization recipe to apply
(e.g.,`dynamic_wi8_afp32`,`weight_only_wi4_afp32`).

### 8. Large Language Models (LM)

Interact with LLM generative models (like Qwen 1.5 or Gemma 4) using native
`litert-lm` CLI.

> [!TIP] **Non-interactive / Background Execution best practice:** When running
> generative LLM inferences using the `litert lm run` command in scripts or in
> the background, the process will hang indefinitely waiting for the next chat
> prompt on standard input (`stdin`). To prevent this and ensure it outputs the
> prompt response and exits immediately, **always redirect stdin from
> `/dev/null`** (i.e., append `< /dev/null` to the command).

```bash
litert lm run <model_path_or_reference_id> < /dev/null
```

**Run with model file path:**
```bash
# Generative LLM models require the path to the compiled .litertlm model file or directory.
# Append < /dev/null to exit immediately after printing the answer.

litert lm run <model_dir>/model.litertlm --prompt "What is edge AI?" <
/dev/null ```

**Download and run with HuggingFace repo:** `bash litert lm run \
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
gemma-4-E2B-it.litertlm \ --prompt="What is the capital of France?" \ <
/dev/null`

### 9. Benchmark

Benchmark LiteRT models on different platforms (Android, Google Cloud, or
Desktop).

**On connected Android device via ADB (CPU, GPU, or NPU):**
```bash
# Benchmark on CPU (Default)
litert benchmark model.tflite --android --cpu

# Benchmark on NPU (Requires compiling for NPU first)

litert benchmark model.tflite --android --npu

# Benchmark on GPU (using OpenCL/OpenGL delegates)
litert benchmark model.tflite --android --gpu
```

**On Macbook (CPU):** `bash litert benchmark my_model_ref --desktop --cpu`

**On Google AI Edge Portal in Google Cloud (GCP):**

> [!IMPORTANT] **Prerequisites for GCP Benchmarking:** 1. Join Google AI Edge
> Portal early access program at: https://ai.google.dev/edge/ai-edge-portal 2.
> Authenticate your terminal session by running: `gcloud auth login` 3.
> Configure the GCP Project ID. You can either: * Set the environment variable:
> `export LITERT_GCP_PROJECT="your-gcp-project-id"` * Or explicitly pass the
> `--gcp-project` option in the command. 4. Configure the Google Cloud Storage
> (GCS) Bucket for model uploading. The CLI resolves it via: * Explicit
> `--gcp-bucket` CLI option. * `LITERT_GCP_BUCKET` environment variable. *
> Default fallback: Automatically creates and uses
> `gs://{gcp_project}-litert-models`.

```bash
# Benchmark on GCP Pixel 7 CPU (using default auto-created project bucket)
litert benchmark model.tflite --gcp --device "pixel 7" --gcp-project "your-gcp-project-id"

# Benchmark on GCP Pixel 7 CPU (specifying custom GCS bucket explicitly)
litert benchmark model.tflite --gcp --device "pixel 7" --gcp-project "your-gcp-project-id" --gcp-bucket "your-custom-bucket"

# Benchmark on multiple devices at once on GPU
litert benchmark model.tflite --gcp --devices "pixel 7, sm-s931u1" --gpu --gcp-project "your-gcp-project-id"
```

### 10. Compile (NPU Offline AOT Compilation)

Apply Ahead-of-Time (AOT) offline compilation to a standard LiteRT (.tflite) model for specific edge SoC target NPUs (e.g., Qualcomm sm8550, MediaTek mt6989).

**Basic target NPU compilation:**
```bash
litert compile my_model.tflite --target sm8750
```

**Compile for multiple NPU targets and export an Android AI Pack (for PODAI
deployment):** `bash litert compile my_model.tflite --target sm8550 --target
mt6989 --export-aipack my_npu_models`

**Compile and specify a custom output directory:** `bash litert compile
my_model.tflite --target sm8750 --output-dir ./compiled`

**Update target SoC metadata configurations from GitHub repository:** ```bash

# Pass 'main' for latest targets, or a version tag like 'v2.1.4'

litert compile --update-targets main ```

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
python litert_cli/litert_test.py
python litert_cli/litert_help_test.py
```

To run comprehensive end-to-end regression tests: `bash
./examples/run_smoke_tests.sh ./examples/run_commands.sh
./examples/run_models.sh`

## Best Practices for Agents

*   Pipe outputs to text files or grep them if you are looking for specific tensor shapes or runtime metrics.
*   **Avoid hanging background processes**: When executing the `litert lm run` command in a script or in the background, **always** append `< /dev/null` to redirect standard input. Otherwise, the process will block indefinitely waiting on stdin.

## 🌟 High-Tier Developer Scenario Prompts

These complex prompts showcase how to combine and leverage this skill. You can
use them directly in agent queries:

### Prompt 1: Dynamic Quantization & Android GPU Benchmarking

> "Download the FP32 EfficientNet model `litert-community/efficientnet_b1` from
> HuggingFace. Quantize it to INT8 dynamic range (`--recipe dynamic_wi8_afp32`),
> then benchmark both the original FP32 model and the newly quantized INT8 model
> on the GPU of my connected Android device. Compare the average latency and
> report the throughput speedup."

### Prompt 2: End-to-End Qwen LLM Conversion & Local Prompt Execution

> "Create a python envinroment with UV, install `litert-cli`, convert the model
> `Qwen/Qwen1.5-0.5B-Chat` from HuggingFace Hub to LiteRT format, and run it
> locally using the prompt 'Explain edge machine learning in one sentence'."

### Prompt 3: NPU AOT Compilation, Inference, and Benchmarking Pipeline

> "Download EfficientNet from huggingface repo
> `litert-community/efficientnet_b1`. On a Linux host machine, offline compile
> (AOT) the model for the `sm8750` target NPU, and output the compiled model
> inside `./models/compiled`. Then, run an on-device inference and benchmark
> using this newly compiled AOT model on the connected Android device's NPU
> (`--npu`). Confirm that the compiled graph loads directly without dynamic JIT
> compilation warmup latency."
