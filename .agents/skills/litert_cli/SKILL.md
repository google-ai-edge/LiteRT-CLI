---
name: litert-cli
description: LiteRT CLI tool to download, convert, quantize, compile (NPU AOT), run, benchmark, and visualize LiteRT models, run LLMs with LiteRT-LM (lm), and manage cached models (import, list, delete, clean).
---

# LiteRT-CLI Agent Skill

This skill allows the agent to download, convert, quantize, compile, run,
benchmark, and visualize LiteRT models, run LLMs (`litert lm`), and manage the
local model cache (`import`, `list`, `delete`, `clean`) using the `litert`
command on desktop, device, or Google Cloud.

## Setup & Prerequisites (Agent Auto-Setup Guide)

Before running any `litert` commands, you must ensure a Python virtual
environment is active and the `litert-cli` package is correctly installed.

LiteRT CLI requires Python 3.10 or newer (3.10 is deprecated; prefer 3.11+, it
is verified in 3.13). Install either the nightly `litert-cli-nightly` or the
stable `litert-cli` package, never both in the same virtual environment.
Command dependencies (extras such as `convert`, `quantize`, `compile`, `lm`,
`visualize`) are installed on first use; to pre-install them (offline or CI),
install e.g. `litert-cli-nightly[convert,lm]` or `litert-cli-nightly[all]`.

### Option 1: Install from Local Clone (Editable Mode)

Use this method if you are developing inside the cloned repository clone:

#### Using `uv` (Recommended - Super Fast)

```bash
# Create a venv with seed packages (critical for dynamic deps.py auto-installers)
uv venv --clear --python=3.13 --seed
source .venv/bin/activate

# Install local clone
uv pip install -e .
```

#### Using standard venv & pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

### Option 2: Install Standalone from PyPI (Nightly)

Use this method if you are installing the published package:

#### Using `uv`

```bash
uv venv --clear --python=3.13 --seed
source .venv/bin/activate
uv pip install litert-cli-nightly
```

#### Using standard venv & pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install litert-cli-nightly
```

> [!TIP]
>
> If you encounter package resolution or network errors with `uv`, set the
> standard PyPI index URL first: `export UV_INDEX_URL=https://pypi.org/simple`

## Core Commands

### 💡 The Model Reference (`model-ref`) System

To avoid handling complex and fragile absolute filesystem paths, the LiteRT CLI
uses a centralized **Model Reference (`model-ref`)** catalog.

When you download or import a model to the centralized cache, you can assign it
a reference alias (and optional sub-references):

*   **Format**: `<alias_name>` or `<alias_name>:<sub_reference>` (e.g.,
    `mobilenet`, `resnet18:gpu`, `efficientnet:int8`).
*   **Default alias**: For HuggingFace downloads, if `--model-ref` is omitted,
    the CLI automatically assigns a flattened repository ID (e.g.,
    `litert-community__MobileNet-v3-large`) as the default alias.

Once a model is registered, **all CLI commands** (including `run`, `benchmark`,
`compile`, `delete`, `list`) accept this `<model_ref>` directly instead of a
file path! The CLI will automatically resolve it to the correct absolute cache
file path on the fly.

Here are some examples of using model references in action:

```bash
# Run inference using the central alias directly
litert run mobilenet --android --cpu

# Benchmark using a specific sub-reference GPU file
litert benchmark resnet18:gpu --android --gpu

# Compile for NPU directly using the reference alias
litert compile efficientnet --target sm8750

# Delete from the central cache (-y skips the confirmation prompt)
litert delete mobilenet -y
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
>
> If `--output` is omitted during HuggingFace downloads, the model is downloaded
> to `~/.cache/litert-cli/models/` and cataloged automatically via
> `metadata.json` (associating it with the repo ID as the `model-ref`). If
> `--output` is provided, it is treated as a standalone folder and is **not**
> cataloged.

### 2. Convert (PyTorch to LiteRT)

Convert a PyTorch or HuggingFace model into a TFLite model (LLMs into a
`.litertlm` bundle).

```bash
# Automated HF model conversion
litert convert Qwen/Qwen3-0.6B --output /tmp/qwen

# Print the resolved export plan (task, target, settings) without exporting
litert convert Qwen/Qwen3-0.6B --dry-run

# Conversion with INT8 Weight-Only Quantization
litert convert Qwen/Qwen3-0.6B --quantize-recipe weight_only_wi8_afp32 --output /tmp/qwen_w8

# Explicitly select the export task (e.g. text-only export of a multimodal model)
litert convert google/gemma-4-E2B-it --task text_generation --output /tmp/gemma4_text

# Export for NPU (uncompiled, no AOT compilation)
litert convert Qwen/Qwen3-0.6B --target npu:qualcomm --output /tmp/qwen_npu

# Use a custom LlmMetadata (chat template, stop tokens) for the .litertlm bundle
litert convert Qwen/Qwen3-0.6B --llm-metadata my_metadata.pbtext --output /tmp/qwen_meta

# From a custom local PyTorch script
litert convert my_model.py --quantize-recipe dynamic_wi8_afp32 --output /tmp/mymodel

# From a script, AOT compiled for Qualcomm SM8750 (repeat --target for more SoCs)
litert convert my_model.py --target sm8750 --output /tmp/mymodel_npu
```

*   **`--target` depends on the mode.** Hugging Face mode (model id or local
    checkpoint directory) takes one of `cpu_gpu` (default), `cpu`, `gpu`,
    `npu`, `npu:<vendor>` (export only, no AOT). Script mode (`.py`) takes NPU
    SoC codenames like `sm8750`, repeatable, and AOT-compiles for each.
    `--export-aipack <dir>` works in script mode only (with `--target`).
*   `--dry-run` and `--task` (`auto` by default) apply to Hugging Face mode.
*   `--llm-metadata` is `auto` by default (canonical LiteRT-LM config for
    Qwen2.5, Qwen3, Qwen3.5, LFM2, MiniCPM5), `none`, or a pbtext path.
*   **Gated Hugging Face models** (e.g. `google/gemma-3-1b-it`) need the
    license accepted on Hugging Face and a login: `hf auth login`, or set
    `HF_TOKEN`.
*   LLM conversion takes lots of memory and disk, and several minutes.

> [!NOTE]
>
> **Custom Python Script Interface (`my_model.py`):** To convert from a custom
> Python script, the file must expose functions to return the instantiated
> PyTorch model and generate sample inputs for tracer graph execution: *
> `--model-func`: Function name returning the model (`torch.nn.Module`).
> Default: `get_model`. * `--input-func`: Function name returning sample trace
> inputs (tuple/dict). Default: `get_args`.

**Minimal Script Example:**

```python
import torch

def get_model() -> torch.nn.Module: return MyPyTorchModel()

def get_args() -> tuple: return (torch.randn(1, 3, 224, 224),)
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

Apply Ahead-of-Time (AOT) offline compilation to a standard TFLite model for
edge SoC target NPUs.

> [!NOTE]
>
> Currently only supported on Linux hosts for Qualcomm targets. Other targets
> are coming soon!

```bash
# Basic offline compile for Qualcomm sm8750 NPU. Writes
# model_Qualcomm_SM8750.tflite into the current directory (or --output-dir).
litert compile model.tflite --target sm8750

# Compile for multiple NPUs and export as Android AI Pack
litert compile model.tflite --target sm8750 --target mt6989 --export-aipack my_npu_models

# Download the latest SoC metadata target maps from GitHub before compiling
# (the model and --target are still required)
litert compile model.tflite --target sm8750 --update-targets main
```

*   The `--export-aipack` directory must be new or empty; the CLI never
    deletes it, so remove a previous AI Pack before re-exporting.

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

*   Output logs are **clean by default**. To enable C++ verbose debug logs, set
    the environment variable:`export LITERT_VERBOSE=1`.

**Android Execution (CPU, GPU, or NPU):**

```bash
# Run on connected Android device (CPU side)

litert run model.tflite --android --cpu

# Enable GPU acceleration with CPU fallback

litert run model.tflite --android --accelerator gpu,cpu

# Enable NPU acceleration with CPU fallback (JIT compilation mode)

litert run standard_model.tflite --android --accelerator npu,cpu
```

*   **NPU Ahead-Of-Time (AOT) execution mode**: Pass an already NPU-compiled
    TFLite model (compiled offline via `litert compile`). The on-device runtime
    loads the compiled binary block directly, avoiding graph-compilation warmup
    overhead:

```bash
litert run resnet18_Qualcomm_SM8750.tflite --android --npu
```

*   `litert run` defaults to `--desktop`; pass `--android` for a device.
*   Fallback order: with `--npu`, `--gpu` and `--cpu` flags the order is always
    NPU, then GPU, then CPU, regardless of flag order. Only `--accelerator
    a,b,...` honors the given order.

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

Benchmark LiteRT models on different platforms (Android, Google Cloud, or
Desktop). Without `--desktop` or `--ddp`, `litert benchmark` targets a
connected Android device (`--android` is the default).

```bash
# Benchmark on connected Android device via CPU
litert benchmark model.tflite --android --cpu

# On the CPU and the GPU (--android, --desktop, --ddp), benchmark_model runs twice: a first process writes the XNNPACK and
# GPU caches (no inference), the measured process reads them and reports the peak memory. The GPU cache needs a
# benchmark_model with --gpu_serialization_dir (nightly has it, 2.2.0 does not). If a process with the cache flags fails,
# benchmark_model runs once without them, as before, and the report says so; LITERT_DISABLE_MODEL_CACHES=1 runs it once
# from the start.

# Benchmark on Android GPU
litert benchmark model.tflite --android --gpu

# Benchmark on Android NPU (JIT compilation mode)
litert benchmark model.tflite --android --npu

# Benchmark compiled AOT model on Android NPU
litert benchmark model_Qualcomm_SM8750.tflite --android --npu

# Benchmark on macOS CPU
litert benchmark my_model_ref --desktop --cpu

# Benchmark on Developer Device Platform (DDP) devices (Google Cloud)
# Note: You must enable the Device Run API in your GCP project, authenticate using
# 'gcloud auth application-default login', and configure --gcp-project.
# Find device ids with 'gcloud beta device-run devices list --project <PROJECT_ID>' (default device is pa3q-35).
# The devices run benchmark_model of LiteRT 2.2.0; set DDP_LITERT_VERSION to pick another release.
litert benchmark model.tflite --ddp --gcp-project "your-gcp-project-id"
litert benchmark model.tflite --ddp --device grizzly-37 --gcp-project "your-gcp-project-id"
litert benchmark model.tflite --ddp --devices "grizzly-37, pa3q-35" --gpu --gcp-project "your-gcp-project-id"

# A .litertlm bundle on DDP devices runs LiteRT-LM's prebuilt benchmark binary (gs://litert/binaries/latest/android_arm64/litert_lm/;
# DDP_LITERT_LM_VERSION picks another version there, DDP_LITERT_LM_DIR a gs:// directory laid out the same way): prefill and decode tokens/s at --prefill-tokens / --decode-tokens
# (default 1024 / 256), --num-iterations times (default 5). The binary runs twice: a warm-up process (one iteration, a cold
# Init) writes the caches beside the bundle, then the measured process loads them; the printed medians leave out its first
# --warmup-runs iterations (default 1) and the summary shows both Init times and the peak memory. The metrics proto and
# provenance land under ~/.cache/litert-cli/ddp/<session>/<job>/.
litert benchmark model.litertlm --ddp --device grizzly-37 --gpu --gcp-project "your-gcp-project-id"
```

### 7. Large Language Models (LM)

Interact with LLM generative models (like Qwen3 or Gemma 4) using native
`litert-lm` utilities. `litert lm` forwards its arguments to `litert-lm`, and
additionally resolves Hugging Face repo ids (see below).

> [!TIP]
>
> **Non-interactive / Background Execution:** When running LLM inferences in
> scripts or background tasks, the process will block waiting for chat prompts
> on `stdin`. To prevent hanging, **always redirect stdin from `/dev/null`**
> (i.e. append `< /dev/null` to the end of command).

```bash
# Run a generative model loading from Hugging Face
litert lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --prompt="What is the capital of France?" < /dev/null

# Run using a local compiled LLM (.litertlm) file
litert lm run gemma-4-E2B-it.litertlm --prompt "Hello, how are you?" < /dev/null

# Benchmark a generative LLM model
litert lm benchmark gemma-4-E2B-it.litertlm
```

**LLMs from Hugging Face ids:** when the first argument after `run` or
`benchmark` is a Hugging Face repo id (`org/name`) that is not a local path:

*   A repo with a `.litertlm` file runs with `--from-huggingface-repo org/name`.
*   A repo with a transformers checkpoint is converted once with `litert
    convert` into `~/.cache/litert-cli/models/<org>__<name>/`, and later runs
    reuse the cached bundle. `litert lm` reads (and does not forward)
    `--quantize` (default `dynamic_wi8_afp32`, `none` for fp32),
    `--prefill-lengths` (default `256`), `--cache-length` (default `4096`) and
    `--reconvert` (convert again even if cached).
*   Gated repos need `hf auth login` or `HF_TOKEN`.

```bash
litert lm run Qwen/Qwen3-0.6B --prompt "What is the capital of France?" < /dev/null
litert lm benchmark Qwen/Qwen3-0.6B
litert lm run Qwen/Qwen3-0.6B --cache-length 8192 --prompt "Hi" < /dev/null
litert lm run Qwen/Qwen3-0.6B --reconvert --prompt "Hi" < /dev/null
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
# Delete a model reference from the cache (-y: no confirmation prompt)
litert delete my_model -y
```

### 12. Clean

Delete the whole local cache (`~/.cache/litert-cli`): all managed models
(`litert list`), the bundles converted by `litert lm`, and downloaded binaries,
plus the LiteRT CLI directory on a connected Android device. Only use it when
the user asks to free disk space, or as a last resort (e.g. NPU library version
mismatch), since models must be downloaded or converted again afterwards.

```bash
# Asks for confirmation; -y skips it
litert clean -y
```

## 🧪 Testing

Agents should run tests after modifying code to ensure no regressions.

To run unit tests locally (from the repository root, with `pytest` installed):

```bash
python -m pytest litert_cli
```

To run the smoke tests (creates a fresh virtual environment, checks the command
help and runs the unit tests; takes no arguments):

```bash
./examples/run_smoke_tests.sh
```

To run comprehensive end-to-end per-command and per-model demos:

```bash
./examples/run_commands.sh
./examples/run_models.sh
```

## Best Practices for Agents

*   Pipe outputs to text files or grep them if you are looking for specific
    tensor shapes or runtime metrics.
*   **Avoid hanging background processes**: When executing the `litert lm run`
    command in a script or in the background, **always** append `< /dev/null` to
    redirect standard input. Otherwise, the process will block indefinitely
    waiting on stdin.
*   **Avoid confirmation prompts**: `litert delete` and `litert clean` ask for
    confirmation, so always pass `-y` when running them non-interactively.
*   **Explore Demos**: Refer to the `examples/` directory to explore
    comprehensive per-command demos (under `examples/commands/`) and
    model-specific demos (under `examples/models/`) for complete automation
    patterns.
*   **Read Troubleshooting Guides**: Refer to the main project `README.md`
    file's **Troubleshooting and tips** section for platform-specific
    environmental setup guides, adb port recoveries, and NPU offline compiler
    clang version requirements.

## 🤖 Example Agent Prompts

These prompts demonstrate how developers can leverage this skill. You can copy
and use them directly in your agent queries:

*   "Download LiteRT model `litert-community/efficientnet_b1` and run it on CPU"
*   "Benchmark LiteRT model `litert-community/efficientnet_b1` on my Android
    GPU"
*   "Compile LiteRT model `litert-community/efficientnet_b1` for NPU target
    `sm8750`"
*   "Visualize LiteRT model `litert-community/efficientnet_b1`"
*   "Download the FP32 EfficientNet model `litert-community/efficientnet_b1`
    from HuggingFace. Quantize it to INT8 dynamic range (`--recipe
    dynamic_wi8_afp32`), then benchmark both the FP32 and INT8 models on my
    Android GPU, comparing the throughput speedup."
*   "Convert the model `Qwen/Qwen3-0.6B` from HuggingFace Hub to LiteRT
    format, and run it locally with the prompt 'Explain edge machine learning in
    one sentence'."
*   "Download `litert-community/efficientnet_b1`, offline compile (AOT) it for
    the `sm8750` target NPU into `./models/compiled`, then run on-device
    inference and benchmark on the NPU, confirming zero JIT warmup latency."
