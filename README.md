# LiteRT CLI (Preview)

A convenient command-line toolkit to streamline
[LiteRT](https://ai.google.dev/edge/litert) related development workflows,
including converting, quantizing, compiling, running, benchmarking and
visualizing LiteRT (TFLite) models on various hardware (CPU / GPU / NPU) across
platforms (desktop, mobile, or cloud).

🚀 [Installation](#-installation) ｜ 📓
[Try Colab](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/examples/litert_cli.ipynb)
| ⚡ [LiteRT 101](https://codelabs.developers.google.com/codelabs/litert-cli/101)
| ⚡ [Quick start](#-quick-start) | 💡 [Commands](#-common-commands) | 🤖
[Coding agent](#-use-in-coding-agent)

> [!NOTE]
>
> It's still an early preview under active development, thus has limited
> platform and feature support, plus possible bugs. We appreciate your patience
> and feedback to help us improve it. Welcome issues and PRs!

LiteRT CLI is built on top of [Google AI Edge](https://ai.google.dev/edge)
stacks, including [LiteRT](https://github.com/google-ai-edge/LiteRT),
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM),
[LiteRT Torch](https://github.com/google-ai-edge/LiteRT-Torch),
[LiteRT Quantizer](https://github.com/google-ai-edge/LiteRT/tree/main/litert_quantizer),
[Developer Device Platform](https://docs.cloud.google.com/developer-device-platform/overview),
and [Model Explorer](https://ai.google.dev/edge/model-explorer).

--------------------------------------------------------------------------------

## ⚙️ Build Status

Build Type                | Status
------------------------- | ------
Unit Tests (Linux, macOS) | [![](https://github.com/google-ai-edge/LiteRT-CLI/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/LiteRT-CLI/actions/workflows/nightly_unittests.yml)
Nightly Release           | [![](https://github.com/google-ai-edge/LiteRT-CLI/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/LiteRT-CLI/actions/workflows/nightly_release.yml)

--------------------------------------------------------------------------------

## 🚀 Installation

Please install `litert-cli-nightly` from PyPI or from local clone. LiteRT CLI
will install the dependencies on-demand, based on which commands to run, to
speed up initial installation.

LiteRT CLI requires Python 3.10 or newer, and is verified in Python 3.13.
Support for Python 3.10 is deprecated (it prints a warning) and will be removed
in the next release, following its upstream end-of-life, so please use Python
3.11 or newer.

We support installation using either
**[uv](https://docs.astral.sh/uv/getting-started/installation/)** (recommended
for ultra-fast dependency resolution) or standard
**[pip](https://pip.pypa.io/)** within a Python virtual environment.

### Option 1: Use uv (recommended)

`uv` is an extremely fast Python package manager written in Rust.

```bash
# 1. Create a virtual environment with Python 3.13.
# TIP: Sometimes setting env var `UV_INDEX_URL=https://pypi.org/simple` helps
# resolve dependency resolution errors, especially inside corporate networks.
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

### Stable vs nightly

*   `litert-cli-nightly` is built every night from `main`, and depends on the
    nightly LiteRT packages (e.g. `ai-edge-litert-nightly`,
    `litert-torch-nightly`). It gets new features and fixes first.
*   `litert-cli` is the stable release, and depends on pinned stable versions
    of the LiteRT packages.

Both packages provide the same `litert` command, so install only one of them
in a virtual environment; don't mix them (or their nightly and stable
dependencies) in the same environment.

--------------------------------------------------------------------------------

## ⚡ Quick start

### 📓 Try Colab

Try
[LiteRT CLI Colab](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/examples/litert_cli.ipynb)
to explore different features quickly.

### 📓 LiteRT 101

[LiteRT 101](https://codelabs.developers.google.com/codelabs/litert-cli/101) is
a hands-on tutorial that walks you through the basics of using the LiteRT CLI
tool.

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

### Run LLMs

You can run LLMs like Gemma4-E2B, using commands like `litert lm run`, which
forwards its arguments to the LiteRT-LM CLI (`litert-lm run`), and also accepts
Hugging Face repo ids. Please follow
[LiteRT-LM CLI](https://ai.google.dev/edge/litert-lm/cli) for detailed
instructions, and [Run and benchmark LLMs](#7-run-and-benchmark-a-generative-llm-model-using-litert-lm-cli)
for what `litert lm` adds.

```bash
litert lm run  \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="What is the capital of France?"

# A Hugging Face repo without a .litertlm file is converted once, then cached
litert lm run Qwen/Qwen3-0.6B --prompt="What is the capital of France?"
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
*   *Convert the model `Qwen/Qwen3-0.6B` from HuggingFace, and run it locally
    using the prompt 'Explain edge machine learning one sentence'*
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

### Optional extras

The dependencies of the heavier commands are installed on first use. To install
them up front, e.g. for offline machines or CI, install the extras with the
package (use `litert-cli[...]` for the stable package, or `.[...]` for a local
clone):

Extra         | Installs dependencies for
------------- | ------------------------------------------------------------
`[convert]`   | `litert convert` (LiteRT Torch, PyTorch, Transformers, ...)
`[quantize]`  | `litert quantize` (LiteRT Quantizer)
`[compile]`   | `litert compile` (NPU AOT compiler SDKs)
`[lm]`        | `litert lm` (LiteRT-LM)
`[visualize]` | `litert visualize` (`ai-edge-model-explorer`)
`[all]`       | All extras

```bash
uv pip install "litert-cli-nightly[convert,lm]"
```

### Environment variables

Variable                                     | Description
-------------------------------------------- | -----------
`LITERT_VERBOSE`                             | Set to `1` to enable verbose logging.
`LITERT_GCP_PROJECT`                         | Default GCP project for `litert benchmark --ddp`, when `--gcp-project` is not set.
`LITERT_GCP_BUCKET`                          | Default GCS bucket for `litert benchmark --ddp`, when `--gcp-bucket` is not set. Defaults to `<project>-devicerun`.
`LITERT_DISABLE_MODEL_CACHES`                | Set to `1` to make `litert benchmark` run `benchmark_model` once, without the XNNPACK and GPU caches.
`DDP_LITERT_VERSION`                         | LiteRT release of the prebuilt `benchmark_model` used on DDP devices (default `2.2.0`; e.g. `nightly`).
`DDP_LITERT_LM_VERSION`, `DDP_LITERT_LM_DIR` | Version (default `latest`), or a whole `gs://` directory, of the prebuilt LiteRT-LM benchmark binary used for `.litertlm` bundles on DDP devices.
`LITERT_LM_MODELS_REF`                       | LiteRT-LM git ref (default `main`) to read the canonical model configs from in `litert convert`; set it to a commit for reproducible conversions.
`LITERT_CLI_ROOT`                            | Local directory for downloaded NPU SDKs and runtime libraries (default `~/.cache/litert-cli/root`).
`LITERT_CLI_ANDROID_ROOT`                    | Working directory on the Android device (default `/data/local/tmp/litert-cli`).
`HF_TOKEN`                                   | Hugging Face token for gated or private models (or run `hf auth login`).

### Troubleshooting and tips

*   Always activate python virtual environment before running `litert` command,
    to avoid conflicts.
*   When `uv` fails to resolve dependencies, try to set below environment
    variable first: `export UV_INDEX_URL=https://pypi.org/simple`.
*   When run fails on GPU using `--gpu` flag, try to add both `--gpu --cpu`
    flags in the command, then `litert run` will try GPU first, and fall back
    to CPU when GPU fails. With the `--npu`, `--gpu` and `--cpu` flags the
    order is always NPU, then GPU, then CPU, whatever order you pass them in;
    use `--accelerator gpu,cpu` to choose the order explicitly.
*   When `litert run` fails on Android device, if the device is not detected,
    try to run `adb kill-server` first.
*   When convert a traditional PyTorch model, you need to write a script to wrap
    it with required functions `get_model` and `get_args`. Check the script
    format in
    [resnet18.py](https://github.com/google-ai-edge/LiteRT-CLI/blob/main/litert_cli/test_data/resnet18.py).
*   `litert convert <hf_id>` supports Hugging Face `AutoModelForCausalLM`
    LLMs, plus these model families:
    *   LLMs and VLMs: Gemma 3, Gemma 3n, Gemma 4, LFM2, LFM2-VL, Qwen3 and
        Qwen3.5.
    *   Speech-to-text: Whisper, Moonshine, Parakeet (CTC and TDT) and
        Qwen3-ASR.
    *   Text-to-speech: Kokoro and Qwen3-TTS.
    *   Text-to-image: FLUX.2 [klein] and Bonsai FLUX.2.
    *   Any-to-any: MiniCPM-o.

    Use `litert convert <hf_id> --dry-run` to see the export plan for a model.
    Other architectures can fail to export.
*   For large models like LLMs, `litert convert` will take large memories and
    disks, and spend multiple minutes. Please make sure you have enough memory
    and disks, and be patient.
*   `litert compile` only supports running on Linux now, and it requires newer
    Clang version `18.x.x` or above. Try `sudo apt install clang libc++-18-dev
    libc++abi-18-dev`. An error file that says `libc++.so.1: cannot open
    shared object file` means the LLVM C++ runtime is missing.
*   When benchmark using `--ddp` flag, you need to 1) Enable the Device Run API
    of
    [Developer Device Platform](https://docs.cloud.google.com/developer-device-platform/overview)
    in a GCP project with billing enabled (the platform is in Preview, and
    sessions are billed to your project); 2) Login using `gcloud auth
    application-default login`; 3) Set your GCP project using
    `--gcp-project=<Your-GCP-Project>`. The devices run the prebuilt
    `benchmark_model` of LiteRT 2.2.0; set the environment variable
    `DDP_LITERT_VERSION` (for example `nightly`) to pick another release. A
    `.litertlm` bundle runs LiteRT-LM's prebuilt benchmark binary from
    `gs://litert/binaries/latest/android_arm64/litert_lm/`, with the shared
    libraries of that directory pushed beside it; `DDP_LITERT_LM_VERSION` picks
    another version there, and `DDP_LITERT_LM_DIR` a whole `gs://` directory
    laid out the same way.
*   When `litert visualize` fails to launch Model Explorer, try to run `litert
    visualize --stop-all` first.
*   Exporting environment variable `LITERT_VERBOSE=1` can enable verbose
    logging.
*   `litert clean` deletes the whole local cache (`~/.cache/litert-cli`),
    including all managed models (see `litert list`), the bundles converted by
    `litert lm`, and the downloaded binaries, plus the LiteRT CLI directory on
    a connected Android device. It asks for confirmation; pass `-y` to skip it.
    Use it to free disk space, or as a last resort for issues like an NPU
    library version mismatch.

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
# Automated HF Conversion (LLMs, VLMs, speech-to-text, text-to-speech, text-to-image, any-to-any)
litert convert Qwen/Qwen3-0.6B --output /tmp/qwen
litert convert google/gemma-4-E2B-it --output /tmp/gemma4
litert convert openai/whisper-tiny --output /tmp/whisper

# Print the resolved export plan without exporting
litert convert Qwen/Qwen3-1.7B --dry-run

# Override recommended export settings
litert convert Qwen/Qwen3-1.7B --quantize dynamic_wi4b32_afp32 \
  --prefill-lengths 128,512 --cache-length 4096 --output /tmp/qwen_w4

# Explicitly select the export task (e.g. text-only export of a multimodal model)
litert convert google/gemma-4-E2B-it --task text_generation --output /tmp/gemma4_text

# Generic Script Injection with INT8 Dynamic Quantization
litert convert my_model.py --quantize-recipe dynamic_wi8_afp32 --output /tmp/mymodel

# Use a custom LlmMetadata (chat template, stop tokens) for the .litertlm bundle
litert convert Qwen/Qwen3-1.7B --llm-metadata my_metadata.pbtext --output /tmp/qwen3

# Export a Hugging Face model for NPU (uncompiled, no AOT compilation)
litert convert Qwen/Qwen3-0.6B --target npu:qualcomm --output /tmp/qwen_npu

# AOT compile a PyTorch script model for Qualcomm SM8750 and SM8650 NPUs
litert convert my_model.py --target sm8750 --target sm8650 --output /tmp/mymodel_npu
```

`litert convert` has two modes, picked from `MODEL_OR_SCRIPT`:

*   **Hugging Face mode** (a Hugging Face model id or a local checkpoint
    directory): `--target` takes a single hardware target: `cpu_gpu`
    (default), `cpu`, `gpu`, `npu`, or `npu:<vendor>` (e.g. `npu:qualcomm`).
    The NPU targets export a model for NPU without AOT compilation. `--dry-run`
    prints the resolved export plan (task, target and export settings) and
    exits. `--task` selects the export task (`auto` by default, which detects
    it from the checkpoint; or `text_generation`, `image_text_to_text`,
    `multimodal_lm`, `automatic_speech_recognition`, `text_to_speech`,
    `text_to_image`, `any_to_any`).
*   **Script mode** (a `.py` file): `--target` takes NPU SoC codenames (e.g.
    `sm8750`), can be repeated, and AOT-compiles the converted model for each
    of them. `--export-aipack <dir>` (script mode only, with `--target`)
    exports an AI Pack instead.

For models with a canonical config in
[LiteRT-LM `models/<family>`](https://github.com/google-ai-edge/LiteRT-LM/tree/main/models),
such as Qwen2.5, Qwen3, Qwen3.5, LFM2 and MiniCPM5, `litert convert` tries to
reuse that config.

*   Gated Hugging Face models (e.g. `google/gemma-3-1b-it`) need you to accept
    their license on Hugging Face and log in first: `hf auth login`, or set
    `HF_TOKEN`.
*   `litert lm run` and `litert lm benchmark` run LLMs only, not other models
    such as ASR.

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
# Basic compilation for specific Qualcomm NPU (e.g., sm8750), which writes
# model_Qualcomm_SM8750.tflite into the current directory (or --output-dir)
litert compile model.tflite --target sm8750

# Compile for multiple targets and export an AI Pack for Android
litert compile model.tflite --target sm8750 --target sm8650 --export-aipack my_npu_models
```

The `--export-aipack` directory must be new or empty; the CLI never deletes an
existing directory, so remove a previous AI Pack yourself before re-exporting.

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
litert run model_Qualcomm_SM8750.tflite --android --npu

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
# On the CPU and the GPU (--android, --desktop, --ddp), benchmark_model runs twice: a first process
# compiles the model and writes the XNNPACK and GPU caches (no inference), then the measured process
# reads them and reports the peak memory too. The GPU cache needs a benchmark_model with
# --gpu_serialization_dir: the nightly binaries have it, 2.2.0 does not. If a process with the cache
# flags fails, benchmark_model runs once without them, as before, and the report says so;
# LITERT_DISABLE_MODEL_CACHES=1 runs it once from the start.

# Without --desktop or --ddp, litert benchmark runs on a connected Android device.

# Benchmark on Android (CPU side)
litert benchmark my_model_ref --android --cpu
litert benchmark model.tflite --android --cpu

# Benchmark on Android NPU (JIT mode)
litert benchmark model.tflite --android --npu

# Benchmark AOT compiled model on Android NPU
litert benchmark model_Qualcomm_SM8750.tflite --android --npu

# Benchmark on Android GPU
litert benchmark model.tflite --android --gpu

# Benchmark on macOS (CPU)
litert benchmark my_model_ref --desktop --cpu

# Benchmark on Developer Device Platform (DDP) devices in Google Cloud. Prerequisites:
# - Enable the Device Run API in your GCP project: gcloud services enable devicerun.googleapis.com
# - Set up authentication by running: gcloud auth application-default login
# - Find device ids by running: gcloud beta device-run devices list --project "your-gcp-project-id"
litert benchmark model.tflite --ddp --gcp-project "your-gcp-project-id"
litert benchmark model.tflite --ddp --device grizzly-37 --gcp-project "your-gcp-project-id"
litert benchmark model.tflite --ddp --devices "grizzly-37, pa3q-35" --gpu --gcp-project "your-gcp-project-id"

# A .litertlm bundle on DDP devices runs LiteRT-LM's benchmark binary twice: a warm-up process (one
# iteration, a cold Init) writes the caches beside the bundle, then the measured process loads them and
# runs --num-iterations times (default 5) at --prefill-tokens / --decode-tokens (default 1024 / 256).
# The printed medians leave out its first --warmup-runs iterations (default 1); the summary also shows
# both Init times and the peak memory.
litert benchmark model.litertlm --ddp --device grizzly-37 --gpu --gcp-project "your-gcp-project-id"
```

### 7. Run and benchmark a generative LLM model using LiteRT-LM CLI

`litert lm` forwards its arguments to `litert-lm` (the `lm` extra is installed
on first use), so `litert lm run ./my_model.litertlm` and `litert-lm run
./my_model.litertlm` achieve the same results. On top of that, `litert lm run`
and `litert lm benchmark` also accept a Hugging Face repo id (see below), and
read the `--quantize`, `--prefill-lengths`, `--cache-length` and `--reconvert`
conversion options themselves. `litert lm --help` prints a short note about
these before the native `litert-lm` help.

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

# Run or benchmark a Hugging Face repo by its id
litert lm run Qwen/Qwen3-0.6B --prompt "What is the capital of France?"
litert lm benchmark Qwen/Qwen3-0.6B

# Convert with a longer KV cache (a second bundle), or rebuild the cached one
litert lm run Qwen/Qwen3-0.6B --cache-length 8192
litert lm run Qwen/Qwen3-0.6B --reconvert
```

When the first argument after `run` or `benchmark` is a Hugging Face repo id
(`org/name`) that is not a local path:

*   A repo with a `.litertlm` file runs with `--from-huggingface-repo org/name`.
*   A repo with a transformers checkpoint is converted once with `litert
    convert` (the `convert` extra is installed on first use) into
    `~/.cache/litert-cli/models/<org>__<name>/`, and later runs reuse the
    bundle. `litert lm` reads the conversion options and does not pass them to
    `litert-lm`: `--quantize` (default `dynamic_wi8_afp32`, `none` for fp32),
    `--prefill-lengths` (default `256`), `--cache-length` (default `4096`) and
    `--reconvert` (convert again even if a bundle is cached).

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
litert list --json

# Show detailed contents of a specific model using model reference.
litert list my_model
litert list my_model --json
```

### 11. Delete a managed model

```bash
# Delete a model from cache (asks for confirmation; -y skips it)
litert delete my_model -y
```

### 12. Clean up all caches

```bash
# Delete the whole local cache, including all managed models (`litert list`)
# and the bundles converted by `litert lm`, plus the LiteRT CLI directory on a
# connected Android device. Asks for confirmation; -y skips it.
litert clean
```
