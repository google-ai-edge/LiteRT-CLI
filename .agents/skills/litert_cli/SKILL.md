---
name: litert-cli
description: LiteRT CLI tool to convert, download, quantize, run, benchmark, and visualize LiteRT models.
---

# LiteRT-CLI Agent Skill

This skill allows the agent to download, convert, quantize, run, benchmark, and visualize LiteRT models using the `litert` command on desktop, device, or Google Cloud

## Setup & Prerequisites

Before running any `litert` commands, an agent should ensure it is in a Python virtual environment and `litert-cli` is installed.

### 1. Check/Create Virtual Environment
Run the following checks:
```bash
# If not in a venv, create and activate one automatically
if [ -z "$VIRTUAL_ENV" ]; then
  python3 -m venv litert-cli-venv
  source litert-cli-venv/bin/activate
fi
```

### 2. Check/Install LiteRT CLI
Run the following checks to ensure the command is available:
```bash
# If litert is not found, try local install from source or fallback to TestPyPI
if ! command -v litert &> /dev/null; then
  if [ -f "pyproject.toml" ]; then
    pip install -e .
  else
    pip install -q -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple litert-cli==0.1.1.dev17
  fi
fi
```

## Core Commands

### 1. Run (Inference)
Run a tflite model locally on desktop or on a adb connected Android device.

**Desktop Execution (CPU and Local GPU):**
```bash
litert run <path_to_model> --desktop --cpu --quiet
```
*   `--quiet`: Silences verbose C++ setup logs. Highly recommended for agents to parse clean output.
*   `--gpu`: Use desktop GPU if available.

**Android Execution:**
```bash
litert run <path_to_model> --android --cpu
```

**Multi-Input Formats (Literals or Arrays):**
```bash
litert run model.tflite --desktop --input inputs="[0.5, 0.5, 0.5]" --print_tensors
```

**Multi-Input Formats (Files - .npy, .raw, .png):**
```bash
litert run model.tflite --desktop --input inputs="test_input.npy" --print_tensors
```

### 2. Quantize
**Standard Selection:**
```bash
litert quantize <path_to_model> --output <output_path>
```

**Dynamic Quantization (int8_dynamic):**
```bash
litert quantize model.tflite --type int8_dynamic --output dynamic.tflite
```

**Static Quantization with Calibration Data:**
```bash
litert quantize model.tflite --type static --calibration-data "calib_data.py" --output static.tflite
```

**Recipe-based Quantization:**
```bash
litert quantize model.tflite --recipe "recipe.json" --output recipe.tflite
```

### 3. Visualize
Launch the Model Explorer to visualize the model structure.

```bash
litert visualize <path_to_model>
```

**Advanced Visualisation Options:**
*   `--no_reuse_server`: Force creation of a NEW server port.
*   `--stop_all`: Kill all existing visualization background servers.

```bash
litert visualize --stop_all
```

### 4. Download
Download public LiteRT models from HuggingFace Hub.

```bash
litert download <repo_id_or_model_name> --output <output_dir>
```

**Filter by File Type:**
```bash
litert download litert-community/MobileNet-v3-large --file "*.tflite" --output ./models
```

### 5. Convert (PyTorch to LiteRT)
Convert a PyTorch or HuggingFace model into a LiteRT model.

**From HuggingFace Model Hub:**
```bash
litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen
```

**From Generic Python Script:**
```bash
litert convert my_model.py --output /tmp/mymodel
```
*   `--model-func`: Name of function that returns the model (`torch.nn.Module`). Default: `get_model`.
*   `--input-func`: Name of function that returns sample inputs. Default: `get_args`.

### 6. Large Language Models (LM)
Interact with LLM generative models using native `litert-lm-cli` or python fallback.

```bash
litert lm run <model_path_or_repo_id>
```

```bash
litert lm run <model_path_or_repo_id> "What is a neural network?"
```

### 7. Benchmark
Benchmark LiteRT models on different platforms (Android or Google Cloud).

**On connected Android device via ADB:**
```bash
litert benchmark model.tflite --android --cpu
```

**On Google Cloud AI Edge Portal devices (e.g., Pixel 7):**
```bash
litert benchmark model.tflite --gcp --device "pixel 7"
```

## Best Practices for Agents

*   Always use the `--quiet` flag when running `litert run` to keep the terminal output clean and easy to parse for automated checks.
*   Pipe outputs to text files or grep them if you are looking for specific tensor shapes or runtime metrics.
