# Copyright 2026 The LiteRT CLI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# LiteRT CLI YamNet Demo & Test Script for Windows

# --- Environment Setup ---
$SCRIPT_DIR = $PSScriptRoot
$REPO_ROOT = (Resolve-Path "$SCRIPT_DIR/../..").Path
$LITERT_CLI_ROOT = Join-Path $env:TEMP "litert_cli_yamnet"

# Source shared utilities
. (Join-Path $SCRIPT_DIR "..\utils.ps1")

Write-Host "${BLUE}${BOLD}==================================================================${NC}"
Write-Host "${BLUE}${BOLD}>>> LiteRT CLI YamNet Demo Script${NC}"
Write-Host "${BLUE}${BOLD}==================================================================${NC}"

# Clean up and create work directory
Write-Host ""
Write-Host "${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
if (Test-Path $LITERT_CLI_ROOT) {
    Remove-Item -Recurse -Force $LITERT_CLI_ROOT
}
New-Item -ItemType Directory -Force -Path $LITERT_CLI_ROOT | Out-Null
Set-Location $LITERT_CLI_ROOT

# Create Python virtual environment
Write-Host "${YELLOW}Creating Python virtual environment...${NC}"
python -m venv venv_yamnet
. .\venv_yamnet\Scripts\Activate.ps1

# Create output directories
$MODEL_DIR = Join-Path $LITERT_CLI_ROOT "models"
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null

# Test data directory
$TEST_DATA_DIR = "$REPO_ROOT/litert_cli/test_data"

# Install litert-cli from source
Write-Host "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

# --- 1. Download YamNet model ---
Run-Case "Download: YamNet TFLite model" {
    litert download "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite" --output "$MODEL_DIR/yamnet"
}

$YAMNET_TFLITE = Join-Path $MODEL_DIR "yamnet/lite-model_yamnet_classification_tflite_1.tflite"
if (-not (Test-Path $YAMNET_TFLITE -PathType Leaf)) {
    Write-Host "${RED}Error: Downloaded model not found at $YAMNET_TFLITE${NC}"
    Exit 1
}

# --- 2. Quantize the YamNet model ---
Run-Case "Quantize: YamNet Dynamic Range INT8" {
    litert quantize "$YAMNET_TFLITE" --recipe dynamic_wi8_afp32 --output "$MODEL_DIR/yamnet/yamnet_int8_dynamic.tflite"
}

Run-Case "Quantize: YamNet Weight-Only INT8" {
    litert quantize "$YAMNET_TFLITE" --recipe weight_only_wi8_afp32 --output "$MODEL_DIR/yamnet/yamnet_int8_weight_only.tflite"
}

# --- 3. Run Inference (Desktop & Android) ---
Run-Case "Run: YamNet FP32 on Desktop (CPU)" {
    litert run "$YAMNET_TFLITE" --desktop --cpu --iterations 1
}

if (Has-DesktopGpu "$YAMNET_TFLITE") {
    Run-Case "Run: YamNet FP32 on Desktop (GPU)" {
        litert run "$YAMNET_TFLITE" --desktop --gpu --iterations 1
    }
} else {
    Write-Host ""
    Write-Host "${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
}

Run-Case "Run: YamNet Dynamic INT8 on Desktop (CPU)" {
    litert run "$MODEL_DIR/yamnet/yamnet_int8_dynamic.tflite" --desktop --cpu --iterations 1
}

if (Has-AndroidDevice) {
    Write-Host ""
    Write-Host "${GREEN}Android device detected. Running Android inference...${NC}"
    Run-Case "Run: YamNet FP32 on Android (CPU)" {
        litert run "$YAMNET_TFLITE" --android --cpu --iterations 1
    }
 
    # Works on Qualcomm NPU SM8750, but not GPU.
    # Run-Case "Run: YamNet FP32 on Android (GPU)" {
    #    litert run "$YAMNET_TFLITE" --android --gpu --iterations 1
    # }

    Run-Case "Run: YamNet Dynamic INT8 on Android (CPU)" {
        litert run "$MODEL_DIR/yamnet/yamnet_int8_dynamic.tflite" --android --cpu --iterations 1
    }
}

# --- 4. Benchmark (Android) ---
if (Has-AndroidDevice) {
    Write-Host ""
    Write-Host "${GREEN}Android device detected. Running Android benchmarks...${NC}"
    Run-Case "Benchmark: YamNet FP32 on Android (CPU)" {
        litert benchmark "$YAMNET_TFLITE" --android
    }

    # Works on Qualcomm NPU SM8750, but not GPU.
    # Run-Case "Benchmark: YamNet FP32 on Android (GPU)" {
    #    litert benchmark "$YAMNET_TFLITE" --android --gpu
    # }

    Run-Case "Benchmark: YamNet Dynamic INT8 on Android" {
        litert benchmark "$MODEL_DIR/yamnet/yamnet_int8_dynamic.tflite" --android
    }
} else {
    Write-Host ""
    Write-Host "${YELLOW}No Android device detected. Skipping benchmarks (litert benchmark only supports Android/GCP).${NC}"
}

# --- 5. Compile (AOT Compilation) ---
# TODO: Add this back when we fix the NPU compile issue.
# Run-Case "Compile: YamNet FP32 for Qualcomm sm8750 NPU" {
#     litert compile "$YAMNET_TFLITE" --target sm8750 --output-dir "$MODEL_DIR/yamnet"
# }

# --- 6. Visualize (Model Explorer) ---
# Run-Case "Visualize: Launch Model Explorer in the background" {
#     litert visualize "$YAMNET_TFLITE"
# }

# Run-Case "Visualize: Stop all Model Explorer servers" {
#     litert visualize --stop-all
# }

# --- Summary Report ---
Print-SummaryReport "YamNet"
