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

# LiteRT CLI EfficientNet Demo & Test Script for Windows

# --- Environment Setup ---
$SCRIPT_DIR = $PSScriptRoot
$REPO_ROOT = (Resolve-Path "$SCRIPT_DIR/../..").Path
$LITERT_CLI_ROOT = Join-Path $env:TEMP "litert_cli_efficientnet"

# Source shared utilities
. (Join-Path $SCRIPT_DIR "..\utils.ps1")

Write-Host "${BLUE}${BOLD}==================================================================${NC}"
Write-Host "${BLUE}${BOLD}>>> LiteRT CLI EfficientNet Demo Script${NC}"
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
python -m venv venv_efficientnet
. .\venv_efficientnet\Scripts\Activate.ps1

# Create output directories
$MODEL_DIR = Join-Path $LITERT_CLI_ROOT "models"
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null

# Test data directory
$TEST_DATA_DIR = "$REPO_ROOT/litert_cli/test_data"

# Install litert-cli from source
Write-Host "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

# --- 1. Download EfficientNet-B1 model ---
Run-Case "Download: EfficientNet-B1 from HuggingFace" {
    litert download litert-community/efficientnet_b1 --file "*.tflite" --output "$MODEL_DIR/efficientnet"
}

# Verify the downloaded model exists
$EFFICIENTNET_TFLITE = Join-Path $MODEL_DIR "efficientnet/efficientnet_b1.tflite"
if (-not (Test-Path $EFFICIENTNET_TFLITE -PathType Leaf)) {
    Write-Host "${RED}Error: Downloaded model not found at $EFFICIENTNET_TFLITE${NC}"
    Exit 1
}

# --- 2. Quantize the EfficientNet model ---
Run-Case "Quantize: EfficientNet Dynamic Range INT8" {
    litert quantize "$EFFICIENTNET_TFLITE" --recipe dynamic_wi8_afp32 --output "$MODEL_DIR/efficientnet/efficientnet_b1_int8_dynamic.tflite"
}

Run-Case "Quantize: EfficientNet Weight-Only INT8" {
    litert quantize "$EFFICIENTNET_TFLITE" --recipe weight_only_wi8_afp32 --output "$MODEL_DIR/efficientnet/efficientnet_b1_int8_weight_only.tflite"
}

# --- 3. Run Inference (Desktop & Android) ---
# Run-Case "Run: EfficientNet FP32 on Desktop (CPU)" {
#     litert run "$EFFICIENTNET_TFLITE" --desktop --cpu --iterations 1
# }

if (Has-DesktopGpu "$EFFICIENTNET_TFLITE") {
    Run-Case "Run: EfficientNet FP32 on Desktop (GPU)" {
        litert run "$EFFICIENTNET_TFLITE" --desktop --gpu --iterations 1
    }
} else {
    Write-Host ""
    Write-Host "${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
}

# Run-Case "Run: EfficientNet Dynamic INT8 on Desktop (CPU)" {
#     litert run "$MODEL_DIR/efficientnet/efficientnet_b1_int8_dynamic.tflite" --desktop --cpu --iterations 1
# }

if (Has-AndroidDevice) {
    Write-Host ""
    Write-Host "${GREEN}Android device detected. Running Android inference...${NC}"
    Run-Case "Run: EfficientNet FP32 on Android (CPU)" {
        litert run "$EFFICIENTNET_TFLITE" --android --cpu --iterations 1
    }

    Run-Case "Run: EfficientNet FP32 on Android (GPU)" {
        litert run "$EFFICIENTNET_TFLITE" --android --gpu --iterations 1
    }

    # If you have Android devices with NPU connected, enable those use cases.
    # Run-Case "Run: EfficientNet FP32 on Android (NPU)" {
    #     litert run "$EFFICIENTNET_TFLITE" --android --npu --iterations 1
    # }

    Run-Case "Run: EfficientNet Dynamic INT8 on Android (CPU)" {
        litert run "$MODEL_DIR/efficientnet/efficientnet_b1_int8_dynamic.tflite" --android --cpu --iterations 1
    }
}

# --- 4. Benchmark (Android) ---
if (Has-AndroidDevice) {
    Write-Host ""
    Write-Host "${GREEN}Android device detected. Running Android benchmarks...${NC}"
    Run-Case "Benchmark: EfficientNet FP32 on Android (CPU)" {
        litert benchmark "$EFFICIENTNET_TFLITE" --android
    }

    Run-Case "Benchmark: EfficientNet FP32 on Android (GPU)" {
        litert benchmark "$EFFICIENTNET_TFLITE" --android --gpu
    }

    # If you have Android devices with NPU connected, enable those use cases.
    # Run-Case "Benchmark: EfficientNet FP32 on Android (NPU)" {
    #     litert benchmark "$EFFICIENTNET_TFLITE" --android --npu
    # }

    Run-Case "Benchmark: EfficientNet Dynamic INT8 on Android" {
        litert benchmark "$MODEL_DIR/efficientnet/efficientnet_b1_int8_dynamic.tflite" --android
    }
} else {
    Write-Host ""
    Write-Host "${YELLOW}No Android device detected. Skipping benchmarks on Android.${NC}"
}

# --- 5. Compile (AOT Compilation) ---
# Run-Case "Compile: EfficientNet FP32 for Qualcomm sm8750 NPU" {
#     litert compile "$EFFICIENTNET_TFLITE" --target sm8750 --output-dir "$MODEL_DIR/efficientnet"
# }
# Run-Case "Compile: EfficientNet FP32 for MediaTek MT6993 NPU" {
#     litert compile "$EFFICIENTNET_TFLITE" --target MT6993 --output-dir "$MODEL_DIR/efficientnet"
# }

# --- 6. Benchmark compiled model ---
# Enable those use cases, or change to your own targets, if you have connected those android
# devices through NPU.
#
# Run-Case "Run Qualcomm compiled EfficientNet" {
#   litert run "$MODEL_DIR/efficientnet/efficientnet_b1_Qualcomm_SM8750.tflite" --android --npu
# }
# Run-Case "Benchmark Qualcomm compiled EfficientNet" {
#   litert benchmark "$MODEL_DIR/efficientnet/efficientnet_b1_Qualcomm_SM8750.tflite" --android --npu
# }

# Run-Case "Run MediaTek compiled EfficientNet" {
#    litert run "$MODEL_DIR/efficientnet/efficientnet_b1_MediaTek_MT6993.tflite" --android --npu
# }
# Run-Case "Benchmark MediaTek compiled EfficientNet" {
#    litert benchmark "$MODEL_DIR/efficientnet/efficientnet_b1_MediaTek_MT6993.tflite" --android --npu
# }

# --- 7. Visualize (Model Explorer) ---
# Run-Case "Visualize: Launch Model Explorer in the background" {
#     litert visualize "$EFFICIENTNET_TFLITE"
# }

# Run-Case "Visualize: Stop all Model Explorer servers" {
#     litert visualize --stop-all
# }

# --- Summary Report ---
Print-SummaryReport "EfficientNet"
