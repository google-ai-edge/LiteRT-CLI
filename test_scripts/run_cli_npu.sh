#!/bin/bash
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

# LiteRT CLI NPU Focus Test Script
set -e

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_npu"

# Source shared utilities
source "$SCRIPT_DIR/models/utils.sh"

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI NPU Focus Test Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv_npu
source venv_npu/bin/activate

# Install litert-cli from source
echo -e "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

# --- 1. Download EfficientNet-B1 model ---
run_case "Download: EfficientNet-B1 from HuggingFace" \
    litert download litert-community/efficientnet_b1 --file "*.tflite" --output "$MODEL_DIR/efficientnet"

EFFICIENTNET_TFLITE="$MODEL_DIR/efficientnet/efficientnet_b1.tflite"

if [ ! -f "$EFFICIENTNET_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $EFFICIENTNET_TFLITE${NC}"
    exit 1
fi

# --- 2. Detect Target SoC ---
echo -e "\n${YELLOW}Detecting Target SoC from connected device...${NC}"
if ! has_android_device; then
    echo -e "${RED}Error: No Android device detected or authorized.${NC}"
    exit 1
fi

# Use python to call our helper function to get the mapped SoC name
TARGET_SOC=$(python3 -c "from litert_cli.core import npu_utils; print(npu_utils.get_soc_target_model())")
echo -e "Detected Target SoC: ${GREEN}$TARGET_SOC${NC}"

if [ "$TARGET_SOC" == "unknown" ]; then
    echo -e "${RED}Error: Could not map detected SoC to a supported target.${NC}"
    exit 1
fi

# --- 3. Run and Benchmark Original Model on NPU ---
# This tests the auto-download of SoC list during run/benchmark!
run_case "Run Original Model on NPU" \
    litert run "$EFFICIENTNET_TFLITE" --android --npu --iterations 1

run_case "Benchmark Original Model on NPU" \
    litert benchmark "$EFFICIENTNET_TFLITE" --android --npu

# --- 4. Compile for Detected Target ---
mkdir -p "$MODEL_DIR/compiled"

run_case "Compile for Detected Target ($TARGET_SOC)" \
    litert compile "$EFFICIENTNET_TFLITE" --target "$TARGET_SOC" --output-dir "$MODEL_DIR/compiled"

# --- 5. Run and Benchmark Compiled Model ---
echo -e "\n${YELLOW}Looking for compiled model...${NC}"
COMPILED_MODEL=$(find "$MODEL_DIR/compiled" -name "*.tflite" | head -n 1)

if [ -z "$COMPILED_MODEL" ]; then
    echo -e "${RED}Error: Compiled model not found.${NC}"
    exit 1
fi

echo -e "Found compiled model: ${GREEN}$COMPILED_MODEL${NC}"

run_case "Run Compiled Model on NPU" \
    litert run "$COMPILED_MODEL" --android --npu --iterations 1

run_case "Benchmark Compiled Model on NPU" \
    litert benchmark "$COMPILED_MODEL" --android --npu

print_summary_report "NPU Tests"
