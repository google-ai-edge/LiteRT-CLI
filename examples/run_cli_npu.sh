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

# LiteRT CLI NPU Focus Demo Script
#
# Usage:
#   Run NPU-focused end-to-end compilation and execution demo:
#   ./examples/run_cli_npu.sh
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

setup_test_env "npu_test" "NPU Focus Demo" "compile"

# --- 1. Download EfficientNet-B1 model ---
run_case "Download: EfficientNet-B1 from HuggingFace" \
    litert download litert-community/efficientnet_b1 --file "*.tflite" --output "models/efficientnet"

EFFICIENTNET_TFLITE="models/efficientnet/efficientnet_b1.tflite"

# --- 2. Detect Target SoC ---
echo -e "\n${YELLOW}Detecting Target SoC from connected device...${NC}"
if ! has_android_device; then
    echo -e "${YELLOW}No Android device detected or authorized. Skipping on-device NPU execution.${NC}"
    TARGET_SOC="sm8750" # Fallback to default Snapdragon 8 Elite for offline compilation
else
    # Use python to call our helper function to get the mapped SoC name
    TARGET_SOC=$(python3 -c "from litert_cli.core import npu_utils; print(npu_utils.get_soc_target_model())")
    echo -e "Detected Target SoC: ${GREEN}$TARGET_SOC${NC}"

    if [ "$TARGET_SOC" == "unknown" ]; then
        echo -e "${YELLOW}Could not map detected SoC to a supported target. Fallback to sm8750.${NC}"
        TARGET_SOC="sm8750"
    else
        # --- 3. Run and Benchmark Original Model on NPU ---
        run_case "Run Original Model on NPU" \
            litert run "$EFFICIENTNET_TFLITE" --android --npu --iterations 1

        run_case "Benchmark Original Model on NPU" \
            litert benchmark "$EFFICIENTNET_TFLITE" --android --npu
    fi
fi

# --- 4. Compile for Detected Target ---
if [[ "$(uname)" != "Linux" ]]; then
    echo -e "\n${YELLOW}Skipping compilation and dependent tests on non-Linux platform ($(uname)). Offline AOT compilation requires Linux.${NC}"
    print_summary_report "NPU Tests"
    exit 0
fi

mkdir -p "models/compiled"

run_case "Compile for Target ($TARGET_SOC)" \
    litert compile "$EFFICIENTNET_TFLITE" --target "$TARGET_SOC" --output-dir "models/compiled"

# --- 5. Run and Benchmark Compiled Model ---
if has_android_device; then
    echo -e "\n${YELLOW}Looking for compiled model...${NC}"
    COMPILED_MODEL=$(find "models/compiled" -name "*.tflite" | head -n 1)

    if [ -z "$COMPILED_MODEL" ]; then
        echo -e "${RED}Error: Compiled model not found.${NC}"
        exit 1
    fi

    echo -e "Found compiled model: ${GREEN}$COMPILED_MODEL${NC}"

    run_case "Run Compiled Model on NPU" \
        litert run "$COMPILED_MODEL" --android --npu --iterations 1

    run_case "Benchmark Compiled Model on NPU" \
        litert benchmark "$COMPILED_MODEL" --android --npu
fi

print_summary_report "NPU Tests"
