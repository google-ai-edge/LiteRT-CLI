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

# LiteRT CLI Compile Commands Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "compile_test" "Compile Commands Demo" "compile"

echo -e "\n${BLUE}${BOLD}--- 1. Compile Commands ---${NC}"

if [[ "$(uname)" != "Linux" ]]; then
    echo -e "\n${YELLOW}Skipping compile test on non-Linux platform ($(uname)). Offline AOT compilation requires Linux.${NC}"
    exit 0
fi

# Download model first
run_case "Download: EfficientNet-B1" \
    litert download litert-community/efficientnet_b1 --file "*.tflite" --output "models/efficientnet"

run_case "Compile: EfficientNet-B1 for Qualcomm sm8750" \
    litert compile "models/efficientnet/efficientnet_b1.tflite" --target sm8750 --output-dir "models/compiled"

print_summary_report "Compile Commands"
