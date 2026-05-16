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

# LiteRT CLI Download Commands Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "download_test" "Download Commands Demo"

echo -e "\n${BLUE}${BOLD}--- 1. Download Commands ---${NC}"

run_case "Download: MobileNet-v3-large (*.tflite only)" \
    litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "models/mobilenet"

run_case "Download: MobileNet-v3-large (Full Repo)" \
    litert download litert-community/MobileNet-v3-large --output "models/mobilenet_full"

run_case "Download: EfficientNet-B1 (Full Repo)" \
    litert download litert-community/efficientnet_b1 --output "models/efficientnet"

print_summary_report "Download Commands"
