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

# LiteRT CLI Benchmark Commands Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "benchmark_test" "Benchmark Commands Demo"

echo -e "\n${BLUE}${BOLD}--- 1. Benchmark Commands ---${NC}"

run_case "Download: MobileNet-v3-large" \
    litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "models/mobilenet"

run_case "Benchmark: MobileNet on Desktop (CPU)" \
    litert benchmark "models/mobilenet/mobilenet_v3_large.tflite" --desktop

if has_android_device; then
    run_case "Benchmark: MobileNet on Android (CPU)" \
        litert benchmark "models/mobilenet/mobilenet_v3_large.tflite" --android

    run_case "Benchmark: MobileNet on Android (GPU)" \
        litert benchmark "models/mobilenet/mobilenet_v3_large.tflite" --android --gpu
else
    echo -e "\n${YELLOW}No Android device detected. Skipping Android benchmarks.${NC}"
fi

print_summary_report "Benchmark Commands"
