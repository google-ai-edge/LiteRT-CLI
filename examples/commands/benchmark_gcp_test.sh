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

# LiteRT CLI Benchmark GCP Commands Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "benchmark_gcp_test" "Benchmark GCP Commands Demo"

echo -e "\n${BLUE}${BOLD}--- 1. Benchmark GCP Commands ---${NC}"

run_case "Download: MobileNet-v3-large" \
    litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "models/mobilenet"

# In open-source environment, GCP benchmarking requires real credentials and billing projects.
# We verify that the CLI successfully parses the parameters and correctly errors out with GCP project missing message.

run_case "Benchmark: GCP NPU JIT Mode (Verify Project Missing Interception)" \
    bash -c "litert benchmark models/mobilenet/mobilenet_v3_large.tflite --gcp --npu --jit --device 'pixel 8' 2>&1 | grep -q 'Missing GCP project'"

run_case "Benchmark: GCP NPU AOT Mode (Verify Project Missing Interception)" \
    bash -c "litert benchmark models/mobilenet/mobilenet_v3_large.tflite --gcp --npu --aot --soc-model SM8750 --device 'pixel 8' 2>&1 | grep -q 'Missing GCP project'"

run_case "Benchmark: GCP GPU Mode (Verify Project Missing Interception)" \
    bash -c "litert benchmark models/mobilenet/mobilenet_v3_large.tflite --gcp --gpu --device 'pixel 8' 2>&1 | grep -q 'Missing GCP project'"

print_summary_report "Benchmark GCP Commands"
