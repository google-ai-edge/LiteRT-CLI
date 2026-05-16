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

GCP_PROJECT=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gcp-project)
      GCP_PROJECT="$2"
      shift 2
      ;;
    --gcp-project=*)
      GCP_PROJECT="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$GCP_PROJECT" ]]; then
    echo -e "${RED}Error: --gcp-project is required for running GCP benchmark tests.${NC}"
    echo -e "${YELLOW}Usage: $0 --gcp-project <YOUR_GCP_PROJECT_ID>${NC}"
    exit 1
fi

setup_test_env "benchmark_gcp_test" "Benchmark GCP Commands Demo"

echo -e "\n${BLUE}${BOLD}--- 1. Benchmark GCP Commands ---${NC}"

run_case "Download: EfficientNet-B1" \
    litert download litert-community/efficientnet_b1 --file "*.tflite" --output "models/efficientnet"

echo -e "\n${BLUE}${BOLD}--- Running Live GCP Benchmarks for Project: $GCP_PROJECT ---${NC}"

run_case "Benchmark: CPU Mode on Google AI Edge Portal" \
    litert benchmark models/efficientnet/efficientnet_b1.tflite --gcp --cpu --device 'pixel 10 pro, sm-s931u1' --gcp-project "$GCP_PROJECT"

run_case "Benchmark: GPU Mode on Google AI Edge Portal" \
    litert benchmark models/efficientnet/efficientnet_b1.tflite --gcp --gpu --device 'sm-s931u1' --gcp-project "$GCP_PROJECT"

run_case "Benchmark: NPU JIT Mode on Google AI Edge Portal" \
    litert benchmark models/efficientnet/efficientnet_b1.tflite --gcp --npu --jit --device 'sm-s931u1' --soc-model SM8750 --gcp-project "$GCP_PROJECT"

print_summary_report "Benchmark GCP Commands"
