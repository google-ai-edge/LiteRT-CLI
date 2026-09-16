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

# LiteRT CLI Benchmark DDP Commands Test Script
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
    echo -e "${RED}Error: --gcp-project is required for running DDP benchmark tests.${NC}"
    echo -e "${YELLOW}Usage: $0 --gcp-project <YOUR_GCP_PROJECT_ID>${NC}"
    exit 1
fi

setup_test_env "benchmark_ddp_test" "Benchmark DDP Commands Demo"

echo -e "\n${BLUE}${BOLD}--- 1. Benchmark DDP Commands ---${NC}"

run_case "Download: MobileNet-v2" \
    litert download litert-community/MobileNet-v2 --file "mobilenet_v2.tflite" --output "models/mobilenet"

echo -e "\n${BLUE}${BOLD}--- Running Live DDP Benchmarks for Project: $GCP_PROJECT ---${NC}"

run_case "Benchmark: CPU Mode on DDP devices" \
    litert benchmark models/mobilenet/mobilenet_v2.tflite --ddp --cpu --devices 'caiman-35, pa3q-35' --gcp-project "$GCP_PROJECT"

run_case "Benchmark: GPU Mode on DDP devices" \
    litert benchmark models/mobilenet/mobilenet_v2.tflite --ddp --gpu --device 'caiman-35' --gcp-project "$GCP_PROJECT"

print_summary_report "Benchmark DDP Commands"
