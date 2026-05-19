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

# LiteRT CLI Convert Commands Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "convert_test" "Convert Commands Demo"

# --- 1. Generic Script Mode (resnet18.py) ---
echo -e "\n${BLUE}${BOLD}--- 1. Generic Script Mode (resnet18.py) ---${NC}"

# 1.1 Base conversion
run_case "Convert: PyTorch ResNet18 Base" \
    litert convert resnet18.py --output models/resnet18_base

# 1.2 Conversion with Model Args (e.g. batch_size=4)
run_case "Convert: PyTorch ResNet18 with Model Args (batch_size=4)" \
    litert convert resnet18.py --output models/resnet18_b4 --model-args "batch_size=4"

# 1.3 Conversion with Quantization (dynamic_wi8_afp32)
run_case "Convert: PyTorch ResNet18 with Dynamic INT8 Recipe" \
    litert convert resnet18.py --output models/resnet18_dyn_wi8 --quantize-recipe dynamic_wi8_afp32

# 1.4 Conversion with Quantization (weight_only_wi8_afp32)
run_case "Convert: PyTorch ResNet18 with Weight-Only INT8 Recipe" \
    litert convert resnet18.py --output models/resnet18_wo_wi8 --quantize-recipe weight_only_wi8_afp32


# --- 2. HuggingFace Mode (Qwen/Qwen1.5-0.5B-Chat) ---
echo -e "\n${BLUE}${BOLD}--- 2. HuggingFace Mode (Qwen/Qwen1.5-0.5B-Chat) ---${NC}"

# 2.1 Base conversion (bundle default)
run_case "Convert: Qwen1.5-0.5B-Chat Base" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output models/qwen_base

# 2.2 Conversion with Custom Prefill and Cache lengths
run_case "Convert: Qwen1.5-0.5B-Chat with Custom Prefill & Cache lengths" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output models/qwen_custom --prefill-lengths "128,512" --cache-length 2048

# 2.3 Conversion without bundle
run_case "Convert: Qwen1.5-0.5B-Chat without Bundle" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output models/qwen_nobundle --no-bundle-litert-lm

# 2.4 Non-CausalLM Architecture Rejection
run_case "Convert: bert-base-uncased (Verify Non-CausalLM Rejection)" \
    bash -c "litert convert google-bert/bert-base-uncased --output 'models/bert_fail' 2>&1 | grep -q 'Currently only AutoModelForCausalLM is supported'"

# --- Summary Report ---
print_summary_report "Convert Commands"
