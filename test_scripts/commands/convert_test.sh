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

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_convert_test"

# Source shared utilities
source "$SCRIPT_DIR/../models/utils.sh"

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI Convert Commands Demo & Test Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment using UV with Python 3.13
echo -e "${YELLOW}Creating Python virtual environment with UV...${NC}"
UV_INDEX_URL=https://pypi.org/simple uv venv --clear --python=3.13 --seed
source .venv/bin/activate

export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"

# Install litert-cli with convert and lm extras from source using UV
echo -e "${YELLOW}Installing litert-cli with convert and lm extras...${NC}"
uv pip install -e "$REPO_ROOT"

# --- 1. Generic Script Mode (resnet18.py) ---
echo -e "\n${BLUE}${BOLD}--- 1. Generic Script Mode (resnet18.py) ---${NC}"

# 1.1 Base conversion
run_case "Convert: PyTorch ResNet18 Base" \
    litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18_base"

# 1.2 Conversion with Quantization (pt2e_dynamic)
run_case "Convert: PyTorch ResNet18 with PT2E Dynamic Quantization" \
    litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18_pt2e" --quantize pt2e_dynamic

# 1.3 Conversion with Model Args (e.g. batch_size=4)
run_case "Convert: PyTorch ResNet18 with Model Args (batch_size=4)" \
    litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18_b4" --model-args "batch_size=4"


# --- 2. HuggingFace Mode (Qwen/Qwen1.5-0.5B-Chat) ---
echo -e "\n${BLUE}${BOLD}--- 2. HuggingFace Mode (Qwen/Qwen1.5-0.5B-Chat) ---${NC}"

# 2.1 Base conversion (bundle default)
run_case "Convert: Qwen1.5-0.5B-Chat Base" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output "$MODEL_DIR/qwen_base"

# 2.2 Conversion with Custom Prefill and Cache lengths
run_case "Convert: Qwen1.5-0.5B-Chat with Custom Prefill & Cache lengths" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output "$MODEL_DIR/qwen_custom" --prefill-lengths "128,512" --cache-length 2048

# 2.3 Conversion without bundle
run_case "Convert: Qwen1.5-0.5B-Chat without Bundle" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output "$MODEL_DIR/qwen_nobundle" --no-bundle-litert-lm

# 2.4 Non-CausalLM Architecture Rejection
run_case "Convert: bert-base-uncased (Verify Non-CausalLM Rejection)" \
    bash -c "litert convert bert-base-uncased --output '$MODEL_DIR/bert_fail' 2>&1 | grep -q 'Currently only AutoModelForCausalLM is supported'"

# 2.5 Non-Whitelisted Model Jinja Template Disabled
run_case "Convert: EleutherAI/pythia-70m (Verify Jinja Template Disabled)" \
    bash -c "litert convert EleutherAI/pythia-70m --output '$MODEL_DIR/pythia' 2>&1 | grep -q 'use_jinja_template.*: False'"


# --- Summary Report ---
print_summary_report "Convert Commands"
