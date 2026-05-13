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

# LiteRT CLI Gemma4 LLM Demo & Test Script
set -e


echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI Gemma4 LLM Demo Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_gemma4"

# Source shared utilities
source "$SCRIPT_DIR/utils.sh"


# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create output directories
export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"


# Create Python virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv_gemma4
source venv_gemma4/bin/activate
pip install --upgrade pip setuptools wheel

# Install litert-cli from source
echo -e "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"




# --- 1. Convert HuggingFace Model google/gemma-4-E2B-it ---
# TODO: Bring this back when we add support for --externalize_embedder in CLI convert command.
# run_case "Convert: HuggingFace google/gemma-4-E2B-it" \
#     litert convert google/gemma-4-E2B-it --output "$MODEL_DIR/gemma4"


# --- 2. Run Gemma4 Generative LLM Model ---
run_case "Run Gemma4: Generative inference with custom prompt" \
    litert lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --prompt="What is the capital of France?"

# --- 3. Benchmark Gemma4 LLM Model ---
run_case "Benchmark Gemma4: Local benchmark of LLM generation" \
    litert lm benchmark gemma-4-E2B-it.litertlm --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm -p 128 -d 128



# --- Summary Report ---
print_summary_report "Gemma4"

