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

# Shared utilities and helpers for LiteRT CLI demo scripts

# Color codes for beautiful output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Run case tracking variables
TOTAL_CASES=0
TOTAL_PASSED=0
TOTAL_FAILED=0
PASSED_CASES=()
FAILED_CASES=()

# Helper for dynamic Android check (supports both authorized and unauthorized devices)
function has_android_device() {
  adb devices 2>/dev/null | grep -E -q "\s+(device|unauthorized)$"
}

# Helper to verify if LiteRT GPU accelerator is supported on Desktop (excluding software emulation like llvmpipe)
function has_desktop_gpu() {
  local model_file="$1"
  local output
  output=$(python3 -c "
import sys
try:
  from ai_edge_litert.compiled_model import CompiledModel
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator
  cm = CompiledModel.from_file('$model_file', HardwareAccelerator.GPU)
except Exception:
  sys.exit(1)
" 2>&1)
  local status=$?

  if [ $status -eq 0 ] && [[ ! "$output" =~ "llvmpipe" && ! "$output" =~ "lavapipe" ]]; then
    return 0
  else
    return 1
  fi
}



# Auto-detect REPO_ROOT from utils.sh location
export UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$UTILS_DIR/.." && pwd)"

# Shared environment initialization for demo scripts
function setup_test_env() {
  local test_name="$1"
  local title="$2"
  local extra_deps="${3:-}"

  echo -e "${BLUE}${BOLD}==================================================================${NC}"
  echo -e "${BLUE}${BOLD}>>> LiteRT CLI $title${NC}"
  echo -e "${BLUE}${BOLD}==================================================================${NC}"

  if [ "${LITERT_CLI_SHARED_VENV:-false}" == "true" ]; then
    export LITERT_CLI_ROOT="/tmp/litert_cli_shared"
    local test_root="$LITERT_CLI_ROOT"
    mkdir -p "$test_root"
    cd "$test_root"

    if [ ! -d ".venv" ] || [ ! -f ".venv/bin/litert" ]; then
      echo -e "\n${YELLOW}Creating Shared Python virtual environment with UV...${NC}"
      UV_INDEX_URL=https://pypi.org/simple uv venv --clear --python=3.13 --seed
      source .venv/bin/activate
      echo -e "${YELLOW}Installing litert-cli with all extras into shared venv...${NC}"
      uv pip install -e "$REPO_ROOT[convert,quantize,lm,compile]"
    else
      echo -e "\n${GREEN}Reusing existing shared virtual environment at $test_root...${NC}"
      source .venv/bin/activate
    fi
  else
    export LITERT_CLI_ROOT="/tmp/litert_cli_$test_name"
    local test_root="$LITERT_CLI_ROOT"

    echo -e "\n${YELLOW}Setting up isolated workspace at: $test_root...${NC}"
    rm -rf "$test_root"
    mkdir -p "$test_root"
    cd "$test_root"

    echo -e "${YELLOW}Creating Isolated Python virtual environment with UV...${NC}"
    UV_INDEX_URL=https://pypi.org/simple uv venv --clear --python=3.13 --seed
    source .venv/bin/activate

    if [ -n "$extra_deps" ]; then
      echo -e "${YELLOW}Installing litert-cli with [$extra_deps] extra...${NC}"
      uv pip install -e "$REPO_ROOT[$extra_deps]"
    else
      echo -e "${YELLOW}Installing litert-cli...${NC}"
      uv pip install -e "$REPO_ROOT"
    fi
  fi

  export MODEL_DIR="$test_root/models"
  export MODELS_CACHE="$test_root/models"
  mkdir -p "$MODEL_DIR"

  export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"
  # Symlink test data directly into workspace for clean command syntax
  ln -sf "$TEST_DATA_DIR/"* .
}


# Robust runner for a test command with isolation and formatting
function run_case() {
    local title="$1"
    shift
    
    echo -e "\n${BLUE}▶ Running:${NC} ${BOLD}$title${NC}"
    echo -e "\033[90mCommand: $*\033[0m"
    echo -e "\033[90m------------------------------------------------------------\033[0m"
    
    set +e
    "$@"
    local status=$?
    set -e
    
    echo -e "\033[90m------------------------------------------------------------\033[0m"
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✔ SUCCESS:${NC} ${GREEN}${BOLD}$title${NC}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
        PASSED_CASES+=("$title")
    else
        echo -e "${RED}✘ FAILED (Exit Code: $status):${NC} ${RED}${BOLD}$title${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_CASES+=("$title")
    fi
    TOTAL_CASES=$((TOTAL_CASES + 1))
    return $status
}

# Prints the final summary report for the demo
function print_summary_report() {
    local model_name="$1"
    local model_name_upper=$(echo "$model_name" | tr '[:lower:]' '[:upper:]')
    
    echo -e "\n${BLUE}${BOLD}==================================================================${NC}"
    echo -e "${BLUE}${BOLD}>>> ${model_name_upper} TEST SUMMARY${NC}"
    echo -e "${BLUE}${BOLD}==================================================================${NC}"
    echo -e "Total Cases Run: ${BOLD}$TOTAL_CASES${NC}"
    echo -e "Passed:          ${GREEN}${BOLD}$TOTAL_PASSED${NC}"
    echo -e "Failed:          ${RED}${BOLD}$TOTAL_FAILED${NC}"
    
    if [ $TOTAL_PASSED -gt 0 ]; then
        echo -e "\n${GREEN}${BOLD}Passed Cases:${NC}"
        for case in "${PASSED_CASES[@]}"; do
            echo -e "  - ${GREEN}$case${NC}"
        done
    fi
    
    if [ $TOTAL_FAILED -gt 0 ]; then
        echo -e "\n${RED}${BOLD}Failed Cases:${NC}"
        for case in "${FAILED_CASES[@]}"; do
            echo -e "  - ${RED}$case${NC}"
        done
        echo -e "${BLUE}${BOLD}==================================================================${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}${BOLD}All ${model_name} CLI commands executed successfully!${NC}"
    echo -e "${BLUE}${BOLD}==================================================================${NC}"
    exit 0
}
