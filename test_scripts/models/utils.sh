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

# Helper to verify if LiteRT GPU accelerator is supported on Desktop (and not falling back to software emulation like llvmpipe)
function has_desktop_gpu() {
  local model_file="$1"
  python3 -c "
import sys
import subprocess
try:
    check_cmd = [
        'python3', '-c',
        f'from ai_edge_litert.compiled_model import CompiledModel; from ai_edge_litert.hardware_accelerator import HardwareAccelerator; cm = CompiledModel.from_file(\"$model_file\", HardwareAccelerator.GPU)'
    ]
    res = subprocess.run(check_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(1)
    if 'llvmpipe' in res.stderr or 'llvmpipe' in res.stdout:
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
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
    
    echo -e "\n${BLUE}${BOLD}==================================================================${NC}"
    echo -e "${BLUE}${BOLD}>>> ${model_name^^} TEST SUMMARY${NC}"
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
