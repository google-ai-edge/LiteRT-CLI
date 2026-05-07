#!/bin/bash
# LiteRT CLI Gemma4 LLM Demo & Test Script
set -e

# Color codes for beautiful output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI Gemma4 LLM Demo Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_gemma4"

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

# Install litert-cli from source
echo -e "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

# Run case helper
TOTAL_CASES=0
TOTAL_PASSED=0
TOTAL_FAILED=0
PASSED_CASES=()
FAILED_CASES=()

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


# --- 1. Convert HuggingFace Model google/gemma-4-E2B-it ---
# TODO: Bring this back when we add support for --externalize_embedder in CLI convert command.
# run_case "Convert: HuggingFace google/gemma-4-E2B-it" \
#     litert convert google/gemma-4-E2B-it --output "$MODEL_DIR/gemma4"


# --- 2. Run Gemma4 Generative LLM Model ---
run_case "Run Gemma4: Generative inference with custom prompt" \
    litert lm run gemma-4-E2B-it.litertlm --prompt "Explain machine learning in one sentence."


# --- 3. Benchmark Gemma4 LLM Model ---
run_case "Benchmark Gemma4: Local benchmark of LLM generation" \
    litert lm benchmark gemma-4-E2B-it.litertlm



# --- Summary Report ---
echo -e "\n${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> GEMMA4 LLM TEST SUMMARY${NC}"
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

echo -e "${GREEN}${BOLD}All Gemma4 LLM CLI commands executed successfully!${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"
exit 0
