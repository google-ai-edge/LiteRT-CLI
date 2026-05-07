#!/bin/bash
# LiteRT CLI ResNet Demo & Test Script
set -e

# Color codes for beautiful output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI ResNet Demo Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_resnet"

# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv_resnet
source venv_resnet/bin/activate

# Create output directories
export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

# Test data directory
export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"

# Install litert-cli from source
echo -e "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

# Helper for dynamic Android check
function has_android_device() {
  adb devices 2>/dev/null | grep -q "[0-9a-zA-Z]\+.*device$"
}

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


# --- 1. Convert PyTorch ResNet18 model to LiteRT ---
run_case "Convert: PyTorch ResNet18 to LiteRT" \
    litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18"

# Verify the converted model exists
RESNET_TFLITE="$MODEL_DIR/resnet18/resnet18.tflite"
if [ ! -f "$RESNET_TFLITE" ]; then
    echo -e "${RED}Error: Converted model not found at $RESNET_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the ResNet18 model ---
run_case "Quantize: ResNet18 Dynamic Range INT8" \
    litert quantize "$RESNET_TFLITE" --type int8_dynamic --output "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite"

run_case "Quantize: ResNet18 Weight-Only INT8" \
    litert quantize "$RESNET_TFLITE" --type int8_weight_only --output "$MODEL_DIR/resnet18/resnet18_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: ResNet18 FP32 on Desktop (CPU)" \
    litert run "$RESNET_TFLITE" --desktop --cpu --iterations 1

run_case "Run: ResNet18 Dynamic INT8 on Desktop (CPU)" \
    litert run "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: ResNet18 FP32 on Android (CPU)" \
        litert run "$RESNET_TFLITE" --android --cpu --iterations 1

    run_case "Run: ResNet18 Dynamic INT8 on Android (CPU)" \
        litert run "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Android) ---
if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: ResNet18 FP32 on Android" \
        litert benchmark "$RESNET_TFLITE" --android

    run_case "Benchmark: ResNet18 Dynamic INT8 on Android" \
        litert benchmark "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks (litert benchmark only supports Android/GCP).${NC}"
fi

# --- 5. Compile (AOT Compilation) ---
# TODO: Add this back when we fix the NPU compile issue.
# run_case "Compile: ResNet18 FP32 for Qualcomm sm8750 NPU" \
#     litert compile "$RESNET_TFLITE" --target sm8750 --output-dir "$MODEL_DIR/resnet18"

# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$RESNET_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all


# --- Summary Report ---
echo -e "\n${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> RESNET TEST SUMMARY${NC}"
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


echo -e "${GREEN}${BOLD}All ResNet CLI commands executed successfully!${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"
exit 0
