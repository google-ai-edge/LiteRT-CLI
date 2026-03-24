#!/bin/bash
set -ex

# This e2e script install your local working space and test all the commands. It assumes:
# 1) You are running on Glinux (haven't tested on Mac and Windows yet)
# 2) You have Android arm64 devices connected to adb if you want to run Android tests.

# Find the repository root from the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set up the environment
export LITERT_CLI_ROOT=${LITERT_CLI_ROOT:-"/tmp/litert_cli_e2e"}
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install litert-cli from local scripts
pip install -q -e "$REPO_ROOT"

# Find test data and scripts from the installed package
TEST_DATA_DIR=$(python3 -c "import os, litert_cli; print(os.path.join(os.path.dirname(litert_cli.__file__), 'test_data'))")
# Discover test_scripts via namespace package discovery, ensure we get a plain path string
TEST_SCRIPTS_DIR=$(python3 -c "import test_scripts; print(list(test_scripts.__path__)[0])")

GENERATE_SCRIPT="$TEST_DATA_DIR/generate_test_inputs.py"

echo "=================================================="
echo "Testing: litert CLI End-to-End Workflows"
echo "=================================================="

# Create output directories
MODEL_DIR=$LITERT_CLI_ROOT/models
mkdir -p $MODEL_DIR
DUMMY_CV_MODEL="$TEST_DATA_DIR/dummy_cv_model.tflite"

function test_download() {
    echo "=== [1] Testing: litert download ==="
    
    echo "--- 1.1: Specific file pattern (*.tflite) ---"
    mkdir -p $MODEL_DIR/mobilenet
    litert download litert-community/MobileNet-v3-large --file "*.tflite" --output $MODEL_DIR/mobilenet

    echo "--- 1.2: Full repository download ---"
    litert download litert-community/MobileNet-v3-large --output $MODEL_DIR/mobilenet_full
}

function test_quantize() {
    echo "=== [2] Testing: litert quantize ==="
    
    STATIC_MODEL="$MODEL_DIR/mobilenet/mobilenet_v3_large.tflite"
    if [ ! -f "$STATIC_MODEL" ]; then
        echo "Downloaded model $STATIC_MODEL not found. Skipping static tests."
        return
    fi

    echo "--- 2.1: Dynamic Quantization ---"
    litert quantize $STATIC_MODEL --type int8_dynamic --output $MODEL_DIR/dynamic.tflite

    echo "--- 2.2: Weight-Only Quantization ---"
    litert quantize $STATIC_MODEL --type int8_weight_only --output $MODEL_DIR/weight_only.tflite

    echo "--- 2.3: Static Range Quantization with calibration data ---"
    litert quantize $STATIC_MODEL --type static --calibration-data "$TEST_DATA_DIR/mobilenet_v3_calib_data.py" --output $MODEL_DIR/static.tflite

    echo "--- 2.4: Recipe-based Quantization ---"
    litert quantize $STATIC_MODEL --recipe "$TEST_DATA_DIR/quantize_recipe.json" --output $MODEL_DIR/recipe.tflite
}

function test_run() {
    echo "=== [3] Testing: litert run ==="
    TEST_MODEL="$TEST_DATA_DIR/dummy_cv_model.tflite"
    if [ ! -f "$TEST_MODEL" ]; then
        echo "Test model not found. Skipping."
        return
    fi

    echo "--- 3.1: Desktop Run ---"
    litert run $TEST_MODEL --desktop --cpu --iterations 2 --print_tensors

    echo "--- 3.2: Android Run (Requires connected device) ---"
    if adb devices | grep -q "[0-9a-zA-Z]\+.*device$"; then
      litert run $TEST_MODEL --android --gpu --iterations 2 --print_tensors
    else
      echo "No Android device found, skipping Android tests."
    fi

    echo "--- 3.2: Multi-Input Formats (Strings, Arrays) ---"
    for target in "--desktop" "--android"; do
        if [ "$target" == "--android" ] && ! adb devices | grep -q "[0-9a-zA-Z]\+.*device$"; then
            continue
        fi
        echo "Running Multi-Input strings/scalars on $target..."
        litert run $TEST_MODEL $target --input inputs="0.5" --print_tensors --iterations 1
        litert run $TEST_MODEL $target --input inputs="[0.5, 0.5, 0.5]" --print_tensors --iterations 1
    done

    echo "--- 3.3: Multi-Input Formats (Files) ---"
    # Ensure dependencies for the helper script are present in the venv
    pip install -q Pillow
    # Generate test input files (.npy, .raw, .png)
    python3 "$GENERATE_SCRIPT" "$LITERT_CLI_ROOT"

    for target in "--desktop" "--android"; do
        if [ "$target" == "--android" ] && ! adb devices | grep -q "[0-9a-zA-Z]\+.*device$"; then
            echo "Skipping $target file test: No device found"
            continue
        fi
        echo "Running Multi-Input files on $target..."
        litert run $TEST_MODEL $target --input inputs="$LITERT_CLI_ROOT/test_input.npy" --print_tensors --iterations 1
        litert run $TEST_MODEL $target --input inputs="$LITERT_CLI_ROOT/test_input.raw" --print_tensors --iterations 1
        litert run $TEST_MODEL $target --input inputs="$LITERT_CLI_ROOT/test_input.png" --print_tensors --iterations 1
        litert run $TEST_MODEL $target --input "$LITERT_CLI_ROOT/test_input.npy" --print_tensors --iterations 1
    done
}

function test_benchmark() {
    echo "=== [4] Testing: litert benchmark ==="
    TEST_MODEL="$TEST_DATA_DIR/dummy_cv_model.tflite"
    if [ ! -f "$TEST_MODEL" ]; then
        echo "Test model not found. Skipping."
        return
    fi
    
    echo "--- 4.1: Android Benchmark (CPU & GPU) ---"
    if adb devices | grep -q "[0-9a-zA-Z]\+.*device$"; then
      litert benchmark $TEST_MODEL --android --cpu
      litert benchmark $TEST_MODEL --android --gpu
    else
      echo "No Android device found, skipping Android benchmark tests."
    fi

    echo "--- 4.2: GCP Benchmark (Dry-run guide) ---"
    # litert benchmark $TEST_MODEL --gcp --gpu --device "pixel 7"
}

function test_convert() {
    echo "=== [5] Testing: litert convert ==="
    mkdir -p $MODEL_DIR/resnet18
    
    echo "--- 5.1: PyTorch script to TFLite (ResNet18) ---"
    litert convert "$TEST_DATA_DIR/resnet18.py" --output $MODEL_DIR/resnet18
 
    if adb devices | grep -q "[0-9a-zA-Z]\+.*device$"; then
      litert benchmark $MODEL_DIR/resnet18/resnet18.tflite --android
    fi

    echo "--- 5.2: HuggingFace Model (Qwen) ---"
    # litert convert Qwen/Qwen1.5-0.5B-Chat --output $MODEL_DIR/qwen0.5b

    echo "--- 5.3: HuggingFace Model with explicit task ---"
    # litert convert Qwen/Qwen1.5-0.5B-Chat --output $MODEL_DIR/qwen0.5b_tasked
}

function test_visualize() {
    echo "=== [6] Testing: litert visualize ==="
    TEST_MODEL="$TEST_DATA_DIR/dummy_cv_model.tflite"
    
    echo "--- 6.1: Standard mode (Starts background server) ---"
    litert visualize $TEST_MODEL

    echo "--- 6.2: Force new server port (no_reuse_server) ---"
    litert visualize $TEST_MODEL --no_reuse_server

    echo "--- 6.3: Clean up and stop all servers ---"
    litert visualize --stop_all
}

# Command line interface
COMMAND=${1:-"--all"}

case "$COMMAND" in
    "download") test_download ;;
    "quantize") test_quantize ;;
    "run") test_run ;;
    "benchmark") test_benchmark ;;
    "convert") test_convert ;;
    "visualize") test_visualize ;;
    "--all")
        test_download
        test_quantize
        test_run
        test_benchmark
        test_convert
        test_visualize
        ;;
    *)
        echo "Usage: $0 [download|quantize|run|benchmark|convert|visualize|--all]"
        exit 1
        ;;
esac

echo "=================================================="
echo "End-to-End test suite '$COMMAND' completed successfully!"
echo "=================================================="
