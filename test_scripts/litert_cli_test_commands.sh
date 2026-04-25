#!/bin/bash
# Shared test commands for LiteRT CLI demo.
# This file should be sourced by driver script which setup environments first.

# Helper for dynamic Android check
function has_android_device() {
  adb devices | grep -q "[0-9a-zA-Z]\+.*device$"
}

function log_section() {
  echo -e "\n=================================================================="
  echo -e ">>> $1"
  echo -e "=================================================================="
}

# --- Model Helper Functions ---
function get_mobilenet() {
    local file="$MODEL_DIR/mobilenet/mobilenet_v3_large.tflite"
    if [ ! -f "$file" ]; then
        echo "Model not found. Will run download first." >&2
        test_download >&2
    fi
    echo "$file"
}

function get_efficientnet() {
    local file="$MODEL_DIR/efficientnet/efficientnet_b1.tflite"
    if [ ! -f "$file" ]; then
        echo "Model not found. Will run download first." >&2
        test_download >&2
    fi
    echo "$file"
}

function get_mobilenet_quant() {
    local file="$MODEL_DIR/dynamic.tflite"
    if [ ! -f "$file" ]; then
        echo "Quantized model not found. Will run quantize first." >&2
        test_quantize >&2
    fi
    echo "$file"
}

function get_compiled_model() {
    local file="$MODEL_DIR/efficientnet_b1_Qualcomm_SM8750.tflite"
    if [ ! -f "$file" ]; then
        echo "Compiled model not found. Will run compile first." >&2
        test_compile >&2
    fi
    echo "$file"
}

function test_download() {
    log_section "Testing: litert download"

    echo "--- Specific file pattern (*.tflite) ---"
    litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "$MODEL_DIR/mobilenet"

    echo "--- Full repository download ---"
    litert download litert-community/MobileNet-v3-large --output "$MODEL_DIR/mobilenet_full"
    litert download litert-community/efficientnet_b1 --output "$MODEL_DIR/efficientnet"
}

function test_quantize() {
    log_section "Testing: litert quantize"
    local mobilenet=$(get_mobilenet)

    echo "--- Dynamic Quantization ---"
    litert quantize "$mobilenet" --type int8_dynamic --output "$MODEL_DIR/dynamic.tflite"

    echo "--- Weight-Only Quantization ---"
    litert quantize "$mobilenet" --type int8_weight_only --output "$MODEL_DIR/weight_only.tflite"

    echo "--- Static Range Quantization with calibration data ---"
    litert quantize "$mobilenet" --type static --calibration-data "$TEST_DATA_DIR/mobilenet_v3_calib_data.py" --output "$MODEL_DIR/static.tflite"

    echo "--- Recipe-based Quantization ---"
    litert quantize "$mobilenet" --recipe "$TEST_DATA_DIR/quantize_recipe.json" --output "$MODEL_DIR/recipe.tflite"
}

function test_compile() {
    log_section "Testing: litert compile"
    local efficientnet=$(get_efficientnet)
    
    if [[ "$(uname)" == "Linux" ]]; then
        litert compile "$efficientnet" --target sm8750 --output-dir "$MODEL_DIR"
    else
        echo "Skipping compile test on non-Linux platform ($(uname))"
    fi
}


function test_run() {
    log_section "Testing: litert run"
    local mobilenet_quant=$(get_mobilenet_quant)
    local efficientnet=$(get_efficientnet)
    local compiled_model=$(get_compiled_model)

    TEST_MODEL="$TEST_DATA_DIR/dummy_cv_model.tflite"

    echo "--- Desktop Run ---"
    litert run "$mobilenet_quant" --desktop --cpu
    litert run "$efficientnet" --desktop --cpu

    echo "--- Android Run (Requires connected device) ---"
    if has_android_device; then
        litert run "$mobilenet_quant" --android --cpu
        litert run "$efficientnet" --android --gpu

        if [[ "$(uname)" == "Linux" ]]; then
            echo "Removing old NPU runtime libraries on device..."
            adb shell rm -f "$LITERT_CLI_ANDROID_ROOT/libQnn*" "$LITERT_CLI_ANDROID_ROOT/libLiteRtDispatch_Qualcomm.so"

            echo "Running NPU compatible model on Android NPU..."
            litert run "$compiled_model" --android --npu
        else
            echo "Skipping Android NPU run test on non-Linux platform ($(uname))"
        fi
    else
        echo "No Android device detected or skipped via flag. Skipping Android execution."
    fi

    echo "--- Multi-Input Formats (Strings, Arrays) ---"
    for target in "--desktop" "--android"; do
        if [ "$target" == "--android" ] && ! has_android_device; then
            continue
        fi
        echo "Running Multi-Input strings/scalars on $target..."
        litert run "$TEST_MODEL" $target --input inputs="0.5" --print-tensors --iterations 1
        litert run "$TEST_MODEL" $target --input inputs="[0.5, 0.5, 0.5]" --print-tensors --iterations 1
    done

    echo "--- Multi-Input Formats (Files) ---"
    echo "Generating test input files..."
    generate_test_inputs

    for target in "--desktop" "--android"; do
        if [ "$target" == "--android" ] && ! has_android_device; then
            continue
        fi
        echo "Running Multi-Input files on $target..."
        litert run "$TEST_MODEL" $target --input inputs="$LITERT_CLI_ROOT/test_input.npy" --print-tensors --iterations 1
        litert run "$TEST_MODEL" $target --input inputs="$LITERT_CLI_ROOT/test_input.raw" --print-tensors --iterations 1
        if [ -f "$LITERT_CLI_ROOT/test_input.png" ]; then
            litert run "$TEST_MODEL" $target --input inputs="$LITERT_CLI_ROOT/test_input.png" --print-tensors --iterations 1
        fi
        litert run "$TEST_MODEL" $target --input "$LITERT_CLI_ROOT/test_input.npy" --print-tensors --iterations 1
    done
}

function test_benchmark() {
    log_section "Testing: litert benchmark"
    local mobilenet_quant=$(get_mobilenet_quant)
    local efficientnet=$(get_efficientnet)
    local compiled_model=$(get_compiled_model)

    if has_android_device; then
        litert benchmark "$mobilenet_quant" --android
        litert benchmark "$efficientnet" --android --gpu
        if [ -f "$compiled_model" ]; then
            litert benchmark "$compiled_model" --android --npu
        fi
    else
        echo "No Android device detected. Skipping Android benchmarks."
    fi

    echo "--- Benchmarking on AI Edge Portal ---"
    litert benchmark "$mobilenet_quant" --gcp
    litert benchmark "$efficientnet" --gcp --gpu
}

function test_convert() {
    log_section "Testing: litert convert"
    litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18"
    litert convert Qwen/Qwen1.5-0.5B-Chat --output $MODEL_DIR/qwen0.5b

    if has_android_device; then
        litert benchmark "$MODEL_DIR/resnet18/resnet18.tflite" --android
    fi
}

function test_visualize() {
    log_section "Testing: litert visualize"
    local mobilenet=$(get_mobilenet)

    if [ ! -f "$mobilenet" ]; then
        echo "Model not found. Please run download first."
        return 1
    fi

    echo "--- Standard mode (Starts background server) ---"
    litert visualize "$mobilenet"

    echo "--- Force new server port (no_reuse_server) ---"
    litert visualize "$mobilenet" --no_reuse_server

    echo "--- Clean up and stop all servers ---"
    litert visualize --stop_all
}

case "$COMMAND" in
    "download") test_download ;;
    "quantize") test_quantize ;;
    "compile") test_compile ;;
    "run") test_run ;;
    "benchmark") test_benchmark ;;
    "convert") test_convert ;;
    "visualize") test_visualize ;;
    "--all")
        test_download
        test_quantize
        test_compile
        test_run
        test_benchmark
        test_convert
        test_visualize
        ;;
    *)
        echo "Usage: $0 [download|quantize|compile|run|benchmark|convert|visualize|--all]"
        exit 1
        ;;
esac
