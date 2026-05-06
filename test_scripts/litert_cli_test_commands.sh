#!/bin/bash
# Shared test commands for LiteRT CLI demo.
# This file should be sourced by driver script which setup environments first.

# Initialize test tracking variables
TOTAL_CASES=0
TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_CASES=()

# Helper for dynamic Android check
function has_android_device() {
  adb devices | grep -q "[0-9a-zA-Z]\+.*device$"
}

function log_section() {
  echo -e "\n=================================================================="
  echo -e ">>> \e[1m$1\e[0m"
  echo -e "=================================================================="
}

# Helper to run a test case with isolation, timing, and status reporting
function run_case() {
    local title="$1"
    shift
    
    echo -e "\n\e[34m▶ Running:\e[0m \e[1;37m$title\e[0m"
    echo -e "\e[90mCommand: $*\e[0m"
    echo -e "\e[90m------------------------------------------------------------\e[0m"
    
    # Execute the command (disabling immediate exit on failure just for the test command)
    set +e
    "$@"
    local status=$?
    set -e
    
    echo -e "\e[90m------------------------------------------------------------\e[0m"
    if [ $status -eq 0 ]; then
        echo -e "\e[32m✔ SUCCESS:\e[0m \e[1;32m$title\e[0m"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
    else
        echo -e "\e[31m✘ FAILED (Exit Code: $status):\e[0m \e[1;31m$title\e[0m"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_CASES+=("$title")
    fi
    TOTAL_CASES=$((TOTAL_CASES + 1))
    return $status
}

function print_test_summary() {
    echo -e "\n=================================================================="
    echo -e ">>> \e[1mTEST SUMMARY\e[0m"
    echo -e "=================================================================="
    echo -e "Total Test Cases Run: \e[1m$TOTAL_CASES\e[0m"
    echo -e "Passed:              \e[1;32m$TOTAL_PASSED\e[0m"
    echo -e "Failed:              \e[1;31m$TOTAL_FAILED\e[0m"
    
    if [ $TOTAL_FAILED -gt 0 ]; then
        echo -e "\n\e[1;31mFailed Test Cases:\e[0m"
        for case in "${FAILED_CASES[@]}"; do
            echo -e "  - \e[1;31m$case\e[0m"
        done
        echo -e "=================================================================="
        return 1
    fi
    echo -e "\e[1;32mAll tests completed successfully!\e[0m"
    echo -e "=================================================================="
    return 0
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

    run_case "Download MobileNet-v3-large (*.tflite only)" \
        litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "$MODEL_DIR/mobilenet"

    run_case "Download MobileNet-v3-large (Full Repo)" \
        litert download litert-community/MobileNet-v3-large --output "$MODEL_DIR/mobilenet_full"

    run_case "Download EfficientNet-B1 (Full Repo)" \
        litert download litert-community/efficientnet_b1 --output "$MODEL_DIR/efficientnet"
}

function test_quantize() {
    log_section "Testing: litert quantize"
    local mobilenet=$(get_mobilenet)

    run_case "Quantize: Dynamic Range INT8" \
        litert quantize "$mobilenet" --type int8_dynamic --output "$MODEL_DIR/dynamic.tflite"

    run_case "Quantize: Weight-Only INT8" \
        litert quantize "$mobilenet" --type int8_weight_only --output "$MODEL_DIR/weight_only.tflite"

    run_case "Quantize: Static Range with Calibration Data" \
        litert quantize "$mobilenet" --type static --calibration-data "$TEST_DATA_DIR/mobilenet_v3_calib_data.py" --output "$MODEL_DIR/static.tflite"

    run_case "Quantize: Recipe-based" \
        litert quantize "$mobilenet" --recipe "$TEST_DATA_DIR/quantize_recipe.json" --output "$MODEL_DIR/recipe.tflite"
}

function test_compile() {
    log_section "Testing: litert compile"
    local efficientnet=$(get_efficientnet)

    if [[ "$(uname)" != "Linux" ]]; then
        echo "Skipping compile test on non-Linux platform ($(uname))"
        return 0
    fi

    run_case "Compile EfficientNet-B1 for Qualcomm sm8750" \
        litert compile "$efficientnet" --target sm8750 --output-dir "$MODEL_DIR"
}

function test_run() {
    log_section "Testing: litert run"
    local mobilenet_quant=$(get_mobilenet_quant)
    local efficientnet=$(get_efficientnet)
    local compiled_model=$(get_compiled_model)

    TEST_MODEL="$TEST_DATA_DIR/dummy_cv_model.tflite"

    run_case "Run: MobileNet Quant on Desktop (CPU)" \
        litert run "$mobilenet_quant" --desktop --cpu

    run_case "Run: EfficientNet on Desktop (CPU)" \
        litert run "$efficientnet" --desktop --cpu

    if has_android_device; then
        run_case "Run: MobileNet Quant on Android (CPU)" \
            litert run "$mobilenet_quant" --android --cpu

        run_case "Run: EfficientNet on Android (GPU)" \
            litert run "$efficientnet" --android --gpu

        if [[ "$(uname)" == "Linux" ]]; then
            echo "Removing old NPU runtime libraries on device..."
            adb shell rm -f "$LITERT_CLI_ANDROID_ROOT/libQnn*" "$LITERT_CLI_ANDROID_ROOT/libLiteRtDispatch_Qualcomm.so"

            if [ -f "$compiled_model" ]; then
                run_case "Run: Compiled EfficientNet on Android (NPU)" \
                    litert run "$compiled_model" --android --npu
            else
                echo "Compiled model not found: $compiled_model. Skipping NPU run."
            fi
        fi
    fi

    for target in "--desktop" "--android"; do
        if [ "$target" == "--android" ] && ! has_android_device; then
            continue
        fi
        run_case "Run: Multi-Input String Scalar ($target)" \
            litert run "$TEST_MODEL" $target --input inputs="0.5" --print-tensors --iterations 1

        run_case "Run: Multi-Input String Array ($target)" \
            litert run "$TEST_MODEL" $target --input inputs="[0.5, 0.5, 0.5]" --print-tensors --iterations 1
    done

    echo "Generating test input files..."
    generate_test_inputs

    for target in "--desktop" "--android"; do
        if [ "$target" == "--android" ] && ! has_android_device; then
            continue
        fi
        run_case "Run: Multi-Input .npy file ($target)" \
            litert run "$TEST_MODEL" $target --input inputs="$LITERT_CLI_ROOT/test_input.npy" --print-tensors --iterations 1

        run_case "Run: Multi-Input .raw file ($target)" \
            litert run "$TEST_MODEL" $target --input inputs="$LITERT_CLI_ROOT/test_input.raw" --print-tensors --iterations 1

        if [ -f "$LITERT_CLI_ROOT/test_input.png" ]; then
            run_case "Run: Multi-Input .png file ($target)" \
                litert run "$TEST_MODEL" $target --input inputs="$LITERT_CLI_ROOT/test_input.png" --print-tensors --iterations 1
        fi

        run_case "Run: Direct positional input ($target)" \
            litert run "$TEST_MODEL" $target --input "$LITERT_CLI_ROOT/test_input.npy" --print-tensors --iterations 1
    done
}

function test_benchmark() {
    log_section "Testing: litert benchmark"
    local mobilenet_quant=$(get_mobilenet_quant)
    local efficientnet=$(get_efficientnet)
    local compiled_model=$(get_compiled_model)

    if has_android_device; then
        run_case "Benchmark: MobileNet Quant on Android (CPU)" \
            litert benchmark "$mobilenet_quant" --android

        run_case "Benchmark: EfficientNet on Android (GPU)" \
            litert benchmark "$efficientnet" --android --gpu

        run_case "Benchmark: EfficientNet on Android (NPU)" \
            litert benchmark "$efficientnet" --android --npu

        if [ -f "$compiled_model" ]; then
            run_case "Benchmark: Compiled EfficientNet on Android (NPU)" \
                litert benchmark "$compiled_model" --android --npu
        else
            echo "Compiled model not found: $compiled_model. Skipping NPU benchmark."
        fi
    fi

    run_case "Benchmark: MobileNet Quant on GCP (AI Edge Portal)" \
        litert benchmark "$mobilenet_quant" --gcp

    run_case "Benchmark: EfficientNet on GCP (AI Edge Portal GPU)" \
        litert benchmark "$efficientnet" --gcp --gpu
}

function test_convert() {
    log_section "Testing: litert convert"
    run_case "Convert: PyTorch ResNet18 Model" \
        litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18"

    run_case "Convert: HuggingFace Qwen 1.5 0.5B Model" \
        litert convert Qwen/Qwen1.5-0.5B-Chat --output $MODEL_DIR/qwen0.5b

    if has_android_device; then
        run_case "Benchmark Converted ResNet18 on Android" \
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

    run_case "Visualize: Standard Mode" \
        litert visualize "$mobilenet"

    run_case "Visualize: Force New Server Port" \
        litert visualize "$mobilenet" --no_reuse_server

    run_case "Visualize: Stop All Servers" \
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

# Print final test summary report
print_test_summary
