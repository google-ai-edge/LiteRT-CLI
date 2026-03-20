#!/bin/bash
set -e
set -x

# LiteRT CLI Demo Script
# 
# Prerequisites:
# 1) Connect an Android device via USB.
# 2) If using a remote machine, ensure adb port forwarding is active:
#    ssh -R 5037:localhost:5037 shuangfeng.c.googlers.com
# 3) You might be asked to set up gcloud credentials for AI Edge Portal.

# If you install from pip, you can skip the following step, which is just for local script test.
#REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Clean up and set up the environment
export LITERT_CLI_ROOT=${LITERT_CLI_ROOT:-"/tmp/litert_cli_demo"}
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install litert-cli from local source or test pypi
# pip install -e "$REPO_ROOT"
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple litert-cli==0.1.1.dev11

# Download Models
# MobileNet V3 for CPU/Quantization tests
litert download litert-community/MobileNet-v3-large --output mobilenet
# EfficientNet B1 for GPU tests (as V3 contains GATHER_ND which isn't GPU-compatible yet)
litert download litert-community/efficientnet_b1 --output efficientnet

# Visualize (Note: This is a blocking command, uncomment to test manually)
# litert visualize mobilenet/tflite_model.tflite

MOBILENET=$(ls mobilenet/*.tflite | head -n 1)
EFFICIENTNET=$(ls efficientnet/*.tflite | head -n 1)

# Quantize
litert quantize "$MOBILENET" --output mobilenet_quant.tflite

# --- Desktop Execution ---
# Run quantized model on Desktop CPU
litert run mobilenet_quant.tflite --desktop --cpu

# Run float model on Desktop GPU (EfficientNet)
litert run "$EFFICIENTNET" --desktop --gpu

# --- Android Execution ---
# Run quantized model on Android CPU
litert run mobilenet_quant.tflite --android --cpu

# Run float model on Android GPU (EfficientNet)
litert run "$EFFICIENTNET" --android --gpu

# Benchmark on android
litert benchmark mobilenet_quant.tflite --android
litert benchmark "$EFFICIENTNET" --android --gpu

# Benchmark on AI Edge Portal
# Need to setup gcloud credentials first.
litert benchmark mobilenet_quant.tflite --gcp
litert benchmark "$EFFICIENTNET" --gcp --gpu

# Convert PyTorch script to TFLite
TEST_DATA_DIR=$(python -c "import os, litert_cli; print(os.path.join(os.path.dirname(litert_cli.__file__), 'test_data'))")
litert convert "$TEST_DATA_DIR/resnet18.py"

# Benchmark ResNet18
litert benchmark resnet18/resnet18.tflite --android

# Convert Qwen from HF (Disabled for speed in demo run)
litert convert Qwen/Qwen1.5-0.5B-Chat

# Run LLM: requires to install litert-lm-cli package, which is not ready yet.
# Check: go/litert-lm-cli-food
# litert lm run Qwen1.5-0.5B-Chat