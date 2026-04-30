#!/bin/bash
# LiteRT CLI Demo Script
set -e

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create Python virtual environment if not exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate

# Install litert-cli with ASR extras
pip install -e "$REPO_ROOT[asr]"

# Helper for dynamic Android check
function has_android_device() {
  adb devices | grep -q "[0-9a-zA-Z]\+.*device$"
}

# Cache directory
MODELS_CACHE="$HOME/.cache/litert-cli/models"
mkdir -p "$MODELS_CACHE"

function download_if_not_exists() {
  local ref=$1
  local repo=$2
  local file_pattern=$3

  # Extract main_ref from ref:sub_ref syntax
  local main_ref=${ref%%:*}

  if [ -d "$MODELS_CACHE/$main_ref" ]; then
    echo -e "\033[32m>>> Model reference '$main_ref' already exists in cache, skipping download.\033[0m"
  else
    echo -e "\033[33m>>> Downloading model '$ref'...\033[0m"
    if [ -n "$file_pattern" ]; then
      litert download "$repo" --file "$file_pattern" --model_ref "$ref"
    else
      litert download "$repo" --model_ref "$ref"
    fi
  fi
}

download_if_not_exists "mobilenetv3" "litert-community/MobileNet-v3-large" "*.tflite"
download_if_not_exists "efficientnet:gpu" "litert-community/efficientnet_b1" "*.tflite"
download_if_not_exists "parakeet:main" "litert-community/parakeet-ctc-0.6b" "*i8.tflite"

echo "Running mobilenetv3 on desktop..."
litert run mobilenetv3

if has_android_device; then
    echo "Running efficientnet:gpu on Android (GPU)..."
    litert run efficientnet:gpu --android --gpu

    echo "Benchmarking mobilenetv3 on Android..."
    litert benchmark mobilenetv3 --android
fi

echo -e "\n=== ASR Models Test ==="
echo "Running parakeet:main on desktop with stream..."
litert run parakeet:main

echo -e "\n\033[32mDemo completed successfully!\033[0m"
