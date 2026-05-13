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

# LiteRT CLI Demo Script
set -e

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define ANSI color codes using \033 for maximum compatibility across Linux and macOS
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create Python virtual environment if not exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
# Upgrade pip and setuptools to ensure build-system requirements can be met.
pip install --upgrade pip setuptools wheel

# Install litert-cli with ASR extras
pip install -e "$REPO_ROOT"

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
    echo -e "${GREEN}>>> Model reference '$main_ref' already exists in cache, skipping download.${NC}"
  else
    echo -e "${YELLOW}>>> Downloading model '$ref'...${NC}"
    if [ -n "$file_pattern" ]; then
      litert download "$repo" --file "$file_pattern" --model-ref "$ref"
    else
      litert download "$repo" --model-ref "$ref"
    fi
  fi
}

download_if_not_exists "mobilenetv3" "litert-community/MobileNet-v3-large" "*.tflite"
download_if_not_exists "efficientnet:gpu" "litert-community/efficientnet_b1" "*.tflite"

echo "Running mobilenetv3 on desktop..."
litert run mobilenetv3

if has_android_device; then
    echo "Running efficientnet:gpu on Android (GPU)..."
    litert run efficientnet:gpu --android --gpu
    litert run efficientnet:gpu --android --npu

    echo "Benchmarking  on Android..."
    litert benchmark efficientnet:gpu --android
    litert benchmark mobilenetv3 --android --npu
fi

echo "Running mobilenetv3 on GCP..."
litert benchmark mobilenetv3 --gcp --devices "pixel 8, sm-s931u1" --gpu

echo -e "\n${GREEN}Demo completed successfully!${NC}"
