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

# ==============================================================================
# LiteRT-CLI PyPI Wheel Build & Integration Test Script
#
# Usage:
#   /tmp/test_litert_cli_wheel.sh [CL_NUMBER]
# ==============================================================================
set -e

CL_ID="${1:-956125631}"
SNAPSHOT_DIR="/google/src/cloud/review/${CL_ID}/google3"

if [ ! -d "${SNAPSHOT_DIR}" ]; then
  echo "Error: Snapshot path ${SNAPSHOT_DIR} does not exist."
  echo "Please verify the CL number."
  exit 1
fi

TIMESTAMP=$(date +%s)
WORK_DIR="/tmp/litert_cli_wheel_test_${TIMESTAMP}"
OSS_SRC_DIR="${WORK_DIR}/oss_src"
DIST_DIR="${WORK_DIR}/dist"
MODELS_DIR="${WORK_DIR}/models"
VENV_DIR="${WORK_DIR}/venv"

echo "======================================================================"
echo " Starting LiteRT-CLI Wheel Test for CL ${CL_ID}"
echo " Workspace: ${WORK_DIR}"
echo "======================================================================"

# 1. Create workspace directories
mkdir -p "${OSS_SRC_DIR}" "${DIST_DIR}" "${MODELS_DIR}"

# 2. Run Copybara locally to transform Google3 source -> Open Source layout
echo ""
echo "[Step 1/5] Running Copybara transformation to OSS format..."
/google/data/ro/teams/copybara/copybara \
  "${SNAPSHOT_DIR}/third_party/py/litert_cli/copy.bara.sky" \
  g3folder_to_gitfolder \
  "${SNAPSHOT_DIR}/.." \
  --folder-dir "${OSS_SRC_DIR}" \
  --force

# 3. Create clean Python virtual environment
echo ""
echo "[Step 2/5] Setting up isolated Python virtual environment..."
cd "${WORK_DIR}"
python3 -m venv --without-pip "${VENV_DIR}"
curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python3" - --extra-index-url https://pypi.org/simple
source "${VENV_DIR}/bin/activate"

# 4. Build PyPI wheel
echo ""
echo "[Step 3/5] Building PyPI .whl package..."
pip install --extra-index-url https://pypi.org/simple build setuptools wheel
cd "${OSS_SRC_DIR}"
python3 -m build --no-isolation --wheel --outdir "${DIST_DIR}"

# 5. Install built wheel + PyPI dependencies
echo ""
echo "[Step 4/5] Installing built wheel into virtualenv..."
pip install --extra-index-url https://pypi.org/simple \
  "${DIST_DIR}"/litert_cli-*.whl \
  ai-edge-quantizer-nightly \
  litert-torch-nightly \
  torch \
  torchvision \
  transformers

# 6. Execute CLI integration test suite
echo ""
echo "[Step 5/5] Running LiteRT CLI Integration Tests..."
cd "${WORK_DIR}"

echo "--- 1. Verify executable ---"
which litert
litert --help

echo "--- 2. litert download ---"
litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "${MODELS_DIR}/mobilenet"

echo "--- 3. litert quantize ---"
litert quantize "${MODELS_DIR}/mobilenet/mobilenet_v3_large.tflite" --recipe dynamic_wi8_afp32 --output "${MODELS_DIR}/mobilenet/dynamic.tflite"

echo "--- 4. litert run (Desktop CPU) ---"
litert run "${MODELS_DIR}/mobilenet/dynamic.tflite" --desktop --cpu

echo "--- 5. litert convert ---"
litert convert "${OSS_SRC_DIR}/litert_cli/test_data/resnet18.py" --output "${MODELS_DIR}/resnet18"

echo "--- 6. litert benchmark (Desktop CPU) ---"
litert benchmark "${MODELS_DIR}/resnet18/resnet18.tflite" --desktop

echo "--- 7. litert convert Qwen ---"
litert convert Qwen/Qwen2.5-0.5B-Instruct --output "${MODELS_DIR}/qwen2.5"

echo ""
echo "======================================================================"
echo " SUCCESS: All LiteRT-CLI PyPI Wheel Integration Tests Passed!"
echo "======================================================================"
