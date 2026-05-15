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

# LiteRT CLI Run Commands Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "run_test" "Run Commands Demo"

echo -e "\n${BLUE}${BOLD}--- 1. Run Commands ---${NC}"

run_case "Download: MobileNet-v3-large" \
    litert download litert-community/MobileNet-v3-large --file "*.tflite" --output "models/mobilenet"

run_case "Run: MobileNet on Desktop (CPU)" \
    litert run "models/mobilenet/mobilenet_v3_large.tflite" --desktop --cpu --iterations 1

if has_desktop_gpu "models/mobilenet/mobilenet_v3_large.tflite"; then
    run_case "Run: MobileNet on Desktop (GPU)" \
        litert run "models/mobilenet/mobilenet_v3_large.tflite" --desktop --gpu --iterations 1
fi

if has_android_device; then
    run_case "Run: MobileNet on Android (CPU)" \
        litert run "models/mobilenet/mobilenet_v3_large.tflite" --android --cpu --iterations 1

    run_case "Run: MobileNet on Android (GPU)" \
        litert run "models/mobilenet/mobilenet_v3_large.tflite" --android --gpu --iterations 1
fi

# Multi-input tests
run_case "Run: Multi-Input String Scalar" \
    litert run dummy_cv_model.tflite --desktop --input inputs="0.5" --print-tensors --iterations 1

run_case "Run: Multi-Input String Array" \
    litert run dummy_cv_model.tflite --desktop --input inputs="[0.5, 0.5, 0.5]" --print-tensors --iterations 1

echo "Generating test input files..."
python3 generate_test_inputs.py .

run_case "Run: Multi-Input .npy file" \
    litert run dummy_cv_model.tflite --desktop --input inputs="test_input.npy" --print-tensors --iterations 1

run_case "Run: Multi-Input .raw file" \
    litert run dummy_cv_model.tflite --desktop --input inputs="test_input.raw" --print-tensors --iterations 1

print_summary_report "Run Commands"
