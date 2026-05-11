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

# LiteRT CLI Gemma4 LLM Demo & Test Script for Windows

# --- Environment Setup ---
$SCRIPT_DIR = $PSScriptRoot
$REPO_ROOT = (Resolve-Path "$SCRIPT_DIR/../..").Path
$LITERT_CLI_ROOT = Join-Path $env:TEMP "litert_cli_gemma4"

# Source shared utilities
. (Join-Path $SCRIPT_DIR "utils.ps1")

Write-Host "${BLUE}${BOLD}==================================================================${NC}"
Write-Host "${BLUE}${BOLD}>>> LiteRT CLI Gemma4 LLM Demo Script${NC}"
Write-Host "${BLUE}${BOLD}==================================================================${NC}"

# Clean up and create work directory
Write-Host ""
Write-Host "${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
if (Test-Path $LITERT_CLI_ROOT) {
    Remove-Item -Recurse -Force $LITERT_CLI_ROOT
}
New-Item -ItemType Directory -Force -Path $LITERT_CLI_ROOT | Out-Null
Set-Location $LITERT_CLI_ROOT

# Create Python virtual environment
Write-Host "${YELLOW}Creating Python virtual environment...${NC}"
python -m venv venv_gemma4
. .\venv_gemma4\Scripts\Activate.ps1

# Create output directories
$MODEL_DIR = Join-Path $LITERT_CLI_ROOT "models"
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null

# Install litert-cli from source
Write-Host "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

# --- 1. Convert HuggingFace Model google/gemma-4-E2B-it ---
# TODO: Bring this back when we add support for --externalize_embedder in CLI convert command.
# Run-Case "Convert: HuggingFace google/gemma-4-E2B-it" {
#     litert convert google/gemma-4-E2B-it --output "$MODEL_DIR/gemma4"
# }

# --- 2. Run Gemma4 Generative LLM Model ---
Run-Case "Run Gemma4: Generative inference with custom prompt" {
    litert lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --prompt="What is the capital of France?"
}

# --- 3. Benchmark Gemma4 LLM Model ---
Run-Case "Benchmark Gemma4: Local benchmark of LLM generation" {
    litert lm benchmark gemma-4-E2B-it.litertlm --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm -p 128 -d 128
}

# --- Summary Report ---
Print-SummaryReport "Gemma4"
