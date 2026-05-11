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

# Shared utilities and helpers for LiteRT CLI demo scripts on Windows

$esc = [char]27
$GREEN = "$esc[0;32m"
$BLUE = "$esc[0;34m"
$YELLOW = "$esc[0;33m"
$RED = "$esc[0;31m"
$NC = "$esc[0m"
$BOLD = "$esc[1m"

$script:TOTAL_CASES = 0
$script:TOTAL_PASSED = 0
$script:TOTAL_FAILED = 0
$script:PASSED_CASES = @()
$script:FAILED_CASES = @()

# Helper for dynamic Android check (supports both authorized and unauthorized devices)
function Has-AndroidDevice {
    try {
        $devices = adb devices 2>$null
        if ($null -ne $devices) {
            foreach ($line in $devices) {
                if ($line -match "\s+(device|unauthorized)$") {
                    return $true
                }
            }
        }
    } catch {}
    return $false
}

# Helper to verify if LiteRT GPU accelerator is supported on Desktop (excluding software emulation like llvmpipe)
function Has-DesktopGpu {
    param([string]$modelFile)
    
    # Double escape backslashes for Python string literal
    $pythonModelPath = $modelFile.Replace('\', '\\')
    
    $pythonCmd = @"
import sys
try:
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    cm = CompiledModel.from_file('$pythonModelPath', HardwareAccelerator.GPU)
except Exception as e:
    sys.exit(1)
"@
    
    try {
        $output = & python -c $pythonCmd 2>&1
        $status = $LASTEXITCODE
        if ($status -eq 0 -and $output -notmatch "llvmpipe" -and $output -notmatch "lavapipe") {
            return $true
        }
    } catch {}
    return $false
}

# Robust runner for a test command with isolation and formatting
function Run-Case {
    param(
        [string]$title,
        [scriptblock]$cmdBlock
    )
    
    $esc = [char]27
    $BLUE = "$esc[0;34m"
    $GREEN = "$esc[0;32m"
    $RED = "$esc[0;31m"
    $NC = "$esc[0m"
    $BOLD = "$esc[1m"
    
    Write-Host ""
    Write-Host "${BLUE}▶ Running:${NC} ${BOLD}$title${NC}"
    $cmdStr = $cmdBlock.ToString().Trim()
    Write-Host "$esc[90mCommand: $cmdStr$NC"
    Write-Host "$esc[90m------------------------------------------------------------$NC"
    
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    
    & $cmdBlock
    $status = $LASTEXITCODE
    
    if ($null -eq $status) {
        if ($?) { $status = 0 } else { $status = 1 }
    }
    
    $ErrorActionPreference = $oldErrorActionPreference
    
    Write-Host "$esc[90m------------------------------------------------------------$NC"
    if ($status -eq 0) {
        Write-Host "${GREEN}✔ SUCCESS:${NC} ${GREEN}${BOLD}$title${NC}"
        $script:TOTAL_PASSED++
        $script:PASSED_CASES += $title
    } else {
        Write-Host "${RED}✘ FAILED (Exit Code: $status):${NC} ${RED}${BOLD}$title${NC}"
        $script:TOTAL_FAILED++
        $script:FAILED_CASES += $title
    }
    $script:TOTAL_CASES++
    return $status
}

# Prints the final summary report for the demo
function Print-SummaryReport {
    param([string]$modelName)
    
    $esc = [char]27
    $BLUE = "$esc[0;34m"
    $GREEN = "$esc[0;32m"
    $RED = "$esc[0;31m"
    $NC = "$esc[0m"
    $BOLD = "$esc[1m"
    
    $upperModel = $modelName.ToUpper()
    
    Write-Host ""
    Write-Host "${BLUE}${BOLD}==================================================================${NC}"
    Write-Host "${BLUE}${BOLD}>>> ${upperModel} TEST SUMMARY${NC}"
    Write-Host "${BLUE}${BOLD}==================================================================${NC}"
    Write-Host "Total Cases Run: ${BOLD}$script:TOTAL_CASES${NC}"
    Write-Host "Passed:          ${GREEN}${BOLD}$script:TOTAL_PASSED${NC}"
    Write-Host "Failed:          ${RED}${BOLD}$script:TOTAL_FAILED${NC}"
    
    if ($script:TOTAL_PASSED -gt 0) {
        Write-Host ""
        Write-Host "${GREEN}${BOLD}Passed Cases:${NC}"
        foreach ($case in $script:PASSED_CASES) {
            Write-Host "  - ${GREEN}$case${NC}"
        }
    }
    
    if ($script:TOTAL_FAILED -gt 0) {
        Write-Host ""
        Write-Host "${RED}${BOLD}Failed Cases:${NC}"
        foreach ($case in $script:FAILED_CASES) {
            Write-Host "  - ${RED}$case${NC}"
        }
        Write-Host "${BLUE}${BOLD}==================================================================${NC}"
        Exit 1
    }
    
    Write-Host ""
    Write-Host "${GREEN}${BOLD}All ${modelName} CLI commands executed successfully!${NC}"
    Write-Host "${BLUE}${BOLD}==================================================================${NC}"
    Exit 0
}
