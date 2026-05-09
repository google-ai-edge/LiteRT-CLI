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

"""Constants for the LiteRT CLI."""

from __future__ import annotations

import os
import types

# Check if we are in Google3 environment
try:
  import google3  # pylint: disable=unused-import,g-import-not-at-top

  _AUTO_IN_GOOGLE3 = True
except ImportError:
  _AUTO_IN_GOOGLE3 = False

# Environment variable names
ENV_LITERT_CLI_ROOT: str = "LITERT_CLI_ROOT"
ENV_LITERT_CLI_ANDROID_ROOT: str = "LITERT_CLI_ANDROID_ROOT"
ENV_LITERT_CLI_FORCE_OSS: str = "LITERT_CLI_FORCE_OSS"
ENV_LITERT_VERBOSE: str = "LITERT_VERBOSE"

DEFAULT_QUIET: bool = os.environ.get(ENV_LITERT_VERBOSE, "1") != "1"

_FORCE_OSS = os.environ.get(ENV_LITERT_CLI_FORCE_OSS, "").lower() in (
    "1",
    "true",
    "yes",
)

IN_GOOGLE3: bool = _AUTO_IN_GOOGLE3 and not _FORCE_OSS

# Cache directory
LITERT_CLI_CACHE_DIR: str = os.path.join(
    os.path.expanduser("~"), ".cache", "litert-cli"
)

LITERT_MODELS_CACHE_DIR: str = os.path.join(LITERT_CLI_CACHE_DIR, "models")

# Default values
_DEFAULT_CLI_ROOT: str = os.path.join(LITERT_CLI_CACHE_DIR, "root")
_DEFAULT_CLI_ANDROID_ROOT: str = "/data/local/tmp/litert-cli"

# Resolved configuration values
LITERT_CLI_ROOT: str = os.environ.get(ENV_LITERT_CLI_ROOT, _DEFAULT_CLI_ROOT)
LITERT_CLI_ANDROID_ROOT: str = os.environ.get(
    ENV_LITERT_CLI_ANDROID_ROOT, _DEFAULT_CLI_ANDROID_ROOT
)

# Downloads and Caching
LITERT_BINARIES_BASE_URL: str = (
    # "https://storage.googleapis.com/litert/binaries/latest"
    "https://storage.googleapis.com/litert/binaries/2.1.4"
)
LITERT_BINARIES_BASE_URL_ANDROID: str = (
    f"{LITERT_BINARIES_BASE_URL}/android_arm64"
)

# NPU and AOT Configuration
# Keep updating the version in sync with LiteRT latest
# release, and reference to the version in
# https://github.com/google-ai-edge/LiteRT/blob/main/third_party/qairt/workspace.bzl#L25
QAIRT_SDK_VERSION: str = "2.44.0.260225"

QAIRT_SDK_URL: str = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    f"Qualcomm_AI_Runtime_Community/All/{QAIRT_SDK_VERSION}/v{QAIRT_SDK_VERSION}.zip"
)
MEDIATEK_SDK_URL: str = (
    "https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/"
    "66f2c33a-2005-4f0b-afef-2053c8654e4f.gz"
)
MEDIATEK_V8_VERSION: str = "v8_0_10"
MEDIATEK_V9_VERSION: str = "v9_0_3"

from litert_cli.core.targets_manager import TargetsManager

_manager = TargetsManager()
_loaded_targets = _manager.load_targets()

_qnn_map = {}
_mtk_map = {}
_aot_map = {}

if _loaded_targets:
    # Reconstruct maps from loaded targets
    _qnn_map = {
        k: v.properties.get("qnn_version", "")
        for k, v in _loaded_targets.items()
        if v.vendor == "qualcomm"
    }

    _mtk_map = {
        k: v.properties.get("recommend_version", "")
        for k, v in _loaded_targets.items()
        if v.vendor == "mediatek"
    }

    _aot_map = {
        k: (v.vendor, v.vendor_id) for k, v in _loaded_targets.items()
    }

QNN_SOC_VERSION_MAP: types.MappingProxyType[str, str] = (
    types.MappingProxyType(_qnn_map)
)

MEDIATEK_SOC_VERSION_MAP: types.MappingProxyType[str, str] = (
    types.MappingProxyType(_mtk_map)
)

AOT_SUPPORTED_TARGETS: types.MappingProxyType[str, tuple[str, str]] = (
    types.MappingProxyType(_aot_map)
)

