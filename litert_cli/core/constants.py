"""Constants for the LiteRT CLI."""

from __future__ import annotations

import os

# Environment variable names
ENV_LITERT_CLI_ROOT: str = "LITERT_CLI_ROOT"
ENV_LITERT_CLI_ANDROID_ROOT: str = "LITERT_CLI_ANDROID_ROOT"

# Default values
_DEFAULT_CLI_ROOT: str = "/tmp/litert-cli"
_DEFAULT_CLI_ANDROID_ROOT: str = "/data/local/tmp/litert-cli"

# Resolved configuration values
LITERT_CLI_ROOT: str = os.environ.get(ENV_LITERT_CLI_ROOT, _DEFAULT_CLI_ROOT)
LITERT_CLI_ANDROID_ROOT: str = os.environ.get(
    ENV_LITERT_CLI_ANDROID_ROOT, _DEFAULT_CLI_ANDROID_ROOT
)

# Downloads and Caching
LITERT_BINARIES_BASE_URL: str = "https://storage.googleapis.com/litert/binaries/latest"
LITERT_BINARIES_BASE_URL_ANDROID: str = f"{LITERT_BINARIES_BASE_URL}/android_arm64"
LITERT_CLI_DOWNLOAD_BASE_URL: str = "https://storage.googleapis.com/litert/tools/cli"
LITERT_CLI_CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache", "litert-cli")

# NPU and AOT Configuration
# TODO: Keep updating the version in sync with LiteRT latest release, and reference to the version in
# https://github.com/google-ai-edge/LiteRT/blob/main/third_party/qairt/workspace.bzl#L25
QAIRT_SDK_VERSION: str = "2.42.0.251225"
QAIRT_SDK_URL: str = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    f"Qualcomm_AI_Runtime_Community/All/{QAIRT_SDK_VERSION}/v{QAIRT_SDK_VERSION}.zip"
)

QNN_SOC_VERSION_MAP: dict[str, int] = {
    "sm8350": 69,
    "sm8450": 69,
    "sm8550": 73,
    "sm8650": 75,
    "sm8750": 79,
    "sm8850": 81,  # Guess
}

AOT_SUPPORTED_TARGETS: dict[str, tuple[str, str]] = {
    # Qualcomm
    "sm8350": ("qualcomm", "SM8350"),
    "sm8450": ("qualcomm", "SM8450"),
    "sm8550": ("qualcomm", "SM8550"),
    "sm8650": ("qualcomm", "SM8650"),
    "sm8750": ("qualcomm", "SM8750"),
    "sm8850": ("qualcomm", "SM8850"),
    "sa8255": ("qualcomm", "SA8255"),
    "sa8295": ("qualcomm", "SA8295"),
    "qnn_all": ("qualcomm", "ALL"),
    # MediaTek
    "mt6853": ("mediatek", "MT6853"),
    "mt6877": ("mediatek", "MT6877"),
    "mt6878": ("mediatek", "MT6878"),
    "mt6879": ("mediatek", "MT6879"),
    "mt6886": ("mediatek", "MT6886"),
    "mt6893": ("mediatek", "MT6893"),
    "mt6895": ("mediatek", "MT6895"),
    "mt6897": ("mediatek", "MT6897"),
    "mt6983": ("mediatek", "MT6983"),
    "mt6985": ("mediatek", "MT6985"),
    "mt6989": ("mediatek", "MT6989"),
    "mt6991": ("mediatek", "MT6991"),
    "mt6993": ("mediatek", "MT6993"),
    "mt8171": ("mediatek", "MT8171"),
    "mt8188": ("mediatek", "MT8188"),
    "mt8189": ("mediatek", "MT8189"),
    "mtk_all": ("mediatek", "ALL"),
}

