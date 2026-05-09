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

"""NPU and AOT utilities for downloading, pushing, and compiling."""

from __future__ import annotations

import pathlib
import subprocess
import zipfile

import click
from litert_cli.core import constants
import requests

from ai_edge_litert.aot.vendors.intel_openvino import target as intel_target
from ai_edge_litert.aot.vendors.mediatek import target as mtk_target
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


def ensure_npu_runtime_libraries(soc_vendor: str = "qualcomm") -> pathlib.Path:
  """Ensures the NPU runtime libraries are downloaded and available locally.

  This checks for a cached version of the runtime SDK first. If it's missing, it
  downloads it from the remote URL specified in constants and extracts it.

  Args:
    soc_vendor: The target SoC vendor ("qualcomm" or "mediatek").

  Returns:
    Path to the litert_npu_runtime_libraries directory (or qairt_sdk directory).

  Raises:
    click.ClickException: If downloading or extracting the libraries fails.
  """
  runtime_dir = pathlib.Path(constants.LITERT_CLI_ROOT)

  if soc_vendor == "mediatek":
    marker_file = runtime_dir / ".mediatek_sdk_extracted.complete"

    if runtime_dir.exists() and marker_file.exists():
      click.echo(f"Found existing MediaTek runtime libraries at {runtime_dir}")
      return runtime_dir

    click.echo(
        f"Downloading MediaTek runtime libraries from {constants.MEDIATEK_SDK_URL}..."
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    gz_path = runtime_dir / "mediatek_sdk.tar.gz"
    tmp_gz_path = gz_path.with_suffix(".tar.gz.tmp")

    try:
      with requests.get(constants.MEDIATEK_SDK_URL, stream=True) as response:
        response.raise_for_status()
        with open(tmp_gz_path, "wb") as f:
          for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        tmp_gz_path.rename(gz_path)

      click.echo(f"Extracting to {runtime_dir}...")
      import tarfile

      with tarfile.open(gz_path, "r:gz") as tar_ref:
        tar_ref.extractall(runtime_dir)

      marker_file.touch()
      gz_path.unlink(missing_ok=True)
    except Exception as e:
      gz_path.unlink(missing_ok=True)
      if tmp_gz_path.exists():
        tmp_gz_path.unlink()
      if marker_file.exists():
        marker_file.unlink()
      raise click.ClickException(
          f"Failed to setup MediaTek runtime libraries: {e}"
      ) from e

    return runtime_dir

  # Check for a marker file to verify if the zip was successfully unpacked.
  marker_file = runtime_dir / ".qairt_sdk_extracted.complete"

  if runtime_dir.exists() and marker_file.exists():
    click.echo(f"Found existing runtime libraries at {runtime_dir}")
    return runtime_dir

  # If directory doesn't exist or marker file is missing, download and unzip
  # the SDK.
  click.echo(
      f"Downloading NPU runtime libraries from {constants.QAIRT_SDK_URL}..."
  )
  runtime_dir.mkdir(parents=True, exist_ok=True)
  zip_path = runtime_dir / "qairt_sdk.zip"

  tmp_zip_path = zip_path.with_suffix(".zip.tmp")

  try:
    with requests.get(constants.QAIRT_SDK_URL, stream=True) as response:
      response.raise_for_status()

      with open(tmp_zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

      tmp_zip_path.rename(zip_path)

    click.echo(f"Extracting to {runtime_dir}...")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
      # Validate against Zip Slip vulnerability.
      for member in zip_ref.infolist():
        # pathlib handles different OS path separators.
        resolved_path = (runtime_dir / member.filename).resolve()
        if not resolved_path.is_relative_to(runtime_dir.resolve()):
          raise click.ClickException(
              f"Unsafe path detected in zip file: {member.filename}"
          )
      zip_ref.extractall(runtime_dir)

    # Create a marker file to indicate successful extraction.
    marker_file.touch()
    zip_path.unlink(missing_ok=True)
  except Exception as e:  # pylint: disable=broad-exception-caught
    zip_path.unlink(missing_ok=True)
    if tmp_zip_path.exists():
      tmp_zip_path.unlink()
    if marker_file.exists():
      marker_file.unlink()
    raise click.ClickException("Failed to setup NPU runtime libraries") from e

  return runtime_dir


def get_soc_target_model(device_id: str | None = None) -> str:
  """Gets the exact SoC target model mapped name from the connected Android device.

  Queries the device via adb for its ro.soc.model property and matches it
  against the AOT_SUPPORTED_TARGETS in constants.

  Args:
    device_id: Optional adb device serial number.

  Returns:
    The matched codename map (e.g. 'sm8550') or 'unknown' if not
    found/supported.
  """
  cmd = (
      ["adb"]
      + (["-s", device_id] if device_id else [])
      + [
          "shell",
          "getprop",
          "ro.soc.model",
      ]
  )

  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    soc_model = result.stdout.strip().lower()
    if not soc_model:
      # Fallback to ro.board.platform
      cmd_fallback = cmd[:-1] + ["ro.board.platform"]
      result_fallback = subprocess.run(
          cmd_fallback, capture_output=True, text=True, check=True
      )
      soc_model = result_fallback.stdout.strip().lower()

    if not soc_model:
      return "unknown"

    for k in constants.AOT_SUPPORTED_TARGETS:
      if k not in ("qnn_all", "mtk_all") and k in soc_model:
        return k

    return soc_model
  except subprocess.CalledProcessError:
    return "unknown"


def push_npu_runtime_libraries(
    device_id: str | None,
    android_root: str = constants.LITERT_CLI_ANDROID_ROOT,
) -> str:
  """Pushes required NPU runtime libraries (QNN/MTK) to the connected Android device.

  Identifies the SoC via `get_soc_target_model`, downloads the necessary SDKs if
  missing, and pushes the correct specific dynamic libraries to the device
  based on the chip version.

  Args:
    device_id: Optional adb device serial number.
    android_root: The remote path on the Android device to push the libraries
      to.

  Returns:
    The remote directory path string where the libraries were pushed.
  """
  target_model = get_soc_target_model(device_id)
  click.echo(f"Identified targeting device SoC Model: {target_model}")

  # Determine the SoC vendor to figure out which libraries to push.
  soc_vendor = "mediatek" if "mt" in target_model else "qualcomm"
  runtime_dir = ensure_npu_runtime_libraries(soc_vendor)

  libs_to_push = []
  if soc_vendor == "qualcomm":
    # Resolve the specific Hexagon DSP version (e.g., 69, 73, 75) mapped to
    # the Qualcomm chip.
    best_version = constants.QNN_SOC_VERSION_MAP.get(target_model)
    if not best_version:
      raise click.ClickException(
          f"No valid Qualcomm runtime version found for SoC: {target_model}"
      )

    click.echo(f"Selected Qualcomm runtime version: v{best_version}")

    # Paths to the unpacked QAIRT SDK on the host.
    src_dir = (
        runtime_dir / f"qairt/{constants.QAIRT_SDK_VERSION}/lib/aarch64-android"
    )
    skel_dir = (
        runtime_dir
        / f"qairt/{constants.QAIRT_SDK_VERSION}/lib/hexagon-v{best_version}/unsigned"
    )

    # Essential QNN backend libraries and stub files needed for execution on
    # the device.
    libs_to_push = [
        src_dir / "libQnnSystem.so",
        src_dir / "libQnnHtp.so",
        src_dir / f"libQnnHtpV{best_version}Stub.so",
        src_dir / "libQnnHtpPrepare.so",
    ]

    skel_file = skel_dir / f"libQnnHtpV{best_version}Skel.so"
    if skel_file.exists():
      libs_to_push.append(skel_file)
    else:
      click.echo(f"Warning: Skeleton file not found: {skel_file}")

  elif soc_vendor == "mediatek":
    from litert_cli.core.targets_manager import TargetsManager

    manager = TargetsManager()
    targets = manager.load_targets()
    target_info = targets.get(target_model)

    if not target_info:
      raise click.ClickException(
          f"No target info found for MediaTek SoC: {target_model}"
      )

    recommend_version = target_info.properties.get("recommend_version", "")
    click.echo(f"Selected MediaTek runtime version: {recommend_version}")

    source_dir = runtime_dir / "neuro_pilot"

    if "v8" in recommend_version:
      lib_path = (
          source_dir
          / f"{constants.MEDIATEK_V8_VERSION}/usdk/lib64/libneuronusdk_adapter.mtk.so"
      )
      if not lib_path.exists():
        raise click.ClickException(f"File not found: {lib_path}")
      libs_to_push = [lib_path]
    elif "v9" in recommend_version:
      lib_path = (
          source_dir
          / f"{constants.MEDIATEK_V9_VERSION}/usdk/lib64/libneuronusdk_adapter.so"
      )
      if not lib_path.exists():
        raise click.ClickException(f"File not found: {lib_path}")
      libs_to_push = [lib_path]
    else:
      raise click.ClickException(
          f"Unsupported MediaTek version '{recommend_version}' for SoC: {target_model}"
      )

  else:
    raise click.ClickException(
        f"Unsupported NPU vendor for device: {target_model}"
    )

  click.echo(f"Pushing runtime libraries to {android_root}...")
  adb_cmd = ["adb", "-s", device_id] if device_id else ["adb"]

  subprocess.run(
      adb_cmd + ["shell", "mkdir", "-p", f"'{android_root}'"], check=True
  )

  for local_file in libs_to_push:
    remote_file_path = f"{android_root}/{local_file.name}"
    # Check if the file already exists on the device using adb shell test.
    check_cmd = adb_cmd + ["shell", "test", "-f", remote_file_path]
    if subprocess.run(check_cmd, check=False).returncode == 0:
      click.echo(f"  Skipping {local_file.name} (already on device)")
      continue

    subprocess.run(
        adb_cmd + ["push", str(local_file), remote_file_path],
        check=True,
        stdout=subprocess.DEVNULL,
    )

  return android_root


def get_aot_target(target_name: str) -> qnn_target.Target | mtk_target.Target | intel_target.Target:
  """Returns the mapped compilation Target object for a given string codename.

  Translates a CLI string input (e.g., 'sm8550') into the underlying Python AOT
  Target object, dynamically importing the vendor-specific SDK modules.

  Args:
    target_name: The codename string of the target (e.g., 'sm8550').

  Returns:
    The LiteRT AOT Target object for AOT compilation.
  """
  target_name = target_name.lower().strip()

  if target_name not in constants.AOT_SUPPORTED_TARGETS:
    raise click.ClickException(f"Unsupported AOT target: {target_name}")

  vendor, model_str = constants.AOT_SUPPORTED_TARGETS[target_name]

  try:
    if vendor == "qualcomm":
      vendor_module = qnn_target
    elif vendor == "mediatek":
      vendor_module = mtk_target
    elif vendor == "intel":
      vendor_module = intel_target
    else:
      raise click.ClickException(f"Unsupported vendor: {vendor}")

    model_enum = getattr(vendor_module.SocModel, model_str)
    return vendor_module.Target(model_enum)
  except AttributeError as e:
    raise click.ClickException("Target model enum not found in SDK") from e
