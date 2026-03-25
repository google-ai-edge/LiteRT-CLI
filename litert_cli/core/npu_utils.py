"""NPU and AOT utilities for downloading, pushing, and compiling."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import zipfile

import click
import requests

from ai_edge_litert.aot.vendors.mediatek import target as mtk_target
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from litert_cli.core import constants

_DEFAULT_LITERT_NPU_RUNTIME_LIBRARIES_URL = (
    "https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.0rc1/"
    "litert_npu_runtime_libraries.zip"
)

_DEFAULT_QAIRT_SDK_URL = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    "Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip"
)

_QNN_SOC_VERSION_MAP = {
    "sm8350": 69,
    "sm8450": 69,
    "sm8550": 73,
    "sm8650": 75,
    "sm8750": 79,
    "sm8850": 81, # Guess
}

# AOT Supported Targets
_SUPPORTED_TARGETS = {
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


def _get_runtime_libraries_dir() -> pathlib.Path:
  """Returns the path to the litert_npu_runtime_libraries directory."""
  return pathlib.Path(constants.LITERT_CLI_ROOT) / "litert_npu_runtime_libraries"


def ensure_runtime_libraries() -> pathlib.Path:
  """Ensures the NPU runtime libraries are available.

  Returns:
    Path to the litert_npu_runtime_libraries directory.
  """
  runtime_dir = _get_runtime_libraries_dir()
  
  # Check if a known .so file exists to verify unpacking
  test_so = (
      runtime_dir
      / "qualcomm_runtime_v73/src/main/jni/arm64-v8a/libLiteRtDispatch_Qualcomm.so"
  )
  
  # A known QNN .so to verify fetching
  test_qnn_so = (
      runtime_dir
      / "qualcomm_runtime_v73/src/main/jni/arm64-v8a/libQnnHtp.so"
  )

  if runtime_dir.exists() and test_so.exists() and test_qnn_so.exists():
    click.echo(f"Found existing runtime libraries at {runtime_dir}")
    return runtime_dir

  # If directory doesn't exist, download and unzip
  if not runtime_dir.exists():
    click.echo(f"Downloading NPU runtime libraries from {_DEFAULT_LITERT_NPU_RUNTIME_LIBRARIES_URL}...")
    zip_path_parent = pathlib.Path(constants.LITERT_CLI_ROOT)
    zip_path_parent.mkdir(parents=True, exist_ok=True)
    zip_path = zip_path_parent / "litert_npu_runtime_libraries.zip"
    
    try:
      response = requests.get(_DEFAULT_LITERT_NPU_RUNTIME_LIBRARIES_URL, stream=True)
      response.raise_for_status()
      with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)
      
      click.echo(f"Extracting to {runtime_dir}...")
      runtime_dir.mkdir(parents=True, exist_ok=True)
      with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(runtime_dir)
        
      os.remove(zip_path)
    except Exception as e:
      if zip_path.exists():
          os.remove(zip_path)
      raise click.ClickException(f"Failed to setup NPU runtime libraries: {e}")

  # Fetch QNN libraries if missing
  if not test_qnn_so.exists():
      _fetch_qualcomm_libraries_python(runtime_dir)

  if not test_qnn_so.exists():
      raise click.ClickException(f"Failed to find expected .so files after setup in {runtime_dir}")

  return runtime_dir


def _fetch_qualcomm_libraries_python(runtime_dir: pathlib.Path):
    """Downloads and extracts QNN libraries using curl and zipfile."""
    tmp_zip = runtime_dir / "qairt_sdk.zip"
    
    click.echo(f"Downloading QAIRT SDK from {_DEFAULT_QAIRT_SDK_URL} using curl...")
    try:
        subprocess.run(["curl", "-L", _DEFAULT_QAIRT_SDK_URL, "-o", str(tmp_zip)], check=True)
    except subprocess.CalledProcessError as e:
         raise click.ClickException(f"Failed to download QAIRT SDK: {e}")
         
    click.echo("Extracting QAIRT SDK...")
    tmp_extract = runtime_dir / "tmp_qairt"
    tmp_extract.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.filename.endswith(".so"):
                    zip_ref.extract(member, tmp_extract)
    except Exception as e:
         raise click.ClickException(f"Failed to extract QAIRT SDK: {e}")
    finally:
        if tmp_zip.exists():
             os.remove(tmp_zip)
             
    # Now copy to correct locations
    source_dir = tmp_extract / "qairt/2.40.0.251030"
    if not source_dir.exists():
         raise click.ClickException("Failed to find QAIRT content in zip.")
         
    versions = [69, 73, 75, 79, 81]
    jni_dir = "src/main/jni/arm64-v8a"
    
    for version in versions:
        dest_jni = runtime_dir / f"qualcomm_runtime_v{version}" / jni_dir
        dest_jni.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Copying libraries to {dest_jni}...")
        
        # Copy shared libs
        try:
            shutil.copy(source_dir / "lib/aarch64-android/libQnnHtp.so", dest_jni)
            shutil.copy(source_dir / "lib/aarch64-android/libQnnSystem.so", dest_jni)
            shutil.copy(source_dir / f"lib/aarch64-android/libQnnHtpV{version}Stub.so", dest_jni)
            
            # libQnnIr.so might be needed
            ir_src = source_dir / "lib/aarch64-android/libQnnIr.so"
            if ir_src.exists():
                shutil.copy(ir_src, dest_jni)
            
            # Hexagon skel
            skel_src = source_dir / f"lib/hexagon-v{version}/unsigned/libQnnHtpV{version}Skel.so"
            if skel_src.exists():
                shutil.copy(skel_src, dest_jni)
        except FileNotFoundError as e:
             click.echo(f"Warning: Failed to copy some files for v{version}: {e}")
            
    shutil.rmtree(tmp_extract)


def get_target_model(device_id: str | None = None) -> str:
    """Gets the target model for the connected device."""
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", device_id])
    cmd.extend(["shell", "getprop", "ro.soc.model"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        soc_model = result.stdout.strip().lower()
        if not soc_model:
             return "unknown"
             
        for k in _SUPPORTED_TARGETS.keys():
            if k != "qnn_all" and k != "mtk_all" and k in soc_model:
                return k
                
        return soc_model
    except subprocess.CalledProcessError:
        return "unknown"


def push_runtime_libraries(device_id: str | None, android_root: str = constants.LITERT_CLI_ANDROID_ROOT) -> str:
    """Pushes runtime libraries to the device and returns the remote path."""
    runtime_dir = ensure_runtime_libraries()
    target_model = get_target_model(device_id)
    click.echo(f"Identified targeting device SoC Model: {target_model}")
    
    soc_vendor = "qualcomm"
    if "mt" in target_model.lower():
        soc_vendor = "mediatek"
        
    if soc_vendor == "qualcomm":
        target_model_key = target_model.lower()
        best_version = None
        for k, v in _QNN_SOC_VERSION_MAP.items():
            if k in target_model_key:
                best_version = v
                break
                
        if not best_version:
             versions = [69, 73, 75, 79, 81]
             for v in reversed(versions):
                  v_dir = runtime_dir / f"qualcomm_runtime_v{v}"
                  if v_dir.exists():
                      test_so = v_dir / "src/main/jni/arm64-v8a/libLiteRtDispatch_Qualcomm.so"
                      if test_so.exists():
                          best_version = v
                          break
                          
        if not best_version:
             raise click.ClickException("No valid Qualcomm runtime version found.")
             
        click.echo(f"Selected Qualcomm runtime version: v{best_version}")
        src_jni_dir = runtime_dir / f"qualcomm_runtime_v{best_version}/src/main/jni/arm64-v8a"
    elif soc_vendor == "mediatek":
         src_jni_dir = runtime_dir / "mediatek_runtime/src/main/jni/arm64-v8a"
    else:
         raise click.ClickException(f"Unsupported NPU vendor for device: {target_model}")
         
    if not src_jni_dir.exists():
         raise click.ClickException(f"Source JNI directory {src_jni_dir} does not exist.")
         
    remote_dispatch_dir = android_root
    
    click.echo(f"Pushing runtime libraries from {src_jni_dir} to {remote_dispatch_dir}...")
    
    subprocess.run(["adb", "shell", f"mkdir -p {remote_dispatch_dir}"], check=True)
    
    for local_file in src_jni_dir.iterdir():
        if local_file.is_file():
             subprocess.run([
                 "adb", "-s", device_id if device_id else "",
                 "push", str(local_file), f"{remote_dispatch_dir}/{local_file.name}"
             ] if device_id else [
                 "adb", "push", str(local_file), f"{remote_dispatch_dir}/{local_file.name}"
             ], check=True, stdout=subprocess.DEVNULL)
             
    return remote_dispatch_dir


def get_target(target_name: str):
  """Returns the mapped compilation Target object for a given string codename."""
  target_name = target_name.lower().strip()
  
  if target_name not in _SUPPORTED_TARGETS:
      raise click.ClickException(f"Unsupported AOT target: {target_name}")
      
  vendor, model_str = _SUPPORTED_TARGETS[target_name]
  
  try:
    if vendor == "qualcomm":
      if not qnn_target:
        raise ImportError("Qualcomm AOT SDK not installed.")
      model_enum = getattr(qnn_target.SocModel, model_str)
      return qnn_target.Target(model_enum)
    elif vendor == "mediatek":
      if not mtk_target:
        raise ImportError("MediaTek AOT SDK not installed.")
      model_enum = getattr(mtk_target.SocModel, model_str)
      return mtk_target.Target(model_enum)
    else:
      raise click.ClickException(f"Unsupported AOT vendor: {vendor}")
  except ImportError as e:
    raise click.ClickException(
        f"Failed to load AOT target {target_name}: {e}\n"
        "Ensure required ai-edge-litert AOT SDKs are installed."
    )
  except AttributeError as e:
    raise click.ClickException(f"Target model enum not found in SDK: {e}")
