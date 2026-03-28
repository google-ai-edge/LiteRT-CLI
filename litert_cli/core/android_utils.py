"""Android utility functions for LiteRT CLI."""

from __future__ import annotations

import pathlib
import subprocess
import urllib.request

import click

from litert_cli.core import constants


def check_adb() -> None:
  """Checks if adb is available.

  Verifies that exactly one authorized device is connected and responding to
  commands.

  Raises:
    click.ClickException: If adb is missing or device is unauthorized/offline.
  """
  try:
    subprocess.run(
        ["adb", "get-state"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
  except FileNotFoundError as e:
    raise click.ClickException(
        "adb command not found. Please ensure Android platform-tools is in"
        " your PATH."
    ) from e
  except subprocess.CalledProcessError as e:
    raise click.ClickException(
        "No Android device found or it is not authorized. Try running 'adb"
        " devices' to check."
    ) from e


def get_android_abi() -> str:
  """Gets the CPU ABI of the connected Android device via adb.

  Returns:
    The CPU ABI string (e.g., 'arm64-v8a').

  Raises:
    click.ClickException: If querying Android ABI fails or adb is missing.
  """
  try:
    result = subprocess.run(
        ["adb", "shell", "getprop", "ro.product.cpu.abi"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    raise click.ClickException(f"Error querying Android ABI: {e}") from e


def _ensure_downloaded_binary(abi: str, tool_name: str) -> pathlib.Path | None:
  """Downloads the pre-built binary for the given ABI if not cached.

  Args:
    abi: The Android CPU ABI (e.g., 'arm64-v8a').
    tool_name: The binary name to download (e.g., 'run_model').

  Returns:
    The absolute local path to the cached binary, or None on failure.
  """
  if "arm64" not in abi:
    raise click.ClickException(
        f"Architecture '{abi}' is not supported for automatic downloading of"
        f" {tool_name}. Only arm64 is supported for these specific binaries."
    )

  download_url = f"{constants.LITERT_BINARIES_BASE_URL_ANDROID}/{tool_name}"

  # Determine cache directory
  cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR) / "binaries" / abi
  cache_dir.mkdir(parents=True, exist_ok=True)

  cached_binary_path = cache_dir / tool_name
  if cached_binary_path.exists():
    return cached_binary_path

  click.secho(f"Downloading {tool_name} for {abi}...", fg="cyan")
  try:
    tmp_cached_file = cached_binary_path.with_suffix(".tmp")
    with urllib.request.urlopen(download_url) as response:
      total_size = int(response.headers.get("Content-Length", 0))

      with click.progressbar(
          length=total_size, label=f"Downloading {tool_name}"
      ) as bar:
        with open(tmp_cached_file, "wb") as f:
          while True:
            buffer = response.read(8192)
            if not buffer:
              break
            f.write(buffer)
            bar.update(len(buffer))

    tmp_cached_file.rename(cached_binary_path)
    # Ensure it is executable
    cached_binary_path.chmod(0o755)
    return cached_binary_path
  except Exception as e:  # pylint: disable=broad-exception-caught
    if cached_binary_path.exists():
      cached_binary_path.unlink()
    tmp_cached_file = cached_binary_path.with_suffix(".tmp")
    if tmp_cached_file.exists():
      tmp_cached_file.unlink()
    click.secho(f"Failed to download {tool_name}: {e}", fg="yellow")
    return None


def find_android_binary(tool_name: str, abi: str) -> pathlib.Path:
  """Locates or downloads an Android executable binary.

  Always downloads from a fixed URL (cached locally).

  Args:
    tool_name: Binary name (e.g. 'run_model').
    abi: Target Android CPU ABI.

  Returns:
    The absolute path to the binary.

  Raises:
    click.ClickException: If the binary could not be found or downloaded.
  """
  downloaded_bin = _ensure_downloaded_binary(abi, tool_name)
  if downloaded_bin and downloaded_bin.exists():
    return downloaded_bin

  raise click.ClickException(
      f"Could not find or download {tool_name} for ABI '{abi}'."
  )


def _ensure_downloaded_library(
    abi: str,
    lib_name: str,
    base_url: str = constants.LITERT_BINARIES_BASE_URL_ANDROID,
) -> pathlib.Path | None:
  """Downloads the pre-built library for the given ABI if not cached.

  Args:
    abi: The Android CPU ABI (e.g., 'arm64-v8a').
    lib_name: The library name to download (e.g., 'libLiteRt.so').
    base_url: The base URL to download the library from.

  Returns:
    The absolute local path to the cached binary, or None on failure.
  """
  if "arm64" not in abi:
    raise click.ClickException(
        f"Architecture '{abi}' is not supported for automatic downloading of"
        f" {lib_name}. Only arm64 is supported for these specific binaries."
    )

  download_url = f"{base_url}/{lib_name}"

  # Determine cache directory
  cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR) / "binaries" / abi
  cache_dir.mkdir(parents=True, exist_ok=True)

  cached_lib_path = cache_dir / lib_name
  if cached_lib_path.exists():
    return cached_lib_path

  click.secho(f"Downloading {lib_name} for {abi}...", fg="cyan")
  try:
    tmp_cached_file = cached_lib_path.with_suffix(".tmp")
    with urllib.request.urlopen(download_url) as response:
      total_size = int(response.headers.get("Content-Length", 0))

      with click.progressbar(
          length=total_size, label=f"Downloading {lib_name}"
      ) as bar:
        with open(tmp_cached_file, "wb") as f:
          while True:
            buffer = response.read(8192)
            if not buffer:
              break
            f.write(buffer)
            bar.update(len(buffer))

    tmp_cached_file.rename(cached_lib_path)
    return cached_lib_path
  except Exception as e:  # pylint: disable=broad-exception-caught
    if cached_lib_path.exists():
      cached_lib_path.unlink()
    tmp_cached_file = cached_lib_path.with_suffix(".tmp")
    if tmp_cached_file.exists():
      tmp_cached_file.unlink()
    click.secho(f"Failed to download {lib_name}: {e}", fg="yellow")
    return None


def find_android_lib(
    lib_name: str,
    abi: str,
    base_url: str = constants.LITERT_BINARIES_BASE_URL_ANDROID,
) -> pathlib.Path:
  """Locates or downloads an Android executable binary.

  Always downloads from a fixed URL (cached locally).

  Args:
    lib_name: Binary name (e.g. 'run_model').
    abi: Target Android CPU ABI.
    base_url: The base URL to download the library from.

  Returns:
    The absolute path to the binary.

  Raises:
    click.ClickException: If the binary could not be found or downloaded.
  """
  downloaded_bin = _ensure_downloaded_library(abi, lib_name, base_url)
  if downloaded_bin and downloaded_bin.exists():
    return downloaded_bin

  raise click.ClickException(
      f"Could not find or download {lib_name} for ABI '{abi}'."
  )

def find_npu_dispatch_lib(soc_vendor: str, abi: str) -> pathlib.Path:
  """Finds and downloads the NPU dispatch library for the given SoC vendor.
  
  Args:
    soc_vendor: The NPU vendor ("qualcomm" or "mediatek").
    abi: Target Android CPU ABI.
    
  Returns:
    The absolute path to the local downloaded backend dispatch library.
  """
  if soc_vendor == "qualcomm":
    lib_name = "libLiteRtDispatch_Qualcomm.so"
  elif soc_vendor == "mediatek":
    lib_name = "libLiteRtDispatch_MediaTek.so"
  else:
    raise click.ClickException(f"Unsupported NPU vendor for dispatch: {soc_vendor}")
    
  return find_android_lib(
      lib_name, abi, base_url=constants.LITERT_CLI_DOWNLOAD_BASE_URL
  )
