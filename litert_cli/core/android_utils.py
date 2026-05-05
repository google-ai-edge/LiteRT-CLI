"""Android utility functions for LiteRT CLI."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import subprocess

import click
import requests

from litert_cli.core import constants


class DownloadError(Exception):
  """A file download failed."""

  pass


def _construct_adb_command(command: Sequence[str], device_id: str | None) -> list[str]:
  """Constructs an adb command list, including the device ID if provided."""
  if device_id:
    return ["adb", "-s", device_id, *command]
  return ["adb", *command]


def check_adb(device_id: str | None = None) -> None:
  """Checks if adb device is available.

  Verifies that exactly one authorized device is connected and responding to
  commands.

  Args:
    device_id: Optional. The serial number of the target device.

  Raises:
    click.ClickException: If adb is missing or device is unauthorized/offline.
  """
  try:
    subprocess.run(
        _construct_adb_command(["get-state"], device_id),
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
    device_info = f" for device {device_id}" if device_id else ""
    raise click.ClickException(
        f"No Android device found{device_info} or it is not authorized. Try"
        " running 'adb devices' to check."
    ) from e


def get_android_abi(device_id: str | None = None) -> str:
  """Gets the CPU ABI of the connected Android device via adb.

  Args:
    device_id: Optional. The serial number of the target device.

  Returns:
    The CPU ABI string (e.g., 'arm64-v8a').

  Raises:
    click.ClickException: If querying Android ABI fails or adb is missing.
  """
  try:
    return subprocess.run(
        _construct_adb_command(
            ["shell", "getprop", "ro.product.cpu.abi"], device_id
        ),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
  except subprocess.CalledProcessError as e:
    device_info = f" for device {device_id}" if device_id else ""
    raise click.ClickException(f"Error querying Android ABI{device_info}") from e


def _ensure_downloaded_file(
    abi: str,
    file_name: str,
    base_url: str,
    *,
    make_executable: bool = False,
) -> pathlib.Path:
  """Downloads a file for the given ABI if not cached.

  Args:
    abi: The Android CPU ABI (e.g., 'arm64-v8a').
    file_name: The file name to download.
    base_url: The base URL to download from.
    make_executable: Whether to make the file executable.

  Returns:
    The absolute local path to the cached file.

  Raises:
    click.ClickException: If the architecture is not supported.
    DownloadError: If the download or file operations fail.
  """
  if "arm64" not in abi:
    raise click.ClickException(
        f"Architecture {abi!r} is not supported for automatic downloading of"
        f" {file_name!r}. Only arm64 is supported for these specific binaries."
    )

  download_url = f"{base_url}/{file_name}"
  cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR) / "binaries" / abi
  cache_dir.mkdir(parents=True, exist_ok=True)

  cached_file_path = cache_dir / file_name
  if cached_file_path.exists():
    return cached_file_path

  click.secho(f"Downloading {file_name!r} for {abi!r}...", fg="cyan")
  tmp_cached_file = cached_file_path.with_suffix(".tmp")
  try:
    with requests.get(download_url, stream=True, timeout=10) as response:
      response.raise_for_status()
      content_length = response.headers.get("Content-Length")
      total_size = int(content_length) if content_length else 0

      bar_length = total_size if total_size > 0 else None
      bar_label = f"Downloading {file_name}"
      if bar_length is None:
        click.secho(
            f"Content-Length header not found for {file_name!r}, using"
            " indeterminate progress bar.",
            fg="yellow",
        )
        bar_label += " (size unknown)"

      with click.progressbar(
          length=bar_length, label=bar_label
      ) as bar:
        with open(tmp_cached_file, "wb") as f:
          for buffer in response.iter_content(chunk_size=8192):
            f.write(buffer)
            bar.update(len(buffer))

    tmp_cached_file.rename(cached_file_path)
    if make_executable:
      cached_file_path.chmod(0o755)
    return cached_file_path
  except (requests.exceptions.RequestException, OSError) as e:
    if cached_file_path.exists():
      cached_file_path.unlink()
    if tmp_cached_file.exists():
      tmp_cached_file.unlink()
    raise DownloadError(
        f"Failed to download {file_name!r} from {download_url!r}"
    ) from e


def _ensure_downloaded_binary(abi: str, tool_name: str) -> pathlib.Path:
  """Downloads the pre-built binary for the given ABI if not cached."""
  return _ensure_downloaded_file(
      abi,
      tool_name,
      constants.LITERT_BINARIES_BASE_URL_ANDROID,
      make_executable=True,
  )


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
  try:
    downloaded_bin = _ensure_downloaded_binary(abi, tool_name)
    if downloaded_bin.exists():
      return downloaded_bin
    else:
      raise click.ClickException(
          f"Downloaded binary {tool_name!r} for ABI {abi!r} does not exist."
      )
  except DownloadError as e:
    raise click.ClickException(
        f"Could not find or download {tool_name!r} for ABI {abi!r}: {e}"
    ) from e


def _ensure_downloaded_library(
    abi: str,
    lib_name: str,
    base_url: str = constants.LITERT_BINARIES_BASE_URL_ANDROID,
) -> pathlib.Path:
  """Downloads the pre-built library for the given ABI if not cached."""
  return _ensure_downloaded_file(abi, lib_name, base_url, make_executable=False)


def find_android_lib(
    lib_name: str,
    abi: str,
    base_url: str = constants.LITERT_BINARIES_BASE_URL_ANDROID,
) -> pathlib.Path:
  """Locates or downloads an Android library.

  Always downloads from a fixed URL (cached locally).

  Args:
    lib_name: library name (e.g. 'libLiteRt.so').
    abi: Target Android CPU ABI.
    base_url: The base URL to download the library from.

  Returns:
    The absolute path to the library.

  Raises:
    click.ClickException: If the library could not be found or downloaded.
  """
  try:
    downloaded_bin = _ensure_downloaded_library(abi, lib_name, base_url)
    if downloaded_bin.exists():
      return downloaded_bin
    else:
      raise click.ClickException(
          f"Downloaded library {lib_name!r} for ABI {abi!r} does not exist."
      )
  except DownloadError as e:
    raise click.ClickException(
        f"Could not find or download {lib_name!r} for ABI {abi!r}: {e}"
    ) from e


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
    raise click.ClickException(
        f"Unsupported NPU vendor for dispatch: {soc_vendor!r}"
    )

  return find_android_lib(
      lib_name, abi, base_url=constants.LITERT_BINARIES_BASE_URL_ANDROID
  )


def find_npu_compiler_plugin_lib(soc_vendor: str, abi: str) -> pathlib.Path:
  """Finds and downloads the NPU compiler plugin library for the given SoC vendor.

  Args:
    soc_vendor: The NPU vendor ("qualcomm" or "mediatek").
    abi: Target Android CPU ABI.

  Returns:
    The absolute path to the local downloaded backend compiler plugin library.
  """
  if soc_vendor == "qualcomm":
    lib_name = "libLiteRtCompilerPlugin_Qualcomm.so"
  elif soc_vendor == "mediatek":
    lib_name = "libLiteRtCompilerPlugin_MediaTek.so"
  else:
    raise click.ClickException(
        f"Unsupported NPU vendor for compiler plugin: {soc_vendor!r}"
    )

  return find_android_lib(
      lib_name, abi, base_url=constants.LITERT_BINARIES_BASE_URL_ANDROID
  )
