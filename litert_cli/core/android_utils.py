"""Android utility functions for LiteRT CLI."""

from __future__ import annotations

import pathlib
import subprocess
import urllib.request

import click


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

  download_url = f"https://storage.googleapis.com/litert/tools/tmp/{tool_name}_android_arm64"

  # Determine cache directory
  cache_dir = pathlib.Path.home() / ".cache" / "litert-cli" / "binaries" / abi
  cache_dir.mkdir(parents=True, exist_ok=True)

  cached_binary_path = cache_dir / tool_name
  if cached_binary_path.exists():
    return cached_binary_path

  click.secho(f"Downloading {tool_name} for {abi}...", fg="cyan")
  try:
    with urllib.request.urlopen(download_url) as response:
      total_size = int(response.headers.get("Content-Length", 0))

      with click.progressbar(
          length=total_size, label=f"Downloading {tool_name}"
      ) as bar:
        with open(cached_binary_path, "wb") as f:
          while True:
            buffer = response.read(8192)
            if not buffer:
              break
            f.write(buffer)
            bar.update(len(buffer))

    # Ensure it is executable
    cached_binary_path.chmod(0o755)
    return cached_binary_path
  except Exception as e:  # pylint: disable=broad-exception-caught
    if cached_binary_path.exists():
      cached_binary_path.unlink()
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
