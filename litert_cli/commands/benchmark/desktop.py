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

"""Desktop Benchmark Module."""

from __future__ import annotations

import pathlib
import platform
import subprocess
import sys

import click
from litert_cli.core import constants
import requests


def _ensure_desktop_binary(tool_name: str) -> pathlib.Path:
  """Downloads the pre-built binary for the current desktop platform if not cached."""
  system = sys.platform.lower()
  machine = platform.machine().lower()

  if system.startswith("linux"):
    if machine in ("arm64", "aarch64"):
      plat = "linux_arm64"
    elif machine in ("x86_64", "amd64"):
      plat = "linux_x86_64"
    else:
      raise click.ClickException(f"Unsupported Linux architecture: {machine}")
  elif system == "darwin":
    if machine in ("arm64", "aarch64"):
      plat = "macos_arm64"
    else:
      raise click.ClickException(
          f"Unsupported macOS architecture: {machine}. Only macos_arm64 is"
          " currently supported."
      )
  elif system == "win32":
    raise click.ClickException("Windows benchmarking is not yet supported.")
  else:
    raise click.ClickException(
        f"Unsupported desktop operating system: {system}"
    )

  base_url = f"{constants.LITERT_BINARIES_BASE_URL}/{plat}"
  download_url = f"{base_url}/{tool_name}"

  cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR) / "binaries" / plat
  cache_dir.mkdir(parents=True, exist_ok=True)

  cached_file_path = cache_dir / tool_name
  if cached_file_path.exists():
    if system == "darwin":
      subprocess.run(
          ["xattr", "-c", str(cached_file_path)],
          check=False,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      subprocess.run(
          ["codesign", "-s", "-", str(cached_file_path)],
          check=False,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
    return cached_file_path

  click.secho(
      f"Downloading {tool_name!r} for desktop platform {plat!r}...", fg="cyan"
  )
  tmp_cached_file = cached_file_path.with_suffix(".tmp")
  try:
    with requests.get(download_url, stream=True, timeout=10) as response:
      response.raise_for_status()
      content_length = response.headers.get("Content-Length")
      total_size = int(content_length) if content_length else 0

      bar_length = total_size if total_size > 0 else None
      bar_label = f"Downloading {tool_name}"
      if bar_length is None:
        click.secho(
            f"Content-Length header not found for {tool_name!r}, using"
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
    cached_file_path.chmod(0o755)
    if system == "darwin":
      subprocess.run(
          ["xattr", "-c", str(cached_file_path)],
          check=False,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      subprocess.run(
          ["codesign", "-s", "-", str(cached_file_path)],
          check=False,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
    return cached_file_path
  except (requests.exceptions.RequestException, OSError) as e:
    if cached_file_path.exists():
      cached_file_path.unlink()
    if tmp_cached_file.exists():
      tmp_cached_file.unlink()
    raise click.ClickException(
        f"Failed to download {tool_name!r} from {download_url!r}: {e}"
    ) from e


def run_desktop(*, model_path: pathlib.Path, accelerator: str) -> None:
  """Runs the benchmark_model binary on the local desktop machine.

  Args:
    model_path: Path to the local LiteRT model file.
    accelerator: Hardware accelerator to use (cpu, gpu, npu).

  Raises:
    click.ClickException: If execution fails.
  """
  click.echo("Preparing to run benchmark on local desktop...")

  if not model_path.exists():
    raise click.ClickException(f"Local model file not found: {model_path}")

  benchmark_bin = _ensure_desktop_binary("benchmark_model")

  click.echo(f"Executing benchmark locally using {benchmark_bin.name}...\n")
  try:
    bench_args = [
        str(benchmark_bin),
        f"--graph={model_path.resolve()}",
    ]
    if accelerator == "gpu":
      bench_args.append("--use_gpu=true")
    elif accelerator == "npu":
      click.secho(
          "Warning: NPU benchmarking via benchmark_model on desktop is not"
          " fully supported yet.",
          fg="yellow",
      )
      bench_args.append("--use_npu=true")

    process = subprocess.Popen(
        bench_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    from litert_cli.core.log_filters import BenchmarkLogFilter

    output_lines = []
    log_filter = BenchmarkLogFilter(constants.DEFAULT_QUIET)

    for line in process.stdout:
      output_lines.append(line)
      if log_filter.should_show(line):
        click.echo(line, nl=False)

    process.wait()
    if process.returncode != 0:
      click.secho(
          f"Execution failed on desktop with exit code {process.returncode}",
          fg="red",
      )
      click.echo("Full output for debugging:")
      for line in output_lines:
        click.echo(line, nl=False)
      raise click.ClickException("Benchmark failed on desktop.")
  except click.ClickException:
    raise
  except Exception as e:
    raise click.ClickException(f"Failed to execute benchmark on desktop: {e}")
