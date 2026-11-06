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
import tempfile

import click
from litert_cli.commands.benchmark import model_caches
from litert_cli.core import android_utils
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

  cache_dir = android_utils.binaries_cache_dir(plat)
  cache_dir.mkdir(parents=True, exist_ok=True)

  cached_file_path = cache_dir / tool_name
  if android_utils.is_cached_binary_fresh(cached_file_path):
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

      with click.progressbar(length=bar_length, label=bar_label) as bar:
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
    if tmp_cached_file.exists():
      tmp_cached_file.unlink()
    if cached_file_path.exists():
      # A stale nightly binary: keep using it rather than failing offline.
      click.secho(
          f"Warning: could not refresh {tool_name!r} ({e}); using the cached"
          " copy.",
          fg="yellow",
      )
      return cached_file_path
    raise click.ClickException(
        f"Failed to download {tool_name!r} from {download_url!r}: {e}"
    ) from e


def _run_benchmark(
    bench_args: list[str], *, show: bool, or_fail: bool = False
) -> tuple[int, list[str]]:
  """Runs benchmark_model and returns its exit code and output lines.

  With `show`, prints the lines the benchmark log filter keeps. With
  `or_fail`, a non-zero exit prints the whole output and raises
  click.ClickException.
  """
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
    if show and log_filter.should_show(line):
      click.echo(line, nl=False)

  process.wait()
  if process.returncode != 0 and or_fail:
    click.secho(
        f"Execution failed on desktop with exit code {process.returncode}",
        fg="red",
    )
    click.echo("Full output for debugging:")
    for line in output_lines:
      click.echo(line, nl=False)
    raise click.ClickException("Benchmark failed on desktop.")
  return process.returncode, output_lines


def _show_benchmark_lines(output_lines: list[str]) -> None:
  """Prints the lines the benchmark log filter keeps."""
  from litert_cli.core.log_filters import BenchmarkLogFilter

  log_filter = BenchmarkLogFilter(constants.DEFAULT_QUIET)
  for line in output_lines:
    if log_filter.should_show(line):
      click.echo(line, nl=False)


def _print_failure(returncode: int, output_lines: list[str]) -> None:
  """Prints the tail of a failed process that the run goes on without."""
  click.secho(
      f"Execution failed on desktop with exit code {returncode}", fg="red"
  )
  tail = output_lines[-model_caches.OUTPUT_TAIL_LINES :]
  hint = (
      ""
      if not constants.DEFAULT_QUIET
      else f" ({constants.ENV_LITERT_VERBOSE}=1 shows all of it)"
  )
  click.echo(f"Last {len(tail)} lines of its output{hint}:")
  for line in tail:
    click.echo(line, nl=False)


def _cache_files(cache_dir: str) -> dict[str, tuple[int, int]]:
  """Size and modification time (ns) of each file in the cache directory.

  A file that disappears between the listing and its stat is left out.
  """
  files = {}
  for path in pathlib.Path(cache_dir).iterdir():
    try:
      stat = path.stat()
    except OSError:
      continue
    if path.is_file():
      files[path.name] = (stat.st_size, stat.st_mtime_ns)
  return files


def _run_with_caches(
    bench_args: list[str], model_name: str, accelerator: str
) -> model_caches.Outcome:
  """Runs the first and the measured process in a fresh cache directory.

  Returns what they did. `fallback` is set when a process exited non-zero
  or the directory could not be made; the caller then runs benchmark_model
  once without the cache flags.
  """
  outcome = model_caches.Outcome()
  try:
    cache_dir = tempfile.TemporaryDirectory(prefix="litert-benchmark-cache-")
  except OSError as e:
    outcome.fallback = f"could not make a cache directory ({e})"
    return outcome
  # A fresh directory per run: the first process always compiles the model
  # and writes the caches, the measured process reads them.
  with cache_dir as d:
    cache_flags = model_caches.cache_args(accelerator, d, model_name)
    click.echo("Writing the model caches (benchmark_model, no inference)...")
    returncode, outcome.first = _run_benchmark(
        bench_args + cache_flags + list(model_caches.WARMUP_ARGS),
        show=not constants.DEFAULT_QUIET,
    )
    if returncode != 0:
      _print_failure(returncode, outcome.first)
      outcome.fallback = f"the first process exited with {returncode}"
      return outcome
    written = _cache_files(d)
    # Shown once it exited 0, so the numbers of a process that fails after
    # printing them are not shown as results (_print_failure shows its last
    # lines); verbose mode streams everything.
    returncode, outcome.measured = _run_benchmark(
        bench_args + cache_flags + [model_caches.PEAK_MEMORY_ARG],
        show=not constants.DEFAULT_QUIET,
    )
    if returncode != 0:
      _print_failure(returncode, outcome.measured)
      outcome.fallback = f"the measured process exited with {returncode}"
      return outcome
    if constants.DEFAULT_QUIET:
      _show_benchmark_lines(outcome.measured)
    after = _cache_files(d)
  outcome.written = {name: size for name, (size, _) in written.items()}
  outcome.rewritten = [
      name for name, stat in after.items() if written.get(name) != stat
  ]
  return outcome


def run_desktop(
    *,
    model_path: pathlib.Path,
    accelerator: str,
    num_runs: int = 50,
    warmup_runs: int = 1,
    min_secs: float = 1.0,
    max_secs: float = 150.0,
    warmup_min_secs: float = 0.5,
    input_layer_value_range: str | None = None,
    signature_key: str | None = None,
) -> None:
  """Runs the benchmark_model binary on the local desktop machine.

  On the CPU and the GPU the binary runs twice (see model_caches): a first
  process writes the model caches, the measured process reads them. When
  one of them fails, the binary runs once more without the cache flags.

  Args:
    model_path: Path to the local LiteRT model file.
    accelerator: Hardware accelerator to use (cpu, gpu, npu).
    num_runs: Target number of benchmark iterations.
    warmup_runs: Number of warmup iterations before benchmarking.
    min_secs: Minimum seconds to run.
    max_secs: Maximum seconds to run.
    warmup_min_secs: Minimum warmup duration in seconds.
    input_layer_value_range: Value range for input layers.
    signature_key: The signature key to benchmark.

  Raises:
    click.ClickException: If execution fails.
  """
  click.echo("Preparing to run benchmark on local desktop...")

  if not model_path.exists():
    raise click.ClickException(f"Local model file not found: {model_path}")

  benchmark_bin = _ensure_desktop_binary("benchmark_model")

  click.echo(f"Executing benchmark locally using {benchmark_bin.name}...\n")
  outcome = None
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

    if num_runs != 50:
      bench_args.append(f"--num_runs={num_runs}")
    if warmup_runs != 1:
      bench_args.append(f"--warmup_runs={warmup_runs}")
    if min_secs != 1.0:
      bench_args.append(f"--min_secs={min_secs}")
    if max_secs != 150.0:
      bench_args.append(f"--max_secs={max_secs}")
    if warmup_min_secs != 0.5:
      bench_args.append(f"--warmup_min_secs={warmup_min_secs}")
    if input_layer_value_range:
      bench_args.append(f"--input_layer_value_range={input_layer_value_range}")
    if signature_key:
      bench_args.append(f"--signature_to_run_for={signature_key}")

    if model_caches.uses_caches(accelerator):
      outcome = _run_with_caches(bench_args, model_path.name, accelerator)
    if outcome is None or outcome.fallback is not None:
      if outcome is not None:
        click.secho(
            "Running benchmark_model once without the cache flags:"
            f" {outcome.fallback}",
            fg="yellow",
        )
      _run_benchmark(bench_args, show=True, or_fail=True)
  except click.ClickException:
    raise
  except Exception as e:
    raise click.ClickException(f"Failed to execute benchmark on desktop: {e}")
  if outcome is not None:
    for line in model_caches.report(accelerator, outcome):
      click.secho(line, fg="green")
  elif accelerator != "npu":
    click.secho(model_caches.DISABLED_LINE, fg="green")
