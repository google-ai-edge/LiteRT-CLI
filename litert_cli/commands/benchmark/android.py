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

"""Android Benchmark Module."""

from __future__ import annotations

import pathlib
import shlex
import subprocess
import uuid

import click
from litert_cli.commands.benchmark import model_caches
from litert_cli.core import android_utils
from litert_cli.core import constants
from litert_cli.core import npu_utils as npu


def run_android(
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
  """Runs the model on an Android device.

  Pushes the model and benchmark_model binary to the device and runs it.

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
    subprocess.CalledProcessError: If any adb command fails on the device.
  """
  click.echo("Preparing to run on Android device via adb...")

  android_utils.check_adb()
  abi = android_utils.get_android_abi()
  click.echo(f"Detected Android device ABI: {abi}")

  cli_android_root = constants.LITERT_CLI_ANDROID_ROOT

  model_name = model_path.name
  remote_model_path = f"{cli_android_root}/{model_name}"

  benchmark_model_bin = android_utils.find_android_binary(
      "benchmark_model", abi
  )

  remote_dispatch_dir = ""
  if accelerator == "npu":
    remote_dispatch_dir = npu.push_npu_runtime_libraries(None, cli_android_root)

    # Download and push SOC-specific LiteRT dispatch and compiler plugin libraries
    target_model = npu.get_soc_target_model(None)
    soc_vendor = "mediatek" if "mt" in target_model else "qualcomm"
    lib_dispatch = android_utils.find_npu_dispatch_lib(soc_vendor, abi)
    lib_compiler = android_utils.find_npu_compiler_plugin_lib(soc_vendor, abi)

    remote_lib_dispatch = f"{cli_android_root}/{lib_dispatch.name}"
    android_utils.push_file_to_device(lib_dispatch, remote_lib_dispatch)

    remote_lib_compiler = f"{cli_android_root}/{lib_compiler.name}"
    android_utils.push_file_to_device(lib_compiler, remote_lib_compiler)

  subprocess.run(["adb", "shell", "mkdir", "-p", cli_android_root], check=True)
  android_utils.push_file_to_device(
      model_path, remote_model_path, label=f"model {model_name}"
  )

  remote_benchmark_model_path = f"{cli_android_root}/benchmark_model"
  android_utils.push_file_to_device(
      benchmark_model_bin, remote_benchmark_model_path
  )
  subprocess.run(
      ["adb", "shell", "chmod", "+x", remote_benchmark_model_path],
      check=True,
  )

  click.echo("Executing benchmark on device...\n")
  outcome = None
  try:
    bench_args = [
        f"{cli_android_root}/benchmark_model",
        f"--graph={shlex.quote(remote_model_path)}",
    ]
    if accelerator == "gpu":
      bench_args.append("--use_gpu=true")
    elif accelerator == "npu":
      bench_args.append("--use_npu=true")
      bench_args.append(f"--dispatch_library_path={shlex.quote(cli_android_root)}")
      bench_args.append(
          f"--compiler_plugin_library_path={shlex.quote(cli_android_root)}"
      )

      if soc_vendor == "mediatek":
        recommend_version = constants.MEDIATEK_SOC_VERSION_MAP.get(
            target_model, ""
        )
        if "v9" in recommend_version:
          bench_args.append("--mediatek_nerun_pilot_version=version9")
        elif "v8" in recommend_version:
          bench_args.append("--mediatek_nerun_pilot_version=version8")

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
      bench_args.append(
          f"--input_layer_value_range={shlex.quote(input_layer_value_range)}"
      )
    if signature_key:
      bench_args.append(f"--signature_to_run_for={shlex.quote(signature_key)}")

    env_vars = ""
    if remote_dispatch_dir:
      quoted_dispatch_dir = shlex.quote(remote_dispatch_dir)
      env_vars = (
          f"LD_LIBRARY_PATH={quoted_dispatch_dir} "
          f"ADSP_LIBRARY_PATH={quoted_dispatch_dir} "
      )

    if model_caches.uses_caches(accelerator):
      outcome = _run_with_caches(env_vars, bench_args, model_name, accelerator)
    if outcome is None or outcome.fallback is not None:
      if outcome is not None:
        click.secho(
            "Running benchmark_model once without the cache flags:"
            f" {outcome.fallback}",
            fg="yellow",
        )
      _, output_lines = _run_on_device(
          env_vars + " ".join(bench_args), show=True, or_fail=True
      )
      if accelerator == "npu":
        npu.warn_if_npu_unused(output_lines)
  except click.ClickException:
    raise
  except Exception as e:
    raise click.ClickException(f"Failed to execute benchmark on device: {e}")
  if outcome is not None:
    for line in model_caches.report(accelerator, outcome):
      click.secho(line, fg="green")
  elif accelerator != "npu":
    click.secho(model_caches.DISABLED_LINE, fg="green")


def _run_with_caches(
    env_vars: str, bench_args: list[str], model_name: str, accelerator: str
) -> model_caches.Outcome:
  """Runs the first and the measured process in a fresh cache directory.

  Returns what they did. `fallback` is set when a process exited non-zero
  or the directory could not be made; the caller then runs benchmark_model
  once without the cache flags. The directory is removed either way.
  """
  outcome = model_caches.Outcome()
  # A fresh directory per run: the first process always compiles the model
  # and writes the caches, the measured process reads them.
  cache_dir = (
      f"{constants.LITERT_CLI_ANDROID_ROOT}/benchmark_cache_"
      f"{uuid.uuid4().hex[:8]}"
  )
  written_mark = f"{cache_dir}.written"
  made = _adb_shell(f"mkdir -p {shlex.quote(cache_dir)}")
  if made.returncode != 0:
    detail = made.stdout.strip().splitlines()
    outcome.fallback = (
        f"'adb shell mkdir -p {cache_dir}' exited with {made.returncode}"
        + (f": {detail[-1]}" if detail else "")
    )
    return outcome
  try:
    cache_flags = [
        shlex.quote(arg)
        for arg in model_caches.cache_args(accelerator, cache_dir, model_name)
    ]
    click.echo("Writing the model caches (benchmark_model, no inference)...")
    returncode, outcome.first = _run_on_device(
        env_vars
        + " ".join(bench_args + cache_flags + list(model_caches.WARMUP_ARGS)),
        show=not constants.DEFAULT_QUIET,
    )
    if returncode != 0:
      _print_failure(returncode, outcome.first)
      outcome.fallback = f"the first process exited with {returncode}"
      return outcome
    listed = _adb_shell(
        f"cd {shlex.quote(cache_dir)} && wc -c * 2>/dev/null;"
        f" touch {shlex.quote(written_mark)}"
    )
    if listed.returncode == 0:
      outcome.written = {}
      for line in listed.stdout.splitlines():
        size, _, name = line.strip().partition(" ")
        if size.isdigit() and name and name != "total":
          outcome.written[name] = int(size)
    # Shown once it exited 0, so the numbers of a process that fails after
    # printing them are not shown as results (_print_failure shows its last
    # lines); verbose mode streams everything.
    returncode, outcome.measured = _run_on_device(
        env_vars
        + " ".join(bench_args + cache_flags + [model_caches.PEAK_MEMORY_ARG]),
        show=not constants.DEFAULT_QUIET,
    )
    if returncode != 0:
      _print_failure(returncode, outcome.measured)
      outcome.fallback = f"the measured process exited with {returncode}"
      return outcome
    if constants.DEFAULT_QUIET:
      _show_benchmark_lines(outcome.measured)
    found = _adb_shell(
        f"find {shlex.quote(cache_dir)} -type f"
        f" -newer {shlex.quote(written_mark)} 2>/dev/null"
    )
    if found.returncode == 0:
      outcome.rewritten = [
          line.strip().rsplit("/", 1)[-1]
          for line in found.stdout.splitlines()
          if line.strip()
      ]
    return outcome
  finally:
    subprocess.run(
        [
            "adb",
            "shell",
            f"rm -rf {shlex.quote(cache_dir)} {shlex.quote(written_mark)}",
        ],
        check=False,
    )


def _adb_shell(command: str) -> subprocess.CompletedProcess[str]:
  """Runs a shell command on the device; its output is in stdout."""
  return subprocess.run(
      ["adb", "shell", command],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      check=False,
  )


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
      f"Execution failed on device with exit code {returncode}", fg="red"
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


def _run_on_device(
    full_command: str, *, show: bool, or_fail: bool = False
) -> tuple[int, list[str]]:
  """Runs benchmark_model on the device; returns its exit code and output.

  With `show`, prints the lines the benchmark log filter keeps. With
  `or_fail`, a non-zero exit prints the whole output and raises
  click.ClickException.
  """
  process = subprocess.Popen(
      ["adb", "shell", full_command],
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
        f"Execution failed on device with exit code {process.returncode}",
        fg="red",
    )
    click.echo("Full output for debugging:")
    for line in output_lines:
      click.echo(line, nl=False)
    raise click.ClickException("Benchmark failed on device.")
  return process.returncode, output_lines
