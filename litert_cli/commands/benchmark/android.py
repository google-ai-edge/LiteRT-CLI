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

import click
from litert_cli.core import android_utils
from litert_cli.core import constants
from litert_cli.core import npu_utils as npu


def run_android(*, model_path: pathlib.Path, accelerator: str) -> None:
  """Runs the model on an Android device.

  Pushes the model and benchmark_model binary to the device and runs it.

  Args:
    model_path: Path to the local LiteRT model file.
    accelerator: Hardware accelerator to use (cpu, gpu, npu).

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
    if (
        subprocess.run(
            ["adb", "shell", f"[ -f {remote_lib_dispatch} ]"], check=False
        ).returncode
        == 0
    ):
      click.echo(f"  Skipping {lib_dispatch.name} (already on device)")
    else:
      click.echo(f"Pushing {lib_dispatch.name} to device...")
      subprocess.run(
          ["adb", "push", str(lib_dispatch), remote_lib_dispatch], check=True
      )

    remote_lib_compiler = f"{cli_android_root}/{lib_compiler.name}"
    if (
        subprocess.run(
            ["adb", "shell", f"[ -f {remote_lib_compiler} ]"], check=False
        ).returncode
        == 0
    ):
      click.echo(f"  Skipping {lib_compiler.name} (already on device)")
    else:
      click.echo(f"Pushing {lib_compiler.name} to device...")
      subprocess.run(
          ["adb", "push", str(lib_compiler), remote_lib_compiler], check=True
      )

  click.echo(f"Pushing model {model_name} to device...")
  subprocess.run(["adb", "shell", "mkdir", "-p", cli_android_root], check=True)
  subprocess.run(
      ["adb", "push", str(model_path), remote_model_path], check=True
  )

  click.echo("Pushing benchmark_model to device...")
  subprocess.run(
      [
          "adb",
          "push",
          str(benchmark_model_bin),
          f"{cli_android_root}/benchmark_model",
      ],
      check=True,
  )
  subprocess.run(
      ["adb", "shell", "chmod", "+x", f"{cli_android_root}/benchmark_model"],
      check=True,
  )

  click.echo("Executing benchmark on device...\n")
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

    env_vars = ""
    if remote_dispatch_dir:
      quoted_dispatch_dir = shlex.quote(remote_dispatch_dir)
      env_vars = (
          f"LD_LIBRARY_PATH={quoted_dispatch_dir} "
          f"ADSP_LIBRARY_PATH={quoted_dispatch_dir} "
      )

    full_command = env_vars + " ".join(bench_args)
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
      if log_filter.should_show(line):
        click.echo(line, nl=False)

    process.wait()
    if process.returncode != 0:
      click.secho(
          f"Execution failed on device with exit code {process.returncode}",
          fg="red",
      )
      click.echo("Full output for debugging:")
      for line in output_lines:
        click.echo(line, nl=False)
      raise click.ClickException("Benchmark failed on device.")
  except Exception as e:
    raise click.ClickException(f"Failed to execute benchmark on device: {e}")
