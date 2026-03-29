"""Android Benchmark Module."""

from __future__ import annotations

import pathlib
import subprocess

import click
from litert_cli.core import android_utils
from litert_cli.core import constants
from litert_cli.core import npu_utils as npu


def run_android(model_path: str, accelerator: str) -> None:
  """Runs the model on an Android device via adb and benchmark_model.

  Args:
    model_path: Path to the local LiteRT model file.
    accelerator: Hardware accelerator to use (cpu, gpu, npu).
  """
  click.echo("Preparing to run on Android device via adb...")

  android_utils.check_adb()
  abi = android_utils.get_android_abi()
  click.echo(f"Detected Android device ABI: {abi}")

  cli_root = constants.LITERT_CLI_ROOT
  cli_android_root = constants.LITERT_CLI_ANDROID_ROOT

  model_path_obj = pathlib.Path(model_path)
  model_name = model_path_obj.name
  remote_model_path = f"{cli_android_root}/{model_name}"

  benchmark_model_bin = android_utils.find_android_binary(
      "benchmark_model", abi
  )

  remote_dispatch_dir = ""
  remote_lib_dispatch = ""
  if accelerator == "npu":
    remote_dispatch_dir = npu.push_npu_runtime_libraries(None, cli_android_root)

    # Download and push SOC-specific LiteRT dispatch library
    target_model = npu.get_soc_target_model(None)
    soc_vendor = "mediatek" if "mt" in target_model else "qualcomm"
    lib_dispatch = android_utils.find_npu_dispatch_lib(soc_vendor, abi)

    remote_lib_dispatch = f"{cli_android_root}/{lib_dispatch.name}"
    if subprocess.run(["adb", "shell", f"[ -f {remote_lib_dispatch} ]"], check=False).returncode == 0:
      click.echo(f"  Skipping {lib_dispatch.name} (already on device)")
    else:
      click.echo(f"Pushing {lib_dispatch.name} to device...")
      subprocess.run(["adb", "push", str(lib_dispatch), remote_lib_dispatch], check=True)

  click.echo(f"Pushing model {model_name} to device...")
  subprocess.run(["adb", "shell", "mkdir", "-p", cli_android_root], check=True)
  subprocess.run(["adb", "push", model_path, remote_model_path], check=True)

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
    bench_cmd = (
        f"{cli_android_root}/benchmark_model --graph={remote_model_path}"
    )
    if accelerator == "gpu":
      bench_cmd += " --use_gpu=true"
    elif accelerator == "npu":
      bench_cmd += (
          f" --use_npu=true --dispatch_library_path={cli_android_root}"
      )

    env_vars = f"LD_LIBRARY_PATH={remote_dispatch_dir} ADSP_LIBRARY_PATH={remote_dispatch_dir} " if remote_dispatch_dir else ""
    subprocess.run(["adb", "shell", env_vars + bench_cmd], check=True)
  except subprocess.CalledProcessError as e:
    click.secho(f"Execution failed on device: {e}", fg="red")
