"""Android Benchmark Module."""

from __future__ import annotations

import pathlib
import subprocess

import click
from litert_cli.core import android_utils
from litert_cli.core import constants


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

  qualcomm_dispatch_so = None
  if accelerator == "npu":
    qualcomm_dispatch_so = (
        pathlib.Path(cli_root)
        / "bin"
        / "android"
        / "libLiteRtDispatch_Qualcomm.so"
    )
    if not qualcomm_dispatch_so.exists():
      click.secho(
          "Warning: Qualcomm dispatch library not found at: "
          f"{qualcomm_dispatch_so}",
          fg="yellow",
      )

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

  if (
      accelerator == "npu"
      and qualcomm_dispatch_so
      and qualcomm_dispatch_so.exists()
  ):
    click.echo("Pushing Qualcomm dispatch library to device...")
    subprocess.run(
        ["adb", "shell", "mkdir", "-p", f"{cli_android_root}/dispatch"],
        check=True,
    )
    subprocess.run(
        [
            "adb",
            "push",
            str(qualcomm_dispatch_so),
            f"{cli_android_root}/dispatch/libLiteRtDispatch_Qualcomm.so",
        ],
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
          f" --use_npu=true --dispatch_library_dir={cli_android_root}/dispatch"
      )

    subprocess.run(["adb", "shell", bench_cmd], check=True)
  except subprocess.CalledProcessError as e:
    click.secho(f"Execution failed on device: {e}", fg="red")
