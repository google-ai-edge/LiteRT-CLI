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

"""Android execution engine for LiteRT models.

Use adb to push the model and the run_model binary to the device, then run the
model on the device.

Usage Examples:
  1. Run a model on an Android device:
     $ litert run /path/to/model.tflite --android

  2. Run with NPU acceleration and CPU fallback:
     $ litert run /path/to/model.tflite --android --npu --cpu
     OR
     $ litert run /path/to/model.tflite --android --accelerator npu,cpu

  3. Run with custom inputs:
     $ litert run /path/to/model.tflite --android --input input_name=value

  4. Run with multiple inputs:
     $ litert run /path/to/model.tflite --android --input input1=value1 --input
     input2=value2

  5. Run with specific signature:
     $ litert run /path/to/model.tflite --android --signature_index 0

  6. Run with multiple iterations:
     $ litert run /path/to/model.tflite --android --iterations 10

  7. Print tensor details:
     $ litert run /path/to/model.tflite --android --print-tensors

  8. Run with sample size:
     $ litert run /path/to/model.tflite --android --sample-size 100
"""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import shlex
import subprocess
import tempfile
import time

import click
from litert_cli.core import android_utils
from litert_cli.core import constants
from litert_cli.core import inputs as inputs_utils
from litert_cli.core import npu_utils as npu


def _prepare_inputs_on_device(
    *,
    model_path: pathlib.Path,
    inputs: Sequence[str],
    signature_index: int,
    android_root: str,
) -> str:
  """Parses inputs locally and pushes files to device, returns remote dir.

  Args:
    model_path: Local path to the LiteRT model file.
    inputs: Tuple of input assignments (e.g. 'name=value').
    signature_index: Signature index to run on device.
    android_root: Remote workspace root path on Android device.

  Returns:
    Remote directory path where inputs are uploaded, or empty if no inputs.

  Raises:
    click.ClickException: If parsing fails or device push fails.
  """
  if not inputs:
    return ""

  try:
    click.echo("Parsing inputs locally before pushing to device...")
    from ai_edge_litert.compiled_model import CompiledModel  # pylint: disable=g-import-not-at-top

    cm = CompiledModel.from_file(str(model_path))
    signatures = cm.get_signature_list()
    if not signatures:
      click.secho("No signatures found in the model for inputs.", fg="yellow")
      return ""

    sig_info = cm.get_signature_by_index(signature_index)
    sig_key = sig_info["key"]
    input_details = cm.get_input_tensor_details(sig_key)

    # 2. Process input strings mapping (e.g., name=value or literal value).
    parsed_inputs = {}
    for inp in inputs:
      if "=" in inp:
        k, v = inp.split("=", 1)
        parsed_inputs[k] = v
      else:
        parsed_inputs["_default_"] = inp

    has_inputs = False
    # 3. Convert literals or files to raw binaries in a local temporary
    # directory.
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = pathlib.Path(temp_dir)
      for name, details in input_details.items():
        input_data_str = parsed_inputs.get(name) or parsed_inputs.get(
            "_default_"
        )
        if input_data_str:
          shape = details.get("shape", [1])
          tensor_type = details.get("dtype", "?")
          np_dtype = inputs_utils.get_np_dtype(tensor_type)

          click.echo(
              f"  Preparing input {name!r} from {input_data_str!r} (shape:"
              f" {shape}, dtype: {tensor_type})"
          )
          data = inputs_utils.parse_input(input_data_str, shape, np_dtype)

          raw_file_path = temp_path / f"{name}.raw"
          data.tofile(raw_file_path)
          has_inputs = True

      if has_inputs:
        # 4. Create remote directory and push input data batch tree to device.
        remote_input_dir = f"{android_root}/inputs_{int(time.time())}"
        click.echo(f"Pushing processed inputs to {remote_input_dir}...")
        subprocess.run(
            ["adb", "shell", f"mkdir -p {shlex.quote(remote_input_dir)}"],
            check=True,
        )
        for local_file in temp_path.iterdir():
          subprocess.run(
              [
                  "adb",
                  "push",
                  local_file,
                  f"{remote_input_dir}/{local_file.name}",
              ],
              check=True,
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
          )
        return remote_input_dir

    return ""

  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(
        f"Failed to prepare inputs for Android model {model_path}: {e!r}"
    ) from e


def run_android(
    *,
    model_path: str,
    inputs: Sequence[str],
    accelerator: str,
    signature_index: int,
    iterations: int,
    print_tensors: bool,
    sample_size: int,
) -> None:
  """Runs the model on an attached Android device using adb and run_model.

  Args:
    model_path: Local path to the LiteRT model file (.tflite).
    inputs: Tuple of input assignments (e.g. 'name=value').
    accelerator: Hardware accelerator ('cpu', 'gpu', 'npu').
    signature_index: Signature index to execute.
    iterations: Number of execute loops for remote runner.
    print_tensors: Whether to print remote stats after execution completes.
    sample_size: Limit execution sample stream print length per tensor.

  Raises:
    click.ClickException: On device error setup or failed execution triggers.
  """
  accel_list = [a.strip().lower() for a in accelerator.split(",") if a.strip()]
  click.echo("Preparing to run on Android device via adb...")
  android_utils.check_adb()

  # Set up Android working directory
  android_root = constants.LITERT_CLI_ANDROID_ROOT
  try:
    subprocess.run(
        ["adb", "shell", f"mkdir -p {shlex.quote(android_root)}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
  except subprocess.CalledProcessError as e:
    raise click.ClickException(
        f"Failed to create directory {android_root} on device: {e!r}"
    ) from e

  # Determine device ABI
  abi = android_utils.get_android_abi()
  click.echo(f"Detected Android device ABI: {abi}")

  # Check if model is compiled for NPU when NPU accelerator is requested
  is_npu_compiled = False
  if "npu" in accel_list:
    target_model = npu.get_soc_target_model(None)
    soc_vendor = "mediatek" if "mt" in target_model else "qualcomm"

    # Check if the input model already contains NPU dispatch bytecode
    try:
      with open(model_path, "rb") as f:
        model_bytes = f.read()
      is_npu_compiled = (
          b"DISPATCH" in model_bytes
          or b"LiteRtBuildStamp" in model_bytes
          or target_model.upper().encode() in model_bytes
      )
    except Exception:
      is_npu_compiled = False

    if not is_npu_compiled:
      # Check if a matching compiled model exists in the same folder or _compiled_models
      candidate_paths = [
          pathlib.Path(model_path).parent
          / f"{pathlib.Path(model_path).stem}_{soc_vendor.capitalize()}_{target_model.upper()}.tflite",
          pathlib.Path(model_path).parent
          / f"{pathlib.Path(model_path).stem}_{target_model}.tflite",
          pathlib.Path(model_path).parent
          / "_compiled_models"
          / f"{pathlib.Path(model_path).stem}_{target_model}.tflite",
      ]
      found_compiled = None
      for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
          found_compiled = str(candidate)
          break

      if found_compiled:
        click.echo(
            f"Using existing NPU compiled model for {target_model}:"
            f" {found_compiled}"
        )
        model_path = found_compiled
        is_npu_compiled = True
      else:
        # Attempt AOT compilation for target SoC
        try:
          from ai_edge_litert.aot import aot_compile as aot_lib  # pylint: disable=g-import-not-at-top
          click.echo(
              f"Compiling model {model_path} for target NPU SoC: {target_model}..."
          )
          aot_target = npu.get_aot_target(target_model)
          compiled_res = aot_lib.aot_compile(
              str(model_path), target=[aot_target], keep_going=False
          )
          compiled_dir = pathlib.Path(model_path).parent
          compiled_res.export(
              str(compiled_dir), model_name=pathlib.Path(model_path).stem
          )
          for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
              model_path = str(candidate)
              is_npu_compiled = True
              break
        except Exception as e:
          click.echo(
              f"Note: AOT compilation skipped ({e}), falling back to on-device"
              " execution."
          )

  # Create remote execution tracking paths
  model_name = pathlib.Path(model_path).name
  remote_model_path = f"{android_root}/{model_name}"

  run_model_bin = android_utils.find_android_binary("run_model", abi)

  # Download libraries
  lib_litert = android_utils.find_android_lib("libLiteRt.so", abi)
  lib_clgl = android_utils.find_android_lib("libLiteRtClGlAccelerator.so", abi)

  # Push model file and runner binary to Android device
  android_utils.push_file_to_device(
      model_path, remote_model_path, label=f"model {model_name}"
  )

  remote_run_model_path = f"{android_root}/run_model"
  android_utils.push_file_to_device(run_model_bin, remote_run_model_path)

  # Push libraries to default path
  remote_lib_litert = f"{android_root}/{lib_litert.name}"
  android_utils.push_file_to_device(lib_litert, remote_lib_litert)

  remote_lib_clgl = f"{android_root}/{lib_clgl.name}"
  android_utils.push_file_to_device(lib_clgl, remote_lib_clgl)

  remote_input_dir = _prepare_inputs_on_device(
      model_path=pathlib.Path(model_path),
      inputs=inputs,
      signature_index=signature_index,
      android_root=android_root,
  )

  # Pass None as device_id to use the default connected device.
  remote_dispatch_dir = (
      npu.push_npu_runtime_libraries(None, android_root)
      if "npu" in accel_list
      else ""
  )

  if "npu" in accel_list:
    # Download and push SOC-specific LiteRT dispatch and compiler plugin libraries
    lib_dispatch = android_utils.find_npu_dispatch_lib(soc_vendor, abi)
    remote_lib_dispatch = f"{android_root}/{lib_dispatch.name}"
    android_utils.push_file_to_device(lib_dispatch, remote_lib_dispatch)

    if not is_npu_compiled:
      lib_compiler = android_utils.find_npu_compiler_plugin_lib(soc_vendor, abi)
      remote_lib_compiler = f"{android_root}/{lib_compiler.name}"
      android_utils.push_file_to_device(lib_compiler, remote_lib_compiler)

  click.echo("Executing on device...\n")

  # Need to make the binary executable before running the model
  subprocess.run(
      ["adb", "shell", f"chmod +x {shlex.quote(remote_run_model_path)}"],
      check=True,
  )

  # Forward valid flags
  run_cmd_args = [remote_run_model_path, f"--graph={remote_model_path}"]
  if accelerator != "cpu":
    run_cmd_args.append(f"--accelerator={accelerator}")
  if remote_dispatch_dir:
    run_cmd_args.append(f"--dispatch_library_dir={remote_dispatch_dir}")
    if not is_npu_compiled:
      run_cmd_args.append(f"--compiler_plugin_library_dir={remote_dispatch_dir}")
  if iterations > 1:
    run_cmd_args.append(f"--iterations={iterations}")
  if signature_index != 0:
    run_cmd_args.append(f"--signature_index={signature_index}")
  if print_tensors:
    run_cmd_args.append("--print_tensors=true")
  run_cmd_args.append(f"--sample_size={sample_size}")
  if remote_input_dir:
    run_cmd_args.append(f"--input_dir={remote_input_dir}")

  try:
    if remote_dispatch_dir:
      env_vars = (
          f"LD_LIBRARY_PATH={shlex.quote(f'{remote_dispatch_dir}:{android_root}')}"
          f" ADSP_LIBRARY_PATH={shlex.quote(remote_dispatch_dir)}"
      )
    else:
      env_vars = ""
    cmd_str = f"{env_vars} " if env_vars else ""
    cmd_str += " ".join(shlex.quote(arg) for arg in run_cmd_args)
    process = subprocess.Popen(
        ["adb", "shell", cmd_str],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    from litert_cli.core.log_filters import RunLogFilter

    output_lines = []
    log_filter = RunLogFilter(constants.DEFAULT_QUIET, print_tensors)

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
      raise click.ClickException("Execution failed on device.")
  except Exception as e:
    raise click.ClickException(f"Failed to execute on device: {e}")
  finally:
    # Cleanup remote input paths if created
    if remote_input_dir:
      click.echo("Clearing remote input files...")
      subprocess.run(
          ["adb", "shell", f"rm -rf {shlex.quote(remote_input_dir)}"],
          check=False,
      )
