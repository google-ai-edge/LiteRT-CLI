"""Android execution engine for LiteRT models.

Use adb to push the model and the run_model binary to the device, then run the
model on the device.

Usage Examples:
  1. Run a model on an Android device:
     $ litert run /path/to/model.tflite --android

  2. Run with custom inputs:
     $ litert run /path/to/model.tflite --android --input input_name=value

  3. Run with multiple inputs:
     $ litert run /path/to/model.tflite --android --input input1=value1 --input
     input2=value2

  4. Run with specific signature:
     $ litert run /path/to/model.tflite --android --signature 0

  5. Run with multiple iterations:
     $ litert run /path/to/model.tflite --android --iterations 10

  6. Print tensor details:
     $ litert run /path/to/model.tflite --android --print-tensors

  7. Run with sample size:
     $ litert run /path/to/model.tflite --android --sample-size 100
"""

from __future__ import annotations

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
    model_path: str,
    inputs: tuple[str, ...],
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
    # 1. Load model structure for Input meta details.
    from ai_edge_litert.compiled_model import CompiledModel  # pylint: disable=g-import-not-at-top

    cm = CompiledModel.from_file(model_path)
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
              f"  Preparing input '{name}' from '{input_data_str}' (shape:"
              f" {shape}, dtype: {tensor_type})"
          )
          data = inputs_utils.parse_input(input_data_str, shape, np_dtype)

          # Write to temp raw file
          raw_file_path = temp_path / f"{name}.raw"
          data.tofile(raw_file_path)
          has_inputs = True

      if has_inputs:
        # 4. Create remote directory and push input data batch tree to device.
        remote_input_dir = f"{android_root}/inputs_{int(time.time())}"
        click.echo(f"Pushing processed inputs to {remote_input_dir}...")
        subprocess.run(
            ["adb", "shell", f"mkdir -p {remote_input_dir}"], check=True
        )
        for local_file in temp_path.iterdir():
          subprocess.run(
              [
                  "adb",
                  "push",
                  str(local_file),
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
        f"Failed to prepare inputs for Android: {e}"
    ) from e


def run_android(
    model_path: str,
    inputs: tuple[str, ...],
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
  click.echo("Preparing to run on Android device via adb...")
  android_utils.check_adb()

  # Set up Android working directory
  android_root = constants.LITERT_CLI_ANDROID_ROOT
  try:
    subprocess.run(
        ["adb", "shell", f"mkdir -p {android_root}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
  except subprocess.CalledProcessError as e:
    raise click.ClickException(
        f"Failed to create directory {android_root} on device: {e}"
    ) from e

  # Create remote execution tracking paths
  model_name = pathlib.Path(model_path).name
  remote_model_path = f"{android_root}/{model_name}"

  # Determine device ABI
  abi = android_utils.get_android_abi()
  click.echo(f"Detected Android device ABI: {abi}")

  run_model_bin = android_utils.find_android_binary("run_model", abi)

  # Push model file and runner binary to Android device
  click.echo(f"Pushing model {model_name} to device...")
  subprocess.run(["adb", "push", model_path, remote_model_path], check=True)

  click.echo("Pushing run_model to device...")
  remote_run_model_path = f"{android_root}/run_model"
  subprocess.run(
      ["adb", "push", str(run_model_bin), remote_run_model_path], check=True
  )

  remote_input_dir = _prepare_inputs_on_device(
      model_path, inputs, signature_index, android_root
  )

  remote_dispatch_dir = ""
  if accelerator == "npu":
    remote_dispatch_dir = npu.push_runtime_libraries(None, android_root)

  click.echo("Executing on device...\n")

  # Need to make the binary executable before drawing execution triggers
  subprocess.run(
      ["adb", "shell", f"chmod +x {remote_run_model_path}"], check=True
  )

  # Forward valid flags
  run_cmd_args = [remote_run_model_path, f"--graph={remote_model_path}"]
  if accelerator != "cpu":
    run_cmd_args.append(f"--accelerator={accelerator}")
  if remote_dispatch_dir:
    run_cmd_args.append(f"--dispatch_library_dir={remote_dispatch_dir}")
  if iterations > 1:
    run_cmd_args.append(f"--iterations={iterations}")
  if signature_index != 0:
    run_cmd_args.append(f"--signature_index={signature_index}")
  if print_tensors:
    run_cmd_args.append("--print_tensors=true")
  if sample_size != 5:
    run_cmd_args.append(f"--sample_size={sample_size}")
  if remote_input_dir:
    run_cmd_args.append(f"--input_dir={remote_input_dir}")

  try:
    env_vars = f"LD_LIBRARY_PATH={remote_dispatch_dir} " if remote_dispatch_dir else ""
    cmd_str = env_vars + " ".join(shlex.quote(arg) for arg in run_cmd_args)
    subprocess.run(
        [
            "adb",
            "shell",
            cmd_str,
        ],
        check=True,
    )
  except subprocess.CalledProcessError as e:
    raise click.ClickException(f"Execution failed on device: {e}") from e
  finally:
    # Cleanup remote paths
    cleanup_cmd = f"rm -f {remote_model_path} {remote_run_model_path}"
    if remote_input_dir:
      cleanup_cmd += f" && rm -rf {remote_input_dir}"
    # Do not cleanup dispatch dir for now as it might be shared or large?
    subprocess.run(["adb", "shell", cleanup_cmd], check=False)
