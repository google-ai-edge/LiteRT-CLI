"""Desktop execution engine for LiteRT models.

Uses CompiledModel to load and run models on desktop (CPU/GPU).

Usage Examples:
  1. Run a model on desktop (CPU):
     $ litert run /path/to/model.tflite --desktop

  2. Run with GPU acceleration:
     $ litert run /path/to/model.tflite --desktop --gpu
     OR
     $ litert run /path/to/model.tflite --desktop --accelerator gpu

  3. Run with custom inputs:
     $ litert run /path/to/model.tflite --desktop --input input_name=value

  4. Run with multiple iterations (benchmark):
     $ litert run /path/to/model.tflite --desktop --iterations 10

  5. Print tensor details:
     $ litert run /path/to/model.tflite --desktop --print-tensors
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import click
from litert_cli.core import inputs as inputs_utils
import numpy as np

if TYPE_CHECKING:
  # Import heavy dependencies only for type hinting to improve CLI startup
  # performance. These are not imported at runtime.
  from ai_edge_litert.compiled_model import CompiledModel
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator


def _parse_inputs_dict(inputs: tuple[str, ...]) -> dict[str, str]:
  """Parses a tuple of input assignments into a dictionary.

  Args:
    inputs: A tuple of input strings, e.g., ('name=value', 'value2').

  Returns:
    A dictionary mapping names to values. Unnamed inputs use '_default_'.
  """
  parsed_inputs = {}
  if inputs:
    for inp in inputs:
      if "=" in inp:
        k, v = inp.split("=", 1)
        parsed_inputs[k] = v
      else:
        parsed_inputs["_default_"] = inp
  return parsed_inputs


def _prepare_inputs(
    cm: CompiledModel, sig_key: str, parsed_inputs: dict[str, str]
) -> dict[str, Any]:
  """Prepares CompiledModel input buffers.

  Loads parsed input assignments or generates random dummy data to load into the
  CompiledModel TensorBuffers.

  Args:
    cm: The loaded CompiledModel structure to interact with.
    sig_key: Signature key describing the input interface.
    parsed_inputs: Dictionary mapping input names to file path/literal strings.

  Returns:
    A dictionary mapping tensor names to their populated TensorBuffers.

  Raises:
    click.ClickException: If input loading or parsing fails.
  """
  inputs_dict = {}
  input_details = cm.get_input_tensor_details(sig_key)

  for name, details in input_details.items():
    shape = details.get("shape", [1])
    tensor_type = details.get("dtype", "?")
    np_dtype = inputs_utils.get_np_dtype(tensor_type)

    input_data_str = parsed_inputs.get(name) or parsed_inputs.get("_default_")

    if input_data_str:
      click.echo(
          f"Loading input '{name}' from '{input_data_str}' (shape:"
          f" {shape}, dtype: {tensor_type})"
      )
      try:
        data = inputs_utils.parse_input(input_data_str, shape, np_dtype)
      except ImportError as ie:
        click.secho(str(ie), fg="red")
        raise click.ClickException("Failed to load input module.") from ie
      except Exception as e:
        click.secho(f"Failed to parse input: {e}", fg="red")
        raise click.ClickException(
            f"Failed to parse input '{name}': {e}"
        ) from e
    else:
      click.echo(
          f"Generating random dummy input '{name}' with shape {shape} and"
          f" dtype {tensor_type}"
      )
      data = np.array(
          np.random.uniform(low=-1.0, high=1.0, size=shape), dtype=np_dtype
      )

    tb = cm.create_input_buffer_by_name(sig_key, name)
    tb.write(data)
    inputs_dict[name] = tb

  return inputs_dict


def _print_outputs(
    outputs_map: dict[str, Any], print_tensors: bool, sample_size: int
) -> None:
  """Prints inference outputs to stdout.

  Iterates through absolute tensor results and applies heuristics for
  classification formatting or raw values flattening details.

  Args:
    outputs_map: Dictionary mapping output names to read-ready TensorBuffers.
    print_tensors: Boolean flag to trigger full tensor stream printing.
    sample_size: Constraint on how many elements to print for large arrays.
  """
  click.echo("Outputs:")
  for out_name, out_tb in outputs_map.items():
    try:
      shape = out_tb.shape if hasattr(out_tb, "shape") else []
      num_elements = np.prod(shape) if shape else 1
      out_np = out_tb.read(num_elements, np.float32)

      if shape:
        out_np = out_np.reshape(shape)

      if print_tensors:
        flat_out = out_np.flatten()
        n_elem = len(flat_out)
        click.echo(f"  {out_name} (shape: {shape}):")

        if n_elem <= sample_size * 2:
          click.echo(f"    {flat_out}")
        else:
          p_start = flat_out[:sample_size]
          p_end = flat_out[-sample_size:]
          click.echo(
              f"    [{' '.join(map(str, p_start))} ..."
              f" {' '.join(map(str, p_end))}]"
          )
      else:
        # Classification inference heuristics fallback
        if (len(shape) == 1 and shape[0] > 1) or (
            len(shape) == 2 and shape[0] == 1 and shape[1] > 1
        ):
          scores = out_np.flatten()
          n_top = min(5, len(scores))
          top_indices = np.argsort(scores)[-n_top:][::-1]

          click.echo(f"  {out_name} (Top {n_top} Predictions):")
          for i, idx in enumerate(top_indices):
            click.echo(f"    {i+1}: index {idx} - score {scores[idx]:.4f}")
        else:
          click.echo(
              f"  {out_name}: mean={np.mean(out_np):.4f},"
              f" min={np.min(out_np):.4f}, max={np.max(out_np):.4f}"
          )
    except Exception:  # pylint: disable=broad-exception-caught
      click.echo(
          f"  {out_name}: [Unable to read data natively without specific"
          " dtype info]"
      )


def run_desktop(
    model_path: str,
    inputs: tuple[str, ...],
    accelerator: str,
    signature_index: int,
    iterations: int,
    print_tensors: bool,
    sample_size: int,
) -> None:
  """Run the model on the desktop target using CompiledModel.

  Args:
    model_path: Local path to the LiteRT model file (.tflite).
    inputs: Tuple of input assignments (e.g., 'name=value').
    accelerator: Hardware accelerator ('cpu', 'gpu', 'npu').
    signature_index: Signature index to execute.
    iterations: Number of execute loops for remote runner.
    print_tensors: Whether to print absolute stats after execution completes.
    sample_size: Limit execution sample stream print length per tensor.

  Raises:
    click.ClickException: On loading failure or inference execution errors.
  """

  click.echo(
      f"Loading model on desktop: {model_path} with {accelerator.upper()}"
  )

  # pylint: disable=g-import-not-at-top,reimported
  from ai_edge_litert.compiled_model import CompiledModel
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator

  hw_accel = HardwareAccelerator.CPU
  if accelerator == "gpu":
    hw_accel = HardwareAccelerator.GPU
  elif accelerator == "npu":
    raise click.ClickException(
        "NPU accelerator is not yet formally supported via desktop API."
    )

  try:
    cm = CompiledModel.from_file(model_path, hw_accel)
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(f"Failed to load CompiledModel: {e}") from e

  signatures = cm.get_signature_list()
  if not signatures:
    click.secho("No signatures found in the model.", fg="yellow")
    return

  try:
    sig_info = cm.get_signature_by_index(signature_index)
    sig_key = sig_info["key"]
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(
        f"Failed to get signature at index {signature_index}: {e}"
    ) from e

  click.echo(f"Using signature: {sig_key}")

  parsed_inputs = _parse_inputs_dict(inputs)
  inputs_dict = _prepare_inputs(cm, sig_key, parsed_inputs)

  click.echo(f"Running inference {iterations} times...")

  run_times = []

  try:
    sig_idx = cm.get_signature_index(sig_key)
    out_buffers = cm.create_output_buffers(sig_idx)
    out_names = signatures[sig_key]["outputs"]
    outputs_map = dict(zip(out_names, out_buffers))

    for _ in range(iterations):
      start_time = time.time()
      cm.run_by_name(sig_key, inputs_dict, outputs_map)
      end_time = time.time()
      run_times.append((end_time - start_time) * 1000)

    if iterations == 1:
      click.echo(f"Inference complete in {run_times[0]:.2f} ms")
    else:
      click.echo(f"Benchmark results ({iterations} iterations):")
      click.echo(f"  First run: {run_times[0]:.2f} ms")
      click.echo(f"  Average: {np.mean(run_times):.2f} ms")
      click.echo(f"  Min: {np.min(run_times):.2f} ms")
      click.echo(f"  Max: {np.max(run_times):.2f} ms")

    _print_outputs(outputs_map, print_tensors, sample_size)

  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(f"Inference failed: {e}") from e
