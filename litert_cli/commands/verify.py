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

"""Command line interface for checking a model against reference outputs.

`litert verify` is the correctness oracle of an authoring loop: after a model
has been re-authored (for example in the LiteRT Tensor API), quantized, or
moved to another backend, it replays reference inputs through the model and
compares every output with the expected values.

References come from one of two places:

* `--inputs inputs.npz --expected outputs.npz`: arrays keyed by signature
  input/output name, produced by anything (a PyTorch script, NumPy, a
  previous `verify --against ... --save DIR`).
* `--against reference.tflite`: another LiteRT model; `verify` feeds both
  models the same inputs (`--inputs`, or seeded random inputs) and compares
  their outputs.

Both models must take inputs of the same dtype. Comparing a model with
quantized (int8/uint8) inputs or outputs against its float original is not
supported yet: `verify` reports an error instead of silently casting.

Usage Examples:
  1. Re-authored model vs goldens dumped from the original framework:
     $ litert verify candidate.tflite --inputs ref/inputs.npz --expected
     ref/outputs.npz

  2. Same model on GPU vs CPU reference, and keep the goldens for later runs:
     $ litert verify model.tflite --against model.tflite --gpu --save ref/

  3. Float model with dynamic-range (weight-only) quantization vs its original,
     looser tolerance:
     $ litert verify model_dr.tflite --against model.tflite --atol 0.05

  4. Machine-readable result (exit code 5 on mismatch):
     $ litert --json verify candidate.tflite --against model.tflite
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import pathlib
import textwrap
from typing import Any

import click
from litert_cli.core import constants
from litert_cli.core import report
import numpy as np

# Schema of the document `litert --json verify` emits.
VERIFY_SCHEMA = "litert.verify.v1"

# Exit code when the model ran but its outputs do not match the reference.
VERIFICATION_FAILED_EXIT_CODE = 5

INPUTS_FILE = "inputs.npz"
OUTPUTS_FILE = "outputs.npz"

# LiteRT tensor type names (as reported by `CompiledModel`) that `verify` can
# move in and out of tensor buffers, and their numpy dtypes. `bfloat16` is
# added when `ml_dtypes` is importable.
_NP_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}
try:
  import ml_dtypes  # pylint: disable=g-import-not-at-top

  _NP_DTYPES["bfloat16"] = np.dtype(ml_dtypes.bfloat16)
except ImportError:  # pragma: no cover
  pass


class VerificationFailed(click.ClickException):
  """The model ran, but at least one output differs from the reference."""

  exit_code = VERIFICATION_FAILED_EXIT_CODE


def _accelerators(accelerator: str | None, cpu: bool, gpu: bool, npu: bool):
  if accelerator:
    return [a.strip().lower() for a in accelerator.split(",") if a.strip()]
  chosen = [a for a, on in (("npu", npu), ("gpu", gpu), ("cpu", cpu)) if on]
  return chosen or ["cpu"]


def np_dtype(tensor_type: str, *, tensor: str) -> np.dtype:
  """Returns the numpy dtype for a LiteRT tensor type name.

  Args:
    tensor_type: Type name reported by `CompiledModel`, e.g. `"float32"`.
    tensor: Tensor name, for the error message.

  Raises:
    click.ClickException: If `verify` cannot handle the type.
  """
  dtype = _NP_DTYPES.get(tensor_type.lower())
  if dtype is None:
    raise click.ClickException(
        f"Tensor {tensor!r} has dtype {tensor_type!r}, which `litert verify`"
        f" does not support yet (supported: {', '.join(sorted(_NP_DTYPES))})."
    )
  return dtype


def random_input(
    rng: np.random.Generator, shape: list[int], dtype: np.dtype
) -> np.ndarray:
  """Seeded random input: bools, integers in [0, 10), or floats in [-1, 1)."""
  if dtype == np.bool_:
    return rng.integers(0, 2, size=shape).astype(np.bool_)
  if np.issubdtype(dtype, np.integer):
    return rng.integers(0, 10, size=shape, dtype=dtype)
  return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def coerce_input(name: str, value: Any, shape: list[int], dtype: np.dtype):
  """Checks a reference input against the model's input `shape` and `dtype`.

  Floats are cast between float widths (goldens are often float64). Any other
  dtype difference, in particular float vs. quantized int8/uint8, is an error:
  casting would silently produce a meaningless comparison.

  Args:
    name: Input name, for error messages.
    value: The reference value.
    shape: Shape the model expects.
    dtype: Dtype the model expects.

  Returns:
    The input as an array of `dtype`.

  Raises:
    click.ClickException: On a shape or dtype mismatch.
  """
  array = np.asarray(value)
  same_kind_float = array.dtype.kind == "f" and dtype.kind == "f"
  if array.dtype != dtype and not same_kind_float:
    raise click.ClickException(
        f"Input {name!r} has dtype {array.dtype} but the model expects"
        f" {dtype}. Comparing models with different input dtypes (e.g. a"
        " fully quantized int8 model against its float original) is not"
        " supported yet."
    )
  array = array.astype(dtype, copy=False)
  if list(array.shape) != shape:
    raise click.ClickException(
        f"Input {name!r} has shape {list(array.shape)} but the model"
        f" expects {shape}."
    )
  return array


def _load_model(model_path: str, accelerators: list[str]) -> Any:
  """Loads a CompiledModel for the requested accelerators (fallback order)."""
  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.compiled_model import CompiledModel
  from ai_edge_litert.compiled_model import Environment
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator
  # pylint: enable=g-import-not-at-top

  flags = {
      "cpu": HardwareAccelerator.CPU,
      "gpu": HardwareAccelerator.GPU,
      "npu": HardwareAccelerator.NPU,
  }
  hw = HardwareAccelerator(0)
  for a in accelerators:
    if a not in flags:
      raise click.ClickException(f"Unsupported hardware accelerator: {a!r}")
    hw |= flags[a]
  env = None
  if constants.IS_INTERNAL_ENV:
    env = Environment.create(runtime_path="")
  return CompiledModel.from_file(model_path, hw, environment=env)


def _signature(cm: Any, model_path: str, index: int) -> tuple[dict, str]:
  signatures = cm.get_signature_list()
  if not signatures:
    raise click.ClickException(
        f"No signatures found in the model: {model_path!r}"
    )
  if not 0 <= index < len(signatures):
    raise click.ClickException(
        f"--signature-index {index} is out of range: the model has"
        f" {len(signatures)} signature(s) ({', '.join(map(repr, signatures))})."
    )
  return signatures, cm.get_signature_by_index(index)["key"]


def run_model(
    model_path: str,
    accelerators: list[str],
    signature_index: int,
    inputs: Mapping[str, np.ndarray] | None,
    *,
    seed: int,
) -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Runs one signature; returns `(signature_key, inputs, outputs)`.

  When `inputs` is None, seeded random inputs matching the signature are
  generated (see `random_input`).
  """
  cm = _load_model(model_path, accelerators)
  signatures, sig_key = _signature(cm, model_path, signature_index)
  input_details = cm.get_input_tensor_details(sig_key)
  rng = np.random.default_rng(seed)

  fed: dict[str, np.ndarray] = {}
  buffers: dict[str, Any] = {}
  for name, details in input_details.items():
    shape = list(details.get("shape", [1]))
    dtype = np_dtype(details.get("dtype", "unknown"), tensor=name)
    if inputs is None:
      array = random_input(rng, shape, dtype)
    elif name not in inputs:
      raise click.ClickException(
          f"Reference inputs have no entry for model input {name!r}."
          f" Available: {', '.join(map(repr, inputs))}."
      )
    else:
      array = coerce_input(name, inputs[name], shape, dtype)
    tb = cm.create_input_buffer_by_name(sig_key, name)
    tb.write(array)
    fed[name] = array
    buffers[name] = tb

  out_names = signatures[sig_key]["outputs"]
  out_buffers = cm.create_output_buffers(cm.get_signature_index(sig_key))
  outputs_by_name = dict(zip(out_names, out_buffers))
  cm.run_by_name(sig_key, buffers, outputs_by_name)

  output_details = cm.get_output_tensor_details(sig_key)
  outputs: dict[str, np.ndarray] = {}
  for name, tb in outputs_by_name.items():
    details = output_details.get(name, {})
    shape = details.get("shape")
    if shape is None and hasattr(tb, "get_tensor_details"):
      shape = tb.get_tensor_details().get("shape")
    dtype = np_dtype(details.get("dtype", "unknown"), tensor=name)
    # A rank-0 tensor has shape [] and one element; keep it a scalar.
    count = int(np.prod(shape)) if shape is not None else 1
    array = tb.read(count, dtype)
    outputs[name] = array.reshape(shape) if shape is not None else array
  return sig_key, fed, outputs


def compare_outputs(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
  """Compares each expected output with the actual one (numpy.isclose)."""
  records = []
  for name in sorted(set(expected) | set(actual)):
    record: dict[str, Any] = {"name": name}
    if name not in expected:
      record.update(passed=False, reason="not in reference")
    elif name not in actual:
      record.update(passed=False, reason="missing from model outputs")
    else:
      got, want = np.asarray(actual[name]), np.asarray(expected[name])
      record["shape"] = list(got.shape)
      record["dtype"] = str(got.dtype)
      if got.shape != want.shape:
        record.update(
            passed=False,
            reason=f"shape {list(got.shape)} != reference {list(want.shape)}",
        )
      elif np.issubdtype(got.dtype, np.number) and np.issubdtype(
          want.dtype, np.number
      ):
        g = got.astype(np.float64, copy=False)
        w = want.astype(np.float64, copy=False)
        close = np.isclose(g, w, atol=atol, rtol=rtol, equal_nan=True)
        # Matching nan/inf pairs must not poison the stats; a nan or inf at
        # a mismatched position is kept, so the stats are non-finite exactly
        # when a mismatch is (and `--json` writes them as null).
        diff = np.where(close, 0.0, np.abs(g - w))
        mismatched = int(np.count_nonzero(~close))
        record.update(
            max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
            mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
            mismatched=mismatched,
            total=int(diff.size),
            passed=mismatched == 0,
        )
        if mismatched:
          record["reason"] = (
              f"{mismatched}/{diff.size} elements outside atol={atol},"
              f" rtol={rtol}"
          )
      else:
        equal = bool(np.array_equal(got, want))
        record.update(passed=equal)
        if not equal:
          record["reason"] = "values differ"
    records.append(record)
  return records


def _load_npz(path: str) -> dict[str, np.ndarray]:
  with np.load(path) as f:
    return {k: f[k] for k in f.files}


@report.supports_json
@click.command(
    "verify",
    help=textwrap.dedent("""\
        Check a model's outputs against a reference.

        MODEL: Path to the LiteRT model (.tflite) or a Model Reference.

        The reference is either a pair of .npz files keyed by signature
        input/output names (--inputs/--expected), or another LiteRT model
        (--against) that is run on the same inputs (--inputs, or seeded
        random inputs). Both models must take inputs of the same dtype;
        comparing a fully quantized int8 model against its float original is
        not supported yet. Exits with code 5 when any output is outside
        tolerance.

        Examples:

          $ litert verify authored.tflite --inputs ref/inputs.npz --expected ref/outputs.npz

          $ litert verify model.tflite --against model.tflite --gpu --save ref/

          $ litert verify model_dr.tflite --against model.tflite --atol 0.05
        """),
)
@click.argument("model", type=str)
@click.option(
    "--inputs",
    "inputs_path",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Inputs to feed (.npz keyed by input name). Required with --expected;"
        " optional with --against."
    ),
)
@click.option(
    "--expected",
    "expected_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Reference outputs (.npz keyed by output name).",
)
@click.option(
    "--against",
    "against",
    type=str,
    help="Reference LiteRT model to run on the same inputs (always on CPU).",
)
@click.option(
    "--save",
    "save_dir",
    type=click.Path(file_okay=False),
    help=(
        "Write the inputs and reference outputs used to DIR as inputs.npz /"
        " outputs.npz."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for random inputs when --inputs is not given.",
)
@click.option(
    "--atol",
    type=float,
    default=1e-5,
    show_default=True,
    help="Absolute tolerance (numpy.isclose semantics).",
)
@click.option(
    "--rtol",
    type=float,
    default=1e-4,
    show_default=True,
    help="Relative tolerance (numpy.isclose semantics).",
)
@click.option(
    "--accelerator",
    type=str,
    help="Comma-separated accelerators for MODEL (e.g. gpu,cpu).",
)
@click.option("--cpu", is_flag=True, help="Run MODEL on CPU.")
@click.option("--gpu", is_flag=True, help="Run MODEL on GPU.")
@click.option("--npu", is_flag=True, help="Run MODEL on NPU.")
@click.option(
    "--signature-index",
    type=int,
    default=0,
    help="Signature to run (in both models). Default is 0.",
)
def verify_cmd(
    model: str,
    inputs_path: str | None,
    expected_path: str | None,
    against: str | None,
    save_dir: str | None,
    seed: int,
    atol: float,
    rtol: float,
    accelerator: str | None,
    cpu: bool,
    gpu: bool,
    npu: bool,
    signature_index: int,
) -> None:
  """Verifies MODEL against reference outputs."""
  reporter = report.get()
  if bool(expected_path) == bool(against):
    raise click.UsageError(
        "Give exactly one reference: --expected (with --inputs) or --against."
    )
  if expected_path and not inputs_path:
    raise click.UsageError("--expected requires --inputs.")

  from litert_cli.core import models as core_models  # pylint: disable=g-import-not-at-top

  def resolve(ref: str) -> str:
    path, _ = core_models.resolve_model_reference(ref)
    if str(path) != ref:
      reporter.echo(f"Resolved model '{ref}' to '{path}'")
    if not os.path.isfile(path):
      raise click.ClickException(
          f"Model file not found: {path} (not a local model file or a model"
          " reference; see `litert list`)."
      )
    return str(path)

  model_path = resolve(model)
  accelerators = _accelerators(accelerator, cpu, gpu, npu)

  try:
    if against:
      against_path = resolve(against)
      given_inputs = _load_npz(inputs_path) if inputs_path else None
      source = f"inputs from {inputs_path}" if inputs_path else f"seed {seed}"
      reporter.echo(f"Running reference {against_path} on CPU ({source})...")
      _, ref_inputs, expected = run_model(
          against_path, ["cpu"], signature_index, given_inputs, seed=seed
      )
      reference_desc = against_path
    else:
      ref_inputs = _load_npz(inputs_path)
      expected = _load_npz(expected_path)
      reference_desc = expected_path

    reporter.echo(
        f"Running {model_path} on {','.join(accelerators)} with the reference"
        " inputs..."
    )
    sig_key, _, actual = run_model(
        model_path, accelerators, signature_index, ref_inputs, seed=seed
    )
  except click.ClickException:
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(f"Inference failed: {e!r}") from e

  saved = None
  if save_dir:
    # Save the reference's own inputs: they are what produced `expected`.
    out = pathlib.Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / INPUTS_FILE, **ref_inputs)
    np.savez(out / OUTPUTS_FILE, **expected)
    saved = {
        "inputs": str(out / INPUTS_FILE),
        "outputs": str(out / OUTPUTS_FILE),
    }
    reporter.echo(f"Saved reference inputs/outputs to {out}")

  records = compare_outputs(actual, expected, atol=atol, rtol=rtol)
  passed = all(r["passed"] for r in records)

  reporter.echo(
      f"Verification against {reference_desc} (atol={atol}, rtol={rtol}):"
  )
  for r in records:
    status = "OK  " if r["passed"] else "FAIL"
    detail = ""
    if "max_abs_diff" in r:
      detail = (
          f" max_abs_diff={r['max_abs_diff']:.3e}"
          f" mean_abs_diff={r['mean_abs_diff']:.3e}"
      )
    if not r["passed"]:
      detail += f" ({r['reason']})"
    reporter.echo(f"  {status} {r['name']}{detail}")

  document: dict[str, Any] = {
      "model": model_path,
      "reference": reference_desc,
      "signature": sig_key,
      "accelerators": accelerators,
      "tolerance": {"atol": atol, "rtol": rtol},
      "passed": passed,
      "outputs": records,
  }
  if saved:
    document["saved"] = saved
  reporter.emit(VERIFY_SCHEMA, document)

  if passed:
    reporter.secho("Verification passed.", fg="green")
  else:
    failed = [r["name"] for r in records if not r["passed"]]
    raise VerificationFailed(
        f"Verification failed for {len(failed)} output(s): {', '.join(failed)}."
    )
