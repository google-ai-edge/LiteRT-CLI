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

"""Shared helpers for running a LiteRT model on the desktop.

`run --desktop` and `verify` (and future `inspect`/`benchmark` JSON work) all
need the same handful of steps: parse the requested accelerators, load a
`CompiledModel`, pick a signature, feed numpy arrays into the input buffers,
run, and read the outputs back as numpy arrays with the right shape and
dtype. Keeping these here means a fix (scalar outputs, dtype handling, ...)
lands in every command at once.

The LiteRT runtime is imported lazily so that `import runtime` stays cheap for
CLI startup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

import click
from litert_cli.core import constants
import numpy as np

# Accelerators a user can name, in the order flags are applied.
ACCELERATORS = ("npu", "gpu", "cpu")

# LiteRT tensor type names (as reported by `CompiledModel`) that can be moved
# in and out of tensor buffers, and their numpy dtypes. `bfloat16` is added
# when `ml_dtypes` is importable.
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


def parse_accelerators(
    spec: str | None = None,
    *,
    cpu: bool = False,
    gpu: bool = False,
    npu: bool = False,
) -> list[str]:
  """Returns the accelerators to try, in fallback order.

  `spec` (a comma-separated `--accelerator` value) wins over the boolean
  flags; the flags are ordered npu, gpu, cpu. With neither, CPU is used.

  Args:
    spec: Comma-separated accelerator names, e.g. `"gpu,cpu"`.
    cpu: `--cpu`.
    gpu: `--gpu`.
    npu: `--npu`.

  Returns:
    A non-empty list of lower-case accelerator names.

  Raises:
    click.ClickException: On an unknown accelerator name.
  """
  if spec:
    chosen = [a.strip().lower() for a in spec.split(",") if a.strip()]
  else:
    flags = {"npu": npu, "gpu": gpu, "cpu": cpu}
    chosen = [a for a in ACCELERATORS if flags[a]]
  for name in chosen:
    if name not in ACCELERATORS:
      raise click.ClickException(
          f"Unsupported hardware accelerator: {name!r} (choose from"
          f" {', '.join(ACCELERATORS)})."
      )
  return chosen or ["cpu"]


def np_dtype(tensor_type: str, *, tensor: str) -> np.dtype:
  """Returns the numpy dtype for a LiteRT tensor type name.

  Args:
    tensor_type: Type name reported by `CompiledModel`, e.g. `"float32"`.
    tensor: Tensor name, for the error message.

  Raises:
    click.ClickException: If the type cannot be moved through a tensor buffer.
  """
  dtype = _NP_DTYPES.get(tensor_type.lower())
  if dtype is None:
    raise click.ClickException(
        f"Tensor {tensor!r} has dtype {tensor_type!r}, which the LiteRT CLI"
        " cannot read or write yet (supported:"
        f" {', '.join(sorted(_NP_DTYPES))})."
    )
  return dtype


def random_input(
    rng: np.random.Generator, shape: Sequence[int], dtype: np.dtype
) -> np.ndarray:
  """Random input: bools, integers in [0, 10), or floats in [-1, 1)."""
  if dtype == np.bool_:
    return rng.integers(0, 2, size=shape).astype(np.bool_)
  if np.issubdtype(dtype, np.integer):
    return rng.integers(0, 10, size=shape, dtype=dtype)
  return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def coerce_input(
    name: str, value: Any, shape: Sequence[int], dtype: np.dtype
) -> np.ndarray:
  """Checks a user-supplied input against the model's `shape` and `dtype`.

  Floats are cast between float widths (goldens are often float64). Any other
  dtype difference, in particular float vs. quantized int8/uint8, is an error:
  casting would silently produce a meaningless run.

  Args:
    name: Input name, for error messages.
    value: The value to feed.
    shape: Shape the model expects.
    dtype: Dtype the model expects.

  Returns:
    The input as a contiguous array of `dtype`.

  Raises:
    click.ClickException: On a shape or dtype mismatch.
  """
  array = np.asarray(value)
  same_kind_float = array.dtype.kind == "f" and dtype.kind == "f"
  if array.dtype != dtype and not same_kind_float:
    raise click.ClickException(
        f"Input {name!r} has dtype {array.dtype} but the model expects"
        f" {dtype}. Feeding a model inputs of a different dtype (e.g. float"
        " data to a fully quantized int8 model) is not supported yet."
    )
  array = array.astype(dtype, copy=False)
  if list(array.shape) != list(shape):
    raise click.ClickException(
        f"Input {name!r} has shape {list(array.shape)} but the model"
        f" expects {list(shape)}."
    )
  return array


def load_model(model_path: str, accelerators: Sequence[str]) -> Any:
  """Loads a `CompiledModel` for the given accelerators (fallback order).

  Args:
    model_path: Local path to the `.tflite` file.
    accelerators: Names from `parse_accelerators`; empty means CPU.

  Returns:
    The loaded `CompiledModel`.
  """
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
  for name in accelerators:
    if name not in flags:
      raise click.ClickException(f"Unsupported hardware accelerator: {name!r}")
    hw |= flags[name]
  if hw == HardwareAccelerator(0):
    hw = HardwareAccelerator.CPU

  env = None
  if constants.IS_INTERNAL_ENV:
    # In the internal environment, fall back to LD_LIBRARY_PATH for loading
    # GPU accelerators from a hermetic .par file.
    env = Environment.create(runtime_path="")
  return CompiledModel.from_file(model_path, hw, environment=env)


def pick_signature(cm: Any, model_path: str, index: int) -> tuple[dict, str]:
  """Returns `(signature_list, signature_key)` for `index`.

  Args:
    cm: The loaded `CompiledModel`.
    model_path: For error messages.
    index: `--signature-index`.

  Raises:
    click.ClickException: If the model has no signatures or `index` is out
      of range (which would otherwise abort the process in the native
      runtime).
  """
  signatures = cm.get_signature_list()
  if not signatures:
    raise click.ClickException(
        f"No signatures found in the model: {model_path!r}"
    )
  if not 0 <= index < len(signatures):
    raise click.ClickException(
        f"--signature-index {index} is out of range: the model has"
        f" {len(signatures)} signature(s)"
        f" ({', '.join(repr(k) for k in signatures)}), so use 0 to"
        f" {len(signatures) - 1}."
    )
  try:
    key = cm.get_signature_by_index(index)["key"]
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(
        f"Failed to get signature at index {index}: {e!r}"
    ) from e
  return signatures, key


def input_specs(cm: Any, sig_key: str) -> dict[str, tuple[list[int], np.dtype]]:
  """Returns `{input_name: (shape, dtype)}` for a signature."""
  specs = {}
  for name, details in cm.get_input_tensor_details(sig_key).items():
    shape = list(details.get("shape", [1]))
    specs[name] = (
        shape,
        np_dtype(details.get("dtype", "unknown"), tensor=name),
    )
  return specs


def write_inputs(
    cm: Any, sig_key: str, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
  """Creates and fills one input buffer per entry of `arrays`."""
  buffers = {}
  for name, array in arrays.items():
    tb = cm.create_input_buffer_by_name(sig_key, name)
    tb.write(array)
    buffers[name] = tb
  return buffers


def create_output_buffers(
    cm: Any, signatures: Mapping[str, Any], sig_key: str
) -> dict[str, Any]:
  """Returns `{output_name: TensorBuffer}` for a signature."""
  out_names = signatures[sig_key]["outputs"]
  out_buffers = cm.create_output_buffers(cm.get_signature_index(sig_key))
  return dict(zip(out_names, out_buffers))


def read_output(name: str, tb: Any, details: Mapping[str, Any]) -> np.ndarray:
  """Reads one output buffer as a numpy array of the tensor's shape and dtype.

  A rank-0 tensor (shape `[]`) is returned as a 0-d array, not as `(1,)`.

  Args:
    name: Output name, for error messages.
    tb: The output `TensorBuffer`.
    details: The entry for `name` from `cm.get_output_tensor_details`.
  """
  shape = details.get("shape")
  if shape is None and hasattr(tb, "get_tensor_details"):
    shape = tb.get_tensor_details().get("shape")
  dtype = np_dtype(details.get("dtype", "unknown"), tensor=name)
  count = int(np.prod(shape)) if shape is not None else 1
  array = tb.read(count, dtype)
  return array.reshape(shape) if shape is not None else array


def read_outputs(
    cm: Any, sig_key: str, outputs_by_name: Mapping[str, Any]
) -> dict[str, np.ndarray]:
  """Reads every output buffer; see `read_output`."""
  output_details = cm.get_output_tensor_details(sig_key)
  return {
      name: read_output(name, tb, output_details.get(name, {}))
      for name, tb in outputs_by_name.items()
  }


def run_once(
    model_path: str,
    accelerators: Sequence[str],
    signature_index: int,
    inputs: Mapping[str, np.ndarray] | None,
    *,
    seed: int = 0,
) -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Loads a model, runs one signature once; returns `(key, inputs, outputs)`.

  Args:
    model_path: Local path to the `.tflite` file.
    accelerators: Names from `parse_accelerators`.
    signature_index: `--signature-index`.
    inputs: Arrays keyed by input name. When None, seeded random inputs matching
      the signature are generated (see `random_input`).
    seed: Seed for the random inputs.

  Raises:
    click.ClickException: On a missing input, or a shape/dtype mismatch.
  """
  cm = load_model(model_path, accelerators)
  signatures, sig_key = pick_signature(cm, model_path, signature_index)
  rng = np.random.default_rng(seed)

  fed: dict[str, np.ndarray] = {}
  for name, (shape, dtype) in input_specs(cm, sig_key).items():
    if inputs is None:
      fed[name] = random_input(rng, shape, dtype)
    elif name not in inputs:
      raise click.ClickException(
          f"Inputs have no entry for model input {name!r}."
          f" Available: {', '.join(map(repr, inputs))}."
      )
    else:
      fed[name] = coerce_input(name, inputs[name], shape, dtype)

  buffers = write_inputs(cm, sig_key, fed)
  outputs_by_name = create_output_buffers(cm, signatures, sig_key)
  cm.run_by_name(sig_key, buffers, outputs_by_name)
  return sig_key, fed, read_outputs(cm, sig_key, outputs_by_name)



# Prefix of the signature output names given to exposed intermediate tensors.
_DUMP_OUTPUT_PREFIX = "__litert_dump_"

# `dump_intermediates` writes the edited model as a single FlatBuffer, which
# cannot exceed 2 GiB.
_MAX_DUMP_MODEL_BYTES = 2**31 - 1


def _text(value: Any) -> str:
  """FlatBuffer object-API strings are bytes; returns them as str."""
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def dump_intermediates(
    model_path: str,
    accelerators: Sequence[str],
    sig_key: str,
    inputs: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
  """Runs one signature with every intermediate tensor exposed as an output.

  The `.tflite` FlatBuffer is edited in memory: every tensor produced by an op
  of the signature's subgraph is appended to the subgraph outputs and to the
  signature's output map, the result is written to a temporary file and run
  through the same `CompiledModel` path as `run_once`, on the same
  accelerators. Because the dumped tensors are graph outputs the runtime
  cannot fuse across them, so values can differ slightly from the plain run.

  Cost: the whole model is loaded into memory and written again to a temporary
  directory, and every intermediate tensor is held in memory at once.

  Args:
    model_path: Local path to the `.tflite` file.
    accelerators: Names from `parse_accelerators`.
    sig_key: Signature key to run.
    inputs: Arrays keyed by input name, already coerced to the model's dtypes.

  Returns:
    `(arrays, records, skipped)`: `arrays` maps tensor name to value;
    `records` lists the same tensors in execution order with `index`, `name`,
    `shape`, `dtype`, the producing `op` (`INPUT` for signature inputs) and
    `op_index`; `skipped` lists the op outputs that are not in `arrays`
    (constants, element types the CLI cannot read back) with `index`, `name`,
    `op`, `op_index`, `dtype` and `reason`. Every op output is in exactly one
    of `records` and `skipped`.

  Raises:
    click.ClickException: If the model cannot be rewritten: it is larger than
      a FlatBuffer can be, or has no signature def named `sig_key` (the
      runtime synthesizes a signature for models without signature defs, but
      there is none in the file to attach the intermediate tensors to).
  """
  # pylint: disable=g-import-not-at-top
  import copy
  import tempfile
  from ai_edge_litert.tools import flatbuffer_utils
  # pylint: enable=g-import-not-at-top

  size = os.path.getsize(model_path)
  if size > _MAX_DUMP_MODEL_BYTES:
    raise click.ClickException(
        f"Cannot dump the intermediate tensors of {model_path!r} ({size}"
        " bytes): the model is rewritten as a single FlatBuffer, which is"
        " limited to 2 GiB."
    )
  model = flatbuffer_utils.read_model(model_path)
  match = next(
      (
          (i, s)
          for i, s in enumerate(model.signatureDefs or [])
          if _text(s.signatureKey) == sig_key
      ),
      None,
  )
  if match is None:
    raise click.ClickException(
        f"Cannot dump the intermediate tensors of {model_path!r}: the file"
        f" has no signature def {sig_key!r} to attach them to (models without"
        " signature defs are not supported)."
    )
  signature_index, signature = match
  signature.outputs = list(signature.outputs or [])
  subgraph = model.subgraphs[signature.subgraphIndex]
  tensors = subgraph.tensors
  subgraph_outputs = list(subgraph.outputs or [])
  output_name_by_index = {
      m.tensorIndex: _text(m.name) for m in signature.outputs
  }

  def is_constant(tensor: Any) -> bool:
    if not tensor.buffer:
      return False
    data = model.buffers[tensor.buffer].data
    return data is not None and len(data) > 0

  arrays: dict[str, np.ndarray] = {}
  records: list[dict[str, Any]] = []
  skipped: list[dict[str, Any]] = []

  def tensor_name(index: int) -> str:
    return _text(tensors[index].name or "") or f"tensor_{index}"

  def unique_name(index: int) -> str:
    name = tensor_name(index)
    return f"{name}#{index}" if name in arrays else name

  for m in signature.inputs or []:
    name = _text(m.name)
    if name in inputs:
      value = np.asarray(inputs[name])
      record = {
          "index": int(m.tensorIndex),
          "name": unique_name(m.tensorIndex),
          "shape": list(value.shape),
          "dtype": str(value.dtype),
          "op": "INPUT",
      }
      arrays[record["name"]] = value
      records.append(record)

  # (signature output name, record) for every op output to read back.
  wanted: list[tuple[str, dict[str, Any]]] = []
  extra: list[tuple[str, int]] = []
  for op_index, op in enumerate(subgraph.operators or []):
    op_name = (
        flatbuffer_utils.opcode_to_name(model, op.opcodeIndex) or "UNKNOWN"
    )
    for index in op.outputs or []:
      if index < 0:
        continue  # Optional output that the op does not produce.
      index = int(index)
      tensor = tensors[index]
      dtype_name = flatbuffer_utils.type_to_name(tensor.type).lower()
      reason = None
      if is_constant(tensor):
        reason = "constant tensor"
      elif dtype_name not in _NP_DTYPES:
        reason = f"dtype {dtype_name} cannot be read back by the LiteRT CLI"
      elif index in subgraph_outputs and index not in output_name_by_index:
        reason = "subgraph output that the signature does not expose"
      if reason:
        skipped.append({
            "index": index,
            "name": tensor_name(index),
            "op_index": op_index,
            "op": op_name,
            "dtype": dtype_name,
            "reason": reason,
        })
        continue
      record = {
          "index": index,
          "name": unique_name(index),
          "op_index": op_index,
          "op": op_name,
      }
      arrays[record["name"]] = np.empty(0)  # Reserve the name.
      if index in output_name_by_index:
        wanted.append((output_name_by_index[index], record))
      else:
        extra.append((f"{_DUMP_OUTPUT_PREFIX}{index}", index))
        wanted.append((extra[-1][0], record))

  if extra:
    # The schema classes are not imported here: a new signature output entry
    # is a copy of an existing one with its name and tensor replaced.
    template = next(
        iter(signature.outputs + list(signature.inputs or [])), None
    )
    if template is None:
      raise click.ClickException(
          f"Cannot dump the intermediate tensors of {model_path!r}: signature"
          f" {sig_key!r} has neither inputs nor outputs."
      )
    subgraph.outputs = subgraph_outputs + [index for _, index in extra]
    for key, index in extra:
      entry = copy.copy(template)
      entry.name = (
          key.encode("utf-8") if isinstance(template.name, bytes) else key
      )
      entry.tensorIndex = index
      signature.outputs.append(entry)

  with tempfile.TemporaryDirectory(prefix="litert_dump_") as tmp:
    path = os.path.join(tmp, "model.tflite")
    flatbuffer_utils.write_model(model, path)
    _, _, outputs = run_once(path, accelerators, signature_index, inputs)

  for key, record in wanted:
    value = outputs[key]
    arrays[record["name"]] = value
    record["shape"] = list(value.shape)
    record["dtype"] = str(value.dtype)
    records.append(record)
  return arrays, records, skipped
