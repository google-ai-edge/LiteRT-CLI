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
  previous `verify --against ... --save DIR`). The reference may cover only
  some of the model's outputs; the others are reported as skipped.
* `--against reference.tflite`: another LiteRT model; `verify` feeds both
  models the same inputs (`--inputs`, or seeded random inputs) and compares
  their outputs.

`--save DIR` keeps the inputs (`inputs.npz`), the reference outputs
(`outputs.npz`) and the verified model's own outputs (`actual.npz`), so one
step's outputs can become the next step's inputs (e.g. prefill -> decode).

`--dump-intermediates DIR` additionally runs the model with every intermediate
tensor exposed as a graph output and writes them all (`intermediates.npz`,
keyed by tensor name, plus an `intermediates.json` index in execution order
with the producing op), so a failing output can be traced back to the first op
that diverges from your own reference of the graph. The dump holds every
activation uncompressed, in memory and then in DIR, and needs a model that
fits in a 2 GiB FlatBuffer and has signature defs. It runs after the
comparison and a failure is only a warning (`intermediates.error` in the JSON
document): the verification result and exit code are not affected.

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
import json
import os
import pathlib
import textwrap
from typing import Any

import click
from litert_cli.core import report
from litert_cli.core import runtime
import numpy as np

# Schema of the document `litert --json verify` emits.
VERIFY_SCHEMA = "litert.verify.v1"

# Exit code when the model ran but its outputs do not match the reference.
VERIFICATION_FAILED_EXIT_CODE = 5

INPUTS_FILE = "inputs.npz"
OUTPUTS_FILE = "outputs.npz"
ACTUAL_FILE = "actual.npz"
INTERMEDIATES_FILE = "intermediates.npz"
INTERMEDIATES_INDEX_FILE = "intermediates.json"


class VerificationFailed(click.ClickException):
  """The model ran, but at least one output differs from the reference."""

  exit_code = VERIFICATION_FAILED_EXIT_CODE


def run_model(
    model_path: str,
    accelerators: list[str],
    signature_index: int,
    inputs: Mapping[str, np.ndarray] | None,
    *,
    seed: int,
) -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Runs one signature; returns `(signature_key, inputs, outputs)`.

  See `runtime.run_once`.
  """
  return runtime.run_once(
      model_path, accelerators, signature_index, inputs, seed=seed
  )



def compare_outputs(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
  """Compares each expected output with the actual one (numpy.isclose).

  Model outputs that the reference does not contain are reported as skipped
  (`passed` is true, `skipped` is true) rather than failed, so a reference may
  cover only the outputs that matter (e.g. logits but not KV-cache tensors).
  Expected outputs that the model does not produce are failures.
  """
  records = []
  for name in sorted(set(expected) | set(actual)):
    record: dict[str, Any] = {"name": name}
    if name not in expected:
      got = np.asarray(actual[name])
      record.update(
          shape=list(got.shape),
          dtype=str(got.dtype),
          passed=True,
          skipped=True,
          reason="not in reference",
      )
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
        # Report the true distance, including for elements within tolerance,
        # so a passing output still shows its margin. Only matching nan/inf
        # pairs (whose difference is nan) are zeroed so they do not poison
        # the stats; a nan or inf at a mismatched position is kept, so the
        # stats are non-finite exactly when a mismatch is (and `--json`
        # writes them as null).
        diff = np.abs(g - w)
        diff = np.where(close & ~np.isfinite(diff), 0.0, diff)
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


def _dump_intermediates(
    reporter: report.Reporter,
    dump_dir: str,
    model_path: str,
    accelerators: list[str],
    sig_key: str,
    inputs: Mapping[str, np.ndarray],
) -> dict[str, Any]:
  """Writes MODEL's intermediate tensors to `dump_dir`.

  See `runtime.dump_intermediates`. A failure does not raise: it is printed as
  a warning and returned as `{"error": ...}`, so that the verification result
  is still reported.

  Args:
    reporter: Where to print progress.
    dump_dir: `--dump-intermediates`.
    model_path: The verified model.
    accelerators: The accelerators it was verified on.
    sig_key: The signature that was verified.
    inputs: The inputs it was fed.

  Returns:
    The `intermediates` entry of the verify document.
  """
  reporter.echo(
      f"Running {model_path} again on {','.join(accelerators)} with every"
      " intermediate tensor exposed as an output..."
  )
  out = pathlib.Path(dump_dir)
  npz_path = out / INTERMEDIATES_FILE
  index_path = out / INTERMEDIATES_INDEX_FILE
  try:
    arrays, records, skipped = runtime.dump_intermediates(
        model_path, accelerators, sig_key, inputs
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    index_doc = {
        "model": model_path,
        "signature": sig_key,
        "accelerators": accelerators,
        "note": (
            "Computed with every intermediate exposed as a graph output, which"
            " prevents fusion across them; values can differ slightly from"
            " the plain run."
        ),
        "tensors": records,
        "skipped": skipped,
    }
    index_path.write_text(json.dumps(index_doc, indent=2) + "\n")
    written = npz_path.stat().st_size + index_path.stat().st_size
  except click.ClickException as e:
    error = e.format_message()
  except Exception as e:  # pylint: disable=broad-exception-caught
    error = repr(e)
  else:
    message = (
        f"Dumped {len(records)} intermediate tensors"
        f" ({written / 2**20:.1f} MiB) to {out}"
    )
    if skipped:
      message += (
          f"; {len(skipped)} could not be dumped (see `skipped` in"
          f" {INTERMEDIATES_INDEX_FILE})"
      )
    reporter.echo(message)
    return {
        "path": str(npz_path),
        "index": str(index_path),
        "count": len(records),
        "skipped": len(skipped),
        "bytes": written,
    }
  reporter.secho(
      f"Warning: could not dump the intermediate tensors: {error}", fg="yellow"
  )
  return {"error": error}


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
    help=(
        "Reference outputs (.npz keyed by output name). May cover only some"
        " of the model's outputs; the rest are skipped."
    ),
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
        "Write the inputs, reference outputs and MODEL's outputs to DIR as"
        " inputs.npz / outputs.npz / actual.npz."
    ),
)
@click.option(
    "--dump-intermediates",
    "dump_dir",
    type=click.Path(file_okay=False),
    help=(
        "Also run MODEL with every intermediate tensor exposed as an output"
        " (same accelerators) and write them to DIR as intermediates.npz"
        " (keyed by tensor name) plus intermediates.json (execution order,"
        " shapes, producing op, tensors that could not be dumped). Use it to"
        " bisect a mismatch against your own reference of the graph. Cost: the"
        " model is loaded into memory and rewritten to a temporary file (it"
        " must fit in a 2 GiB FlatBuffer and have signature defs), and every"
        " activation is held in memory and written to DIR uncompressed. A"
        " failed dump is a warning: it does not change the verification"
        " result or the exit code."
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
    dump_dir: str | None,
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
  accelerators = runtime.parse_accelerators(
      accelerator, cpu=cpu, gpu=gpu, npu=npu
  )

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
    sig_key, fed_inputs, actual = run_model(
        model_path, accelerators, signature_index, ref_inputs, seed=seed
    )
  except click.ClickException:
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise click.ClickException(f"Inference failed: {e!r}") from e

  saved = None
  if save_dir:
    # Save the reference's own inputs: they are what produced `expected`.
    # MODEL's outputs go next to them so a later step can consume them (for
    # example a decode signature fed with the KV cache a prefill run emitted).
    out = pathlib.Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / INPUTS_FILE, **ref_inputs)
    np.savez(out / OUTPUTS_FILE, **expected)
    np.savez(out / ACTUAL_FILE, **actual)
    saved = {
        "inputs": str(out / INPUTS_FILE),
        "outputs": str(out / OUTPUTS_FILE),
        "actual": str(out / ACTUAL_FILE),
    }
    reporter.echo(f"Saved reference inputs/outputs and model outputs to {out}")

  if not set(actual) & set(expected):
    raise click.ClickException(
        f"Reference {reference_desc} has none of the model's outputs"
        f" ({', '.join(sorted(actual))}); nothing to verify."
    )
  records = compare_outputs(actual, expected, atol=atol, rtol=rtol)
  passed = all(r["passed"] for r in records)

  reporter.echo(
      f"Verification against {reference_desc} (atol={atol}, rtol={rtol}):"
  )
  for r in records:
    if r.get("skipped"):
      status = "SKIP"
    else:
      status = "OK  " if r["passed"] else "FAIL"
    detail = ""
    if "max_abs_diff" in r:
      detail = (
          f" max_abs_diff={r['max_abs_diff']:.3e}"
          f" mean_abs_diff={r['mean_abs_diff']:.3e}"
      )
    if not r["passed"] or r.get("skipped"):
      detail += f" ({r['reason']})"
    reporter.echo(f"  {status} {r['name']}{detail}")

  # After the comparison, and never fatal: the dump is a debugging aid and
  # must not hide the verification result or change the exit code.
  intermediates = None
  if dump_dir:
    intermediates = _dump_intermediates(
        reporter, dump_dir, model_path, accelerators, sig_key, fed_inputs
    )

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
  if intermediates:
    document["intermediates"] = intermediates
  reporter.emit(VERIFY_SCHEMA, document)

  if passed:
    reporter.secho("Verification passed.", fg="green")
  else:
    failed = [r["name"] for r in records if not r["passed"]]
    raise VerificationFailed(
        f"Verification failed for {len(failed)} output(s): {', '.join(failed)}."
    )
