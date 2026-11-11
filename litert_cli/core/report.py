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

"""Machine-readable output for LiteRT CLI commands.

The CLI has a root-level `--json` option (also `LITERT_JSON=1`). When it is set:

* stdout carries exactly one JSON document, written by `Reporter.emit`, and
* all human-oriented prose written through `Reporter.echo` / `Reporter.secho`
  goes to stderr instead of stdout.

This keeps `litert --json <command> | jq` composable for scripts and coding
agents without changing what humans see by default.

Commands opt in by decorating themselves with `@supports_json`, replacing
`click.echo` with `report.get().echo`, and calling `emit` exactly once, as
their final step, with their result. The root command rejects `--json` with a
usage error (exit code 2) for commands that have not opted in, so a consumer
never has to guess whether prose on stdout is "unsupported" or "broken".

The v1 output contract is:

* Exit code 0: stdout holds exactly one JSON document and nothing else.
* Nonzero exit code: stdout is empty. The error is reported on stderr as in
  text mode. Structured error documents are not part of v1; consumers must
  check the exit code before parsing stdout.
* Every document carries a `schema` field of the form `litert.<command>.v<N>`
  so consumers can detect incompatible changes. `schema` is reserved and may
  not appear in the document a command passes to `emit`.
* Documents are strict JSON: non-finite floats (`nan`, `inf`) are written as
  `null`, never as the bare `NaN` / `Infinity` tokens that `jq` and most JSON
  parsers reject.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from typing import Any

import click

# Key under which the root command stores the `Reporter` in `ctx.obj`.
REPORTER_KEY = "reporter"

# Attribute set on `click.Command` objects that honor `--json`.
_SUPPORTS_JSON_ATTR = "litert_supports_json"

# Document key reserved for the schema identifier.
SCHEMA_KEY = "schema"


def _normalize(value: Any) -> Any:
  """Converts `value` into plain JSON types, replacing non-finite floats.

  Handles the containers and leaf types commands commonly put in a document
  (dataclasses, paths, sets, numpy scalars and arrays). Anything else is
  returned unchanged so that `json.dumps` raises `TypeError` for it.

  Args:
    value: An arbitrary value from a result document.

  Returns:
    A value made only of dicts, lists, strings, finite numbers, booleans and
    None, or the original value if it is not a known type.
  """
  if isinstance(value, bool) or value is None:
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, (int, str)):
    return value
  if isinstance(value, dict):
    return {key: _normalize(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_normalize(item) for item in value]
  if isinstance(value, (set, frozenset)):
    return [_normalize(item) for item in sorted(value)]
  if isinstance(value, pathlib.PurePath):
    return str(value)
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return _normalize(dataclasses.asdict(value))
  if hasattr(value, "tolist"):  # numpy scalars and arrays
    return _normalize(value.tolist())
  return value


def dumps(document: dict[str, Any]) -> str:
  """Serializes a document as strict, indented JSON.

  Args:
    document: The mapping to serialize.

  Returns:
    The JSON text, without a trailing newline.

  Raises:
    TypeError: If the document contains a value that cannot be serialized.
    ValueError: If a non-finite float survived normalization. This indicates
      a bug in `_normalize` rather than in the calling command.
  """
  return json.dumps(_normalize(document), indent=2, allow_nan=False)


class Reporter:
  """Routes command output according to the active output mode.

  One `Reporter` lives in the root command's `ctx.obj` for the whole
  invocation, so it can enforce that at most one document is emitted.

  Attributes:
    json_mode: Whether `--json` is active for this invocation.
    emitted: Whether `emit` has already written a document.
  """

  def __init__(self, json_mode: bool = False):
    self.json_mode = json_mode
    self.emitted = False

  def echo(self, message: Any = None, **kwargs: Any) -> None:
    """Drop-in for `click.echo` that keeps stdout clean in JSON mode."""
    if self.json_mode:
      kwargs["err"] = True
    click.echo(message, **kwargs)

  def secho(self, message: Any = None, **kwargs: Any) -> None:
    """Drop-in for `click.secho` that keeps stdout clean in JSON mode."""
    if self.json_mode:
      kwargs["err"] = True
    click.secho(message, **kwargs)

  def emit(self, schema: str, document: dict[str, Any]) -> None:
    """Writes the command's result document to stdout in JSON mode.

    In text mode nothing is written: commands keep rendering their
    human-readable output through `echo`, so adopting `emit` never changes the
    default UX. The single-document rule is still enforced in text mode so a
    double `emit` is caught by ordinary tests, not only under `--json`.

    Args:
      schema: Schema identifier, e.g. `"litert.list.v1"`.
      document: JSON-serializable mapping with the command's result. Must not
        contain a `schema` key.

    Raises:
      ValueError: If `document` contains the reserved `schema` key.
      RuntimeError: If a document has already been emitted in this
        invocation.
      TypeError: If `document` contains a value that cannot be serialized.
    """
    if SCHEMA_KEY in document:
      raise ValueError(f"`{SCHEMA_KEY}` is reserved in emitted documents.")
    if self.emitted:
      raise RuntimeError(
          "A JSON document was already emitted; stdout may hold only one."
      )
    if not self.json_mode:
      self.emitted = True
      return
    text = dumps({SCHEMA_KEY: schema, **document})
    self.emitted = True
    click.echo(text)


def install(ctx: click.Context, json_mode: bool) -> Reporter:
  """Creates the invocation's `Reporter` and stores it in `ctx.obj`.

  Called once by the root command. Tests that build their own root group can
  call it from their root callback to mirror `litert`'s wiring.

  Args:
    ctx: The root command's context.
    json_mode: Whether `--json` was passed.

  Returns:
    The installed `Reporter`.
  """
  ctx.ensure_object(dict)
  reporter = Reporter(json_mode=json_mode)
  ctx.obj[REPORTER_KEY] = reporter
  return reporter


def get(ctx: click.Context | None = None) -> Reporter:
  """Returns the `Reporter` for the current (or given) click context.

  Outside of a click context, or under a root command that has not called
  `install`, returns a text-mode `Reporter` so helpers stay usable from tests
  and scripts.

  Args:
    ctx: A click context; defaults to the current one.
  """
  if ctx is None:
    ctx = click.get_current_context(silent=True)
  if ctx is None:
    return Reporter()
  root_obj = ctx.find_root().obj
  if isinstance(root_obj, dict):
    reporter = root_obj.get(REPORTER_KEY)
    if isinstance(reporter, Reporter):
      return reporter
  return Reporter()


def is_json_mode(ctx: click.Context | None = None) -> bool:
  """Returns whether `--json` was passed to the root `litert` command."""
  return get(ctx).json_mode


def supports_json(command: click.Command) -> click.Command:
  """Decorator declaring that a command honors `--json`.

  Apply it outside `@click.command()` (i.e. above it) so it receives the
  `click.Command` object. The root command rejects `--json` for commands that
  are not marked.

  Args:
    command: The command to mark.

  Returns:
    The same command, marked.
  """
  setattr(command, _SUPPORTS_JSON_ATTR, True)
  return command


def command_supports_json(command: click.Command | None) -> bool:
  """Returns whether `command` was marked with `supports_json`."""
  return bool(getattr(command, _SUPPORTS_JSON_ATTR, False))


def check_json_supported(ctx: click.Context, command: click.Command) -> None:
  """Fails with a usage error if `--json` is set but `command` ignores it.

  Args:
    ctx: The root command's context, after its options were parsed.
    command: The resolved subcommand.

  Raises:
    click.UsageError: If JSON mode is requested and unsupported.
  """
  if ctx.params.get("output_json") and not command_supports_json(command):
    raise click.UsageError(
        f"'{command.name}' does not support --json yet. Run it without --json"
        " (or unset LITERT_JSON) to get its text output.",
        ctx=ctx,
    )
