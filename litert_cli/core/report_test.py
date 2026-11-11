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

"""Tests for the `--json` output contract and the `Reporter` helper."""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import pathlib
import types
from unittest import mock

from absl.testing import absltest
import click
from click import testing
from litert_cli import litert
from litert_cli.core import report


def _runner() -> testing.CliRunner:
  """A runner that keeps stdout and stderr apart across click versions."""
  try:
    return testing.CliRunner(mix_stderr=False)  # click < 8.2
  except TypeError:
    return testing.CliRunner()  # click >= 8.2 never mixes them


def _probe_group() -> click.Group:
  """A root group mirroring `litert`'s `--json` wiring, with a probe command."""

  @click.group()
  @click.option("--json", "output_json", is_flag=True, envvar="LITERT_JSON")
  @click.pass_context
  def root(ctx: click.Context, output_json: bool) -> None:
    report.install(ctx, json_mode=output_json)

  @report.supports_json
  @root.command()
  @click.pass_context
  def probe(ctx: click.Context) -> None:
    reporter = report.get(ctx)
    reporter.echo("prose")
    reporter.secho("styled prose", fg="green")
    reporter.emit(
        "litert.probe.v1",
        {"path": pathlib.Path("/tmp/model.tflite"), "tags": {"b", "a"}},
    )

  @report.supports_json
  @root.command("probe-twice")
  @click.pass_context
  def probe_twice(ctx: click.Context) -> None:
    reporter = report.get(ctx)
    reporter.emit("litert.probe.v1", {"n": 1})
    reporter.emit("litert.probe.v1", {"n": 2})

  return root


def _json_reporter() -> report.Reporter:
  return report.Reporter(json_mode=True)


class ReporterTest(absltest.TestCase):

  def test_text_mode_prints_prose_to_stdout_and_emits_nothing(self) -> None:
    result = _runner().invoke(_probe_group(), ["probe"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(result.stdout, "prose\nstyled prose\n")
    self.assertEqual(result.stderr, "")

  def test_json_mode_keeps_stdout_to_a_single_document(self) -> None:
    result = _runner().invoke(_probe_group(), ["--json", "probe"])

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(
        json.loads(result.stdout),
        {
            "schema": "litert.probe.v1",
            "path": "/tmp/model.tflite",
            "tags": ["a", "b"],
        },
    )
    self.assertEqual(result.stderr, "prose\nstyled prose\n")

  def test_json_mode_can_be_enabled_through_the_environment(self) -> None:
    result = _runner().invoke(
        _probe_group(), ["probe"], env={"LITERT_JSON": "1"}
    )

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(json.loads(result.stdout)["schema"], "litert.probe.v1")

  def test_get_returns_the_same_reporter_for_the_whole_invocation(
      self,
  ) -> None:
    seen: list[report.Reporter] = []

    @click.group()
    @click.pass_context
    def root(ctx: click.Context) -> None:
      seen.append(report.install(ctx, json_mode=True))

    @root.command()
    @click.pass_context
    def probe(ctx: click.Context) -> None:
      seen.append(report.get(ctx))
      seen.append(report.get())

    self.assertEqual(_runner().invoke(root, ["probe"]).exit_code, 0)
    self.assertLen(seen, 3)
    self.assertIs(seen[0], seen[1])
    self.assertIs(seen[0], seen[2])

  def test_get_outside_of_a_click_context_defaults_to_text_mode(self) -> None:
    self.assertFalse(report.get().json_mode)
    self.assertFalse(report.is_json_mode())

  def test_emit_rejects_unserializable_values(self) -> None:
    with self.assertRaises(TypeError):
      _json_reporter().emit("litert.x.v1", {"value": object()})

  def test_emit_rejects_the_reserved_schema_key(self) -> None:
    for json_mode in (False, True):
      with self.subTest(json_mode=json_mode):
        reporter = report.Reporter(json_mode=json_mode)
        with self.assertRaisesRegex(ValueError, "reserved"):
          reporter.emit("litert.x.v1", {"schema": "litert.other.v9"})

  def test_emit_more_than_once_fails(self) -> None:
    result = _runner().invoke(_probe_group(), ["--json", "probe-twice"])

    self.assertNotEqual(result.exit_code, 0)
    self.assertIsInstance(result.exception, RuntimeError)
    self.assertEqual(
        json.loads(result.stdout), {"schema": "litert.probe.v1", "n": 1}
    )

  def test_emit_more_than_once_fails_in_text_mode_too(self) -> None:
    result = _runner().invoke(_probe_group(), ["probe-twice"])

    self.assertNotEqual(result.exit_code, 0)
    self.assertIsInstance(result.exception, RuntimeError)

  def test_dumps_writes_non_finite_floats_as_null(self) -> None:
    @dataclasses.dataclass
    class Stats:
      mean: float
      samples: tuple[float, ...]

    document = {
        "nan": math.nan,
        "inf": math.inf,
        "neg_inf": -math.inf,
        "finite": 1.5,
        "nested": {"values": [1.0, math.nan, (math.inf, 2)]},
        "stats": Stats(mean=math.nan, samples=(0.5, math.inf)),
        "flags": [True, False, None],
    }

    self.assertEqual(
        json.loads(report.dumps(document)),
        {
            "nan": None,
            "inf": None,
            "neg_inf": None,
            "finite": 1.5,
            "nested": {"values": [1.0, None, [None, 2]]},
            "stats": {"mean": None, "samples": [0.5, None]},
            "flags": [True, False, None],
        },
    )

  def test_dumps_handles_numpy_values(self) -> None:
    try:
      import numpy as np  # pylint: disable=g-import-not-at-top
    except ImportError:
      self.skipTest("numpy is not available")

    document = {
        "scalar": np.float32("nan"),
        "array": np.array([1.0, np.inf, 3.0]),
        "count": np.int64(4),
    }

    self.assertEqual(
        json.loads(report.dumps(document)),
        {"scalar": None, "array": [1.0, None, 3.0], "count": 4},
    )

  def test_dumps_output_is_strict_json(self) -> None:
    text = report.dumps({"x": math.nan, "y": [math.inf]})

    self.assertNotIn("NaN", text)
    self.assertNotIn("Infinity", text)


class LitertJsonOptionTest(absltest.TestCase):
  """`litert --json` reaches lazily loaded subcommands."""

  def _install_fake_list(
      self, mock_import_module, command: click.Command
  ) -> None:
    mock_mod = mock.create_autospec(types.ModuleType)
    mock_mod.list_cmd = command
    mock_import_module.return_value = mock_mod

  @mock.patch.object(importlib, "import_module", autospec=True)
  def test_root_json_option_is_visible_to_subcommands(
      self, mock_import_module
  ) -> None:
    seen: list[bool] = []

    @report.supports_json
    @click.command("list")
    @click.pass_context
    def fake_list(ctx: click.Context) -> None:
      seen.append(report.is_json_mode(ctx))

    self._install_fake_list(mock_import_module, fake_list)

    runner = _runner()
    self.assertEqual(runner.invoke(litert.cli, ["list"]).exit_code, 0)
    self.assertEqual(runner.invoke(litert.cli, ["--json", "list"]).exit_code, 0)

    self.assertEqual(seen, [False, True])

  @mock.patch.object(importlib, "import_module", autospec=True)
  def test_json_is_a_usage_error_for_commands_that_do_not_support_it(
      self, mock_import_module
  ) -> None:
    invoked: list[bool] = []

    @click.command("list")
    def fake_list() -> None:
      invoked.append(True)
      click.echo("prose that would corrupt stdout")

    self._install_fake_list(mock_import_module, fake_list)

    runner = _runner()
    self.assertEqual(runner.invoke(litert.cli, ["list"]).exit_code, 0)

    result = runner.invoke(litert.cli, ["--json", "list"])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertEqual(result.stdout, "")
    self.assertIn("'list' does not support --json yet", result.stderr)
    self.assertLen(invoked, 1)

  def test_json_option_is_documented_in_help(self) -> None:
    result = _runner().invoke(litert.cli, ["--help"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("--json", result.output)

  def test_help_and_version_still_work_under_json(self) -> None:
    runner = _runner()
    self.assertEqual(
        runner.invoke(litert.cli, ["--json", "--help"]).exit_code, 0
    )
    self.assertEqual(
        runner.invoke(litert.cli, ["--json", "--version"]).exit_code, 0
    )


if __name__ == "__main__":
  absltest.main()
