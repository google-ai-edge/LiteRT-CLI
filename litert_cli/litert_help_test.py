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

from __future__ import annotations

import importlib.metadata
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from click import testing
from litert_cli import litert

# `lm` has no click --help of its own: it forwards --help to the native
# `litert-lm` binary, and first auto-installs the `lm` extra when it is missing,
# so it cannot be checked hermetically here.
_SKIPPED_COMMANDS = frozenset({"lm"})

# Every registered command, so newly added commands are covered automatically.
_COMMANDS = [
    cmd
    for cmd in litert.cli.list_commands(None)
    if cmd not in _SKIPPED_COMMANDS
]


class LitertHelpTest(parameterized.TestCase):

  def test_main_help(self):
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, ["--help"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("LiteRT CLI", result.output)

  def test_version(self):
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, ["--version"])
    self.assertEqual(result.exit_code, 0)
    self.assertStartsWith(result.output, "litert ")

  @parameterized.parameters(
      ("litert-cli-nightly", "1.2.3 (litert-cli-nightly)"),
      ("litert-cli", "1.2.3 (litert-cli)"),
      (None, "unknown (not installed as a package)"),
  )
  def test_cli_version_names_the_installed_package(self, installed, expected):
    def version(dist):
      if dist == installed:
        return "1.2.3"
      raise importlib.metadata.PackageNotFoundError(dist)

    with mock.patch.object(importlib.metadata, "version", side_effect=version):
      self.assertEqual(litert._cli_version(), expected)  # pylint: disable=protected-access

  def test_every_listed_command_resolves(self):
    for cmd in litert.cli.list_commands(None):
      with self.subTest(cmd=cmd):
        self.assertIsNotNone(litert.cli.get_command(None, cmd))

  @parameterized.named_parameters((cmd, cmd) for cmd in _COMMANDS)
  def test_subcommands_help(self, cmd):
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, [cmd, "--help"])
    self.assertEqual(
        result.exit_code,
        0,
        msg=f"Command '{cmd}' failed to load --help. Output: {result.output}",
    )
    self.assertIn("Usage:", result.output)


if __name__ == "__main__":
  absltest.main()
