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

import importlib
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import click
from click import testing
from litert_cli import litert


class LitertTest(parameterized.TestCase):

  def test_lazy_cli_list_commands(self) -> None:
    cli_group = litert.LazyCLI()
    commands = cli_group.list_commands(None)
    self.assertEqual(
        commands,
        [
            'benchmark',
            'clean',
            'compile',
            'convert',
            'delete',
            'download',
            'import',
            'list',
            'lm',
            'quantize',
            'run',
            'visualize',
        ],
    )

  @mock.patch.object(importlib, 'import_module', autospec=True)
  def test_lazy_cli_get_command(self, mock_import_module) -> None:
    # Mock the module and command object
    mock_mod = mock.create_autospec(types.ModuleType)
    mock_cmd = mock.create_autospec(click.Command)
    mock_mod.download_cmd = mock_cmd
    mock_import_module.return_value = mock_mod

    cli_group = litert.LazyCLI()
    cmd = cli_group.get_command(None, 'download')

    self.assertEqual(cmd, mock_cmd)
    mock_import_module.assert_called_once_with('litert_cli.commands.download')

  def test_lazy_cli_get_command_not_found(self) -> None:
    cli_group = litert.LazyCLI()
    cmd = cli_group.get_command(None, 'unknown_command')
    self.assertIsNone(cmd)

  @parameterized.parameters(
      'LiteRT CLI.',
      'download',
      'visualize',
      'clean',
      'compile',
      'convert',
      'lm',
      'quantize',
      'run',
  )
  def test_cli_help(self, expected_string) -> None:
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, ['--help'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn(expected_string, result.output)


if __name__ == '__main__':
  absltest.main()
