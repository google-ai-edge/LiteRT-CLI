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
import os
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import click
from click import testing
from litert_cli import litert
from litert_cli.core import constants


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

  def test_run_reports_a_missing_model_file(self) -> None:
    runner = testing.CliRunner()
    result = runner.invoke(
        litert.cli, ['run', 'no_such_model.tflite', '--desktop', '--cpu']
    )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn('Model file not found: no_such_model.tflite', result.output)

  def test_run_rejects_a_directory(self) -> None:
    model_dir = self.create_tempdir().full_path
    result = testing.CliRunner().invoke(litert.cli, ['run', model_dir])
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn(f'Model file not found: {model_dir}', result.output)


class ModelCacheCommandsTest(parameterized.TestCase):
  """`delete`, `import` and `clean` must stay inside the CLI cache."""

  def setUp(self):
    super().setUp()
    self.cache = self.create_tempdir('cache').full_path
    self.models = self.create_tempdir('cache/models').full_path
    self.enter_context(
        mock.patch.object(constants, 'LITERT_CLI_CACHE_DIR', self.cache)
    )
    self.enter_context(
        mock.patch.object(constants, 'LITERT_MODELS_CACHE_DIR', self.models)
    )
    self.enter_context(
        mock.patch.object(
            constants, 'LITERT_CLI_ROOT', os.path.join(self.cache, 'root')
        )
    )

  @parameterized.parameters('..', '.', '', 'a\\b')
  def test_delete_rejects_refs_outside_the_models_cache(self, ref) -> None:
    result = testing.CliRunner().invoke(litert.cli, ['delete', ref, '-y'])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('not a valid model reference name', result.output)
    self.assertTrue(os.path.isdir(self.models))

  def test_delete_removes_a_flattened_ref(self) -> None:
    os.makedirs(os.path.join(self.models, 'org__name'))
    result = testing.CliRunner().invoke(
        litert.cli, ['delete', 'org/name', '-y']
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertFalse(os.path.exists(os.path.join(self.models, 'org__name')))

  def test_delete_rejects_a_sub_reference(self) -> None:
    os.makedirs(os.path.join(self.models, 'foo'))
    result = testing.CliRunner().invoke(litert.cli, ['delete', 'foo:enc', '-y'])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('sub-reference', result.output)
    self.assertTrue(os.path.isdir(os.path.join(self.models, 'foo')))

  def test_list_shows_a_model(self) -> None:
    os.makedirs(os.path.join(self.models, 'org__name'))
    with open(os.path.join(self.models, 'org__name', 'm.tflite'), 'w') as f:
      f.write('x')
    result = testing.CliRunner().invoke(litert.cli, ['list', 'org/name'])
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('m.tflite', result.output)

  def test_import_rejects_a_ref_outside_the_models_cache(self) -> None:
    model = self.create_tempfile('m.tflite', content='x').full_path
    result = testing.CliRunner().invoke(
        litert.cli, ['import', model, '--model-ref', '..']
    )
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertFalse(os.path.exists(os.path.join(self.cache, 'm.tflite')))

  def test_clean_asks_before_deleting(self) -> None:
    result = testing.CliRunner().invoke(litert.cli, ['clean'], input='n\n')
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('every managed model', result.output)
    self.assertIn('Aborted.', result.output)
    self.assertTrue(os.path.isdir(self.models))


if __name__ == '__main__':
  absltest.main()
