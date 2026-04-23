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
            'download',
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
