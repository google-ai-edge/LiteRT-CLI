"""Smoke test for LiteRT CLI commands loading."""

import os
import sys

# Add project root to path so litert_cli package is discoverable
# TODO(shuangfeng): Remove them once migrate to official paths.
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_DIR))

from absl.testing import absltest
from click.testing import CliRunner
from litert_cli import litert


class CommandsHelpSmokeTest(absltest.TestCase):
  """Smoke test verifying all commands can display --help."""

  def test_main_help(self):
    runner = CliRunner()
    result = runner.invoke(litert.cli, ["--help"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("LiteRT CLI", result.output)

  def test_all_subcommands_help(self):
    runner = CliRunner()
    commands = [
        "download",
        "convert",
        "visualize",
        "quantize",
        "run",
        "lm",
        "benchmark",
    ]
    for cmd in commands:
      with self.subTest(command=cmd):
        result = runner.invoke(litert.cli, [cmd, "--help"])
        if cmd == "lm" and result.exit_code != 0:
          # `lm` is a transparent proxy. If `litert-lm` is not on PATH,
          # it intentionally fails with exit 1 and prints warning.
          self.assertIn("'litert-lm' executable not found", result.output)
        else:
          self.assertEqual(
              result.exit_code, 0, msg=f"Command '{cmd}' failed to load --help"
          )


if __name__ == "__main__":
  absltest.main()
