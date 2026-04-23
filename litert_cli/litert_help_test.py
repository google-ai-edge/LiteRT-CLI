from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from click import testing
from litert_cli import litert


class LitertHelpTest(parameterized.TestCase):

  def test_main_help(self):
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, ["--help"])
    self.assertEqual(result.exit_code, 0)
    self.assertIn("LiteRT CLI", result.output)

  @parameterized.named_parameters(
      ("download", "download"),
      ("visualize", "visualize"),
      ("quantize", "quantize"),
      ("run", "run"),
      ("clean", "clean"),
      ("compile", "compile"),
  )
  def test_subcommands_help(self, cmd):
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, [cmd, "--help"])
    self.assertEqual(
        result.exit_code,
        0,
        msg=f"Command '{cmd}' failed to load --help. Output: {result.output}",
    )


if __name__ == "__main__":
  absltest.main()
