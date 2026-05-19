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

"""LiteRT CLI.

This module provides the main entry point for the LiteRT CLI, using lazy-loaded
commands for improved startup performance.
"""

import importlib
import sys
import click


class LazyCLI(click.Group):
  """Lazy loading command group.

  Only imports the actual module when the command is invoked.
  """

  def invoke(self, ctx: click.Context):
    try:
      return super().invoke(ctx)
    except Exception:
      from litert_cli.core import utils

      utils.restore_stderr()
      raise

  def list_commands(self, ctx: click.Context) -> list[str]:
    del self
    return [
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
    ]

  def get_command(
      self, ctx: click.Context, cmd_name: str
  ) -> click.Command | None:
    del self
    routes = {
        'benchmark': ('litert_cli.commands.benchmark.cli', 'benchmark_cmd'),
        'clean': ('litert_cli.commands.clean', 'clean_cmd'),
        'compile': ('litert_cli.commands.compile', 'compile_cmd'),
        'convert': ('litert_cli.commands.convert.cli', 'convert_cmd'),
        'delete': ('litert_cli.commands.delete', 'delete_cmd'),
        'download': ('litert_cli.commands.download', 'download_cmd'),
        'import': ('litert_cli.commands.import', 'import_cmd'),
        'list': ('litert_cli.commands.list', 'list_cmd'),
        'lm': ('litert_cli.commands.lm', 'lm_cmd'),
        'quantize': ('litert_cli.commands.quantize', 'quantize_cmd'),
        'run': ('litert_cli.commands.run.cli', 'run_cmd'),
        'visualize': ('litert_cli.commands.visualize', 'visualize_cmd'),
    }
    if cmd_name not in routes:
      return None

    module_path, obj_name = routes[cmd_name]
    try:
      mod = importlib.import_module(module_path)
      return getattr(mod, obj_name)
    except (ImportError, AttributeError) as e:
      raise click.ClickException(
          f"Failed to load command '{cmd_name}': {e}"
      ) from e


@click.command(cls=LazyCLI)
@click.option(
    '--enable-model-plugins',
    is_flag=True,
    default=False,
    hidden=True,
    help='Enable experimental model-specific plugin handlers.',
)
@click.pass_context
def cli(ctx: click.Context, enable_model_plugins: bool) -> None:
  """Runs the LiteRT CLI."""
  ctx.ensure_object(dict)
  if enable_model_plugins:
    from litert_cli.core import constants

    constants.ENABLE_MODEL_PLUGINS = True


def main_bridge(unused_argv):
  cli.main(args=sys.argv[1:])


if __name__ == '__main__':
  try:
    # Try to run in internal environment
    from absl import app  # pylint: disable=g-import-not-at-top

    app.run(main_bridge, argv=[sys.argv[0]])
  except ImportError:
    cli()
