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
@click.pass_context
def cli(ctx: click.Context) -> None:
  """Runs the LiteRT CLI."""
  ctx.ensure_object(dict)


def main_bridge(unused_argv):
  cli.main(args=sys.argv[1:])


if __name__ == '__main__':
  try:
    # Try to run in google3
    from absl import app  # pylint: disable=g-import-not-at-top

    app.run(main_bridge, argv=[sys.argv[0]])
  except ImportError:
    cli()
