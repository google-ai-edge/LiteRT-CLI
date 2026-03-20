"""LiteRT CLI.

This module provides the main entry point for the LiteRT CLI, using lazy-loaded
commands for improved startup performance.
"""

from __future__ import annotations

import importlib

import click


class LazyCLI(click.Group):
  """A click.Group that lazy-loads command implementations from other modules.

  This improves CLI startup performance by avoiding the overhead of importing
  heavy dependencies (like torch, tensorflow, or huggingface_hub) until a
  specific command is actually executed.
  """

  def list_commands(self, ctx: click.Context) -> list[str]:
    """Returns a list of available command names."""
    return [
        'download',
        'convert',
        'visualize',
        'quantize',
        'run',
        'lm',
        'benchmark',
    ]

  def get_command(
      self, ctx: click.Context, cmd_name: str
  ) -> click.Command | None:
    """Loads and returns the requested command object.

    Args:
      ctx: The click context (unused).
      cmd_name: The name of the command to load.

    Returns:
      A click.Command instance if found, or None otherwise.
    """
    routes = {
        'download': ('litert_cli.commands.download', 'download_cmd'),
        'convert': ('litert_cli.commands.convert.cli', 'convert_cmd'),
        'quantize': ('litert_cli.commands.quantize', 'quantize_cmd'),
        'run': ('litert_cli.commands.run.cli', 'run_cmd'),
        'visualize': ('litert_cli.commands.visualize', 'visualize_cmd'),
        'lm': ('litert_cli.commands.lm', 'lm_cmd'),
        'benchmark': ('litert_cli.commands.benchmark.cli', 'benchmark_cmd'),
    }

    if cmd_name not in routes:
      return None

    module_path, obj_name = routes[cmd_name]
    try:
      mod = importlib.import_module(module_path)
      return getattr(mod, obj_name)
    except (ImportError, AttributeError) as e:
      click.secho(f"Failed to load command '{cmd_name}': {e}", fg='red')
      return None


@click.command(cls=LazyCLI)
@click.pass_context
def cli(ctx: click.Context) -> None:
  """The main entry point for the LiteRT CLI."""
  ctx.ensure_object(dict)


if __name__ == '__main__':
  cli()
