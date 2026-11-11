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
import importlib.metadata
import sys
import click


def _environment_error_message(error: BaseException) -> str | None:
  """Returns a one-line message for an error caused by the environment.

  Failed subprocesses (e.g. adb), file system errors and network errors are
  reported without a traceback unless LITERT_VERBOSE=1. Returns None for any
  other error, which is raised as is.

  Args:
    error: The error raised by a command.
  """
  import os  # pylint: disable=g-import-not-at-top
  import subprocess  # pylint: disable=g-import-not-at-top
  import urllib.error  # pylint: disable=g-import-not-at-top

  if os.environ.get('LITERT_VERBOSE') == '1':
    return None
  hint = ' (set LITERT_VERBOSE=1 for the full traceback)'
  if isinstance(error, subprocess.CalledProcessError):
    cmd = error.cmd if isinstance(error.cmd, str) else ' '.join(error.cmd)
    stderr = error.stderr
    if isinstance(stderr, bytes):
      stderr = stderr.decode(errors='replace')
    tail = f': {stderr.strip().splitlines()[-1]}' if stderr else ''
    return f'`{cmd}` exited with {error.returncode}{tail}{hint}'
  if isinstance(error, urllib.error.URLError):
    return f'Network error: {error.reason}{hint}'
  if isinstance(error, OSError):
    return f'{error}{hint}'
  try:
    import requests  # pylint: disable=g-import-not-at-top
  except ImportError:
    return None
  if isinstance(error, requests.RequestException):
    return f'Network error: {error}{hint}'
  return None


class LazyCLI(click.Group):
  """Lazy loading command group.

  Only imports the actual module when the command is invoked.
  """

  def invoke(self, ctx: click.Context):
    try:
      return super().invoke(ctx)
    except BaseException as e:
      from litert_cli.core import utils  # pylint: disable=g-import-not-at-top

      utils.restore_stderr()
      message = _environment_error_message(e)
      if message is None:
        raise
      raise click.ClickException(message) from e

  def resolve_command(
      self, ctx: click.Context, args: list[str]
  ) -> tuple[str | None, click.Command | None, list[str]]:
    cmd_name, cmd, args = super().resolve_command(ctx, args)
    if cmd is not None:
      from litert_cli.core import report  # pylint: disable=g-import-not-at-top

      report.check_json_supported(ctx, cmd)
    return cmd_name, cmd, args

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
        'verify',
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
        'verify': ('litert_cli.commands.verify', 'verify_cmd'),
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


def _cli_version() -> str:
  """Returns the installed LiteRT CLI package and version."""
  for dist in ('litert-cli-nightly', 'litert-cli'):
    try:
      return f'{importlib.metadata.version(dist)} ({dist})'
    except importlib.metadata.PackageNotFoundError:
      pass
  return 'unknown (not installed as a package)'


@click.command(cls=LazyCLI)
@click.version_option(
    version=_cli_version(), prog_name='litert', message='%(prog)s %(version)s'
)
@click.option(
    '--enable-model-plugins',
    is_flag=True,
    default=False,
    hidden=True,
    help='Enable experimental model-specific plugin handlers.',
)
@click.option(
    '--json',
    'output_json',
    is_flag=True,
    default=False,
    envvar='LITERT_JSON',
    help=(
        'Print results as a single JSON document on stdout (prose goes to'
        ' stderr). Commands that do not support it yet fail with a usage'
        ' error.'
    ),
)
@click.pass_context
def cli(
    ctx: click.Context, enable_model_plugins: bool, output_json: bool
) -> None:
  """Runs the LiteRT CLI."""
  from litert_cli.core import report  # pylint: disable=g-import-not-at-top

  report.install(ctx, json_mode=output_json)
  _warn_if_deprecated_python()
  if enable_model_plugins:
    from litert_cli.core import constants

    constants.ENABLE_MODEL_PLUGINS = True


def _warn_if_deprecated_python() -> None:
  """Prints a deprecation banner when running on an end-of-life Python."""
  # Python 3.10 reached upstream end-of-life on 2026-10-01 (PEP 619). Support
  # for it is deprecated and will be removed in the next release.
  if sys.version_info < (3, 11):
    click.secho(
        'WARNING: litert-cli support for Python 3.10 is deprecated and will be'
        " removed in the next release, following Python 3.10's upstream"
        ' end-of-life (https://peps.python.org/pep-0619/). Please upgrade to'
        ' Python 3.11 or newer.',
        fg='yellow',
        err=True,
    )


def main_bridge(unused_argv):
  cli.main(args=sys.argv[1:])


if __name__ == '__main__':
  try:
    # Try to run in internal environment
    from absl import app  # pylint: disable=g-import-not-at-top

    app.run(main_bridge, argv=[sys.argv[0]])
  except ImportError:
    cli()
