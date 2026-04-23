"""Command line interface for visualizing LiteRT models.

This module provides the `litert visualize` command, which launches the
Model Explorer web application to inspect the architecture of TFLite models.

Key Features:
- Background Execution: The web server runs entirely in the background,
  instantly returning control of the terminal to the user.
- URL Printing: The command calculates and prints a click-able URL directly
  in the terminal, bypassing IDE port-forwarding interceptions that often
  strip URL query parameters.
- Server Reuse: By default, multiple visualizations reuse the same server
  instance, preventing port exhaustion and reducing memory footprint.
"""

from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
import textwrap
import urllib.parse

import click

from ..core import deps


def _is_port_in_use(port_num: int) -> bool:
  """Checks if a port is in use on localhost.

  Args:
    port_num: Port to check.

  Returns:
    True if the port is bound and in use, False otherwise.
  """
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    return s.connect_ex(('localhost', port_num)) == 0


def _find_available_port(start_port: int, max_attempts: int = 20) -> int:
  """Finds the first available port on localhost starting from start_port.

  Args:
    start_port: Port to start searching from.
    max_attempts: Maximum number of ports to check.

  Returns:
    The first available port found.
  """
  return next(
      (
          p
          for p in range(start_port, start_port + max_attempts)
          if not _is_port_in_use(p)
      ),
      start_port,
  )


@click.command(
    'visualize',
    help=textwrap.dedent("""\
      Runs model explorer to visualize the model architecture.

      MODEL_PATH: Path to the LiteRT model (.tflite) to visualize.

      Examples:

        1. Visualize a single model (starts a new server if none exists):

          $ litert visualize /path/to/model.tflite

        2. Visualize a different model (reuses existing server, refreshes browser):

          $ litert visualize /path/to/another_model.tflite

        3. Compare two models side-by-side (forces a new server on a new port):

          $ litert visualize /path/to/model_A.tflite
          $ litert visualize /path/to/model_B.tflite --no_reuse_server

        4. Clean up and stop all running Model Explorer servers:

          $ litert visualize --stop_all
      """),
)
@click.argument(
    'model_path',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=False,
)
@click.option(
    '--reuse_server/--no_reuse_server',
    default=True,
    help='Reuse running Model Explorer server.',
)
@click.option(
    '--stop_all',
    is_flag=True,
    help='Stop all running Model Explorer servers.',
)
@deps.require_extra('visualize')
def visualize_cmd(
    model_path: pathlib.Path | None, *, reuse_server: bool, stop_all: bool
) -> None:
  """Runs model explorer to visualize the model architecture.

  Args:
    model_path: Path to the LiteRT model (.tflite) to visualize.
    reuse_server: Whether to reuse an already running Model Explorer server.
    stop_all: If True, stops all running Model Explorer servers instead.
  """
  if stop_all:
    click.echo('Attempting to stop all running Model Explorer servers...')
    try:
      # Use pkill to find and terminate all model-explorer processes
      subprocess.run(
          ['pkill', '-f', 'model-explorer'],
          check=True,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      click.secho(
          'Successfully stopped all Model Explorer servers.',
          fg='green',
      )
    except subprocess.CalledProcessError:
      click.secho('No running Model Explorer servers found.', fg='yellow')
    return

  if not model_path:
    raise click.UsageError('Missing argument "MODEL_PATH".')

  click.echo(
      f'Starting Model Explorer visualization for {model_path} in the'
      ' background...'
  )
  # Check for available port so we can print the exact URL
  port = (
      8080
      if reuse_server and _is_port_in_use(8080)
      else _find_available_port(8080)
  )

  # Build the exact model explorer data URL
  data = {'models': [{'url': str(model_path)}]}
  data_param = urllib.parse.quote(json.dumps(data))
  url = f'http://localhost:{port}/?data={data_param}'

  model_explorer_bin = None
  if sys.executable:
    python_dir = pathlib.Path(sys.executable).parent
    # Assume model_explorer binary is in the same directory as the python
    model_explorer_bin = python_dir / 'model-explorer'

  cmd = [
      str(model_explorer_bin)
      if model_explorer_bin and model_explorer_bin.exists()
      else 'model-explorer',
      str(model_path),
      '--no_open_in_browser',
  ]
  if reuse_server:
    cmd.append('--reuse_server')

  # Launch as a fully detached daemon so the terminal isn't blocked.
  try:
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    click.secho(
        '\nModel Explorer server is running in the background.', fg='green'
    )
    click.secho(
        '► Please click the link below to view your model:',
        fg='cyan',
        bold=True,
    )
    click.secho(f'\n    {url}\n')
    if reuse_server:
      click.secho(
          'It will reuse the existing server instance if one is already'
          ' running, avoiding port conflicts.',
          fg='cyan',
      )
  except FileNotFoundError:
    click.secho(
        '\nError: "model-explorer" command not found.', fg='red', bold=True
    )
    click.secho(
        'Please make sure Model Explorer is installed and available in your'
        ' PATH.',
        fg='yellow',
    )
    click.secho(
        'You can install it via: pip install model-explorer',
        fg='cyan',
    )
