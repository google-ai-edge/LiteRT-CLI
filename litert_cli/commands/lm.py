"""CLI interface for LiteRT-LM."""

from __future__ import annotations

import subprocess
import sys

import click


@click.command(
    name="lm",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        help_option_names=[],
    ),
    help="""LiteRT-LM related commands.

This command is a transparent proxy to the native `litert-lm-cli` package.
Any wildcard arguments used here are forwarded directly to the actual engine.

Examples:

  Display the native litert-lm help:

    $ litert lm --help

  Run a generative LLM model:

    $ litert lm run model.litertlm
    $ litert lm run gemma3-1b

  Benchmark a generative LLM model:

    $ litert lm benchmark gemma3-1b
""",
)
@click.pass_context
def lm_cmd(ctx: click.Context) -> None:
  """LiteRT-LM related commands.

  This command is a transparent proxy to the native `litert-lm-cli` package.
  Any wildcard arguments used here are forwarded directly to the actual engine.

  Args:
    ctx: click Context object containing forwarded arguments.
  """
  try:
    result = subprocess.run(["litert-lm"] + ctx.args, check=False)
    sys.exit(result.returncode)
  except FileNotFoundError:
    click.secho("Error: 'litert-lm' executable not found in PATH.", fg="red")
    click.secho(
        "Please install 'litert-lm-cli' to use this command.", fg="yellow"
    )
    sys.exit(1)
