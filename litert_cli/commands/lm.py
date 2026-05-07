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

"""Command to call LiteRT-LM CLI tool."""

from __future__ import annotations

import subprocess
import sys

import click
from litert_cli.core import deps


@click.command(
    name="lm",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        help_option_names=[],
    ),
    help="""LiteRT-LM CLI commands.

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
@deps.require_extra("lm")
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
