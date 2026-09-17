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

import pathlib
import shutil
import subprocess
import sys

import click
from litert_cli.core import deps


def _resolve_litert_lm() -> str | None:
  """Locates the `litert-lm` executable.

  Console scripts are installed next to the running interpreter, which is not
  necessarily on PATH: that is the case for a virtualenv that has not been
  activated, for `uv run`, and for pipx-style installs. Look there first and
  only then fall back to PATH.

  Returns:
    An absolute path or a PATH-resolvable name, or None if not found.
  """
  candidate = pathlib.Path(sys.executable).parent / "litert-lm"
  if candidate.exists():
    return str(candidate)
  return shutil.which("litert-lm")


@click.command(
    name="lm",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        help_option_names=[],
    ),
    help="""LiteRT-LM CLI commands.

This command is a transparent proxy to the native `litert-lm` CLI.
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

  This command is a transparent proxy to the native `litert-lm` CLI.
  Any wildcard arguments used here are forwarded directly to the actual engine.

  Args:
    ctx: click Context object containing forwarded arguments.
  """
  exe = _resolve_litert_lm()
  if exe is None:
    click.secho(
        "Error: 'litert-lm' executable not found.", fg="red", err=True
    )
    click.secho(
        "Install it with: pip install litert-lm-nightly", fg="yellow", err=True
    )
    sys.exit(1)

  result = subprocess.run([exe] + ctx.args, check=False)
  sys.exit(result.returncode)
