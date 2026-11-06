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
from litert_cli.commands import lm_hf
from litert_cli.core import deps


def _help_preamble() -> str:
  """Returns the LiteRT CLI part of `litert lm --help`."""
  prefill = ",".join(map(str, lm_hf.DEFAULT_PREFILL_LENGTHS))
  return f"""\
LiteRT CLI `lm`: forwards to the native `litert-lm` CLI (its help follows).

For `litert lm run|benchmark <org/name>` with a Hugging Face repo id:
  - a repo with a .litertlm file runs with --from-huggingface-repo;
  - a transformers checkpoint is converted once with `litert convert` and
    cached under ~/.cache/litert-cli/models/<org>__<name>/.
  Conversion options, read by LiteRT CLI and not passed to litert-lm:
    --quantize RECIPE         default {lm_hf.DEFAULT_RECIPE} ('none' for fp32)
    --prefill-lengths N[,N]   default {prefill}
    --cache-length N          default {lm_hf.DEFAULT_CACHE_LENGTH}
    --reconvert               convert again even if a bundle is cached
"""


@click.command(
    name="lm",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        help_option_names=[],
    ),
    help="""LiteRT-LM CLI commands.

This command forwards its arguments to the native `litert-lm` CLI (from the
`litert-lm` / `litert-lm-nightly` package). In addition, when the model of
`run` or `benchmark` is a Hugging Face repo id (org/name), it runs the repo's
.litertlm file, or converts the repo once with `litert convert` and caches the
bundle.

Examples:

  Display the native litert-lm help:

    $ litert lm --help

  Run a generative LLM model:

    $ litert lm run model.litertlm
    $ litert lm run gemma3-1b

  Run a Hugging Face model (converted once if the repo has no .litertlm):

    $ litert lm run Qwen/Qwen3-0.6B --prompt "What is the capital of France?"

  Benchmark a generative LLM model:

    $ litert lm benchmark gemma3-1b
""",
)
@deps.require_extra("lm")
@click.pass_context
def lm_cmd(ctx: click.Context) -> None:
  """LiteRT-LM related commands.

  This command forwards its arguments to the native `litert-lm` CLI, after
  resolving a Hugging Face repo id given to `run` or `benchmark`.

  Args:
    ctx: click Context object containing forwarded arguments.
  """
  args = ctx.args
  if args[:1] in (["--help"], ["-h"]):
    click.echo(_help_preamble())
  if lm_hf.should_resolve(args):
    args = lm_hf.resolve_args(args)
  try:
    result = subprocess.run(["litert-lm"] + args, check=False)
    sys.exit(result.returncode)
  except FileNotFoundError:
    click.secho("Error: 'litert-lm' executable not found in PATH.", fg="red")
    click.secho(
        "Install it with `pip install litert-lm-nightly` (or reinstall"
        " `litert-cli-nightly[lm]`).",
        fg="yellow",
    )
    sys.exit(1)
