"""CLI module for the `litert convert` command.

This command handles converting PyTorch and Hugging Face models into LiteRT
models using various conversion paths like automated HF export, generic script
injection, and native generative API re-authoring.
"""

from __future__ import annotations

import pathlib

import click
from litert_cli.core import deps


@click.command(
    "convert",
    help="""Convert a PyTorch model into a LiteRT model.

MODEL_OR_SCRIPT: Hugging Face model ID or path to a PyTorch script.

Examples:

  Automated HF Conversion:

    $ litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen

  Generic Script Injection:

    $ litert convert my_model.py --output /tmp/mymodel
""",
)
@deps.require_extra("torch")
@click.argument("model_or_script", type=str, required=True)
@click.option(
    "--task",
    type=str,
    required=False,
    help="Target task (e.g., text_generation).",
)
@click.option(
    "--output",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=False,
    default=None,
    help="Directory to save the converted TFLite model.",
)
@click.option(
    "--model-func",
    type=str,
    default="get_model",
    help=(
        "Name of the function in the --script that returns a torch.nn.Module"
        " and optionally a quantization config. Default: 'get_model'."
    ),
)
@click.option(
    "--input-func",
    type=str,
    default="get_args",
    help=(
        "Name of the function in the --script that returns (True) sample_args "
        "and/or kwargs. Default: 'get_args'."
    ),
)
def convert_cmd(
    model_or_script: str,
    task: str | None,
    output: pathlib.Path | None,
    model_func: str,
    input_func: str,
) -> None:
  r"""Converts a PyTorch model into a LiteRT model.

  Args:
    model_or_script: Hugging Face model ID or path to a PyTorch script.
    task: Target task for HF conversion.
    output: Output directory for the converted model.
    model_func: Function to retrieve the model in 'script' mode.
    input_func: Function to retrieve sample inputs in 'script' mode.
  """

  if not output:
    if model_or_script.endswith(".py"):
      base_name = pathlib.Path(model_or_script).stem
    else:
      base_name = model_or_script.split("/")[-1]
    output = pathlib.Path.cwd() / base_name

  if model_or_script.endswith(".py"):
    from litert_cli.commands.convert import generic  # pylint: disable=g-import-not-at-top

    generic.convert_generic_script(
        model_or_script, model_func, input_func, str(output)
    )
  else:
    from litert_cli.commands.convert import huggingface  # pylint: disable=g-import-not-at-top

    huggingface.convert_huggingface(model_or_script, task, str(output))
