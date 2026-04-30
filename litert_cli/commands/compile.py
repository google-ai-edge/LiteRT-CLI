"""CLI module for the `litert compile` command.

This command applies NPU AOT compilation to a standard LiteRT (.tflite) model.
"""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import shutil
import textwrap

import click
from litert_cli.core import deps
from litert_cli.core import npu_utils


@click.command(
    "compile",
    help=textwrap.dedent("""\
        Apply AOT (Ahead-of-Time) compilation for NPUs to a TFLite model.

        model_path: Path to a valid .tflite model.

        Examples:

          Basic Compilation for specific NPU:

            $ litert compile my_model.tflite --target sm8450

          Compile for multiple targets and export AI Pack for Android:

            $ litert compile my_model.tflite --target sm8550 --target mt6989 \
               --export-aipack my_npu_models
        """),
)
@click.argument("model_path", type=str)
@click.option(
    "--target",
    type=str,
    multiple=True,
    required=True,
    help="One or more NPU target codenames (e.g., sm8450).",
)
@click.option(
    "--export-aipack",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=False,
    default=None,
    help=(
        "If specified, exports an AI Pack directory for PODAI instead of"
        " standard .tflite."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=False,
    default=None,
    help=(
        "Directory to save the compiled TFLite model. Defaults to current"
        " directory."
    ),
)
@deps.require_extra("compile")
def compile_cmd(
    model_path: str,
    target: Sequence[str],
    export_aipack: pathlib.Path | None,
    output_dir: pathlib.Path | None,
) -> None:
  """Compiles a tflite model with NPU AOT backends.

  Args:
    model_path: Path to the input tflite model or Model Reference.
    target: List of target SoCs or acceleration backends.
    export_aipack: Path to export the compiled model as an AI Pack.
    output_dir: Directory to save the compiled TFLite model.

  Raises:
    click.ClickException: If compilation or export fails.
  """
  from ai_edge_litert.aot import aot_compile as aot_lib
  from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export
  from litert_cli.core import models as core_models

  resolved_model_path, _ = core_models.resolve_model_reference(model_path)
  if str(resolved_model_path) != str(model_path):
    click.echo(f"Resolved model '{model_path}' to '{resolved_model_path}'")

  resolved_model_path = pathlib.Path(resolved_model_path)

  click.echo(
      f"Compiling model {resolved_model_path} for targets: {', '.join(target)}"
  )

  aot_targets = [npu_utils.get_aot_target(t) for t in target]

  try:
    compiled_models = aot_lib.aot_compile(
        str(resolved_model_path),
        target=aot_targets,
        keep_going=False,
    )
    resolved_output_dir = output_dir or pathlib.Path.cwd()

    base_name = resolved_model_path.stem

    if export_aipack:
      click.echo(f"Exporting AI Pack to: {export_aipack}")
      if export_aipack.exists():
        if export_aipack.is_dir():
          shutil.rmtree(export_aipack)
        else:
          export_aipack.unlink()
      export_aipack.mkdir(parents=True, exist_ok=True)
      try:
        ai_pack_export.export(
            compiled_models, str(export_aipack), base_name, "model"
        )
      except Exception as e:
        raise click.ClickException(f"Failed to export AI Pack: {e!r}") from e
    else:
      click.echo(f"Exporting compiled TFLite to: {resolved_output_dir}")
      try:
        compiled_models.export(str(resolved_output_dir), model_name=base_name)
      except Exception as e:
        raise click.ClickException(f"Failed to export models: {e!r}") from e
  except click.ClickException:
    raise
  except Exception as e:
    raise click.ClickException(
        f"AOT Compilation of '{resolved_model_path}' for targets"
        f" {', '.join(target)} failed: {e!r}"
    ) from e

  click.secho("AOT Compilation Completed Successfully!", fg="green")
