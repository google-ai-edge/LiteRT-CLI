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

"""CLI module for the `litert compile` command.

This command applies NPU AOT compilation to a standard LiteRT (.tflite) model.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import pathlib
import shutil
import textwrap

import click
from litert_cli.core import constants
from litert_cli.core import deps
from litert_cli.core import npu_utils
from litert_cli.core import utils
from litert_cli.core.targets_manager import TargetsManager


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
    "--update-targets",
    type=str,
    required=False,
    default=None,
    help=(
        "Update SoC target lists from GitHub. Pass 'main' for latest, or a"
        " version tag like 'v2.1.4'."
    ),
)
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
    update_targets: str | None,
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

  # Quiet if default is true
  if constants.DEFAULT_QUIET:
    utils.enable_quiet_mode()

  # Initialize targets
  manager = TargetsManager()

  # Handle update or first-run download
  if update_targets:
    manager.download_targets(version=update_targets)
    importlib.reload(constants)
  else:
    # Check if cache exists
    if not manager.load_targets():
      click.echo("No target cache found. Downloading default target lists...")
      try:
        manager.download_targets(version="main")
        importlib.reload(constants)
      except Exception as e:
        click.echo(f"Warning: Failed to download default targets: {e}")
        click.echo("Falling back to built-in static target lists.")

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
    unsupported = [
        t
        for t in target
        if not any(k in t.lower() for k in ("sm", "qnn", "qualcomm"))
    ]
    if unsupported:
      raise click.ClickException(
          f"AOT Compilation failed for target(s) {', '.join(unsupported)}: "
          "Currently, only the Qualcomm platform is fully supported for offline AOT compilation."
      ) from e

    raise click.ClickException(
        f"AOT Compilation of '{resolved_model_path}' for targets"
        f" {', '.join(target)} failed: {e!r}"
    ) from e

  click.secho("AOT Compilation Completed Successfully!", fg="green")
