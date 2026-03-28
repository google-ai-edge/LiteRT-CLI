"""CLI module for the `litert compile` command.

This command handles purely applying NPU AOT compilation to standard
LiteRT (.tflite) models natively mapping strings to registered backend compilers.
"""

from __future__ import annotations

import pathlib
import shutil

import click
from litert_cli.core import deps
from litert_cli.core import npu_utils

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export


@click.command(
    "compile",
    help="""Apply AOT (Ahead-of-Time) compilation for NPUs to a TFLite model.

MODEL_PATH: Path to a valid .tflite model.

Examples:

  Basic Compilation for specific NPU:

    $ litert compile my_model.tflite --target sm8450

  Compile for multiple targets and export AI Pack for Android:

    $ litert compile my_model.tflite --target sm8550 --target mt6989 --export-aipack my_npu_models
""",
)
@deps.require_extra("run")
@click.argument("model_path", type=str, required=True)
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
    help="If specified, exports an AI Pack directory for PODAI instead of standard .tflite.",
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
    help="Directory to save the compiled TFLite model. Defaults to current directory.",
)
def compile_cmd(
    model_path: str,
    target: tuple[str, ...],
    export_aipack: pathlib.Path | None,
    output_dir: pathlib.Path | None,
) -> None:
    """Compiles a tflite model with NPU AOT backends."""
    if not pathlib.Path(model_path).exists():
        raise click.ClickException(f"Model file not found: {model_path}")

    click.echo(f"Compiling model {model_path} for targets: {', '.join(target)}")
    
    aot_targets = [npu_utils.get_aot_target(t) for t in target]
    
    try:
        compiled_models = aot_lib.aot_compile(
            model_path,
            target=aot_targets,
            keep_going=False,
        )
    except Exception as e:
        raise click.ClickException(f"AOT Compilation failed: {e}")
        
    if not output_dir:
        output_dir = pathlib.Path.cwd()
        
    base_name = pathlib.Path(model_path).stem
        
    if export_aipack:
        click.echo(f"Exporting AI Pack to: {export_aipack}")
        shutil.rmtree(export_aipack, ignore_errors=True)
        export_aipack.mkdir(parents=True, exist_ok=True)
        try:
            ai_pack_export.export(
                compiled_models, str(export_aipack), base_name, "model"
            )
        except Exception as e:
            raise click.ClickException(f"Failed to export AI Pack: {e}")
    else:
        click.echo(f"Exporting compiled TFLite to: {output_dir}")
        try:
            compiled_models.export(str(output_dir), model_name=base_name)
        except Exception as e:
            raise click.ClickException(f"Failed to export models: {e}")
            
    click.secho("AOT Compilation Completed Successfully!", fg="green")
