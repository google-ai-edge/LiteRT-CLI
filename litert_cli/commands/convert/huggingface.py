"""Hugging Face automated export logic for LiteRT conversion.

This module provides the implementation for the 'hf' mode of the
`litert convert` command, orchestrating the download and conversion of
models directly from the Hugging Face Hub.
"""

from __future__ import annotations

import click
import pathlib
import shutil

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export
from litert_cli.core import npu_utils


def convert_huggingface(
    model: str,
    output: str,
    target: tuple[str, ...],
    export_aipack: pathlib.Path | None,
) -> None:
  """Conversion logic using HuggingFace Automated Export (export_hf).

  Args:
    model: The Hugging Face model ID (e.g., Qwen/Qwen1.5-0.5B-Chat).
    output: The directory to save the converted model.
    target: NPU targets to apply AOT compilation.
    export_aipack: Output directory to export the AI Pack for PODAI.
  """
  # Lazy load the export module to avoid importing torch and other heavy
  # dependencies when the litert CLI is merely invoked for --help.
  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf import export as hf_export

  click.echo(
      f"Starting conversion for model '{model}''"
  )

  try:
    # Call the auto-export function from litert_torch.
    # It automatically saves to the output.
    hf_export.export(
        model=model,
        output_dir=output,
    )

    if target:
      output_dir = pathlib.Path(output)
      # Find the generated tflite
      tflite_files = list(output_dir.glob("*.tflite"))
      if not tflite_files:
        raise FileNotFoundError(f"No .tflite files found in HF export output: {output_dir}")
      target_tflite = tflite_files[0]
      base_name = target_tflite.stem

      click.echo(f"Compiling converted model {target_tflite} for targets: {', '.join(target)}")
      aot_targets = [npu_utils.get_target(t) for t in target]
      
      compiled_models = aot_lib.aot_compile(
          str(target_tflite),
          target=aot_targets,
          keep_going=False,
      )
      
      if export_aipack:
          export_dir = pathlib.Path(export_aipack)
          click.echo(f"Exporting AI Pack to: {export_dir}")
          shutil.rmtree(export_dir, ignore_errors=True)
          export_dir.mkdir(parents=True, exist_ok=True)
          ai_pack_export.export(compiled_models, str(export_dir), base_name, "model")
      else:
          # Overwrite the original tflite output with the compiled models
          click.echo(f"Exporting compiled model over original in {output_dir}")
          compiled_models.export(str(output_dir), model_name=base_name)

    click.echo(f"Successfully converted and saved model to {output}")

  except Exception as e:
    click.echo(f"Error during conversion: {e}")
    # Re-raise to let click handle the error exit
    raise e
