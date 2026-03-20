"""Hugging Face automated export logic for LiteRT conversion.

This module provides the implementation for the 'hf' mode of the
`litert convert` command, orchestrating the download and conversion of
models directly from the Hugging Face Hub.
"""

from __future__ import annotations

import click


def convert_huggingface(model: str, task: str | None, output: str) -> None:
  """Conversion logic using HuggingFace Automated Export (export_hf).

  Args:
    model: The Hugging Face model ID (e.g., Qwen/Qwen1.5-0.5B-Chat).
    task: The target task for conversion (e.g., text_generation).
    output: The directory to save the converted model.
  """
  # Lazy load the export module to avoid importing torch and other heavy
  # dependencies when the litert CLI is merely invoked for --help.
  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf import export as hf_export

  export_task = task if task else "text_generation"

  click.echo(
      f"Starting conversion for model '{model}' with task '{export_task}'"
  )

  try:
    # Call the auto-export function from litert_torch.
    # It automatically saves to the output.
    hf_export.export(
        model=model,
        output_dir=output,
    )

    click.echo(f"Successfully converted and saved model to {output}")

  except Exception as e:
    click.echo(f"Error during conversion: {e}")
    # Re-raise to let click handle the error exit
    raise e
