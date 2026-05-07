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

"""Command line interface for listing managed models in LiteRT cache."""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import click

from ..core.constants import LITERT_MODELS_CACHE_DIR


@click.command(
    "list",
    help="List all managed models or detailed contents of a specific model.",
)
@click.argument("model_ref", required=False)
def list_cmd(model_ref: str | None) -> None:
  """Lists managed models. If MODEL_REF is provided, shows detailed contents."""
  cache_dir = pathlib.Path(LITERT_MODELS_CACHE_DIR)

  if not cache_dir.exists() or not cache_dir.is_dir():
    click.echo("No managed models found (cache directory does not exist).")
    return

  # Case 1: List detailed contents of a specific model
  if model_ref:
    # Flatten for directory check
    ref_flat = model_ref.replace("/", "__") if "/" in model_ref else model_ref
    model_dir = cache_dir / ref_flat

    if not model_dir.exists() or not model_dir.is_dir():
      click.secho(f"Error: Managed model '{model_ref}' not found.", fg="red")
      sys.exit(1)

    metadata_file = model_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
      try:
        with open(metadata_file, "r") as f:
          metadata = json.load(f)
      except Exception:
        pass

    click.echo(
        "Details for managed model:"
        f" {click.style(model_ref, fg='green', bold=True)}"
    )
    if metadata:
      click.echo(f"  HF ID:      {metadata.get('hf_id', 'N/A')}")
      click.echo(f"  Source:     {metadata.get('source', 'N/A')}")
      click.echo(f"  Created At: {metadata.get('created_at', 'N/A')}")

    click.echo("\nFiles in model directory:")
    sub_refs = metadata.get("sub_references", {})
    # Reverse mapping for display
    file_to_subrefs = collections.defaultdict(list)
    for sub, info in sub_refs.items():
      f = info.get("file")
      if f:
        file_to_subrefs[f].append(sub)

    for item in sorted(model_dir.iterdir()):
      if item.name == "metadata.json":
        continue

      size_kb = item.stat().st_size / 1024
      suffix = ""
      if item.name in file_to_subrefs:
        subs = ", ".join(file_to_subrefs[item.name])
        suffix = f" {click.style(f'[{subs}]', fg='cyan')}"

      click.echo(f"  - {item.name:<30} ({size_kb:>8.1f} KB){suffix}")
    return

  # Case 2: List all managed models (Default)
  models = [d for d in cache_dir.iterdir() if d.is_dir()]

  if not models:
    click.echo("No managed models found in cache.")
    return

  click.echo(f"Managed models in {cache_dir}:")
  click.echo("-" * 60)

  for model_dir in sorted(models, key=lambda x: x.name):
    ref = model_dir.name
    metadata_file = model_dir / "metadata.json"
    hf_id = "N/A"
    source = "N/A"
    sub_refs = {}

    if metadata_file.exists():
      try:
        with open(metadata_file, "r") as f:
          metadata = json.load(f)
          # Use original model_ref from metadata if available
          ref = metadata.get("model_ref", ref)
          hf_id = metadata.get("hf_id", "N/A")
          source = metadata.get("source", "N/A")
          sub_refs = metadata.get("sub_references", {})
      except Exception:
        pass

    click.echo(f"Ref: {click.style(ref, fg='green', bold=True)}")
    click.echo(f"  HF ID:  {hf_id}")
    click.echo(f"  Source: {source}")

    if sub_refs:
      click.echo("  Sub-references:")
      for sub_ref, info in sub_refs.items():
        file_name = info.get("file", "N/A")
        click.echo(f"    - {sub_ref} -> {file_name}")

    click.echo("-" * 60)
