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

import json
import pathlib
import sys
from typing import Any

import click

from ..core.constants import LITERT_MODELS_CACHE_DIR


def _load_metadata(model_dir: pathlib.Path) -> dict[str, Any]:
  """Loads model metadata, returning an empty dict if metadata is unavailable."""
  metadata_file = model_dir / "metadata.json"
  if not metadata_file.exists():
    return {}
  try:
    with open(metadata_file, "r") as f:
      metadata = json.load(f)
      return metadata if isinstance(metadata, dict) else {}
  except Exception:
    return {}


def _model_summary(model_dir: pathlib.Path) -> dict[str, Any]:
  """Builds a summary for a cached model directory."""
  metadata = _load_metadata(model_dir)
  return {
      "ref": metadata.get("model_ref", model_dir.name),
      "hf_id": metadata.get("hf_id", "N/A"),
      "source": metadata.get("source", "N/A"),
      "created_at": metadata.get("created_at", "N/A"),
      "sub_references": metadata.get("sub_references", {}),
  }


def _model_details(model_ref: str, model_dir: pathlib.Path) -> dict[str, Any]:
  """Builds detailed information for a cached model directory."""
  summary = _model_summary(model_dir)
  sub_refs = summary["sub_references"]
  file_to_subrefs = {}
  for sub_ref, info in sub_refs.items():
    file_name = info.get("file") if isinstance(info, dict) else None
    if file_name:
      file_to_subrefs.setdefault(file_name, []).append(sub_ref)

  files = []
  for item in sorted(model_dir.iterdir()):
    if item.name == "metadata.json":
      continue
    files.append({
        "name": item.name,
        "size_bytes": item.stat().st_size,
        "is_dir": item.is_dir(),
        "sub_references": file_to_subrefs.get(item.name, []),
    })

  summary["ref"] = summary.get("ref") or model_ref
  summary["files"] = files
  return summary


@click.command(
    "list",
    help="List all managed models or detailed contents of a specific model.",
)
@click.argument("model_ref", required=False)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output machine-readable JSON.",
)
def list_cmd(model_ref: str | None, as_json: bool) -> None:
  """Lists managed models. If MODEL_REF is provided, shows detailed contents."""
  cache_dir = pathlib.Path(LITERT_MODELS_CACHE_DIR)

  if not cache_dir.exists() or not cache_dir.is_dir():
    if as_json:
      click.echo(json.dumps([] if model_ref is None else {}, indent=2))
      return
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

    details = _model_details(model_ref, model_dir)
    if as_json:
      click.echo(json.dumps(details, indent=2))
      return

    metadata = _load_metadata(model_dir)
    click.echo(
        "Details for managed model:"
        f" {click.style(model_ref, fg='green', bold=True)}"
    )
    if metadata:
      click.echo(f"  HF ID:      {details.get('hf_id', 'N/A')}")
      click.echo(f"  Source:     {details.get('source', 'N/A')}")
      click.echo(f"  Created At: {details.get('created_at', 'N/A')}")

    click.echo("\nFiles in model directory:")
    for item in details["files"]:
      size_kb = item["size_bytes"] / 1024
      suffix = ""
      if item["sub_references"]:
        subs = ", ".join(item["sub_references"])
        suffix = f" {click.style(f'[{subs}]', fg='cyan')}"

      click.echo(f"  - {item['name']:<30} ({size_kb:>8.1f} KB){suffix}")
    return

  # Case 2: List all managed models (Default)
  models = [d for d in cache_dir.iterdir() if d.is_dir()]

  if not models:
    if as_json:
      click.echo(json.dumps([], indent=2))
      return
    click.echo("No managed models found in cache.")
    return

  summaries = [_model_summary(model_dir) for model_dir in sorted(models)]

  if as_json:
    click.echo(json.dumps(summaries, indent=2))
    return

  click.echo(f"Managed models in {cache_dir}:")
  click.echo("-" * 60)

  for summary in summaries:
    click.echo(f"Ref: {click.style(summary['ref'], fg='green', bold=True)}")
    click.echo(f"  HF ID:  {summary['hf_id']}")
    click.echo(f"  Source: {summary['source']}")

    if summary["sub_references"]:
      click.echo("  Sub-references:")
      for sub_ref, info in summary["sub_references"].items():
        file_name = info.get("file", "N/A")
        click.echo(f"    - {sub_ref} -> {file_name}")

    click.echo("-" * 60)
