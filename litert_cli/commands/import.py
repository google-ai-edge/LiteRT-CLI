"""Command line interface for importing local models into LiteRT cache."""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil

import click

from ..core import constants


@click.command(
    "import",
    help="Import a local file or directory into the centralized cache.",
)
@click.argument(
    "file_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--model-ref",
    required=True,
    help="Reference name for the model (e.g. 'my_model:enc').",
)
@click.option("--hf-id", help="Hugging Face ID to associate with the model.")
def import_cmd(
    file_path: pathlib.Path, model_ref: str, hf_id: str | None
) -> None:
  """Imports a local file or directory into the centralized cache.

  Args:
    file_path: Path to the local file or directory to import.
    model_ref: Reference name for management.
    hf_id: Optional Hugging Face ID.
  """
  # Parse model_ref for sub-reference if using ':' syntax
  main_ref = model_ref
  resolved_sub_ref = None

  if ":" in model_ref:
    parts = model_ref.split(":", 1)
    main_ref = parts[0]
    resolved_sub_ref = parts[1]

  # Flatten main_ref for directory name consistency (e.g. a/b -> a__b)
  main_ref_flat = main_ref.replace("/", "__") if "/" in main_ref else main_ref
  output_path = pathlib.Path(constants.LITERT_MODELS_CACHE_DIR) / main_ref_flat
  output_path.mkdir(parents=True, exist_ok=True)

  click.echo(f"Importing {file_path} to {output_path}...")

  try:
    if file_path.is_dir():
      # Copy directory content, not the directory itself, to avoid nested
      # folders
      for item in file_path.iterdir():
        if item.is_dir():
          shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
        else:
          shutil.copy(item, output_path / item.name)
    else:
      shutil.copy(file_path, output_path / file_path.name)

    click.secho(f"Successfully imported to {output_path}", fg="green")

    # Save metadata
    metadata_file = output_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
      try:
        with open(metadata_file, "r") as f:
          metadata = json.load(f)
      except (IOError, json.JSONDecodeError):
        pass

    metadata.update({
        "model_ref": main_ref,
        "hf_id": hf_id,
        "source": "imported",
        "import_path": str(file_path.resolve()),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })

    # Handle sub-references
    if resolved_sub_ref:
      if "sub_references" not in metadata:
        metadata["sub_references"] = {}

      # Use the file name in the destination directory
      file_to_map = file_path.name
      metadata["sub_references"][resolved_sub_ref] = {
          "file": file_to_map,
          "hf_id": hf_id,
      }
      click.echo(
          f"Mapped sub-reference '{resolved_sub_ref}' to file '{file_to_map}'"
      )

    with open(metadata_file, "w") as f:
      json.dump(metadata, f, indent=2)
    click.secho(f"Metadata saved to {metadata_file}", fg="green")

  except Exception as e:
    raise click.ClickException(f"Failed to import model: {e}") from e
