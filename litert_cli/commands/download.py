"""Command line interface for downloading LiteRT models from huggingface.

External packages like requests, or huggingface_hub are optional and will be
auto-installed when the command is invoked.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import textwrap
import urllib.parse

import click
import huggingface_hub
import huggingface_hub.utils
import requests

from ..core import constants
from ..core import deps


def _download_from_url(model_path: str, output_dir: str | pathlib.Path) -> None:
  """Downloads a model directly from a given URL.

  Args:
    model_path: The direct URL to the model file.
    output_dir: The directory where the model will be saved.
  """
  click.echo(f"Downloading direct URL: {model_path} to {output_dir}")

  parsed_url = urllib.parse.urlparse(model_path)
  filename = pathlib.Path(parsed_url.path).name
  if not filename:
    filename = "downloaded_model.tflite"

  filepath = pathlib.Path(output_dir) / filename

  try:
    with requests.get(model_path, stream=True) as response:
      response.raise_for_status()

      total_size = int(response.headers.get("content-length", 0))
      chunk_size = 8192

      with click.progressbar(
          length=total_size,
          label=f"Downloading {filename}",
          empty_char=" ",
          fill_char=click.style("#", fg="green"),
      ) as bar:
        with open(filepath, "wb") as f:
          for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
              f.write(chunk)
              bar.update(len(chunk))

    click.secho(f"Downloaded to {filepath}", fg="green")
  except requests.exceptions.RequestException as e:
    raise click.ClickException(f"Failed to download URL: {e}") from e


def _download_from_hf(
    repo_id: str,
    output_dir: str | pathlib.Path,
    file_pattern: str | None = None,
    token: str | None = None,
) -> None:
  """Downloads a model repository or specific files from HuggingFace Hub.

  Args:
    repo_id: The HuggingFace Hub repository ID.
    output_dir: The directory where the files will be saved.
    file_pattern: Optional glob pattern to filter files to download.
    token: Optional HuggingFace access token.
  """

  click.echo(f"Downloading from HuggingFace Hub: {repo_id}")

  kwargs = {
      "local_dir": output_dir,
  }

  if token:
    kwargs["token"] = token
  if file_pattern:
    kwargs["allow_patterns"] = file_pattern

  try:
    downloaded_path = huggingface_hub.snapshot_download(
        repo_id=repo_id, **kwargs
    )
    click.secho(f"Successfully downloaded to {downloaded_path}", fg="green")
  except huggingface_hub.utils.HfHubHTTPError as e:
    if "401" in str(e):
      raise click.ClickException(
          f"Failed to download from HuggingFace: {e}\n"
          "This model might require a token. Use --token"
      ) from e
    raise click.ClickException(
        f"Failed to download from HuggingFace: {e}"
    ) from e
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Catching broad exception as fallback for other huggingface_hub errors
    raise click.ClickException(
        f"An error occurred during HuggingFace download: {e}"
    ) from e


@click.command(
    "download",
    help=textwrap.dedent("""\
    Download models from URL or HuggingFace.

    MODEL_PATH: Direct URL to a file, or a HuggingFace Hub repository ID.

    Examples:

      Download a direct URL to the current directory:

        $ litert download https://example.com/model.tflite

      Download all .tflite files from a HuggingFace repo:

        $ litert download Qwen/Qwen1.5-0.5B-Chat --file "*.tflite"

      Download from a private repository using a token:

        $ litert download MyOrg/PrivateModel --token hf_your_token
    """),
)
@click.argument("model_path")
@click.option(
    "--model-ref",
    help=(
        "Reference name for the model in centralized cache (e.g."
        " 'mobilenet:v3')."
    ),
)
@click.option(
    "--output",
    help="Specify output directory (defaults to centralized cache).",
)
@click.option(
    "--file",
    "file_pattern",
    help="Only download specific files in the HF repository (e.g. '*.tflite')",
)
@click.option("--token", help="Private model requiring a Token")
@click.option(
    "--hf-id", help="Optional Hugging Face ID to associate with the model."
)
@deps.require_extra("download")
def download_cmd(
    model_path: str,
    model_ref: str | None,
    output: str | None,
    file_pattern: str | None,
    token: str | None,
    hf_id: str | None,
) -> None:
  """Downloads models from URL or HuggingFace.

  Args:
    model_path: Direct URL to a file, or a HuggingFace Hub repository ID.
    model_ref: Optional reference name for the model.
    output: Optional custom directory to save files.
    file_pattern: Glob pattern to filter downloaded files from HF.
    token: HuggingFace access token for private repositories.
    hf_id: Optional Hugging Face ID to associate with the model.
  """
  # Resolve output directory and handle sub-ref in model_ref
  if model_ref and ":" in model_ref:
    ref, resolved_sub_ref = model_ref.split(":", 1)
  else:
    ref = model_ref
    resolved_sub_ref = None

  if output:
    output_path = pathlib.Path(output)
  else:
    # Use centralized cache by default for HF models
    if not model_path.startswith(("http://", "https://")):
      # Default to hf_id as ref if not provided
      ref = ref or model_path.replace("/", "__")
      # Flatten ref for directory name consistency
      ref_flat = ref.replace("/", "__") if "/" in ref else ref
      output_path = pathlib.Path(constants.LITERT_MODELS_CACHE_DIR) / ref_flat
    else:
      output_path = pathlib.Path(".")

  output_path.mkdir(parents=True, exist_ok=True)

  if model_path.startswith(("http://", "https://")):
    _download_from_url(model_path, output_path)
  else:
    _download_from_hf(model_path, output_path, file_pattern, token)

    # Save metadata if it's a managed download (non-URL)
    if not model_path.startswith(("http://", "https://")):
      final_ref = ref or model_path.replace("/", "__")

      # Load existing metadata if it exists (to preserve other sub-refs)
      metadata_file = output_path / "metadata.json"
      metadata = {}
      try:
        with open(metadata_file, "r") as f:
          metadata = json.load(f)
      except (IOError, json.JSONDecodeError):
        # If the file doesn't exist or is malformed, start with empty metadata.
        pass

      # Update or create metadata
      metadata.setdefault("files", [])
      metadata.setdefault("sub_references", {})

      metadata.update({
          "model_ref": final_ref,
          "hf_id": hf_id or model_path,
          "source": "huggingface",
          "created_at": (
              datetime.datetime.now(datetime.timezone.utc).isoformat()
          ),
      })

      if file_pattern and file_pattern not in metadata["files"]:
        metadata["files"].append(file_pattern)

      # Handle sub-references
      if resolved_sub_ref:
        sub_references = metadata["sub_references"]

        # If we downloaded a specific file, find it in the directory!
        # Since file_pattern might be a glob, we search for matching files.
        matched_files = []
        if file_pattern:
          # Glob search in output_path
          matched_files = list(output_path.glob(file_pattern))

        file_to_map = ""
        if matched_files:
          # Use the first matched file for now
          file_to_map = matched_files[0].name
        elif file_pattern and "*" not in file_pattern:
          # If it's not a glob, assume it's the file name
          file_to_map = file_pattern

        sub_references[resolved_sub_ref] = {
            "file": file_to_map,
            "hf_id": hf_id or model_path,
        }
        click.echo(
            f"Mapped sub-reference '{resolved_sub_ref}' to file '{file_to_map}'"
        )

      with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
      click.secho(f"Metadata saved to {metadata_file}", fg="green")
