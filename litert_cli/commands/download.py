"""Command line interface for downloading LiteRT models from huggingface.

External packages like requests, or huggingface_hub are optional and will be
auto-installed when the command is invoked.
"""

from __future__ import annotations

import pathlib

import click
from litert_cli.core import deps


def _download_from_url(model_path: str, output_dir: str | pathlib.Path) -> None:
  """Downloads a model directly from a given URL.

  Args:
    model_path: The direct URL to the model file.
    output_dir: The directory where the model will be saved.
  """
  try:
    import requests  # pylint: disable=g-import-not-at-top
  except ImportError:
    click.secho(
        "The 'requests' library is required for URL downloads. "
        "Please install the 'download' extra.",
        fg="red",
    )
    return

  click.echo(f"Downloading direct URL: {model_path} to {output_dir}")

  filename = model_path.split("/")[-1] or "downloaded_model.tflite"

  filepath = pathlib.Path(output_dir) / filename

  try:
    with requests.get(model_path, stream=True) as response:
      response.raise_for_status()

      with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

    click.secho(f"Downloaded to {filepath}", fg="green")
  except requests.exceptions.RequestException as e:
    click.secho(f"Failed to download URL: {e}", fg="red")


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
  try:
    # pylint: disable=g-import-not-at-top
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
  except ImportError:
    click.secho(
        "The 'huggingface_hub' library is required for HuggingFace downloads. "
        "Please install the 'download' extra.",
        fg="red",
    )
    return

  click.echo(f"Downloading from HuggingFace Hub: {repo_id}")

  kwargs = {
      "local_dir": output_dir,
  }

  if token:
    kwargs["token"] = token
  if file_pattern:
    kwargs["allow_patterns"] = file_pattern

  try:
    downloaded_path = snapshot_download(repo_id=repo_id, **kwargs)
    click.secho(f"Successfully downloaded to {downloaded_path}", fg="green")
  except HfHubHTTPError as e:
    click.secho(f"Failed to download from HuggingFace: {e}", fg="red")
    if "401" in str(e):
      click.secho("This model might require a token. Use --token", fg="yellow")
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Catching broad exception as fallback for other huggingface_hub errors
    click.secho(f"An error occurred during HuggingFace download: {e}", fg="red")


@click.command(
    "download",
    help="""Download models from URL or HuggingFace.

MODEL_PATH: Direct URL to a file, or a HuggingFace Hub repository ID.

Examples:

  Download a direct URL to the current directory:

    $ litert download https://example.com/model.tflite

  Download all .tflite files from a HuggingFace repo:

    $ litert download Qwen/Qwen1.5-0.5B-Chat --file "*.tflite"

  Download from a private repository using a token:

    $ litert download MyOrg/PrivateModel --token hf_your_token
""",
)
@click.argument("model_path")
@click.option("--output", default=".", help="Specify output directory")
@click.option(
    "--file",
    "file_pattern",
    help="Only download specific files in the HF repository (e.g. '*.tflite')",
)
@click.option("--token", help="Private model requiring a Token")
@deps.require_extra("download")
def download_cmd(
    model_path: str,
    output: str,
    file_pattern: str | None,
    token: str | None,
) -> None:
  """Downloads models from URL or HuggingFace.

  Args:
    model_path: Direct URL to a file, or a HuggingFace Hub repository ID.
    output: Directory to save the downloaded model files.
    file_pattern: Glob pattern to filter downloaded files from HF.
    token: HuggingFace access token for private repositories.
  """
  pathlib.Path(output).mkdir(parents=True, exist_ok=True)

  if model_path.startswith(("http://", "https://")):
    _download_from_url(model_path, output)
  else:
    _download_from_hf(model_path, output, file_pattern, token)
