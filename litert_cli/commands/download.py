"""Command line interface for downloading LiteRT models from huggingface.

External packages like requests, or huggingface_hub are optional and will be
auto-installed when the command is invoked.
"""

from __future__ import annotations

import pathlib
import textwrap
import urllib.parse

import click
import huggingface_hub
import huggingface_hub.utils
import requests

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
