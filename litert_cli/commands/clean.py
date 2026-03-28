"""Clean command for LiteRT CLI."""

import pathlib
import shutil
import subprocess

import click
from litert_cli.core import android_utils
from litert_cli.core import constants


@click.command(name="clean")
def clean_cmd() -> None:
  """Cleans up local caches, downloaded files, and remote Android directories."""
  click.echo("Cleaning LiteRT CLI workspace...")

  # 1. Clean local downloaded CLI root qairt directory (e.g., /tmp/litert-cli/qairt)
  qairt_dir = pathlib.Path(constants.LITERT_CLI_ROOT) / "qairt"
  if qairt_dir.exists() and qairt_dir.is_dir():
    click.echo(f"Removing local qairt workspace: {qairt_dir}")
    try:
      shutil.rmtree(qairt_dir)
    except Exception as e:
      click.secho(f"Warning: Failed to remove {qairt_dir}: {e}", fg="yellow")

  # 2. Clean local cache directory (e.g., ~/.cache/litert-cli)
  cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR)
  if cache_dir.exists() and cache_dir.is_dir():
    click.echo(f"Removing local cache: {cache_dir}")
    try:
      shutil.rmtree(cache_dir)
    except Exception as e:
      click.secho(f"Warning: Failed to remove {cache_dir}: {e}", fg="yellow")

  # 3. Clean remote Android directory
  try:
    # Check if adb is setup and available
    android_utils.check_adb()
    android_root = constants.LITERT_CLI_ANDROID_ROOT
    click.echo(f"Removing remote Android workspace via adb: {android_root}")

    subprocess.run(
        ["adb", "shell", f"rm -rf {android_root}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
  except click.ClickException:
    click.echo(
        "No active Android device found via adb. Skipping remote cleanup."
    )

  click.secho("Cleanup complete!", fg="green")
