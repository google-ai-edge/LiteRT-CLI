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

"""Clean command for LiteRT CLI."""

from __future__ import annotations

import pathlib
import shlex
import shutil
import subprocess

import click
from litert_cli.core import android_utils
from litert_cli.core import constants


@click.command(name="clean")
def clean_cmd() -> None:
  """Cleans up local caches, downloaded files, and remote Android directories."""

  click.echo("Cleaning LiteRT CLI workspace...")

  # 1. Clean local downloaded CLI root qairt directory
  qairt_dir = pathlib.Path(constants.LITERT_CLI_ROOT) / "qairt"
  try:
    shutil.rmtree(qairt_dir)
    click.echo(f"Removing local qairt workspace: {qairt_dir}")
  except FileNotFoundError:
    pass
  except OSError as e:
    click.secho(f"Warning: Failed to remove {qairt_dir}: {e}", fg="yellow")

  # 2. Clean local cache directory (e.g., ~/.cache/litert-cli)
  cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR)
  try:
    shutil.rmtree(cache_dir)
    click.echo(f"Removing local cache: {cache_dir}")
  except FileNotFoundError:
    pass
  except OSError as e:
    click.secho(f"Warning: Failed to remove {cache_dir}: {e}", fg="yellow")

  # 3. Clean remote Android directory
  try:
    android_utils.check_adb()
    android_root = constants.LITERT_CLI_ANDROID_ROOT
    click.echo(f"Removing remote Android workspace via adb: {android_root}")

    subprocess.run(
        ["adb", "shell", f"rm -rf {shlex.quote(android_root)}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
  except click.ClickException:
    click.echo(
        "No active Android device found via adb. Skipping remote cleanup."
    )

  click.secho("Cleanup complete!", fg="green")
