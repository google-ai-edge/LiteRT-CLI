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
from typing import NamedTuple

import click
from litert_cli.core import android_utils
from litert_cli.core import constants


class _CleanTarget(NamedTuple):
  """A local filesystem cleanup target."""

  label: str
  path: pathlib.Path


def _remove_path(target: _CleanTarget, dry_run: bool) -> None:
  """Removes a local cleanup target if it exists."""
  if dry_run:
    click.echo(f"Would remove {target.label}: {target.path}")
    return

  try:
    shutil.rmtree(target.path)
    click.echo(f"Removing {target.label}: {target.path}")
  except FileNotFoundError:
    pass
  except OSError as e:
    click.secho(
        f"Warning: Failed to remove {target.path}: {e}", fg="yellow"
    )


def _clean_android_workspace(dry_run: bool) -> None:
  """Removes the remote Android workspace, if a device is available."""
  android_root = constants.LITERT_CLI_ANDROID_ROOT
  if dry_run:
    click.echo(f"Would remove remote Android workspace: {android_root}")
    return

  try:
    android_utils.check_adb()
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


@click.command(name="clean")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be removed without deleting anything.",
)
@click.option(
    "--models",
    "clean_models",
    is_flag=True,
    help="Remove only the managed model cache.",
)
@click.option(
    "--binaries",
    "clean_binaries",
    is_flag=True,
    help="Remove only downloaded LiteRT binaries.",
)
@click.option(
    "--targets",
    "clean_targets",
    is_flag=True,
    help="Remove only cached SoC target metadata.",
)
@click.option(
    "--root",
    "clean_root",
    is_flag=True,
    help="Remove only the local LiteRT CLI root workspace.",
)
@click.option(
    "--android",
    "clean_android",
    is_flag=True,
    help="Remove only the remote Android workspace.",
)
def clean_cmd(
    dry_run: bool,
    clean_models: bool,
    clean_binaries: bool,
    clean_targets: bool,
    clean_root: bool,
    clean_android: bool,
) -> None:
  """Cleans up local caches, downloaded files, and remote Android directories."""

  selected_local_targets = []
  any_selector = any((
      clean_models,
      clean_binaries,
      clean_targets,
      clean_root,
      clean_android,
  ))

  if not any_selector:
    selected_local_targets.append(
        _CleanTarget("local cache", pathlib.Path(constants.LITERT_CLI_CACHE_DIR))
    )
    clean_android = True
  else:
    cache_dir = pathlib.Path(constants.LITERT_CLI_CACHE_DIR)
    if clean_models:
      selected_local_targets.append(
          _CleanTarget(
              "managed model cache",
              pathlib.Path(constants.LITERT_MODELS_CACHE_DIR),
          )
      )
    if clean_binaries:
      selected_local_targets.append(
          _CleanTarget("downloaded binaries", cache_dir / "binaries")
      )
    if clean_targets:
      selected_local_targets.append(
          _CleanTarget("SoC target metadata", cache_dir / "targets")
      )
    if clean_root:
      selected_local_targets.append(
          _CleanTarget(
              "local root workspace", pathlib.Path(constants.LITERT_CLI_ROOT)
          )
      )

  action = "Dry run: checking" if dry_run else "Cleaning"
  click.echo(f"{action} LiteRT CLI workspace...")

  for target in selected_local_targets:
    _remove_path(target, dry_run)

  if clean_android:
    _clean_android_workspace(dry_run)

  done_message = "Dry run complete!" if dry_run else "Cleanup complete!"
  click.secho(done_message, fg="green")
