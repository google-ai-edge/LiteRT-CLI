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

"""Command line interface for deleting managed models from LiteRT cache."""

from __future__ import annotations

import shutil

import click
from litert_cli.core import models


@click.command(
    "delete", help="Delete a managed model from the centralized cache."
)
@click.argument("ref")
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation.")
def delete_cmd(ref: str, yes: bool) -> None:
  """Deletes a managed model from the centralized cache.

  Args:
    ref: The model reference to delete.
    yes: Whether to skip confirmation prompt.
  """
  if ":" in ref:
    raise click.BadParameter(
        f"'{ref}' names a sub-reference; `delete` removes whole models only."
        f" Use '{ref.split(':', 1)[0]}' to delete the model.",
        param_hint="ref",
    )
  cache_path = models.cache_dir_for_ref(ref)

  if not cache_path.exists() or not cache_path.is_dir():
    raise click.ClickException(f"Model reference '{ref}' not found in cache.")

  if not yes:
    if not click.confirm(f"Are you sure you want to delete model '{ref}'?"):
      click.echo("Aborted.")
      return

  try:
    shutil.rmtree(cache_path)
    click.secho(f"Successfully deleted model '{ref}' from cache.", fg="green")
  except OSError as e:
    raise click.ClickException(f"Failed to delete model: {e}") from e
