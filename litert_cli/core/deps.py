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

"""Dependency management module for LiteRT CLI.

This module provides utilities to manage optional dependencies for the LiteRT
CLI. It allows for on-the-fly installation of dependencies when a command that
requires them is invoked.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import importlib.metadata
import importlib.util
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import click
from immutabledict import immutabledict

# Distribution names this CLI may be published under, most specific first.
_CLI_DISTRIBUTIONS = ("litert-cli-nightly", "litert-cli")

# Map extra names to the distribution names used to check whether the optional
# dependency is already installed. Every key here must also exist in
# [project.optional-dependencies] in pyproject.toml, otherwise the install
# command below resolves to a non-existent extra and silently does nothing.
_PACKAGE_BY_EXTRA = immutabledict({
    "convert": ("litert-torch-nightly", "litert-torch"),
    "lm": ("litert-lm-nightly", "litert-lm"),
    "download": ("huggingface-hub",),
    "run": ("ai-edge-litert-nightly", "ai-edge-litert"),
    "compile": (
        "ai-edge-litert-sdk-qualcomm-nightly",
        "ai-edge-litert-sdk-qualcomm",
    ),
    "visualize": ("ai-edge-model-explorer",),
    "quantize": ("ai-edge-quantizer-nightly", "ai-edge-quantizer"),
    "image": ("Pillow",),
    "asr": ("librosa",),
})


def _installed_cli_distribution() -> str | None:
  """Returns the distribution name this CLI was pip-installed as.

  Returns:
    The distribution name (e.g. 'litert-cli-nightly'), or None when the CLI is
    not running from a pip-installed distribution. The latter happens for
    hermetic builds that bundle their dependencies, where installing anything
    at runtime is both unnecessary and unsafe.
  """
  for dist in _CLI_DISTRIBUTIONS:
    try:
      importlib.metadata.version(dist)
      return dist
    except importlib.metadata.PackageNotFoundError:
      continue
  return None


def _is_extra_installed(packages_to_check: tuple[str, ...]) -> bool:
  """Returns True if any of the given distributions is installed."""
  for pkg in packages_to_check:
    try:
      importlib.metadata.version(pkg)
      return True
    except importlib.metadata.PackageNotFoundError:
      continue
  return False


def ensure_extra(extra_name: str, *, silent: bool = False) -> bool:
  """Ensures the required extra dependency is installed.

  Args:
    extra_name: The name of the extra dependency (e.g. 'convert', 'image').
    silent: If True, suppresses output and returns False instead of exiting on
      failure. If False, prints status and exits with code 1 on failure.

  Returns:
    True if the dependency is already installed or successfully installed now.

  Raises:
    click.Abort: If `silent` is False and the dependency cannot be ensured
      (e.g., unknown extra or installation failure).
  """
  packages_to_check = _PACKAGE_BY_EXTRA.get(extra_name)

  if not packages_to_check:
    if not silent:
      click.secho(
          f"Internal error: Unknown extra '{extra_name}'", fg="red", err=True
      )
      raise click.Abort()
    return False

  if _is_extra_installed(packages_to_check):
    return True

  cli_package = _installed_cli_distribution()
  project_root = pathlib.Path(__file__).resolve().parents[1]
  pyproject_path = project_root / "pyproject.toml"

  if pyproject_path.exists():
    # Running from a source checkout: install the extra from local source.
    target = f".[{extra_name}]"
    cwd = str(project_root)
  elif cli_package is not None:
    target = f"{cli_package}[{extra_name}]"
    cwd = None
  else:
    # Not pip-installed and not a source checkout, i.e. a hermetic build that
    # bundles its dependencies. Nothing to install; let the caller proceed and
    # surface a normal ImportError if the assumption is wrong.
    return True

  if not silent:
    click.secho(
        f"[*] Initializing '{extra_name}' components for the first time...",
        fg="cyan",
    )

  uv_path = shutil.which("uv")
  if uv_path:
    cmd = [uv_path, "pip", "install", "--python", sys.executable, target]
    cmd_str = f"uv pip install -q {target}"
  else:
    cmd = [sys.executable, "-m", "pip", "install", target]
    cmd_str = f"pip install -q {target}"

  if not silent:
    click.echo(f"    Running: {cmd_str}")

  try:
    subprocess.check_call(
        cmd,
        cwd=cwd,
        stdout=subprocess.DEVNULL if silent else None,
        stderr=subprocess.DEVNULL if silent else None,
    )
    importlib.invalidate_caches()
  except (subprocess.CalledProcessError, OSError) as e:
    if silent:
      return False
    click.secho(
        f"\n[!] Failed to auto-install '{extra_name}' components.",
        fg="red",
        bold=True,
        err=True,
    )
    click.secho(f"    {e}", fg="red", err=True)
    click.secho(
        f"Please install it manually with: {cmd_str}", fg="yellow", err=True
    )
    raise click.Abort() from e

  # A zero exit status is not proof that anything was installed: `uv pip
  # install pkg[does-not-exist]` succeeds silently. Verify before claiming so.
  if not _is_extra_installed(packages_to_check):
    if silent:
      return False
    click.secho(
        f"\n[!] '{extra_name}' components still missing after installing"
        f" {target}.",
        fg="red",
        bold=True,
        err=True,
    )
    click.secho(
        "    Expected one of: " + ", ".join(packages_to_check),
        fg="red",
        err=True,
    )
    raise click.Abort()

  if not silent:
    click.secho(
        f"[+] Successfully installed '{extra_name}' components!\n",
        fg="green",
    )

  return True


def require_extra(
    extra_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Click command decorator: Ensures required extra dependency.

  Auto-installs using pip if it's missing. Exits on failure.

  Args:
    extra_name: The name of the extra dependency to ensure is installed.

  Returns:
    The decorated function.
  """

  def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      # Dependencies are ready, proceed with the command
      ensure_extra(extra_name, silent=False)
      return f(*args, **kwargs)

    return wrapper

  return decorator
