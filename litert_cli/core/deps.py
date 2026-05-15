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
from litert_cli.core import constants

# Map extra names to Python module names to check if the optional
# dependency is already installed.
_PACKAGE_BY_EXTRA = immutabledict({
    "convert": ("litert-torch-nightly", "litert-torch"),
    "torch": ("litert-torch-nightly", "litert-torch"),
    "lm": ("litert-lm-nightly", "litert-lm"),
    "download": ("huggingface-hub",),
    "run": ("ai-edge-litert-nightly", "ai-edge-litert"),
    "compile": (
        "ai-edge-litert-sdk-qualcomm-nightly",
        "ai-edge-litert-sdk-qualcomm",
    ),
    "visualize": ("model-explorer",),
    "quantize": ("ai-edge-quantizer-nightly", "ai-edge-quantizer"),
    "image": ("Pillow",),
    "asr": ("librosa",),
})


def ensure_extra(extra_name: str, *, silent: bool = False) -> bool:
  """Ensures the required extra dependency is installed.

  Args:
    extra_name: The name of the extra dependency (e.g. 'torch', 'image').
    silent: If True, suppresses output and returns False instead of exiting on
      failure. If False, prints status and exits with code 1 on failure.

  Returns:
    True if the dependency is already installed or successfully installed now.

  Raises:
    click.Abort: If `silent` is False and the dependency cannot be ensured
      (e.g., unknown extra or installation failure).
  """
  if constants.IS_INTERNAL_ENV:
    return True

  packages_to_check = _PACKAGE_BY_EXTRA.get(extra_name)

  if not packages_to_check:
    if not silent:
      click.secho(f"Internal error: Unknown extra '{extra_name}'", fg="red")
      raise click.Abort()
    return False

  for pkg in packages_to_check:
    try:
      importlib.metadata.version(pkg)
      return True
    except importlib.metadata.PackageNotFoundError:
      pass

  if not silent:
    click.secho(
        f"[*] Initializing '{extra_name}' components for the first time...",
        fg="cyan",
    )

  project_root = pathlib.Path(__file__).resolve().parents[1]
  pyproject_path = project_root / "pyproject.toml"

  # If pyproject.toml exists, install from local source
  if pyproject_path.exists():
    target = f".[{extra_name}]"
    cwd = str(project_root)
  else:
    # Otherwise, install from pypi, checking if nightly CLI is installed
    cli_package = "litert-cli"
    try:
      importlib.metadata.version("litert-cli-nightly")
      cli_package = "litert-cli-nightly"
    except importlib.metadata.PackageNotFoundError:
      pass
    target = f"{cli_package}[{extra_name}]"
    cwd = None

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

    if not silent:
      click.secho(
          f"[+] Successfully installed '{extra_name}' components!\n",
          fg="green",
      )
  except subprocess.CalledProcessError as e:
    if not silent:
      click.secho(
          f"\n[!] Failed to auto-install '{extra_name}' components.",
          fg="red",
          bold=True,
      )
      click.secho(
          "Please try installing it manually or check your network connection.",
          fg="yellow",
      )
      raise click.Abort() from e
    return False

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
