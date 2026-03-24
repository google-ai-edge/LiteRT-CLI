"""Dependency management module for LiteRT CLI.

This module provides utilities to manage optional dependencies for the LiteRT
CLI. It allows for on-the-fly installation of dependencies when a command that
requires them is invoked.
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import subprocess
import sys
from typing import Any, Callable

import click

# Map extra names to Python module names to check if the optional
# dependency is already installed.
_MODULE_BY_EXTRA = {
    "torch": "litert_torch",
    "lm": "litert_lm",
    "download": "huggingface_hub",
    "run": "ai_edge_litert",
    "visualize": "model_explorer",
    "quantize": "ai_edge_quantizer",
    "image": "PIL",
}


def ensure_extra(extra_name: str, silent: bool = False) -> bool:
  """Ensures the required extra dependency is installed.

  Args:
    extra_name: The name of the extra dependency (e.g. 'torch', 'image').
    silent: If True, suppresses output and returns False instead of exiting on
      failure. If False, prints status and exits with code 1 on failure.

  Returns:
    True if the dependency is already installed or successfully installed now.
  """
  module_to_check = _MODULE_BY_EXTRA.get(extra_name)

  if not module_to_check:
    if not silent:
      click.secho(f"Internal error: Unknown extra '{extra_name}'", fg="red")
      raise click.Abort()
    return False

  # Check if the module is installed
  if importlib.util.find_spec(module_to_check) is None:
    if not silent:
      click.secho(
          f"[*] Initializing '{extra_name}' components for the first time...",
          fg="cyan",
      )

    project_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject_path = project_root / "pyproject.toml"

    # If pyproject.toml exists, install from local source
    if pyproject_path.exists():
      target = f".[{extra_name}]"
      cwd = str(project_root)
    else:
      # Otherwise, install from pypi
      target = f"litert-cli[{extra_name}]"
      cwd = None

    if not silent:
      click.echo(f"    Running: pip install -q {target}")

    try:
      subprocess.check_call(
          [sys.executable, "-m", "pip", "install", "-q", target],
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
            "Please try installing it manually or check your network"
            " connection.",
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
