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

"""Plugin discovery for model specific handlers."""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
from typing import Any

import click
from immutabledict import immutabledict
from litert_cli.core import constants

from . import base

# Cache file path
_MODEL_REGISTRY_CACHE = os.path.join(
    constants.LITERT_MODELS_CACHE_DIR, "model_registry.json"
)

# Global list of instantiated plugins (lazy loaded)
_MODEL_PLUGINS: list[base.ModelIntentHandler] = []
# Global registry of discovered plugin metadata
_MODEL_REGISTRY: tuple[immutabledict[str, Any], ...] = ()


def _extract_metadata(module: Any, module_name: str) -> list[dict[str, Any]]:
  """Extracts plugin metadata from a module without full instantiation if possible."""
  metadata_list = []
  for name in dir(module):
    obj = getattr(module, name)
    if (
        isinstance(obj, type)
        and issubclass(obj, base.ModelIntentHandler)
        and obj is not base.ModelIntentHandler
    ):
      # Extract static attributes used for matching
      metadata = {
          "module": module_name,
          "class": obj.__name__,
          "hf_ids": getattr(obj, "HF_MODEL_IDS", []),
          "intents": getattr(
              obj, "SUPPORTED_INTENTS", ["run", "convert", "evaluate"]
          ),
      }
      metadata_list.append(metadata)
  return metadata_list


def discover_models(force_rescan: bool = False) -> None:
  """Recursively discover models. Uses local cache if available."""
  global _MODEL_REGISTRY

  if not force_rescan and os.path.exists(_MODEL_REGISTRY_CACHE):
    try:
      with open(_MODEL_REGISTRY_CACHE, "r") as f:
        loaded_registry = json.load(f)
        _MODEL_REGISTRY = tuple(immutabledict(d) for d in loaded_registry)
      return
    except (json.JSONDecodeError, IOError):
      pass  # Fallback to rescan if cache is corrupted

  # Perform full scan
  registry_list = []
  package_path = __path__[0]
  package_name = __name__

  # Walk and import to extract metadata
  for _, name, _ in pkgutil.walk_packages([package_path], package_name + "."):
    if name == "litert_cli.models.base":
      continue
    try:
      module = importlib.import_module(name)
      registry_list.extend(_extract_metadata(module, name))
    except Exception as e:  # pylint: disable=broad-except
      click.echo(f"Failed to scan module {name}: {e}", err=True)

  _MODEL_REGISTRY = tuple(immutabledict(d) for d in registry_list)

  # Save to local record
  try:
    os.makedirs(os.path.dirname(_MODEL_REGISTRY_CACHE), exist_ok=True)
    with open(_MODEL_REGISTRY_CACHE, "w") as f:
      json.dump(registry_list, f, indent=2)
  except IOError as e:
    click.echo(f"Failed to save model registry: {e}", err=True)


def dispatch_model_intent(intent: str, model_id: str, **kwargs: Any) -> Any:
  """Finds a model plugin that can handle the intent and executes it (lazy loaded)."""
  if not _MODEL_REGISTRY:
    discover_models()

  # Check already instantiated plugins first
  for plugin in _MODEL_PLUGINS:
    if plugin.can_handle(intent, model_id):
      return plugin.handle(intent, model_id, **kwargs)

  # Look up in registry metadata
  for meta in _MODEL_REGISTRY:
    # Check if this plugin potentially handles the intent and model
    if intent not in meta["intents"]:
      continue
    if model_id not in meta["hf_ids"]:
      continue

    # Found a match! Now lazily import and instantiate
    try:
      module = importlib.import_module(meta["module"])
      cls = getattr(module, meta["class"])
      plugin = cls()
      _MODEL_PLUGINS.append(plugin)

      click.echo(
          f"Plugin {plugin.__class__.__name__} is handling intent"
          f" '{intent}' for model '{model_id}'"
      )
      return plugin.handle(intent, model_id, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      click.echo(
          f"Failed to instantiate model {meta['class']} from"
          f" {meta['module']}: {e}",
          err=True,
      )

  return None
