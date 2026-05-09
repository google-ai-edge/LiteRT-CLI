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

"""Manager for loading and syncing supported SoC targets."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import urllib.request

import click


@dataclass
class SoCTarget:
  """Unified structure for a supported SoC target."""

  soc_id: str  # Unified lowercase ID, e.g., "sm8850", "mt6895z"
  vendor: str  # Vendor name, e.g., "qualcomm", "mediatek"
  vendor_id: str  # Vendor specific ID, e.g., "SM8850", "MT6895Z_A/TCZA"
  properties: Dict[str, str] = field(default_factory=dict)


class TargetsManager:
  """Manages downloading, caching and parsing of supported SoC lists."""

  def __init__(self, cache_dir: Optional[str] = None):
    from litert_cli.core.constants import LITERT_CLI_CACHE_DIR
    self.cache_dir = cache_dir or os.path.join(LITERT_CLI_CACHE_DIR, "targets")

    # URL templates, supporting version replacement
    self.urls = {
        "qualcomm": (
            "https://raw.githubusercontent.com/google-ai-edge/LiteRT/"
            "{version}/litert/vendors/qualcomm/supported_soc.csv"
        ),
        "mediatek": (
            "https://raw.githubusercontent.com/google-ai-edge/LiteRT/"
            "{version}/litert/vendors/mediatek/supported_soc.csv"
        ),
        "intel": (
            "https://raw.githubusercontent.com/google-ai-edge/LiteRT/"
            "{version}/litert/vendors/intel_openvino/supported_soc.csv"
        ),
    }

  def get_csv_path(self, vendor: str) -> str:
    """Get the local cache path for a vendor's CSV file."""
    return os.path.join(self.cache_dir, vendor, "supported_soc.csv")

  def download_targets(
      self, version: str = "main", vendor: Optional[str] = None
  ):
    """Download targets from GitHub.

    Args:
        version: The git tag or branch (e.g., "main", "v2.1.4").
        vendor: Specific vendor to download, or None for all.
    """
    vendors = [vendor] if vendor else self.urls.keys()
    for v in vendors:
      url = self.urls[v].format(version=version)
      dest_path = self.get_csv_path(v)
      os.makedirs(os.path.dirname(dest_path), exist_ok=True)

      click.echo(f"Downloading {v} targets from {url}...")
      try:
        # Using standard urllib to avoid external dependencies like requests
        urllib.request.urlretrieve(url, dest_path)
        click.echo(f"Saved to {dest_path}")
      except Exception as e:
        click.echo(f"Failed to download {v} targets: {e}")
        raise

  def load_targets(self) -> Dict[str, SoCTarget]:
    """Load and parse all cached targets.

    Returns:
        A dictionary mapping unified soc_id to SoCTarget.
    """
    targets = {}
    for vendor in self.urls.keys():
      path = self.get_csv_path(vendor)
      if os.path.exists(path):
        if vendor == "qualcomm":
          targets.update(self._parse_qualcomm(path))
        elif vendor == "mediatek":
          targets.update(self._parse_mediatek(path))
        elif vendor == "intel":
          targets.update(self._parse_intel(path))
    return targets

  def _parse_qualcomm(self, path: str) -> Dict[str, SoCTarget]:
    """Parse Qualcomm CSV."""
    targets = {}
    with open(path, "r", encoding="utf-8") as f:
      reader = csv.reader(f)
      for row in reader:
        # Skip empty lines or headers/comments starting with '#'
        if not row or row[0].strip().startswith("#"):
          continue

        # Schema: manufacturer,model,runtime_library_version,soc_model
        # Example: Qualcomm,SM8850,v81,87
        if len(row) >= 3:
          model = row[1].strip()
          version = row[2].strip()

          soc_id = model.lower()
          targets[soc_id] = SoCTarget(
              soc_id=soc_id,
              vendor="qualcomm",
              vendor_id=model,
              properties={"qnn_version": version.lstrip("v")},
          )
    return targets

  def _parse_mediatek(self, path: str) -> Dict[str, SoCTarget]:
    """Parse MediaTek CSV."""
    targets = {}
    with open(path, "r", encoding="utf-8") as f:
      reader = csv.reader(f)
      for row in reader:
        if not row or row[0].strip().startswith("#"):
          continue

        # Schema: manufacturer,model,Recommend Version,Compatible Version
        # Example: Mediatek,MT6895Z_A/TCZA,v8,"v8, v9"
        if len(row) >= 2:
          model = row[1].strip()
          recommend = row[2].strip() if len(row) > 2 else ""
          compatible = row[3].strip() if len(row) > 3 else ""

          soc_id = model.lower()
          targets[soc_id] = SoCTarget(
              soc_id=soc_id,
              vendor="mediatek",
              vendor_id=model,
              properties={
                  "recommend_version": recommend,
                  "compatible_version": compatible,
              },
          )
    return targets

  def _parse_intel(self, path: str) -> Dict[str, SoCTarget]:
    """Parse Intel CSV."""
    targets = {}
    with open(path, "r", encoding="utf-8") as f:
      reader = csv.reader(f)
      for row in reader:
        if not row or row[0].strip().startswith("#"):
          continue

        # Schema: manufacturer,SOC Name, NPU version, Platform name, Recommend OpenVINO Version,Compatible OpenVINO Version
        # Example: Intel, LNL, NPU4000, Intel NPU for Intel Core Ultra processors..., ,
        if len(row) >= 2:
          soc_name = row[1].strip()
          npu_version = row[2].strip() if len(row) > 2 else ""
          platform_name = row[3].strip() if len(row) > 3 else ""
          recommend = row[4].strip() if len(row) > 4 else ""
          compatible = row[5].strip() if len(row) > 5 else ""

          soc_id = soc_name.lower()
          targets[soc_id] = SoCTarget(
              soc_id=soc_id,
              vendor="intel",
              vendor_id=soc_name,
              properties={
                  "npu_version": npu_version,
                  "platform_name": platform_name,
                  "recommend_version": recommend,
                  "compatible_version": compatible,
              },
          )
    return targets
