#!/usr/bin/env python3
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

"""Log filters for LiteRT CLI commands to prevent terminal noise."""


class BenchmarkLogFilter:
  """Filters output of litert benchmark command."""

  def __init__(self, default_quiet: bool):
    self.default_quiet = default_quiet

  def should_show(self, line: str) -> bool:
    """Determines if a line should be shown in the output."""
    is_core_info = (
        "benchmark_litert_model" in line
        or "compiler_plugin.cc" in line
        or ("Replacing" in line and "node(s) with delegate" in line)
    )
    return not self.default_quiet or is_core_info


class RunLogFilter:
  """Filters output of litert run command."""

  def __init__(self, default_quiet: bool, print_tensors: bool):
    self.default_quiet = default_quiet
    self.print_tensors = print_tensors

  def should_show(self, line: str) -> bool:
    """Determines if a line should be shown in the output."""
    is_core_info = (
        "run_model.cc" in line
        or "compiler_plugin.cc" in line
        or ("Replacing" in line and "node(s) with delegate" in line)
    )
    return not self.default_quiet or is_core_info or self.print_tensors
