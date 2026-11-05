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

"""Tests for npu_utils."""

import pathlib
from unittest import mock

from absl.testing import absltest
from litert_cli.core import npu_utils

from ai_edge_litert.aot.vendors import fallback_backend
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


class GetAotTargetsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A fixed target list instead of the downloaded one.
    self.enter_context(mock.patch.object(npu_utils, "_ensure_targets_loaded"))
    self.enter_context(
        mock.patch.object(
            npu_utils.constants,
            "AOT_SUPPORTED_TARGETS",
            {"sm8750": ("qualcomm", "SM8750")},
        )
    )

  def test_compiled_models_only(self):
    targets = npu_utils.get_aot_targets(("sm8750",))
    self.assertLen(targets, 1)
    self.assertIsInstance(targets[0], qnn_target.Target)

  def test_ai_pack_adds_the_fallback_target(self):
    targets = npu_utils.get_aot_targets(("sm8750",), export_aipack=True)
    self.assertLen(targets, 2)
    self.assertIsInstance(targets[0], qnn_target.Target)
    self.assertIsInstance(targets[1], fallback_backend.FallbackTarget)


class CompileFailureHintTest(absltest.TestCase):

  def _error_for(self, details: str) -> ValueError:
    error_file = pathlib.Path(self.create_tempdir().full_path) / "x.error"
    error_file.write_text(details)
    return ValueError(
        f"apply_plugin failed to apply plugin. See {error_file} for details."
    )

  def test_missing_libcxx_names_the_packages(self):
    error = self._error_for(
        "Could not load shared library /x/libLiteRtCompilerPlugin_Qualcomm.so:"
        " libc++.so.1: cannot open shared object file: No such file or"
        " directory.\n"
    )
    self.assertIn("libc++1-18", npu_utils.compile_failure_hint(error))

  def test_other_failures_have_no_hint(self):
    self.assertIsNone(
        npu_utils.compile_failure_hint(self._error_for("Some other error\n"))
    )
    self.assertIsNone(npu_utils.compile_failure_hint(ValueError("boom")))
    self.assertIsNone(
        npu_utils.compile_failure_hint(
            ValueError("See /no/such/file.error for details.")
        )
    )


if __name__ == "__main__":
  absltest.main()
