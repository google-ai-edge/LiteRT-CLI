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
import click
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


class AiPackDirTest(absltest.TestCase):

  def test_new_and_empty_directories_are_accepted(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "empty").mkdir()
    npu_utils.check_ai_pack_dir(root / "new")
    npu_utils.check_ai_pack_dir(root / "empty")

  def test_existing_contents_are_never_replaced(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("x")
    (root / "f").write_text("x")
    for path in (root / "src", root / "f"):
      with self.assertRaisesRegex(click.UsageError, "not empty"):
        npu_utils.check_ai_pack_dir(path)
    self.assertTrue((root / "src" / "main.py").exists())


class SdkPackageQairtDirTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.package = self.create_tempdir()
    self.package.create_file("__init__.py")
    self.package.create_file("data/lib/aarch64-android/libQnnSystem.so")
    spec = mock.MagicMock(origin=str(pathlib.Path(self.package) / "__init__.py"))
    self.enter_context(
        mock.patch.object(
            npu_utils.importlib.util, "find_spec", return_value=spec
        )
    )
    self.versions = {}
    self.enter_context(
        mock.patch.object(
            npu_utils.importlib.metadata, "version", side_effect=self._version
        )
    )

  def _version(self, dist):
    if dist not in self.versions:
      raise npu_utils.importlib.metadata.PackageNotFoundError(dist)
    return self.versions[dist]

  def _binaries(self, version):
    return mock.patch.object(
        npu_utils.constants, "LITERT_BINARIES_VERSION", version
    )

  def test_nightly_package_for_nightly_binaries(self):
    self.versions["ai-edge-litert-sdk-qualcomm-nightly"] = "2.4.0.dev20261104"
    with self._binaries("nightly"):
      self.assertEqual(
          npu_utils._sdk_package_qairt_dir(),
          pathlib.Path(self.package) / "data",
      )

  def test_stable_package_of_the_same_release(self):
    self.versions["ai-edge-litert-sdk-qualcomm"] = "2.2.0"
    with self._binaries("2.2.0"):
      self.assertIsNotNone(npu_utils._sdk_package_qairt_dir())

  def test_package_of_another_release_is_not_used(self):
    self.versions["ai-edge-litert-sdk-qualcomm"] = "2.1.6"
    with self._binaries("2.2.0"):
      self.assertIsNone(npu_utils._sdk_package_qairt_dir())
    self.versions = {"ai-edge-litert-sdk-qualcomm": "2.2.0"}
    with self._binaries("nightly"):
      self.assertIsNone(npu_utils._sdk_package_qairt_dir())

  def test_old_nightly_package_is_not_used(self):
    self.versions["ai-edge-litert-sdk-qualcomm-nightly"] = "2.4.0.dev20260101"
    self.package.create_file(
        "data/sdk.yaml", "product: QAIRT\nversion: 2.47.0\nbuild_id: 260601114230\n"
    )
    with self._binaries("nightly"), mock.patch.object(
        npu_utils.constants, "QAIRT_SDK_VERSION", "2.51.1.261002"
    ):
      self.assertIsNone(npu_utils._sdk_package_qairt_dir())

  def test_nightly_package_as_new_as_the_map_is_used(self):
    self.versions["ai-edge-litert-sdk-qualcomm-nightly"] = "2.4.0.dev20261104"
    self.package.create_file(
        "data/sdk.yaml", "product: QAIRT\nversion: 2.51.1\nbuild_id: 261002102551\n"
    )
    with self._binaries("nightly"), mock.patch.object(
        npu_utils.constants, "QAIRT_SDK_VERSION", "2.51.1.261002"
    ):
      self.assertIsNotNone(npu_utils._sdk_package_qairt_dir())

  def test_no_package(self):
    with self._binaries("nightly"):
      self.assertIsNone(npu_utils._sdk_package_qairt_dir())

  def test_package_without_android_libraries_is_not_used(self):
    self.versions["ai-edge-litert-sdk-qualcomm-nightly"] = "2.4.0.dev20261104"
    (pathlib.Path(self.package) / "data/lib/aarch64-android/libQnnSystem.so").unlink()
    with self._binaries("nightly"):
      self.assertIsNone(npu_utils._sdk_package_qairt_dir())

  def test_qualcomm_libs_come_from_the_package(self):
    sdk = pathlib.Path(self.package) / "data"
    with mock.patch.object(
        npu_utils, "_sdk_package_qairt_dir", return_value=sdk
    ), mock.patch.object(
        npu_utils.constants, "QNN_SOC_VERSION_MAP", {"sm8750": "79"}
    ):
      libs = npu_utils._get_qualcomm_libs(pathlib.Path("/unused"), "sm8750")
    self.assertEqual(libs[0], sdk / "lib/aarch64-android/libQnnSystem.so")

  def test_downloaded_sdk_without_the_package(self):
    with mock.patch.object(
        npu_utils, "_sdk_package_qairt_dir", return_value=None
    ), mock.patch.object(
        npu_utils.constants, "QAIRT_SDK_VERSION", "2.51.1.261002"
    ):
      self.assertEqual(
          npu_utils._qairt_dir(pathlib.Path("/root")),
          pathlib.Path("/root/qairt/2.51.1.261002"),
      )


class WarnIfNpuUnusedTest(absltest.TestCase):

  def test_no_warning_when_the_dispatch_delegate_ran(self):
    lines = [
        "VERBOSE: Replacing 1 out of 1 node(s) with delegate (DispatchDelegate)"
        " node, yielding 1 partitions for subgraph 0 (main).\n"
    ]
    self.assertFalse(npu_utils.warn_if_npu_unused(lines))

  def test_warning_names_the_reason(self):
    lines = [
        "ERROR: [qnn_manager.cc:408] Qnn System library version 1.11.0 is"
        " mismatched. The minimum supported version is 1.15.0.\n",
        "VERBOSE: Replacing 36 out of 37 node(s) with delegate"
        " (TfLiteXNNPackDelegate) node.\n",
    ]
    with mock.patch.object(npu_utils.click, "secho") as secho:
      self.assertTrue(npu_utils.warn_if_npu_unused(lines))
    printed = " ".join(c.args[0] for c in secho.call_args_list)
    self.assertIn("did not run on the NPU", printed)
    self.assertIn("is mismatched", printed)
    self.assertNotIn("XNNPack", printed)


if __name__ == "__main__":
  absltest.main()
