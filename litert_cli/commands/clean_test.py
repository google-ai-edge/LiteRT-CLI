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

from __future__ import annotations

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from click import testing

from litert_cli.commands import clean


class CleanCmdTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmpdir.cleanup)
    self.cache_dir = pathlib.Path(self.tmpdir.name) / "cache"
    self.models_dir = self.cache_dir / "models"
    self.binaries_dir = self.cache_dir / "binaries"
    self.targets_dir = self.cache_dir / "targets"
    self.root_dir = self.cache_dir / "root"
    self.android_root = "/data/local/tmp/litert-cli-test"

  def _invoke(self, args):
    runner = testing.CliRunner()
    patches = [
        mock.patch.object(
            clean.constants, "LITERT_CLI_CACHE_DIR", str(self.cache_dir)
        ),
        mock.patch.object(
            clean.constants, "LITERT_MODELS_CACHE_DIR", str(self.models_dir)
        ),
        mock.patch.object(
            clean.constants, "LITERT_CLI_ROOT", str(self.root_dir)
        ),
        mock.patch.object(
            clean.constants, "LITERT_CLI_ANDROID_ROOT", self.android_root
        ),
    ]
    with patches[0], patches[1], patches[2], patches[3]:
      return runner.invoke(clean.clean_cmd, args)

  def _write_file(self, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")

  def test_dry_run_models_does_not_delete(self):
    self._write_file(self.models_dir / "model.tflite")

    result = self._invoke(["--models", "--dry-run"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertTrue(self.models_dir.exists())
    self.assertIn("Would remove managed model cache", result.output)
    self.assertIn("Dry run complete!", result.output)

  def test_models_removes_only_model_cache(self):
    self._write_file(self.models_dir / "model.tflite")
    self._write_file(self.binaries_dir / "run_model")

    result = self._invoke(["--models"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertFalse(self.models_dir.exists())
    self.assertTrue(self.binaries_dir.exists())
    self.assertIn("Removing managed model cache", result.output)

  @mock.patch.object(clean.subprocess, "run", autospec=True)
  @mock.patch.object(clean.android_utils, "check_adb", autospec=True)
  def test_default_removes_cache_and_android_workspace(
      self, mock_check_adb, mock_run
  ):
    self._write_file(self.models_dir / "model.tflite")

    result = self._invoke([])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertFalse(self.cache_dir.exists())
    mock_check_adb.assert_called_once_with()
    mock_run.assert_called_once()
    self.assertIn(self.android_root, mock_run.call_args.args[0][2])
    self.assertIn("Cleanup complete!", result.output)

  @mock.patch.object(clean.android_utils, "check_adb", autospec=True)
  def test_android_dry_run_does_not_require_device(self, mock_check_adb):
    result = self._invoke(["--android", "--dry-run"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    mock_check_adb.assert_not_called()
    self.assertIn("Would remove remote Android workspace", result.output)


if __name__ == "__main__":
  absltest.main()
