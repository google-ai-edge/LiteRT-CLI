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

"""Unit tests for the Android utility functions in LiteRT CLI."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
import click
from litert_cli.core import android_utils
import requests

# Parse flags to avoid UnparsedFlagAccessError when running with pytest
try:
  flags.FLAGS(sys.argv)
except flags.UnrecognizedFlagError:
  pass


class CheckAdbTest(absltest.TestCase):

  @mock.patch("subprocess.run", autospec=True)
  def test_check_adb_success(self, mock_run: mock.MagicMock) -> None:
    mock_run.return_value = mock.MagicMock(returncode=0)
    android_utils.check_adb()
    mock_run.assert_called_once_with(
        ["adb", "get-state"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

  @mock.patch("subprocess.run", autospec=True)
  def test_check_adb_missing_raises_click_exception(
      self, mock_run: mock.MagicMock
  ) -> None:
    mock_run.side_effect = FileNotFoundError()
    with self.assertRaisesRegex(click.ClickException, "adb command not found"):
      android_utils.check_adb()


class GetAndroidAbiTest(absltest.TestCase):

  @mock.patch("subprocess.run", autospec=True)
  def test_get_android_abi_success(self, mock_run: mock.MagicMock) -> None:
    mock_run.return_value = mock.MagicMock(stdout="arm64-v8a\n", returncode=0)
    abi = android_utils.get_android_abi()
    self.assertEqual(abi, "arm64-v8a")

  @mock.patch("subprocess.run", autospec=True)
  def test_get_android_abi_failure_raises_click_exception(
      self, mock_run: mock.MagicMock
  ) -> None:
    mock_run.side_effect = subprocess.CalledProcessError(1, "adb")
    with self.assertRaisesRegex(click.ClickException, "Error querying Android ABI"):
      android_utils.get_android_abi()


class FindAndroidBinaryTest(absltest.TestCase):

  @mock.patch("pathlib.Path.exists", autospec=True)
  def test_find_android_binary_success(
      self, mock_exists: mock.MagicMock
  ) -> None:
    with mock.patch(
        "litert_cli.core.android_utils._ensure_downloaded_binary",
        autospec=True,
    ) as mock_download:
      mock_download.return_value = pathlib.Path("/cached/run_model")
      mock_exists.return_value = True
      path = android_utils.find_android_binary("run_model", "arm64-v8a")
    self.assertEqual(path, pathlib.Path("/cached/run_model"))

  def test_find_android_binary_failure_raises_click_exception(self) -> None:
    with mock.patch(
        "litert_cli.core.android_utils._ensure_downloaded_binary",
        autospec=True,
    ) as mock_download:
      mock_download.side_effect = android_utils.DownloadError("Failed to download")
      with self.assertRaisesRegex(
          click.ClickException, "Could not find or download"
      ):
        android_utils.find_android_binary("run_model", "arm64-v8a")


class EnsureDownloadedBinaryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = pathlib.Path(self.create_tempdir().full_path)
    self.enter_context(
        mock.patch(
            "litert_cli.core.android_utils.constants.LITERT_CLI_CACHE_DIR",
            new=str(self.test_dir),
        )
    )


  @mock.patch("requests.get", autospec=True)
  @mock.patch("click.progressbar", autospec=True)
  def test_ensure_downloaded_binary_success(
      self, mock_progressbar, mock_get
  ):
    mock_response = mock.MagicMock()
    mock_response.headers = {"Content-Length": "20"}
    mock_response.iter_content.return_value = iter([b"dummy_binary_content"])
    mock_get.return_value.__enter__.return_value = mock_response

    mock_bar = mock.MagicMock()
    mock_progressbar.return_value.__enter__.return_value = mock_bar

    path = android_utils._ensure_downloaded_binary("arm64-v8a", "run_model")

    self.assertIsNotNone(path)
    self.assertTrue(path.exists())
    self.assertEqual(path.read_bytes(), b"dummy_binary_content")
    self.assertTrue(os.access(path, os.X_OK))
    mock_progressbar.assert_called_once()

  def test_ensure_downloaded_binary_unsupported_abi(self):
    with self.assertRaisesRegex(
        click.ClickException, "Architecture 'x86' is not supported"
    ):
      android_utils._ensure_downloaded_binary("x86", "run_model")

  @mock.patch("requests.get", autospec=True)
  @mock.patch("click.progressbar", autospec=True)
  def test_ensure_downloaded_binary_failure_cleans_up(
      self, mock_progressbar, mock_get
  ):
    mock_get.side_effect = requests.exceptions.RequestException("Network error")

    with self.assertRaises(android_utils.DownloadError):
      android_utils._ensure_downloaded_binary("arm64-v8a", "run_model")

    mock_progressbar.assert_not_called()
    cache_dir = self.test_dir / "binaries" / "arm64-v8a"
    self.assertFalse((cache_dir / "run_model").exists())
    self.assertFalse((cache_dir / "run_model.tmp").exists())


class EnsureDownloadedLibraryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = pathlib.Path(self.create_tempdir().full_path)
    self.enter_context(
        mock.patch(
            "litert_cli.core.android_utils.constants.LITERT_CLI_CACHE_DIR",
            new=str(self.test_dir),
        )
    )

  @mock.patch("requests.get", autospec=True)
  @mock.patch("click.progressbar", autospec=True)
  def test_ensure_downloaded_library_success(
      self, mock_progressbar, mock_get
  ):
    mock_response = mock.MagicMock()
    mock_response.headers = {"Content-Length": "17"}
    mock_response.iter_content.return_value = iter([b"dummy_lib_content"])
    mock_get.return_value.__enter__.return_value = mock_response

    mock_bar = mock.MagicMock()
    mock_progressbar.return_value.__enter__.return_value = mock_bar

    path = android_utils._ensure_downloaded_library("arm64-v8a", "libTest.so")

    self.assertIsNotNone(path)
    self.assertTrue(path.exists())
    self.assertEqual(path.read_bytes(), b"dummy_lib_content")
    mock_progressbar.assert_called_once()

  def test_ensure_downloaded_library_unsupported_abi(self):
    with self.assertRaisesRegex(
        click.ClickException, "Architecture 'x86' is not supported"
    ):
      android_utils._ensure_downloaded_library("x86", "libTest.so")


class FindNpuDispatchLibTest(absltest.TestCase):

  @mock.patch("litert_cli.core.android_utils.find_android_lib", autospec=True)
  def test_find_npu_dispatch_lib_qualcomm(self, mock_find_lib):
    mock_find_lib.return_value = pathlib.Path(
        "/path/to/libLiteRtDispatch_Qualcomm.so"
    )
    path = android_utils.find_npu_dispatch_lib("qualcomm", "arm64-v8a")
    self.assertEqual(
        path, pathlib.Path("/path/to/libLiteRtDispatch_Qualcomm.so")
    )
    mock_find_lib.assert_called_once_with(
        "libLiteRtDispatch_Qualcomm.so", "arm64-v8a", base_url=mock.ANY
    )

  @mock.patch("litert_cli.core.android_utils.find_android_lib", autospec=True)
  def test_find_npu_dispatch_lib_mediatek(self, mock_find_lib):
    mock_find_lib.return_value = pathlib.Path(
        "/path/to/libLiteRtDispatch_MediaTek.so"
    )
    path = android_utils.find_npu_dispatch_lib("mediatek", "arm64-v8a")
    self.assertEqual(
        path, pathlib.Path("/path/to/libLiteRtDispatch_MediaTek.so")
    )
    mock_find_lib.assert_called_once_with(
        "libLiteRtDispatch_MediaTek.so", "arm64-v8a", base_url=mock.ANY
    )

  def test_find_npu_dispatch_lib_unsupported(self):
    with self.assertRaisesRegex(
        click.ClickException, "Unsupported NPU vendor"
    ):
      android_utils.find_npu_dispatch_lib("unknown_vendor", "arm64-v8a")


if __name__ == "__main__":
  absltest.main()
