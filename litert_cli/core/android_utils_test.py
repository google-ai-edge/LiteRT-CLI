"""Unit tests for the Android utility functions in LiteRT CLI."""

import pathlib
from unittest import mock

from absl.testing import absltest
import click
from litert_cli.core import android_utils


class CheckAdbTest(absltest.TestCase):

  @mock.patch("subprocess.run")
  def test_check_adb_success(self, mock_run: mock.MagicMock) -> None:
    """Tests that check_adb calls subprocess.run successfully."""
    mock_run.return_value = mock.MagicMock(returncode=0)
    android_utils.check_adb()
    mock_run.assert_called_once()

  @mock.patch("subprocess.run")
  def test_check_adb_missing_raises_click_exception(
      self, mock_run: mock.MagicMock
  ) -> None:
    """Tests that check_adb raises ClickException when adb is missing."""
    mock_run.side_effect = FileNotFoundError()
    with self.assertRaisesRegex(click.ClickException, "adb command not found"):
      android_utils.check_adb()


class GetAndroidAbiTest(absltest.TestCase):

  @mock.patch("subprocess.run")
  def test_get_android_abi_success(self, mock_run: mock.MagicMock) -> None:
    """Tests that get_android_abi returns the correct ABI string."""
    mock_run.return_value = mock.MagicMock(stdout="arm64-v8a\n", returncode=0)
    abi = android_utils.get_android_abi()
    self.assertEqual(abi, "arm64-v8a")


class FindAndroidBinaryTest(absltest.TestCase):

  @mock.patch("pathlib.Path.exists")
  def test_find_android_binary_success(
      self, mock_exists: mock.MagicMock
  ) -> None:
    """Tests that find_android_binary successfully returns the cached path."""
    with mock.patch(
        "litert_cli.core.android_utils._ensure_downloaded_binary"
    ) as mock_download:
      mock_download.return_value = pathlib.Path("/cached/run_model")
      mock_exists.return_value = True
      path = android_utils.find_android_binary("run_model", "arm64-v8a")
    self.assertEqual(path, pathlib.Path("/cached/run_model"))

  def test_find_android_binary_failure_raises_click_exception(self) -> None:
    """Tests that find_android_binary raises ClickException on failure."""
    with mock.patch(
        "litert_cli.core.android_utils._ensure_downloaded_binary"
    ) as mock_download:
      mock_download.return_value = None
      with self.assertRaisesRegex(
          click.ClickException, "Could not find or download"
      ):
        android_utils.find_android_binary("run_model", "arm64-v8a")


if __name__ == "__main__":
  absltest.main()
