"""Unit tests for the LiteRT CLI dependency management system."""

from unittest import mock

from absl.testing import absltest
import click
from litert_cli.core import deps


class EnsureExtraTest(absltest.TestCase):
  """Tests for ensure_extra."""

  @mock.patch("importlib.util.find_spec")
  @mock.patch("subprocess.check_call")
  def test_ensure_extra_already_installed(
      self, mock_check_call: mock.MagicMock, mock_find_spec: mock.MagicMock
  ) -> None:
    """Tests that ensure_extra returns True when dependency is installed."""
    mock_find_spec.return_value = mock.MagicMock()
    self.assertTrue(deps.ensure_extra("convert", silent=True))
    mock_check_call.assert_not_called()

  @mock.patch("importlib.util.find_spec")
  @mock.patch("subprocess.check_call")
  def test_ensure_extra_installs_missing(
      self, mock_check_call: mock.MagicMock, mock_find_spec: mock.MagicMock
  ) -> None:
    """Tests that ensure_extra installs missing dependency."""
    mock_find_spec.side_effect = [None, mock.MagicMock()]

    with mock.patch("pathlib.Path.exists") as mock_exists:
      mock_exists.return_value = False
      self.assertTrue(deps.ensure_extra("convert", silent=True))

    mock_check_call.assert_called_once()
    call_args = mock_check_call.call_args[0][0]
    self.assertIn("pip", call_args)
    self.assertIn("litert-cli[convert]", call_args)

  def test_ensure_extra_unknown_extra_returns_false_silent(self) -> None:
    """Tests ensure_extra returns False in silent mode."""
    self.assertFalse(deps.ensure_extra("unknown_ext_xyz", silent=True))

  def test_ensure_extra_unknown_extra_raises_abort_not_silent(self) -> None:
    """Tests that ensure_extra raises click.Abort for unknown extra."""
    with self.assertRaises(click.Abort):
      deps.ensure_extra("unknown_ext_xyz", silent=False)


class RequireExtraDecoratorTest(absltest.TestCase):
  """Tests for require_extra decorator."""

  @mock.patch("litert_cli.core.deps.ensure_extra")
  def test_require_extra_calls_ensure_extra(
      self, mock_ensure_extra: mock.MagicMock
  ) -> None:
    """Tests that require_extra decorator calls ensure_extra."""
    mock_ensure_extra.return_value = True

    @deps.require_extra("convert")
    def dummy_func(x: int) -> int:
      return x + 1

    self.assertEqual(dummy_func(2), 3)
    mock_ensure_extra.assert_called_once_with("convert", silent=False)


if __name__ == "__main__":
  absltest.main()
