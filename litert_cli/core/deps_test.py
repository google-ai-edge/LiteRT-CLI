import importlib.util
import subprocess
from unittest import mock

from absl.testing import absltest
import click
from litert_cli.core import constants
from litert_cli.core import deps


class DepsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(constants, 'IN_GOOGLE3', False))

  @mock.patch.object(importlib.util, 'find_spec', autospec=True)
  def test_ensure_extra_already_installed(self, mock_find_spec):
    # Simulate module is already installed
    mock_find_spec.return_value = object()

    result = deps.ensure_extra('torch')

    self.assertTrue(result)
    mock_find_spec.assert_called_once_with('litert_torch')

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.util, 'find_spec', autospec=True)
  def test_ensure_extra_not_installed_success(
      self, mock_find_spec, mock_check_call
  ):
    # Simulate module is NOT installed
    mock_find_spec.return_value = None
    mock_check_call.return_value = 0

    result = deps.ensure_extra('torch')

    self.assertTrue(result)
    mock_find_spec.assert_called_once_with('litert_torch')
    mock_check_call.assert_called_once()

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.util, 'find_spec', autospec=True)
  def test_ensure_extra_not_installed_failure(
      self, mock_find_spec, mock_check_call
  ):
    # Simulate module is NOT installed and pip install fails
    mock_find_spec.return_value = None
    mock_check_call.side_effect = subprocess.CalledProcessError(
        1, 'pip install'
    )

    with self.assertRaises(click.Abort):
      deps.ensure_extra('torch')

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.util, 'find_spec', autospec=True)
  def test_ensure_extra_silent_failure(self, mock_find_spec, mock_check_call):
    # Simulate module is NOT installed and pip install fails, but silent=True
    mock_find_spec.return_value = None
    mock_check_call.side_effect = subprocess.CalledProcessError(
        1, 'pip install'
    )

    result = deps.ensure_extra('torch', silent=True)

    self.assertFalse(result)

  @mock.patch.object(deps, 'ensure_extra', autospec=True)
  def test_require_extra_decorator(self, mock_ensure_extra):
    @deps.require_extra('torch')
    def my_func():
      return 'Hello'

    result = my_func()

    self.assertEqual(result, 'Hello')
    mock_ensure_extra.assert_called_once_with('torch', silent=False)


if __name__ == '__main__':
  absltest.main()
