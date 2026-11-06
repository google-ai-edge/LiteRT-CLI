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

import importlib.metadata
import subprocess
from unittest import mock

from absl.testing import absltest
import click
from litert_cli.core import constants
from litert_cli.core import deps


class DepsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(constants, 'IS_INTERNAL_ENV', False))
    # Install from PyPI, as for an installed package, not a source checkout.
    self.enter_context(
        mock.patch.object(deps, '_source_checkout_root', return_value=None)
    )

  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_ensure_extra_already_installed(self, mock_version):
    # Simulate package is already installed
    mock_version.return_value = '0.1.0'

    result = deps.ensure_extra('torch')

    self.assertTrue(result)
    mock_version.assert_called_once_with('litert-torch-nightly')

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_ensure_extra_not_installed_success(
      self, mock_version, mock_check_call
  ):
    # Simulate package is NOT installed
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    mock_check_call.return_value = 0

    result = deps.ensure_extra('torch')

    self.assertTrue(result)
    mock_version.assert_has_calls([
        mock.call('litert-torch-nightly'),
        mock.call('litert-torch'),
        mock.call('litert-cli-nightly'),
    ])
    mock_check_call.assert_called_once()

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_source_checkout_stays_editable(self, mock_version, mock_check_call):
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    with mock.patch.object(
        deps, '_source_checkout_root', return_value=deps.pathlib.Path('/repo')
    ):
      self.assertTrue(deps.ensure_extra('quantize'))
    cmd = mock_check_call.call_args.args[0]
    self.assertEqual(cmd[-2:], ['-e', '.[quantize]'])
    self.assertEqual(mock_check_call.call_args.kwargs['cwd'], '/repo')

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_ensure_extra_not_installed_failure(
      self, mock_version, mock_check_call
  ):
    # Simulate package is NOT installed and pip install fails
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    mock_check_call.side_effect = subprocess.CalledProcessError(
        1, 'pip install'
    )

    with self.assertRaises(click.Abort):
      deps.ensure_extra('torch')

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_ensure_extra_silent_failure(self, mock_version, mock_check_call):
    # Simulate package is NOT installed and pip install fails, but silent=True
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    mock_check_call.side_effect = subprocess.CalledProcessError(
        1, 'pip install'
    )

    result = deps.ensure_extra('torch', silent=True)

    self.assertFalse(result)

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_visualize_checks_the_distribution_it_installs(
      self, mock_version, mock_check_call
  ):
    mock_version.return_value = '0.1.0'

    self.assertTrue(deps.ensure_extra('visualize'))

    mock_version.assert_called_once_with('ai-edge-model-explorer')
    mock_check_call.assert_not_called()

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_compile_needs_every_vendor_sdk(self, mock_version, mock_check_call):
    def version(pkg):
      if 'qualcomm' in pkg:
        return '0.1.0'
      raise importlib.metadata.PackageNotFoundError(pkg)

    mock_version.side_effect = version

    self.assertTrue(deps.ensure_extra('compile'))

    mock_check_call.assert_called_once()

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
