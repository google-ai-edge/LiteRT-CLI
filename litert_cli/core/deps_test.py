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

import collections
import importlib.metadata
import pathlib
import re
import subprocess
from unittest import mock

from absl.testing import absltest
import click
from litert_cli.core import deps


class DepsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Pretend the CLI was pip-installed. Tests that care about the hermetic
    # build path override this explicitly.
    self.mock_cli_dist = self.enter_context(
        mock.patch.object(
            deps,
            '_installed_cli_distribution',
            autospec=True,
            return_value='litert-cli-nightly',
        )
    )

  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_ensure_extra_already_installed(self, mock_version):
    # Simulate package is already installed
    mock_version.return_value = '0.1.0'

    result = deps.ensure_extra('convert')

    self.assertTrue(result)
    mock_version.assert_called_once_with('litert-torch-nightly')

  @mock.patch.object(deps, '_is_extra_installed', autospec=True)
  @mock.patch.object(subprocess, 'check_call', autospec=True)
  def test_ensure_extra_not_installed_success(
      self, mock_check_call, mock_is_installed
  ):
    # Missing before the install, present after it.
    mock_is_installed.side_effect = [False, True]
    mock_check_call.return_value = 0

    result = deps.ensure_extra('convert')

    self.assertTrue(result)
    mock_check_call.assert_called_once()
    self.assertIn(
        'litert-cli-nightly[convert]', mock_check_call.call_args.args[0]
    )

  @mock.patch.object(deps, '_is_extra_installed', autospec=True)
  @mock.patch.object(subprocess, 'check_call', autospec=True)
  def test_ensure_extra_not_installed_failure(
      self, mock_check_call, mock_is_installed
  ):
    mock_is_installed.return_value = False
    mock_check_call.side_effect = subprocess.CalledProcessError(
        1, 'pip install'
    )

    with self.assertRaises(click.Abort):
      deps.ensure_extra('convert')

  @mock.patch.object(deps, '_is_extra_installed', autospec=True)
  @mock.patch.object(subprocess, 'check_call', autospec=True)
  def test_ensure_extra_silent_failure(self, mock_check_call, mock_is_installed):
    mock_is_installed.return_value = False
    mock_check_call.side_effect = subprocess.CalledProcessError(
        1, 'pip install'
    )

    result = deps.ensure_extra('convert', silent=True)

    self.assertFalse(result)

  @mock.patch.object(deps, '_is_extra_installed', autospec=True)
  @mock.patch.object(subprocess, 'check_call', autospec=True)
  def test_ensure_extra_installer_succeeds_but_installs_nothing(
      self, mock_check_call, mock_is_installed
  ):
    # `uv pip install pkg[does-not-exist]` exits 0 without installing
    # anything. A zero exit status must not be reported as success.
    mock_is_installed.return_value = False
    mock_check_call.return_value = 0

    with self.assertRaises(click.Abort):
      deps.ensure_extra('convert')

    self.assertFalse(deps.ensure_extra('convert', silent=True))

  @mock.patch.object(subprocess, 'check_call', autospec=True)
  @mock.patch.object(importlib.metadata, 'version', autospec=True)
  def test_ensure_extra_hermetic_build_does_not_install(
      self, mock_version, mock_check_call
  ):
    # A hermetic build is not registered as a pip distribution; its optional
    # dependencies are already bundled and must never be installed at runtime.
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    self.mock_cli_dist.return_value = None
    self.enter_context(
        mock.patch.object(pathlib.Path, 'exists', return_value=False)
    )

    result = deps.ensure_extra('convert')

    self.assertTrue(result)
    mock_check_call.assert_not_called()

  def test_unknown_extra_aborts(self):
    with self.assertRaises(click.Abort):
      deps.ensure_extra('no-such-extra')

    self.assertFalse(deps.ensure_extra('no-such-extra', silent=True))

  def test_package_map_matches_distribution_metadata(self):
    """Guards the two silent failure modes this map has had.

    1. An extra named here that the distribution does not declare: the
       generated install target resolves to a non-existent extra, which uv
       accepts with exit status 0 while installing nothing.
    2. A distribution name here that the extra does not actually pull in (the
       map used to look for 'model-explorer'; the real name is
       'ai-edge-model-explorer'). The installed-check then never succeeds and
       the CLI reinstalls on every invocation.
    """
    for dist_name in deps._CLI_DISTRIBUTIONS:
      try:
        md = importlib.metadata.metadata(dist_name)
        break
      except importlib.metadata.PackageNotFoundError:
        continue
    else:
      self.skipTest('litert-cli is not installed as a distribution')

    declared_extras = set(md.get_all('Provides-Extra') or [])
    self.assertContainsSubset(set(deps._PACKAGE_BY_EXTRA), declared_extras)

    def normalize(name):
      return re.sub(r'[-_.]+', '-', name).lower()

    requires_by_extra = collections.defaultdict(set)
    for req in md.get_all('Requires-Dist') or []:
      match = re.search(r'extra\s*==\s*[\'"]([^\'"]+)[\'"]', req)
      if not match:
        continue
      dep = re.split(r'[\s;\[<>=!~]', req.strip(), maxsplit=1)[0]
      requires_by_extra[match.group(1)].add(normalize(dep))

    for extra, candidates in deps._PACKAGE_BY_EXTRA.items():
      normalized = {normalize(c) for c in candidates}
      self.assertTrue(
          normalized & requires_by_extra[extra],
          f"none of {sorted(candidates)} is pulled in by extra '{extra}'"
          f' (it requires {sorted(requires_by_extra[extra])})',
      )

  @mock.patch.object(deps, '_is_extra_installed', autospec=True)
  def test_require_extra_decorator(self, mock_is_installed):
    mock_is_installed.return_value = True

    @deps.require_extra('convert')
    def my_func():
      return 'Hello'

    result = my_func()

    self.assertEqual(result, 'Hello')


if __name__ == '__main__':
  absltest.main()
