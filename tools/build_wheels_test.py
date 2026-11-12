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

import importlib
import os
import pathlib
import re

from absl.testing import absltest

try:
  from litert_cli.tools import build_wheels  # pylint: disable=g-import-not-at-top
except ImportError:
  # In the open-source layout, tools/ sits next to the litert_cli package,
  # so build_wheels is a top-level module.
  build_wheels = importlib.import_module("build_wheels")

_TORCH_NIGHTLY = (
    f'"litert-torch-nightly{build_wheels.NIGHTLY_SPECIFIERS["litert-torch"]}"'
)
_TORCH_STABLE = (
    f'"litert-torch{build_wheels.STABLE_DEPENDENCIES["litert-torch"]}"'
)
_PYPROJECT = f"""\
[project]
name = "litert-cli"
dependencies = [
    "ai-edge-litert-nightly",
]

[project.optional-dependencies]
convert = [
    {_TORCH_NIGHTLY},
]
all = [
    "litert-cli[convert]"
]
"""


class ConfigureBuildTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = self.create_tempdir()
    tmp.create_file("pyproject.toml", _PYPROJECT)
    tmp.create_file("VERSION", "0.3.0\n")
    cwd = os.getcwd()
    os.chdir(tmp.full_path)
    self.addCleanup(os.chdir, cwd)

  def _pyproject(self) -> str:
    return pathlib.Path("pyproject.toml").read_text()

  def test_nightly_build_keeps_the_specifier(self):
    build_wheels.configure_build(is_nightly=True)

    content = self._pyproject()
    self.assertIn('name = "litert-cli-nightly"', content)
    self.assertIn(_TORCH_NIGHTLY, content)

  def test_stable_build_pins_the_stable_version(self):
    build_wheels.configure_build(is_nightly=False)

    content = self._pyproject()
    self.assertIn(_TORCH_STABLE, content)
    self.assertNotIn("-nightly", content)

  def test_restore_after_a_build_gives_back_the_source(self):
    for is_nightly in (False, True):
      with self.subTest(is_nightly=is_nightly):
        build_wheels.configure_build(is_nightly=is_nightly)
        build_wheels.configure_build(is_nightly=is_nightly, restore=True)

        self.assertEqual(self._pyproject(), _PYPROJECT)
        self.assertEqual(pathlib.Path("VERSION").read_text(), "0.3.0\n")

  def test_repo_pyproject_uses_the_specifiers(self):
    content = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
    for pkg in build_wheels.STABLE_DEPENDENCIES:
      specifier = build_wheels.NIGHTLY_SPECIFIERS.get(pkg, "")
      expected = f'"{pkg}-nightly{specifier}"'
      for found in re.findall(rf'"{re.escape(pkg)}-nightly[^"]*"', content):
        self.assertEqual(found, expected)


if __name__ == "__main__":
  absltest.main()
