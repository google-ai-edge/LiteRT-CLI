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

"""pytest configuration shared by all LiteRT CLI unit tests.

The unit tests are absltest test cases. `absltest.main()` parses absl flags
before running them, but pytest does not, and absltest helpers such as
`create_tempdir()` read absl flags (e.g. `--test_tmpdir`) and raise
`UnparsedFlagAccessError` when the flags are unparsed. Mark absl flags as
parsed once, with their default values, so any test file can run on its own
under pytest (`pytest path/to/foo_test.py`, `pytest -k ...`). pytest's own
command line options are deliberately not handed to absl.

Running a test file directly (`python foo_test.py`) does not load this file;
`absltest.main()` parses the flags there.
"""

from absl import flags


def pytest_configure(config):
  del config  # Unused.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()
