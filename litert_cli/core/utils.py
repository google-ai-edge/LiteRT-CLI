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

"""Utility functions for LiteRT CLI."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import os


@contextlib.contextmanager
def silence_stderr() -> Iterator[None]:
  """Silences file descriptor 2 (stderr) temporarily."""
  new_target = os.open(os.devnull, os.O_WRONLY)
  old_stderr = os.dup(2)
  os.dup2(new_target, 2)
  os.close(new_target)
  try:
    yield
  finally:
    os.dup2(old_stderr, 2)
    os.close(old_stderr)


_OLD_STDERR: int | None = None


def enable_quiet_mode() -> None:
  """Directly silences file descriptor 2 (stderr) for the rest of the process."""
  global _OLD_STDERR
  if _OLD_STDERR is None:
    _OLD_STDERR = os.dup(2)
  new_target = os.open(os.devnull, os.O_WRONLY)
  os.dup2(new_target, 2)
  os.close(new_target)


def restore_stderr() -> None:
  """Restores file descriptor 2 (stderr) to its original target."""
  global _OLD_STDERR
  if _OLD_STDERR is not None:
    os.dup2(_OLD_STDERR, 2)
    os.close(_OLD_STDERR)
    _OLD_STDERR = None
