"""Utility functions for LiteRT CLI."""

import os
from contextlib import contextmanager

@contextmanager
def silence_stderr():
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
