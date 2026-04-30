"""Base interface for model intent handlers in litert-cli."""

from __future__ import annotations

from typing import Any


class ModelIntentHandler:
  """Base interface for model intent handlers.

  These handlers decide whether to handle a specific command (intent) for a
  model, and how to execute it.
  """

  def can_handle(self, intent: str, model_id: str) -> bool:
    """Check if this plugin should handle the given intent and model.

    Args:
      intent: The command name (e.g., 'convert', 'run', 'evaluate').
      model_id: The model identifier or path.

    Returns:
      True if this handler can process the request, or False otherwise.
    """
    raise NotImplementedError()

  def handle(self, intent: str, model_id: str, **kwargs: Any) -> Any:
    """Execute the command for the given model.

    Args:
      intent: The command name.
      model_id: The model identifier.
      **kwargs: Dynamic arguments passed from the CLI command.

    Returns:
      The result of the execution or None.
    """
    raise NotImplementedError()

  def get_model_help(self) -> str:
    """Return help text for model specific parameters."""
    raise NotImplementedError()
