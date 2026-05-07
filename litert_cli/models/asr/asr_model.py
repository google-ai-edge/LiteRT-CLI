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

"""Base class of ASR models."""

from __future__ import annotations

import abc
from typing import Any

import click

from .. import base


class AsrModel(base.ModelIntentHandler):
  """An abstract base for ASR models that can be converted to LiteRT.

  Handles ASR related CLI intents generally by dispatching to specific methods.
  """

  # Subclasses should override these
  HF_MODEL_IDS: list[str] = []
  SUPPORTED_INTENTS: list[str] = ["run"]

  def can_handle(self, intent: str, model_id: str) -> bool:
    """Check if this handler can process the request."""
    is_model_match = model_id in self.HF_MODEL_IDS
    return is_model_match and intent in self.SUPPORTED_INTENTS

  def handle(self, intent: str, model_id: str, **kwargs: Any) -> Any:
    """Execute the command for the given model by dispatching to methods."""
    if kwargs.get("model_help"):
      click.echo(self.get_model_help())
      return "Help shown"

    method_name = f"_handle_{intent}"
    if hasattr(self, method_name):
      return getattr(self, method_name)(model_id=model_id, **kwargs)
    else:
      raise ValueError(f"Unsupported intent: {intent}")

  def get_model_help(self) -> str:
    """Return help text for model specific parameters."""
    return f"Specific help for {self.__class__.__name__} not available."

  @abc.abstractmethod
  def get_encoder(self) -> Any:
    """Returns the encoder to be converted to 'encode' subgraph in LiteRT."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decoder(self) -> Any:
    """Returns the decoder to be converted to 'decode' subgraph in LiteRT."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_processor(self) -> Any:
    """Returns the processor to convert audio to model inputs."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_encoder_sample_input(
      self, processed_audio: dict[str, Any]
  ) -> tuple[Any, ...]:
    """Builds the encoder inputs as sample args for conversion."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decoder_sample_input(
      self, encoder_output: Any, num_tokens: int
  ) -> tuple[Any, ...]:
    """Builds the decoder inputs as sample args for conversion."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decode_start_token_id(self) -> int:
    """Returns the token ID to indicate the start of decoding."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decode_stop_token_id(self) -> int:
    """Returns the token ID to indicate the end of decoding."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_mask_dtype(self) -> Any:
    """Returns the dtype of the attention mask of decoding."""
    raise NotImplementedError()

  @abc.abstractmethod
  def run_original_model(
      self, processed_audio: dict[str, Any]
  ) -> dict[str, Any]:
    """Runs the original model and returns the logits and sequences."""
    raise NotImplementedError()
