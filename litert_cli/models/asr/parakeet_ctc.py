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

"""Wrapper for ParakeetCTC model."""

from __future__ import annotations

import textwrap
from typing import Any, override

from . import asr_model


class ParakeetCTC(asr_model.AsrModel):
  """Wrapper for ParakeetCTC model."""

  HF_MODEL_IDS = [
      'nvidia/parakeet-ctc-0.6b',
      'litert-community/parakeet-ctc-0.6b',
  ]

  INPUT_MILLISECONDS = 5000
  LOG_MEL_SPECTRO = {'nFFT': 512, 'transpose': True, 'preemphasis': 0.97}
  MASK_TYPE = 'bool'

  def __init__(self, model_id: str = 'nvidia/parakeet-ctc-0.6b'):
    self.model_id = model_id
    self._model = None
    self._processor = None

  def get_processor(self) -> Any:
    """Lazy loads the processor."""
    if self._processor is None:
      import transformers

      self._processor = transformers.AutoProcessor.from_pretrained(
          self.model_id
      )
    return self._processor

  def _get_model(self) -> Any:
    """Lazy loads the PyTorch model."""
    if self._model is None:
      import transformers

      print(f'Loading PyTorch model from HF: {self.model_id}')
      self._model = (
          transformers.AutoModelForCTC.from_pretrained(self.model_id)
          .float()
          .eval()
      )
    return self._model

  # --- ModelIntentHandler implementation ---

  def get_model_help(self) -> str:
    """Returns help text for model specific parameters."""
    return textwrap.dedent("""\
        Parakeet CTC Model Specific Help:
        Supported --model-params:
          - wav_path: Path to the audio file (overrides --input).
          - duration: Audio duration to load in seconds (default: 5.0).
          - offset: Audio offset to skip in seconds (default: 0.0).
    """)

  def _handle_run(self, model_id: str = None, **kwargs: Any) -> str:
    """Runs inference on a sample audio file."""
    from .runner import ASRRunner

    runner = ASRRunner(model_id, **kwargs)
    return runner.run_full_pipeline(self, **kwargs)

  def preprocess_inputs(
      self, processed_inputs: dict[str, Any]
  ) -> dict[str, Any]:
    """Model-specific preprocessing to handle feature truncation."""
    features = processed_inputs['input_features']
    frames_axis = 1 if features.shape[1] > features.shape[2] else 2

    # NOTE: Input shape is [1, 501, 80], but the model expect [1, 500, 80].
    if features.shape[frames_axis] > 500:
      if frames_axis == 1:
        processed_inputs['input_features'] = features[:, :500, :]
        if 'attention_mask' in processed_inputs:
          processed_inputs['attention_mask'] = processed_inputs[
              'attention_mask'
          ][:, :500]
      else:
        processed_inputs['input_features'] = features[:, :, :500]
        if 'attention_mask' in processed_inputs:
          processed_inputs['attention_mask'] = processed_inputs[
              'attention_mask'
          ][:, :500]
    return processed_inputs

  def postprocess_tokens(self, tokens: Any) -> list[list[int]]:
    """Model-specific token postprocessing (CTC style collapsing)."""
    collapsed_tokens = []
    for batch in tokens:
      new_batch = []
      previous_id = -1
      for token_id in batch:
        if token_id != previous_id:
          new_batch.append(int(token_id))
          previous_id = token_id
      collapsed_tokens.append(new_batch)
    return collapsed_tokens

  # --- AsrModel hooks ---

  @override
  def get_encoder(self) -> Any:
    return self._get_model()

  @override
  def get_decoder(self) -> Any:
    """CTC models do not have a decoder subgraph."""
    return None

  @override
  def get_encoder_sample_input(
      self, processed_audio: dict[str, Any]
  ) -> tuple[Any, ...]:
    features = processed_audio['input_features']
    if features.shape[1] > 500:
      features = features[:, :500, :]
    return features, processed_audio['attention_mask'][:, :500]

  @override
  def get_decoder_sample_input(
      self, encoder_output: Any, num_tokens: int
  ) -> tuple[Any, ...]:
    return ()

  @override
  def get_decode_start_token_id(self) -> int:
    return -1

  @override
  def get_decode_stop_token_id(self) -> int:
    return -1

  @override
  def get_mask_dtype(self) -> Any:
    import torch

    return torch.bool

  @override
  def run_original_model(self, processed_audio: dict[str, Any]) -> Any:
    return self._get_model().generate(
        **processed_audio, return_dict_in_generate=True
    )
