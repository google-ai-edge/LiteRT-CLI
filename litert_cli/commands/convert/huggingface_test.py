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

from unittest import mock

from absl.testing import absltest
from litert_cli.commands.convert import huggingface


class HuggingfaceTest(absltest.TestCase):

  @mock.patch("transformers.AutoConfig.from_pretrained")
  @mock.patch("litert_torch.generative.export_hf.export.export")
  def test_convert_huggingface_standard_causallm(
      self, mock_export, mock_from_pretrained
  ):
    # Setup mocks
    mock_config = mock.MagicMock()
    mock_config.architectures = ["Qwen2ForCausalLM"]
    mock_from_pretrained.return_value = mock_config

    # Run conversion
    huggingface.convert_huggingface(
        model="Qwen/Qwen1.5-0.5B-Chat",
        output="/tmp/output",
        target=(),
        export_aipack=None,
    )

    # Assert standard causal LM export parameters
    mock_export.assert_called_once_with(
        model="Qwen/Qwen1.5-0.5B-Chat",
        output_dir="/tmp/output",
        task="text_generation",
        quantization_recipe=None,
        prefill_lengths=[256],
        cache_length=4096,
        bundle_litert_lm=True,
        trust_remote_code=False,
        use_jinja_template=False,
    )

  @mock.patch("transformers.AutoConfig.from_pretrained")
  @mock.patch("litert_torch.generative.export_hf.export.export")
  def test_convert_huggingface_gemma3_vlm(
      self, mock_export, mock_from_pretrained
  ):
    # Setup mocks
    mock_config = mock.MagicMock()
    mock_config.architectures = ["Gemma3ForConditionalGeneration"]
    mock_from_pretrained.return_value = mock_config

    # Run conversion
    huggingface.convert_huggingface(
        model="google/gemma-3-4b-it",
        output="/tmp/output",
        target=(),
        export_aipack=None,
    )

    # Assert Gemma 3 VLM export parameters
    mock_export.assert_called_once_with(
        model="google/gemma-3-4b-it",
        output_dir="/tmp/output",
        task="image_text_to_text",
        quantization_recipe=None,
        prefill_lengths=[256],
        cache_length=4096,
        bundle_litert_lm=True,
        trust_remote_code=False,
        use_jinja_template=False,
        export_vision_encoder=True,
        externalize_embedder=True,
    )

  @mock.patch("transformers.AutoConfig.from_pretrained")
  @mock.patch("litert_torch.generative.export_hf.export.export")
  def test_convert_huggingface_gemma4_vlm(
      self, mock_export, mock_from_pretrained
  ):
    # Setup mocks
    mock_config = mock.MagicMock()
    mock_config.architectures = ["Gemma4ForConditionalGeneration"]
    mock_from_pretrained.return_value = mock_config

    # Run conversion
    huggingface.convert_huggingface(
        model="google/gemma-4-E2B-it",
        output="/tmp/output",
        target=(),
        export_aipack=None,
    )

    # Assert Gemma 4 VLM export parameters
    mock_export.assert_called_once_with(
        model="google/gemma-4-E2B-it",
        output_dir="/tmp/output",
        task="image_text_to_text",
        quantization_recipe=None,
        prefill_lengths=[256],
        cache_length=4096,
        bundle_litert_lm=True,
        trust_remote_code=False,
        use_jinja_template=True,
        export_vision_encoder=True,
        jinja_chat_template_override="litert-community/gemma-4-E2B-it-litert-lm",
        externalize_embedder=True,
    )

  @mock.patch("transformers.AutoConfig.from_pretrained")
  @mock.patch("litert_torch.generative.export_hf.export.export")
  def test_convert_huggingface_unsupported_architecture(
      self, mock_export, mock_from_pretrained
  ):
    # Setup mocks
    mock_config = mock.MagicMock()
    mock_config.architectures = ["BertModel"]
    mock_from_pretrained.return_value = mock_config

    # Assert that conversion raises ValueError
    with self.assertRaises(ValueError):
      huggingface.convert_huggingface(
          model="google/unsupported-model",
          output="/tmp/output",
          target=(),
          export_aipack=None,
      )

    # Assert that export was never called
    mock_export.assert_not_called()


if __name__ == "__main__":
  absltest.main()
