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

"""Tests for the high-level `litert convert <hf_id>` options."""

from absl.testing import absltest
from absl.testing import parameterized
from litert_cli.commands.convert import hf_options


class NormalizeTaskTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, None),
      ("auto", None),
      ("AUTO", None),
      ("llm", "text_generation"),
      ("vlm", "image_text_to_text"),
      ("asr", "automatic_speech_recognition"),
      ("tts", "text_to_speech"),
      ("t2i", "text_to_image"),
      ("omni", "any_to_any"),
      ("text_to_image", "text_to_image"),
  )
  def test_normalize_task(self, value, expected):
    self.assertEqual(hf_options.normalize_task(value), expected)

  def test_unknown_task_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown task 'video'"):
      hf_options.normalize_task("video")


class ParseQuantizeTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, None),
      ("none", ""),
      ("Float", ""),
      ("fp32", ""),
      (" dynamic_wi8_afp32 ", "dynamic_wi8_afp32"),
  )
  def test_parse_quantize(self, value, expected):
    self.assertEqual(hf_options.parse_quantize(value), expected)


class ParsePrefillLengthsTest(absltest.TestCase):

  def test_parses_comma_separated_lengths(self):
    self.assertEqual(hf_options.parse_prefill_lengths("128, 1024"), [128, 1024])

  def test_unset_is_none(self):
    self.assertIsNone(hf_options.parse_prefill_lengths(None))

  def test_invalid_value_raises(self):
    with self.assertRaisesRegex(ValueError, "comma-separated integers"):
      hf_options.parse_prefill_lengths("128,long")


class ParseExportArgsTest(absltest.TestCase):

  def test_parses_scalars_and_json(self):
    self.assertEqual(
        hf_options.parse_export_args([
            "use_sdpa_composite=False",
            "--input-sec=2.5",
            "stateful_after=4",
            "litert_lm_model_type_override=qwen3",
            'extra_kwargs={"max_seq_len": 256}',
            "cache_implementation=none",
        ]),
        {
            "use_sdpa_composite": False,
            "input_sec": 2.5,
            "stateful_after": 4,
            "litert_lm_model_type_override": "qwen3",
            "extra_kwargs": {"max_seq_len": 256},
            "cache_implementation": None,
        },
    )

  def test_float_quantization_recipe_becomes_empty_string(self):
    self.assertEqual(
        hf_options.parse_export_args(
            ["vision_encoder_quantization_recipe=none"]
        ),
        {"vision_encoder_quantization_recipe": ""},
    )

  def test_missing_value_raises(self):
    with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
      hf_options.parse_export_args(["use_sdpa_composite"])

  def test_reserved_key_raises(self):
    with self.assertRaisesRegex(ValueError, "use --task"):
      hf_options.parse_export_args(["task=text_generation"])


class BuildOverridesTest(absltest.TestCase):

  def test_only_set_options_are_returned(self):
    self.assertEqual(
        hf_options.build_overrides(cache_length=4096), {"cache_length": 4096}
    )

  def test_maps_options_to_export_hf_arguments(self):
    self.assertEqual(
        hf_options.build_overrides(
            quantize="none",
            prefill_lengths="128,512",
            bundle_litert_lm=False,
            gpu_optimizations=False,
            externalize_weights=True,
        ),
        {
            "quantization_recipe": "",
            "prefill_lengths": [128, 512],
            "bundle_litert_lm": False,
            "apply_gpu_composites": False,
            "externalize_weights": True,
        },
    )

  def test_export_args_win_over_options(self):
    self.assertEqual(
        hf_options.build_overrides(
            cache_length=4096, export_args={"cache_length": 8192}
        ),
        {"cache_length": 8192},
    )


if __name__ == "__main__":
  absltest.main()
