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

"""Tests for `litert convert <hf_id>` (litert-torch `export_hf` is faked)."""

import dataclasses
import sys
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
import click
from litert_cli.commands.convert import huggingface
from litert_cli.commands.convert import litert_lm_config


@dataclasses.dataclass(frozen=True)
class _FakePlan:
  task: str
  export_kwargs: dict[str, Any]
  model_type: str | None = None
  model: str = "org/model"


class _FakeExportHf:
  """Records `export(..., dry_run=...)` calls and returns a fixed plan."""

  def __init__(self, plan):
    self.plan = plan
    self.dry_run_calls = []
    self.export_calls = []

  def export(self, model, output_dir=None, *, dry_run=False, **kwargs):
    if dry_run:
      self.dry_run_calls.append((model, output_dir, kwargs))
      return self.plan
    self.export_calls.append((model, output_dir, kwargs))
    return self.plan


class ConvertHuggingFaceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output = self.create_tempdir().full_path
    self.tmp_files = self.create_tempdir()
    self.metadata = self.tmp_files.create_file("meta.pbtext").full_path
    self.write_metadata = self.enter_context(
        mock.patch.object(
            litert_lm_config,
            "write_metadata_override",
            autospec=True,
            return_value=self.metadata,
        )
    )

  def _run(self, fake, **kwargs):
    with mock.patch.object(huggingface, "_export_hf", return_value=fake):
      return huggingface.convert_huggingface(
          "org/model", self.output, target=(), export_aipack=None, **kwargs
      )

  def test_omits_unset_options_so_export_hf_uses_defaults(self):
    fake = _FakeExportHf(
        _FakePlan("automatic_speech_recognition", {"input_sec": 30.0})
    )

    self._run(fake)

    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(
        kwargs,
        {
            "task": "auto",
            "target": "cpu_gpu",
            "trust_remote_code": False,
        },
    )

  def test_passes_explicit_options_to_export_hf(self):
    fake = _FakeExportHf(
        _FakePlan("automatic_speech_recognition", {"input_sec": 30.0})
    )

    self._run(
        fake,
        quantize="none",
        prefill_lengths="128, 512",
        cache_length=2048,
        bundle_litert_lm=False,
        task="automatic_speech_recognition",
    )

    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(
        kwargs,
        {
            "task": "automatic_speech_recognition",
            "target": "cpu_gpu",
            "trust_remote_code": False,
            "quantization_recipe": None,
            "prefill_lengths": [128, 512],
            "cache_length": 2048,
            "bundle_litert_lm": False,
        },
    )

  def test_missing_model_package_names_the_pip_package(self):
    fake = _FakeExportHf(_FakePlan("automatic_speech_recognition", {}))
    fake.export = mock.Mock(
        side_effect=[
            fake.plan,
            ModuleNotFoundError("No module named 'nemo'", name="nemo"),
        ]
    )

    with self.assertRaises(click.ClickException) as cm:
      self._run(fake)

    self.assertIn(
        "needs the Python package 'nemo_toolkit[asr]'", cm.exception.message
    )
    self.assertIn('pip install "nemo_toolkit[asr]"', cm.exception.message)

  def test_other_import_errors_are_not_reported_as_missing_packages(self):
    # Version skew: the package is installed but lacks a name.
    error = ImportError(
        "cannot import name 'X' from 'transformers'", name="transformers"
    )
    fake = _FakeExportHf(_FakePlan("automatic_speech_recognition", {}))
    fake.export = mock.Mock(side_effect=[fake.plan, error])
    with self.assertRaises(ImportError):
      self._run(fake)

  def test_forwards_target_to_export_hf(self):
    fake = _FakeExportHf(_FakePlan("text_generation", {"cache_length": 1024}))

    with mock.patch.object(huggingface, "_export_hf", return_value=fake):
      huggingface.convert_huggingface(
          "org/model",
          self.output,
          target=("npu:qualcomm",),
          export_aipack=None,
      )

    _, _, dry_run_kwargs = fake.dry_run_calls[0]
    self.assertEqual(dry_run_kwargs["target"], "npu:qualcomm")
    _, _, export_kwargs = fake.export_calls[0]
    self.assertEqual(export_kwargs["target"], "npu:qualcomm")

  def test_multiple_targets_raises_usage_error(self):
    fake = _FakeExportHf(_FakePlan("text_generation", {}))

    with mock.patch.object(huggingface, "_export_hf", return_value=fake):
      with self.assertRaisesRegex(click.UsageError, "single --target"):
        huggingface.convert_huggingface(
            "org/model",
            self.output,
            target=("cpu_gpu", "npu:qualcomm"),
            export_aipack=None,
        )

  def test_not_implemented_target_raises_usage_error(self):
    fake = _FakeExportHf(_FakePlan("text_generation", {}))
    fake.export = mock.create_autospec(
        fake.export,
        side_effect=NotImplementedError("npu:qualcomm:sm8750 not implemented"),
    )

    with mock.patch.object(huggingface, "_export_hf", return_value=fake):
      with self.assertRaisesRegex(
          click.UsageError, "npu:qualcomm:sm8750 not implemented"
      ):
        huggingface.convert_huggingface(
            "org/model",
            self.output,
            target=("npu:qualcomm:sm8750",),
            export_aipack=None,
        )

  def test_dry_run_resolves_without_exporting_or_fetching(self):
    fake = _FakeExportHf(
        _FakePlan(
            "text_generation",
            {"cache_length": 4096},
            model_type="qwen3",
        )
    )

    self._run(fake, dry_run=True)

    self.assertLen(fake.dry_run_calls, 1)
    self.assertEmpty(fake.export_calls)
    self.write_metadata.assert_not_called()

  def test_llm_family_adds_canonical_metadata(self):
    fake = _FakeExportHf(
        _FakePlan(
            "text_generation",
            {
                "cache_length": 32771,
                "llm_metadata_max_num_tokens_override": 4096,
            },
            model_type="qwen3",
        )
    )

    self._run(fake)

    self.write_metadata.assert_called_once_with("qwen3", 4096, mock.ANY)
    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(kwargs["litert_lm_llm_metadata_override"], self.metadata)
    self.assertTrue(kwargs["use_jinja_template"])

  def test_user_cache_length_override_updates_canonical_metadata(self):
    fake = _FakeExportHf(
        _FakePlan(
            "text_generation",
            {
                "cache_length": 8192,
                "llm_metadata_max_num_tokens_override": 4096,
            },
            model_type="qwen3",
        )
    )

    self._run(fake, cache_length=8192)

    self.write_metadata.assert_called_once_with("qwen3", 8192, mock.ANY)

  def test_extension_supplied_metadata_is_preserved(self):
    fake = _FakeExportHf(
        _FakePlan(
            "text_generation",
            {"litert_lm_llm_metadata_override": "/ext/meta.pbtext"},
            model_type="qwen3",
        )
    )

    self._run(fake)

    self.write_metadata.assert_not_called()
    _, _, kwargs = fake.export_calls[0]
    self.assertNotIn("litert_lm_llm_metadata_override", kwargs)

  def test_llm_metadata_none_keeps_generated_metadata(self):
    fake = _FakeExportHf(
        _FakePlan(
            "text_generation",
            {
                "litert_lm_llm_metadata_override": "qwen3",
                "use_jinja_template": True,
            },
            model_type="qwen3",
        )
    )

    self._run(fake, llm_metadata="none")

    _, _, kwargs = fake.export_calls[0]
    self.assertIsNone(kwargs["litert_lm_llm_metadata_override"])
    self.assertFalse(kwargs["use_jinja_template"])
    self.write_metadata.assert_not_called()

  def test_custom_llm_metadata_file_overrides_extension_default(self):
    fake = _FakeExportHf(
        _FakePlan(
            "text_generation",
            {"litert_lm_llm_metadata_override": "qwen3"},
            model_type="qwen3",
        )
    )

    self._run(fake, llm_metadata=self.metadata)

    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(kwargs["litert_lm_llm_metadata_override"], self.metadata)
    self.assertTrue(kwargs["use_jinja_template"])
    self.write_metadata.assert_not_called()

  def test_llm_without_canonical_files_uses_converted_template(self):
    fake = _FakeExportHf(_FakePlan("text_generation", {}))

    self._run(fake)

    _, _, kwargs = fake.export_calls[0]
    self.assertNotIn("use_jinja_template", kwargs)
    self.assertNotIn("litert_lm_llm_metadata_override", kwargs)

  def test_missing_llm_metadata_file_raises(self):
    fake = _FakeExportHf(_FakePlan("text_generation", {}))

    with self.assertRaisesRegex(click.BadParameter, "not found"):
      self._run(fake, llm_metadata="/no/such/metadata.pbtext")

  def test_non_llm_task_gets_no_litert_lm_files(self):
    fake = _FakeExportHf(
        _FakePlan("text_to_image", {}, model_type="bonsai_flux2")
    )

    self._run(fake)

    _, _, kwargs = fake.export_calls[0]
    self.assertNotIn("use_jinja_template", kwargs)
    self.write_metadata.assert_not_called()

  def test_resolve_target_handles_string_and_whitespace_inputs(self):
    self.assertEqual(
        huggingface._resolve_target("npu:mediatek"), "npu:mediatek"
    )
    self.assertEqual(huggingface._resolve_target("   "), "cpu_gpu")
    self.assertEqual(huggingface._resolve_target(("   ", "")), "cpu_gpu")

  def test_export_hf_imports_litert_torch_lazily(self):
    fake_export = types.ModuleType("litert_torch.generative.export_hf.export")
    fake_export.export = lambda *args, **kwargs: None
    fake_export_hf = types.ModuleType("litert_torch.generative.export_hf")
    fake_export_hf.export = fake_export
    fake_generative = types.ModuleType("litert_torch.generative")
    fake_generative.export_hf = fake_export_hf
    fake_litert_torch = types.ModuleType("litert_torch")
    fake_litert_torch.generative = fake_generative

    with mock.patch.dict(
        sys.modules,
        {
            "litert_torch": fake_litert_torch,
            "litert_torch.generative": fake_generative,
            "litert_torch.generative.export_hf": fake_export_hf,
            "litert_torch.generative.export_hf.export": fake_export,
        },
    ):
      module = huggingface._export_hf()

    self.assertIs(module, fake_export)
    self.assertTrue(callable(module.export))


if __name__ == "__main__":
  absltest.main()
