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

"""Tests for `litert convert <hf_id>` (litert-torch `auto_export` is faked)."""

import dataclasses
import pathlib
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
  litert_lm_model_family: str | None = None
  litert_lm_chat_template: str | None = None
  runtime_model_name: str | None = None

  def summary_rows(self):
    return [("task", self.task, "model")]


class _FakeAutoExport:
  """Records `resolve` / `export` calls and returns a fixed plan."""

  def __init__(self, plan, bundle_name=None):
    self.plan = plan
    self.bundle_name = bundle_name
    self.resolve_calls = []
    self.export_calls = []

  def resolve(self, model, **kwargs):
    self.resolve_calls.append((model, kwargs))
    return self.plan

  def export(self, model, output_dir, **kwargs):
    self.export_calls.append((model, output_dir, kwargs))
    if self.bundle_name:
      pathlib.Path(output_dir, self.bundle_name).write_bytes(b"")
    return self.plan


class ConvertHuggingFaceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output = self.create_tempdir().full_path
    self.tmp_files = self.create_tempdir()
    self.metadata = self.tmp_files.create_file("meta.pbtext").full_path
    self.template = self.tmp_files.create_file("template.jinja").full_path
    self.write_metadata = self.enter_context(
        mock.patch.object(
            litert_lm_config,
            "write_metadata_override",
            autospec=True,
            return_value=self.metadata,
        )
    )
    self.write_template = self.enter_context(
        mock.patch.object(
            litert_lm_config,
            "write_canonical_file",
            autospec=True,
            return_value=self.template,
        )
    )

  def _run(self, fake, **kwargs):
    with mock.patch.object(huggingface, "_auto_export", return_value=fake):
      return huggingface.convert_huggingface(
          "org/model", self.output, target=(), export_aipack=None, **kwargs
      )

  def test_passes_only_explicit_options_to_auto_export(self):
    fake = _FakeAutoExport(
        _FakePlan("automatic_speech_recognition", {"input_sec": 30.0})
    )

    self._run(fake, quantize="none", task="asr", export_args=("input_sec=10",))

    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(
        kwargs,
        {
            "task": "automatic_speech_recognition",
            "preset": "recommended",
            "extension_modules": None,
            "quantization_recipe": "",
            "input_sec": 10,
        },
    )

  def test_dry_run_resolves_without_exporting_or_fetching(self):
    fake = _FakeAutoExport(
        _FakePlan(
            "text_generation",
            {"cache_length": 32768},
            litert_lm_model_family="qwen3",
        )
    )

    self._run(fake, dry_run=True)

    self.assertLen(fake.resolve_calls, 1)
    self.assertEmpty(fake.export_calls)
    self.write_metadata.assert_not_called()

  def test_llm_family_adds_canonical_metadata(self):
    fake = _FakeAutoExport(
        _FakePlan(
            "text_generation",
            {"cache_length": 32768},
            litert_lm_model_family="qwen3",
        ),
        bundle_name="model.litertlm",
    )

    self._run(fake)

    self.write_metadata.assert_called_once_with("qwen3", 32768, mock.ANY)
    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(kwargs["litert_lm_llm_metadata_override"], self.metadata)
    self.assertTrue(kwargs["use_jinja_template"])

  def test_chat_template_is_fetched_from_litert_lm_models(self):
    fake = _FakeAutoExport(
        _FakePlan(
            "image_text_to_text",
            {},
            litert_lm_chat_template="gemma4/chat_template.jinja",
        )
    )

    self._run(fake)

    self.write_template.assert_called_once_with(
        "gemma4", "chat_template.jinja", mock.ANY
    )
    _, _, kwargs = fake.export_calls[0]
    self.assertEqual(kwargs["jinja_chat_template_override"], self.template)
    self.assertNotIn("litert_lm_llm_metadata_override", kwargs)

  def test_llm_metadata_none_keeps_generated_metadata(self):
    fake = _FakeAutoExport(
        _FakePlan("text_generation", {}, litert_lm_model_family="qwen3")
    )

    self._run(fake, llm_metadata="none")

    _, _, kwargs = fake.export_calls[0]
    self.assertNotIn("litert_lm_llm_metadata_override", kwargs)
    self.write_metadata.assert_not_called()

  def test_missing_llm_metadata_file_raises(self):
    fake = _FakeAutoExport(_FakePlan("text_generation", {}))

    with self.assertRaisesRegex(click.BadParameter, "not found"):
      self._run(fake, llm_metadata="/no/such/metadata.pbtext")

  def test_non_llm_task_gets_no_litert_lm_files(self):
    fake = _FakeAutoExport(
        _FakePlan("text_to_image", {}, runtime_model_name="bonsai-flux2"),
        bundle_name="model.litertlm",
    )

    self._run(fake)

    _, _, kwargs = fake.export_calls[0]
    self.assertNotIn("use_jinja_template", kwargs)
    self.write_metadata.assert_not_called()

  def test_reserved_export_arg_is_a_usage_error(self):
    fake = _FakeAutoExport(_FakePlan("text_generation", {}))

    with self.assertRaisesRegex(click.UsageError, "use --output"):
      self._run(fake, export_args=("output_dir=/tmp/x",))

  def test_unknown_task_is_a_usage_error(self):
    fake = _FakeAutoExport(_FakePlan("text_generation", {}))

    with self.assertRaisesRegex(click.UsageError, "Unknown task"):
      self._run(fake, task="video")


if __name__ == "__main__":
  absltest.main()
