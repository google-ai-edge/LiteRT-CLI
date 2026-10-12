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

"""Tests for the canonical LiteRT-LM config lookup (no network: urlopen is faked)."""

import io
import os
import pathlib
import types
from unittest import mock
import urllib.error

from absl.testing import absltest
from absl.testing import parameterized
from litert_cli.commands.convert import litert_lm_config

_PBTEXT = """\
max_num_tokens: 32768
llm_model_type {
  qwen3 {
  }
}
jinja_prompt_template: "{{ messages }}"
"""


def _config(model_type=None, text_model_type=None, name=""):
  text_config = (
      types.SimpleNamespace(model_type=text_model_type)
      if text_model_type
      else None
  )
  return types.SimpleNamespace(
      model_type=model_type, text_config=text_config, _name_or_path=name
  )


class DetectFamilyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("qwen3", "Qwen/Qwen3-1.7B", _config("qwen3"), "qwen3"),
      ("qwen2_5", "Qwen/Qwen2.5-1.5B-Instruct", _config("qwen2"), "qwen2_5"),
      ("qwen3_5", "Qwen/Qwen3.5-2B", _config("qwen3_5"), "qwen3_5"),
      (
          "qwen3_5_text_config",
          "Qwen/Qwen3.5-2B",
          _config("other", "qwen3_5_text"),
          "qwen3_5",
      ),
      ("lfm2", "LiquidAI/LFM2-1.2B", _config("lfm2"), "lfm2"),
      ("minicpm5_by_id", "openbmb/MiniCPM5-1B", _config("llama"), "minicpm5"),
      (
          "minicpm5_by_local_path",
          "/models/ckpt",
          _config("llama", name="openbmb/MiniCPM5-2B"),
          "minicpm5",
      ),
      ("unknown_llama", "meta-llama/Llama-3.2-1B", _config("llama"), None),
      ("no_config", "Qwen/Qwen3-1.7B", None, None),
  )
  def test_detect_family(self, model_id, config, expected):
    self.assertEqual(litert_lm_config.detect_family(model_id, config), expected)


class MaxNumTokensTest(absltest.TestCase):

  def test_replaces_existing_value(self):
    out = litert_lm_config.with_max_num_tokens(_PBTEXT, 4096)
    self.assertIn("max_num_tokens: 4096\n", out)
    self.assertNotIn("32768", out)
    self.assertIn("jinja_prompt_template", out)

  def test_appends_when_missing(self):
    out = litert_lm_config.with_max_num_tokens(
        'jinja_prompt_template: "x"', 2048
    )
    self.assertTrue(out.endswith("max_num_tokens: 2048\n"))

  def test_ignores_nested_fields(self):
    pbtext = "sampler_params {\n  max_num_tokens: 7\n}\n"
    out = litert_lm_config.with_max_num_tokens(pbtext, 4096)
    self.assertIn("  max_num_tokens: 7\n", out)
    self.assertTrue(out.endswith("max_num_tokens: 4096\n"))


class FetchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cache = self.create_tempdir().full_path
    self.enter_context(
        mock.patch.dict(os.environ, {"XDG_CACHE_HOME": self.cache}, clear=False)
    )
    os.environ.pop("LITERT_LM_MODELS_REF", None)

  def _urlopen(self, text=_PBTEXT):
    # A fresh response per call, since each call reads the body once.
    return mock.patch.object(
        litert_lm_config.urllib.request,
        "urlopen",
        side_effect=lambda *a, **k: io.BytesIO(text.encode()),
    )

  def _cached(self, ref, family="qwen3"):
    return (
        pathlib.Path(self.cache)
        / "litert-cli"
        / "litert_lm_models"
        / ref
        / f"{family}.pbtext"
    )

  def test_default_ref_is_main_and_refetched(self):
    with self._urlopen() as urlopen:
      text = litert_lm_config.fetch_canonical_metadata("qwen3")
      litert_lm_config.fetch_canonical_metadata("qwen3")
    self.assertEqual(text, _PBTEXT)
    # `main` moves, so every call fetches.
    self.assertEqual(urlopen.call_count, 2)
    url = urlopen.call_args.args[0]
    self.assertIn("/main/models/qwen3/LlmMetadataProto.pbtext", url)
    self.assertEqual(self._cached("main").read_text(), _PBTEXT)

  def test_moving_ref_offline_uses_cache(self):
    with self._urlopen("old config"):
      litert_lm_config.fetch_canonical_metadata("qwen3")
    with mock.patch.object(
        litert_lm_config.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
      self.assertEqual(
          litert_lm_config.fetch_canonical_metadata("qwen3"), "old config"
      )

  def test_commit_ref_served_from_cache(self):
    with self._urlopen() as urlopen:
      litert_lm_config.fetch_canonical_metadata("qwen3", ref="4a363c07")
      litert_lm_config.fetch_canonical_metadata("qwen3", ref="4a363c07")
    urlopen.assert_called_once()
    self.assertIn("/4a363c07/models/qwen3/", urlopen.call_args.args[0])
    self.assertEqual(self._cached("4a363c07").read_text(), _PBTEXT)

  def test_env_ref_override(self):
    with mock.patch.dict(os.environ, {"LITERT_LM_MODELS_REF": "4a363c07"}):
      with self._urlopen() as urlopen:
        litert_lm_config.fetch_canonical_metadata("lfm2")
    self.assertIn("/4a363c07/models/lfm2/", urlopen.call_args.args[0])

  def test_network_error_without_cache_raises_runtime_error(self):
    with mock.patch.object(
        litert_lm_config.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
      with self.assertRaisesRegex(RuntimeError, "qwen3"):
        litert_lm_config.fetch_canonical_metadata("qwen3")

  def test_write_metadata_override(self):
    out_dir = self.create_tempdir().full_path
    with mock.patch.object(
        litert_lm_config.urllib.request,
        "urlopen",
        return_value=io.BytesIO(_PBTEXT.encode()),
    ):
      path = litert_lm_config.write_metadata_override("qwen3", 4096, out_dir)
    text = pathlib.Path(path).read_text()
    self.assertIn("max_num_tokens: 4096", text)
    self.assertIn("qwen3", text)


if __name__ == "__main__":
  absltest.main()
