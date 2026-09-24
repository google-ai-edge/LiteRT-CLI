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

from __future__ import annotations

import json
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from click import testing

from litert_cli.commands import list as list_module


class ListCmdTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmpdir.cleanup)
    self.cache_dir = pathlib.Path(self.tmpdir.name)

  def _invoke(self, args):
    runner = testing.CliRunner()
    with mock.patch.object(
        list_module, "LITERT_MODELS_CACHE_DIR", str(self.cache_dir)
    ):
      return runner.invoke(list_module.list_cmd, args)

  def _write_model(self):
    model_dir = self.cache_dir / "mobilenet"
    model_dir.mkdir()
    (model_dir / "model.tflite").write_bytes(b"model")
    (model_dir / "labels.txt").write_text("label\n")
    metadata = {
        "model_ref": "mobilenet",
        "hf_id": "litert-community/MobileNet-v3-large",
        "source": "huggingface",
        "created_at": "2026-09-24T00:00:00+00:00",
        "sub_references": {
            "fp32": {
                "file": "model.tflite",
                "hf_id": "litert-community/MobileNet-v3-large",
            },
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    return model_dir

  def test_list_all_json(self):
    self._write_model()

    result = self._invoke(["--json"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads(result.output)
    self.assertLen(data, 1)
    self.assertEqual(data[0]["ref"], "mobilenet")
    self.assertEqual(data[0]["hf_id"], "litert-community/MobileNet-v3-large")
    self.assertEqual(data[0]["source"], "huggingface")
    self.assertEqual(data[0]["sub_references"]["fp32"]["file"], "model.tflite")

  def test_list_model_json_includes_files(self):
    self._write_model()

    result = self._invoke(["mobilenet", "--json"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads(result.output)
    self.assertEqual(data["ref"], "mobilenet")
    self.assertCountEqual(
        [file_info["name"] for file_info in data["files"]],
        ["labels.txt", "model.tflite"],
    )
    model_file = next(
        file_info
        for file_info in data["files"]
        if file_info["name"] == "model.tflite"
    )
    self.assertEqual(model_file["size_bytes"], 5)
    self.assertEqual(model_file["sub_references"], ["fp32"])

  def test_list_empty_cache_json(self):
    result = self._invoke(["--json"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.assertEqual(json.loads(result.output), [])

  def test_list_json_ignores_malformed_metadata(self):
    model_dir = self.cache_dir / "bad_metadata"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text("{")

    result = self._invoke(["--json"])

    self.assertEqual(result.exit_code, 0, msg=result.output)
    data = json.loads(result.output)
    self.assertLen(data, 1)
    self.assertEqual(data[0]["ref"], "bad_metadata")
    self.assertEqual(data[0]["hf_id"], "N/A")
    self.assertEqual(data[0]["source"], "N/A")
    self.assertEqual(data[0]["sub_references"], {})


if __name__ == "__main__":
  absltest.main()
