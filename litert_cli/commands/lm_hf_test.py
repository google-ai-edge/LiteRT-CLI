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

"""Tests for Hugging Face repo ids in `litert lm run|benchmark`.

No network, torch or litert-lm: the Hub API, the conversion process and the
litert-lm process are faked.
"""

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import click
from click import testing
import huggingface_hub
import huggingface_hub.utils
from litert_cli import litert
from litert_cli.commands import lm_hf
from litert_cli.core import constants
from litert_cli.core import deps

_VARIANT = "litertlm-dynamic_wi8_afp32-p256-c4096"

# pylint: disable=line-too-long
_MAMBA_LOG = """\
Traceback (most recent call last):
  File "/venv/bin/litert", line 10, in <module>
    sys.exit(cli())
  File "/venv/site-packages/litert_torch/generative/export_hf/export.py", line 370, in export
    exported_model_artifacts = run_export_tasks(
  File "/venv/site-packages/litert_torch/generative/export_hf/core/cache.py", line 629, in create_from_config
    model_config.linear_key_head_dim * model_config.linear_num_key_heads
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/venv/site-packages/transformers/configuration_utils.py", line 652, in __getattribute__
    return super().__getattribute__(key)
AttributeError: 'MambaConfig' object has no attribute 'linear_key_head_dim'
"""

_IMPORT_LOG = """\
Traceback (most recent call last):
  File "/repo/litert_cli/commands/convert/huggingface.py", line 70, in convert_huggingface
    from litert_torch.generative.export_hf import export as hf_export
  File "/venv/site-packages/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py", line 41, in <module>
    from tensorflow.lite.python import schema_py_generated as schema_fb
ModuleNotFoundError: No module named 'tensorflow'
"""
# pylint: enable=line-too-long


def _tempdir(test):
  # tempfile rather than create_tempdir(): absl flags are not parsed under
  # pytest.
  path = tempfile.mkdtemp()
  test.addCleanup(shutil.rmtree, path, ignore_errors=True)
  return pathlib.Path(path)


def _hub_error(cls, message):
  """Builds a huggingface_hub HTTP error without an HTTP response object."""
  error = cls.__new__(cls)
  Exception.__init__(error, message)
  return error


def _model_info(files, sha="a" * 40, gated=False, parameters=751_632_384):
  return types.SimpleNamespace(
      sha=sha,
      siblings=[
          types.SimpleNamespace(rfilename=name, size=size)
          for name, size in files
      ],
      gated=gated,
      safetensors=types.SimpleNamespace(total=parameters),
  )


_CHECKPOINT = [("config.json", 726), ("model.safetensors", 1_503_300_328)]


class LooksLikeHfRepoTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = _tempdir(self)
    self.enter_context(
        mock.patch.dict(os.environ, {"LITERT_LM_DIR": str(self.tmp / "lm")})
    )
    cwd = os.getcwd()
    os.chdir(self.tmp)
    self.addCleanup(os.chdir, cwd)

  @parameterized.parameters(
      ("org/name", True),
      ("Qwen/Qwen3-0.6B", True),
      ("litert-community/gemma-4-E2B-it-litert-lm", True),
      ("x", False),
      ("model.litertlm", False),
      ("./x.litertlm", False),
      ("../x", False),
      ("~/x", False),
      ("/abs/path", False),
      ("a/b/c", False),
      ("org/name.litertlm", False),
      ("org/", False),
      ("/name", False),
  )
  def test_reference_shape(self, reference, expected):
    self.assertEqual(lm_hf.looks_like_hf_repo(reference), expected)

  def test_existing_local_path_is_not_a_repo(self):
    (self.tmp / "models" / "qwen3").mkdir(parents=True)
    self.assertFalse(lm_hf.looks_like_hf_repo("models/qwen3"))

  def test_imported_litert_lm_model_is_not_a_repo(self):
    imported = self.tmp / "lm" / "models" / "org--name"
    imported.mkdir(parents=True)
    (imported / "model.litertlm").write_bytes(b"x")
    self.assertFalse(lm_hf.looks_like_hf_repo("org/name"))
    self.assertTrue(lm_hf.looks_like_hf_repo("org/other"))


class ShouldResolveTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tmp = str(_tempdir(self))
    self.enter_context(mock.patch.dict(os.environ, {"LITERT_LM_DIR": tmp}))

  @parameterized.parameters(
      (["run", "org/name", "--prompt", "hi"], True),
      (["benchmark", "org/name", "-p", "128"], True),
      (["run", "org/name"], True),
      (["run", "--from-huggingface-repo", "org/name"], False),
      (["run", "org/name", "--from-huggingface-repo=org/other"], False),
      (["run", "org/name", "--help"], False),
      (["run", "org/name", "-h"], False),
      (["run", "./model.litertlm"], False),
      (["run", "--prompt", "x", "org/name"], False),
      (["list"], False),
      (["import", "org/name"], False),
      (["--help"], False),
      (["run"], False),
      ([], False),
  )
  def test_should_resolve(self, args, expected):
    self.assertEqual(lm_hf.should_resolve(args), expected)


class SplitConvertOptionsTest(absltest.TestCase):

  def test_defaults(self):
    options, rest, found = lm_hf.split_convert_options(["--prompt", "hi"])
    self.assertEqual(options, lm_hf.ConvertOptions())
    self.assertEqual(rest, ["--prompt", "hi"])
    self.assertEqual(found, [])
    self.assertEqual(options.variant, _VARIANT)

  def test_takes_out_conversion_options(self):
    options, rest, found = lm_hf.split_convert_options(
        [
            "--quantize",
            "weight_only_wi4_afp32",
            "--prompt",
            "a=b",
            "--prefill-lengths=128,512",
            "--cache-length",
            "2048",
            "--reconvert",
            "--thinking",
            "false",
        ]
    )
    self.assertEqual(
        options,
        lm_hf.ConvertOptions(
            recipe="weight_only_wi4_afp32",
            prefill_lengths=(128, 512),
            cache_length=2048,
            reconvert=True,
        ),
    )
    self.assertEqual(rest, ["--prompt", "a=b", "--thinking", "false"])
    self.assertEqual(
        found,
        ["--quantize", "--prefill-lengths", "--cache-length", "--reconvert"],
    )
    self.assertEqual(
        options.variant, "litertlm-weight_only_wi4_afp32-p128_512-c2048"
    )

  def test_quantize_recipe_alias_with_equals(self):
    options, _, _ = lm_hf.split_convert_options(["--quantize-recipe=none"])
    self.assertEqual(options.recipe, lm_hf.NO_QUANTIZATION)
    self.assertEqual(options.variant, "litertlm-none-p256-c4096")

  def test_missing_value(self):
    with self.assertRaisesRegex(click.UsageError, "requires a value"):
      lm_hf.split_convert_options(["--cache-length"])

  def test_invalid_values(self):
    for args in (
        ["--cache-length", "abc"],
        ["--cache-length", "0"],
        ["--prefill-lengths", "128,x"],
        ["--prefill-lengths", ""],
    ):
      with self.subTest(args=args):
        with self.assertRaises(click.UsageError):
          lm_hf.split_convert_options(args)


class LmCommandTest(parameterized.TestCase):
  """`litert lm` resolves Hugging Face repo ids and forwards the rest as is."""

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(deps, "ensure_extra", autospec=True))
    self.run = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    self.run.return_value = subprocess.CompletedProcess([], 0)
    self.resolve = self.enter_context(
        mock.patch.object(lm_hf, "resolve_args", autospec=True)
    )
    tmp = str(_tempdir(self))
    self.enter_context(mock.patch.dict(os.environ, {"LITERT_LM_DIR": tmp}))

  def test_repo_id_is_resolved_before_forwarding(self):
    self.resolve.return_value = ["run", "/cache/bundle", "--prompt", "hi"]
    result = testing.CliRunner().invoke(
        litert.cli, ["lm", "run", "org/name", "--prompt", "hi"]
    )
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.resolve.assert_called_once_with(["run", "org/name", "--prompt", "hi"])
    self.run.assert_called_once_with(
        ["litert-lm", "run", "/cache/bundle", "--prompt", "hi"], check=False
    )

  @parameterized.parameters(
      (["run", "./model.litertlm", "--prompt", "hi"],),
      (["run", "--from-huggingface-repo=org/name", "--prompt", "hi"],),
      (["run", "--prompt", "x", "org/name"],),
      (["benchmark", "gemma3-1b"],),
      (["list"],),
      (["--help"],),
  )
  def test_other_arguments_are_forwarded_unchanged(self, args):
    result = testing.CliRunner().invoke(litert.cli, ["lm"] + args)
    self.assertEqual(result.exit_code, 0, msg=result.output)
    self.resolve.assert_not_called()
    self.run.assert_called_once_with(["litert-lm"] + args, check=False)


class ResolveArgsTest(absltest.TestCase):
  """Hub lookups, the conversion cache and the failure messages."""

  def setUp(self):
    super().setUp()
    self.tmp = _tempdir(self)
    self.models = self.tmp / "models"
    self.enter_context(
        mock.patch.object(constants, "LITERT_CLI_CACHE_DIR", str(self.tmp))
    )
    self.enter_context(
        mock.patch.object(
            constants, "LITERT_MODELS_CACHE_DIR", str(self.models)
        )
    )
    self.ensure_extra = self.enter_context(
        mock.patch.object(deps, "ensure_extra", autospec=True)
    )
    self.model_info = self.enter_context(
        mock.patch.object(huggingface_hub.HfApi, "model_info", autospec=True)
    )
    self.convert = self.enter_context(
        mock.patch.object(subprocess, "run", autospec=True)
    )
    self.convert.side_effect = self._fake_conversion()
    self.echoed = []
    self.enter_context(
        mock.patch.object(
            click,
            "echo",
            side_effect=lambda message="", **_: self.echoed.append(message),
        )
    )

  def _fake_conversion(self, returncode=0, log="", write_bundle=True):
    def fake_run(cmd, stdin=None, stderr=None, env=None, check=False):
      del env, check
      self.assertEqual(stdin, subprocess.DEVNULL)
      stderr.write(log.encode())
      output = pathlib.Path(cmd[cmd.index("--output") + 1])
      if write_bundle and returncode == 0:
        (output / "model.litertlm").write_bytes(b"x" * 1000)
      return subprocess.CompletedProcess(cmd, returncode)

    return fake_run

  def _output(self):
    return "\n".join(str(m) for m in self.echoed)

  def _bundle_dir(self, variant=_VARIANT):
    return self.models / "org__name" / variant

  def test_repo_with_litertlm_file_is_forwarded(self):
    self.model_info.return_value = _model_info(
        [("model.litertlm", 10), ("README.md", 1)]
    )
    args = lm_hf.resolve_args(
        ["run", "org/name", "--quantize", "x", "--prompt", "hi"]
    )
    self.assertEqual(
        args,
        ["run", "--from-huggingface-repo", "org/name", "--prompt", "hi"],
    )
    self.convert.assert_not_called()
    self.assertIn("Ignoring --quantize", self._output())

  def test_converts_once_then_reuses_the_bundle(self):
    self.model_info.return_value = _model_info(_CHECKPOINT)

    args = lm_hf.resolve_args(["run", "org/name", "--prompt", "hi"])

    bundle_dir = self._bundle_dir()
    self.assertEqual(args, ["run", str(bundle_dir), "--prompt", "hi"])
    self.assertTrue((bundle_dir / "model.litertlm").is_file())
    self.assertTrue((bundle_dir / "convert.log").is_file())
    self.assertEqual(
        [p.name for p in bundle_dir.parent.iterdir() if ".tmp-" in p.name], []
    )
    self.ensure_extra.assert_called_once_with("convert")
    cmd = self.convert.call_args.args[0]
    self.assertEqual(
        cmd[1:5], ["-m", "litert_cli.litert", "convert", "org/name"]
    )
    for flag, value in (
        ("--quantize", "dynamic_wi8_afp32"),
        ("--prefill-lengths", "256"),
        ("--cache-length", "4096"),
    ):
      self.assertEqual(cmd[cmd.index(flag) + 1], value)
    metadata = json.loads((bundle_dir.parent / "metadata.json").read_text())
    self.assertEqual(metadata["hf_id"], "org/name")
    self.assertEqual(metadata["source"], "converted")
    record = metadata["conversions"][_VARIANT]
    self.assertEqual(record["revision"], "a" * 40)
    self.assertEqual(record["recipe"], "dynamic_wi8_afp32")
    self.assertEqual(record["prefill_lengths"], [256])
    self.assertEqual(record["cache_length"], 4096)
    self.assertEqual(record["bundle_bytes"], 1000)
    self.assertIn("has no .litertlm file; converting", self._output())
    self.assertIn("Converted in", self._output())

    self.echoed.clear()
    args = lm_hf.resolve_args(["benchmark", "org/name", "-p", "128"])
    self.assertEqual(args, ["benchmark", str(bundle_dir), "-p", "128"])
    self.assertEqual(self.convert.call_count, 1)
    self.assertIn("Using the bundle converted on", self._output())

  def test_keeps_metadata_of_litert_download(self):
    model_dir = self.models / "org__name"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_ref": "my_ref", "source": "huggingface"})
    )
    self.model_info.return_value = _model_info(_CHECKPOINT)
    lm_hf.resolve_args(["run", "org/name"])
    metadata = json.loads((model_dir / "metadata.json").read_text())
    self.assertEqual(metadata["model_ref"], "my_ref")
    self.assertEqual(metadata["source"], "huggingface")
    self.assertIn(_VARIANT, metadata["conversions"])

  def test_quantize_none_converts_without_recipe(self):
    self.model_info.return_value = _model_info(_CHECKPOINT)
    args = lm_hf.resolve_args(["run", "org/name", "--quantize", "none"])
    self.assertEqual(
        args, ["run", str(self._bundle_dir("litertlm-none-p256-c4096"))]
    )
    self.assertNotIn("--quantize", self.convert.call_args.args[0])

  def test_newer_revision_keeps_the_cached_bundle(self):
    self.model_info.return_value = _model_info(_CHECKPOINT, sha="a" * 40)
    lm_hf.resolve_args(["run", "org/name"])
    self.model_info.return_value = _model_info(_CHECKPOINT, sha="b" * 40)
    self.echoed.clear()

    args = lm_hf.resolve_args(["run", "org/name"])

    self.assertEqual(args, ["run", str(self._bundle_dir())])
    self.assertEqual(self.convert.call_count, 1)
    self.assertIn(
        "has changed since the bundle was converted (revision aaaaaaa, now"
        " bbbbbbb)",
        self._output(),
    )
    self.assertIn("--reconvert", self._output())

  def test_reconvert_rebuilds_the_bundle(self):
    self.model_info.return_value = _model_info(_CHECKPOINT, sha="a" * 40)
    lm_hf.resolve_args(["run", "org/name"])
    self.model_info.return_value = _model_info(_CHECKPOINT, sha="b" * 40)

    args = lm_hf.resolve_args(["run", "org/name", "--reconvert"])

    self.assertEqual(args, ["run", str(self._bundle_dir())])
    self.assertEqual(self.convert.call_count, 2)
    metadata = json.loads(
        (self.models / "org__name" / "metadata.json").read_text()
    )
    self.assertEqual(metadata["conversions"][_VARIANT]["revision"], "b" * 40)

  def test_stale_tmp_dirs_are_removed(self):
    model_dir = self.models / "org__name"
    dead = model_dir / f"{_VARIANT}.tmp-111"
    alive = model_dir / f"{_VARIANT}.tmp-222"
    for path in (dead, alive):
      path.mkdir(parents=True)
      (path / "model.litertlm").write_bytes(b"partial")
    self.enter_context(
        mock.patch.object(
            lm_hf, "_pid_alive", autospec=True, side_effect=lambda p: p == 222
        )
    )
    self.model_info.return_value = _model_info(_CHECKPOINT)

    lm_hf.resolve_args(["run", "org/name"])

    self.assertFalse(dead.exists())
    self.assertTrue(alive.exists())

  def test_failed_conversion_is_reported_and_not_cached(self):
    self.model_info.side_effect = [
        _model_info(_CHECKPOINT),
        _model_info([("README.md", 1)]),  # litert-community/name: no bundle
    ]
    self.convert.side_effect = self._fake_conversion(
        returncode=1, log=_MAMBA_LOG
    )

    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])

    message = cm.exception.message
    self.assertIn(
        "Conversion of org/name failed: AttributeError: 'MambaConfig' object"
        " has no attribute 'linear_key_head_dim'",
        message,
    )
    self.assertIn(
        "in litert_torch/generative/export_hf/core/cache.py, line 629, in"
        " create_from_config",
        message,
    )
    kept_log = self.tmp / "logs" / f"org__name-{_VARIANT}.log"
    self.assertIn(f"full log: {kept_log}", message)
    self.assertEqual(kept_log.read_text(), _MAMBA_LOG)
    self.assertIn("LITERT_VERBOSE=1 litert convert org/name", message)
    self.assertIn("export_hf may not support this architecture", message)
    self.assertNotIn("a converted bundle exists", message)
    self.assertIsNone(lm_hf.cached_bundle("org/name", lm_hf.ConvertOptions()))
    # Nothing else was in the model dir, so `litert list` does not show it.
    self.assertFalse((self.models / "org__name").exists())

  def test_failed_conversion_keeps_other_bundles(self):
    self.model_info.return_value = _model_info(_CHECKPOINT)
    lm_hf.resolve_args(["run", "org/name"])
    self.model_info.side_effect = [
        _model_info(_CHECKPOINT),
        _hub_error(huggingface_hub.utils.RepositoryNotFoundError, "404"),
    ]
    self.convert.side_effect = self._fake_conversion(
        returncode=1, log=_MAMBA_LOG
    )

    with self.assertRaises(click.ClickException):
      lm_hf.resolve_args(["run", "org/name", "--cache-length", "8192"])

    self.assertEqual(
        sorted(p.name for p in (self.models / "org__name").iterdir()),
        [_VARIANT, "metadata.json"],
    )

  def test_failure_points_to_a_litert_community_bundle(self):
    self.model_info.side_effect = [
        _model_info(_CHECKPOINT),
        _model_info([("name.litertlm", 10)]),
    ]
    self.convert.side_effect = self._fake_conversion(
        returncode=1, log=_MAMBA_LOG
    )
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    self.assertIn(
        "a converted bundle exists: litert lm run litert-community/name",
        cm.exception.message,
    )

  def test_import_error_is_an_install_problem(self):
    self.model_info.side_effect = [
        _model_info(_CHECKPOINT),
        _hub_error(huggingface_hub.utils.RepositoryNotFoundError, "404"),
    ]
    self.convert.side_effect = self._fake_conversion(
        returncode=1, log=_IMPORT_LOG
    )
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    message = cm.exception.message
    self.assertIn("ModuleNotFoundError: No module named 'tensorflow'", message)
    self.assertIn(
        "this is an install problem, not a problem with the model", message
    )
    self.assertIn("reinstall the 'convert' extra", message)
    self.assertNotIn("export_hf may not support", message)

  def test_killed_conversion(self):
    self.model_info.side_effect = [
        _model_info(_CHECKPOINT),
        _hub_error(huggingface_hub.utils.RepositoryNotFoundError, "404"),
    ]
    self.convert.side_effect = self._fake_conversion(returncode=-9)
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    self.assertIn("stopped by SIGKILL", cm.exception.message)
    self.assertIn("may have run out of memory", cm.exception.message)

  def test_success_exit_without_bundle_is_a_failure(self):
    self.model_info.side_effect = [
        _model_info(_CHECKPOINT),
        _hub_error(huggingface_hub.utils.RepositoryNotFoundError, "404"),
    ]
    self.convert.side_effect = self._fake_conversion(write_bundle=False)
    with self.assertRaisesRegex(
        click.ClickException, "finished without writing model.litertlm"
    ):
      lm_hf.resolve_args(["run", "org/name"])

  def test_repo_not_found(self):
    self.model_info.side_effect = _hub_error(
        huggingface_hub.utils.RepositoryNotFoundError, "401 Client Error."
    )
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    self.assertIn("is not a local path", cm.exception.message)
    self.assertIn("hf auth login", cm.exception.message)
    self.convert.assert_not_called()

  def test_gated_repo_without_access(self):
    self.model_info.return_value = _model_info(_CHECKPOINT, gated="manual")
    self.enter_context(
        mock.patch.object(
            huggingface_hub,
            "get_hf_file_metadata",
            autospec=True,
            side_effect=_hub_error(
                huggingface_hub.utils.GatedRepoError, "401 Client Error."
            ),
        )
    )
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    self.assertIn("is gated", cm.exception.message)
    self.assertIn("https://huggingface.co/org/name", cm.exception.message)
    self.convert.assert_not_called()

  def test_repo_without_checkpoint_or_bundle(self):
    self.model_info.return_value = _model_info(
        [("model-Q4_K_M.gguf", 10), ("README.md", 1)]
    )
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    self.assertIn(
        "no .litertlm file and no transformers config.json (files:"
        " model-Q4_K_M.gguf, README.md)",
        cm.exception.message,
    )

  def test_offline_uses_the_cached_bundle(self):
    self.model_info.return_value = _model_info(_CHECKPOINT)
    lm_hf.resolve_args(["run", "org/name"])
    self.model_info.side_effect = _hub_error(
        huggingface_hub.utils.OfflineModeIsEnabled, "offline mode is enabled"
    )
    self.echoed.clear()

    args = lm_hf.resolve_args(["run", "org/name", "--prompt", "hi"])

    self.assertEqual(args, ["run", str(self._bundle_dir()), "--prompt", "hi"])
    self.assertIn(
        "Could not reach the Hugging Face Hub (HF_HUB_OFFLINE is set)",
        self._output(),
    )
    self.assertIn("Using the bundle converted on", self._output())

  def test_offline_without_cached_bundle(self):
    self.model_info.side_effect = ConnectionError("offline mode is enabled")
    with self.assertRaises(click.ClickException) as cm:
      lm_hf.resolve_args(["run", "org/name"])
    self.assertIn("Could not reach the Hugging Face Hub", cm.exception.message)
    self.assertIn(
        "no bundle converted with these options", cm.exception.message
    )


class SummarizeLogTest(absltest.TestCase):

  def test_exception_and_deepest_litert_torch_frame(self):
    self.assertEqual(
        lm_hf.summarize_log("noise\n" + _MAMBA_LOG + "trailing line\n"),
        (
            "AttributeError: 'MambaConfig' object has no attribute"
            " 'linear_key_head_dim'",
            "litert_torch/generative/export_hf/core/cache.py, line 629, in"
            " create_from_config",
        ),
    )

  def test_last_traceback_of_a_chain(self):
    log = (
        _IMPORT_LOG
        + "\nDuring handling of the above exception, another exception"
        " occurred:\n\n" + _MAMBA_LOG
    )
    exception, _ = lm_hf.summarize_log(log)
    self.assertTrue(exception.startswith("AttributeError"))

  def test_without_litert_torch_frame(self):
    log = (
        "Traceback (most recent call last):\n"
        '  File "/repo/litert_cli/commands/convert/huggingface.py", line 108,'
        " in convert_huggingface\n"
        "    raise ValueError(\n"
        "ValueError: Currently only AutoModelForCausalLM is supported.\n"
    )
    self.assertEqual(
        lm_hf.summarize_log(log),
        ("ValueError: Currently only AutoModelForCausalLM is supported.", None),
    )

  def test_no_traceback(self):
    self.assertEqual(lm_hf.summarize_log("LLVM ERROR: abort\n"), (None, None))


class EstimateTest(absltest.TestCase):

  def test_smallest_measured_conversion_is_the_floor(self):
    self.assertEqual(lm_hf.estimate(350_000_000), (10.7, 43, True))
    self.assertIn(
        "expect about 11 GB of RAM and under a minute, plus the download",
        lm_hf.estimate_line(751_632_384),
    )

  def test_interpolates_between_measured_conversions(self):
    ram, seconds, measured = lm_hf.estimate(2_031_739_904)
    self.assertEqual(
        (round(ram, 1), round(seconds), measured), (25.1, 94, True)
    )
    self.assertIn(
        "expect about 33 GB of RAM and about 2 minutes",
        lm_hf.estimate_line(3_000_000_000),
    )

  def test_above_measured_conversions(self):
    self.assertEqual(lm_hf.estimate(8_000_000_000), (40.8, 202, False))
    self.assertIn(
        "expect more than 41 GB of RAM and 3 minutes, plus the download"
        " (measured up to 4 B parameters",
        lm_hf.estimate_line(8_000_000_000),
    )

  def test_ram_warning(self):
    self.assertIn(
        "this machine has 16.0 GB of RAM",
        lm_hf.ram_warning(16_000_000_000, 25.1),
    )
    self.assertIsNone(lm_hf.ram_warning(128_000_000_000, 40.8))
    self.assertIsNone(lm_hf.ram_warning(None, 10.7))


if __name__ == "__main__":
  absltest.main()
