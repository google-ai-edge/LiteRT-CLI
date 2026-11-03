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

"""Tests for the DDP benchmark target (no network, no gcloud)."""

import contextlib
import http.client
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import time
from unittest import mock
import urllib.error
import uuid

from absl.testing import absltest
import click
from click import testing
from litert_cli.commands.benchmark import cli as benchmark_cli
from litert_cli.commands.benchmark import ddp
from litert_cli.commands.benchmark import model_caches
from litert_cli.core import constants

_ROOT = constants.LITERT_CLI_ANDROID_ROOT
_UUID = uuid.UUID("abcd1234-0000-0000-0000-000000000000")
_SESSION_NAME = "litert-cli-benchmark-abcd1234"
_BINARY = "gs://litert/binaries/2.2.0/android_arm64/benchmark_model"
_CREATE_BUCKET = ["gcloud", "storage", "buckets", "create"]
# The benchmark flags run_ddp requires; the CLI defaults live in cli.py.
_BENCH_KWARGS = dict(
    num_runs=50,
    warmup_runs=1,
    min_secs=1.0,
    max_secs=150.0,
    warmup_min_secs=0.5,
    input_layer_value_range=None,
    signature_key=None,
    timeout=None,
)
_CREATE_RESPONSE = {
    "name": "projects/p/locations/global/operations/op-1",
    "metadata": {"target": "projects/p/locations/global/sessions/s-1"},
    "done": False,
}
_RUNNING_RESPONSE = {
    "name": "projects/p/locations/global/operations/op-1",
    "done": False,
}
_LOGCAT = (
    "noise line\nI benchmark_litert_model: Inference timings in us: Init: 1\n"
)
_LM_BINARY_DIR = "gs://litert/binaries/latest/android_arm64/litert_lm"
# `gcloud storage ls <dir>/` as printed for the published directory.
_LM_LISTING = (
    f"{_LM_BINARY_DIR}/\n\n{_LM_BINARY_DIR}/:\n{_LM_BINARY_DIR}/\n"
    f"{_LM_BINARY_DIR}/embedding_litert_lm_main\n"
    f"{_LM_BINARY_DIR}/libLiteRtDispatch_Qualcomm.so\n"
    f"{_LM_BINARY_DIR}/libLiteRtOpenClAccelerator.so\n"
    f"{_LM_BINARY_DIR}/libLiteRtTopKOpenClSampler.so\n"
    f"{_LM_BINARY_DIR}/litert_lm_advanced_main\n"
    f"{_LM_BINARY_DIR}/litert_lm_main\n"
    f"{_LM_BINARY_DIR}/qairt_version.txt\n"
)
_LM_LIBS = [
    "libLiteRtDispatch_Qualcomm.so",
    "libLiteRtOpenClAccelerator.so",
    "libLiteRtTopKOpenClSampler.so",
]


def _lm_block(
    ttft: str,
    prefill: str,
    decode: str,
    *,
    pid: str = "8446",
    init: str = "2811.82",
    peak: str | None = None,
) -> str:
  """One BenchmarkInfo block as LiteRT-LM's binary logs it (logcat form).

  With `peak`, the peak memory lines of --report_peak_memory_footprint follow
  the block, as the binary logs them after each iteration.
  """
  prefix = f"09-19 11:22:12.427  {pid}  {pid} I native  : "
  block = (
      f"{prefix}I0000 00:00:1789784532.427623    {pid} litert_lm_lib.cc:424]"
      " BenchmarkInfo:\n"
      f"{prefix}    - Init Total: {init} ms\n"
      f"{prefix}  Time to first token: {ttft} s\n"
      f"{prefix}      Prefill Speed: {prefill} tokens/sec.\n"
      f"{prefix}      Decode Speed: {decode} tokens/sec.\n"
  )
  if peak is not None:
    block += (
        f"{prefix}I0000 00:00:1789784533.000000    {pid} litert_lm_lib.cc:474]"
        f" Peak system ram usage: {peak}MB.\n"
        f"{prefix}I0000 00:00:1789784533.000000    {pid} litert_lm_lib.cc:475]"
        " Memory usage: max resident set size/physical footprint = 1.00 MB\n"
        f"{prefix}I0000 00:00:1789784533.000000    {pid} litert_lm_lib.cc:477]"
        f" Peak private footprint: {peak}MB.\n"
    )
  return block


def _lm_aggregated(pid: str, iterations: int) -> str:
  return (
      f"09-19 11:22:13.000  {pid}  {pid} I native  : I0000 00:00:1789784533.0"
      f"    {pid} litert_lm_lib.cc:537] Aggregated BenchmarkInfo (median of"
      f" {iterations} iterations):\n"
      f"09-19 11:22:13.000  {pid}  {pid} I native  :       Prefill Speed:"
      " 999.00 tokens/sec.\n"
      f"09-19 11:22:13.000  {pid}  {pid} I native  :       Decode Speed:"
      " 999.00 tokens/sec.\n"
  )


_LM_LOGCAT = (
    "09-19 11:22:08.986  8446  8446 I litert  : [gpu_registry.cc:135]"
    " Dynamically loaded GPU accelerator(libLiteRtOpenClAccelerator.so)"
    " registered.\n"
    "noise line\n"
    + _lm_block("0.16", "491.23", "32.09")
    + _lm_block("0.14", "516.38", "65.98")
    + _lm_block("0.12", "500.00", "60.00")
    + _lm_aggregated("8446", 3)
)
# The run script's two processes: the warm-up process (one iteration, the cold
# Init) and the measured one, both with the peak memory lines.
_LM_LOGCAT_TWO_PROCESSES = (
    "09-19 11:22:08.986  8400  8400 I litert  : [gpu_registry.cc:135]"
    " Dynamically loaded GPU accelerator(libLiteRtOpenClAccelerator.so)"
    " registered.\n"
    + _lm_block("0.20", "480.00", "30.00", pid="8400", peak="1643.35")
    + _lm_aggregated("8400", 1)
    + _lm_block("0.16", "491.23", "32.09", init="900.50", peak="1500.10")
    + _lm_block("0.14", "516.38", "65.98", init="900.50", peak="1568.53")
    + _lm_block("0.12", "500.00", "60.00", init="900.50", peak="1568.74")
    + _lm_aggregated("8446", 3)
)


def _job_report(name: str, result: str = "PASSED") -> dict:
  prefix = f"gs://p-devicerun/litert-cli/sessions/s-1/{name}/e-1"
  return {
      "displayName": name,
      "result": {"resultType": result},
      "executionReports": [{
          "outputFiles": [
              {"gcsOutputFile": {"path": f"{prefix}/{f}"}}
              for f in (
                  "artifacts/data/local/tmp/results.pb",
                  "artifacts/data/local/tmp/runtime_info.pb",
                  "logcat.txt",
              )
          ]
      }],
  }


def _done_response(
    session_result: str = "PASSED", jobs: tuple[dict, ...] = ()
) -> dict:
  return {
      "name": "projects/p/locations/global/operations/op-1",
      "done": True,
      "response": {
          "sessionReport": {
              "result": {"resultType": session_result},
              "jobReports": list(jobs) or [_job_report("cpu-caiman-35")],
          }
      },
  }


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
  return urllib.error.HTTPError(
      "https://devicerun.googleapis.com/x", code, "err", None, io.BytesIO(body)
  )


class _FakeCloud:
  """Fakes gcloud and the Device Run API for one run.

  Attributes:
    commands: every gcloud command run, in order.
    requests: every urllib request, in order (the POST first).
    events: "post", "poll" and "sleep" in the order they happened.
  """

  def __init__(
      self,
      poll_results=(),
      *,
      create_response=None,
      ls_stderr=None,
      create_stderr=None,
      upload_fails=False,
      upload_fails_for=(),
      cp_fails_for=(),
      tokens=("token-1",),
      lm_listing=_LM_LISTING,
      lm_ls_stderr=None,
      logcat=_LOGCAT,
      record=None,
  ):
    self.poll_results = list(poll_results)
    self.create_response = create_response or _CREATE_RESPONSE
    self.ls_stderr = ls_stderr
    self.create_stderr = create_stderr
    self.upload_fails = upload_fails
    self.upload_fails_for = upload_fails_for
    self.cp_fails_for = cp_fails_for
    self.tokens = list(tokens)
    self.lm_listing = lm_listing
    self.lm_ls_stderr = lm_ls_stderr
    self.logcat = logcat
    self.record = record
    self.commands = []
    self.requests = []
    self.events = []
    self.sleeps = []

  def run(self, cmd, **kwargs):
    self.commands.append(cmd)
    if cmd[:3] == ["gcloud", "storage", "ls"] and cmd[3].startswith(
        "gs://litert/"
    ):
      # The listing of the published LiteRT-LM binaries.
      if self.lm_ls_stderr:
        return subprocess.CompletedProcess(cmd, 1, "", self.lm_ls_stderr)
      return subprocess.CompletedProcess(cmd, 0, self.lm_listing, "")
    if cmd[:3] == ["gcloud", "storage", "ls"] and self.ls_stderr:
      return subprocess.CompletedProcess(cmd, 1, "", self.ls_stderr)
    if cmd[:4] == _CREATE_BUCKET and self.create_stderr:
      return subprocess.CompletedProcess(cmd, 1, "", self.create_stderr)
    is_cp = cmd[:3] == ["gcloud", "storage", "cp"]
    is_upload = is_cp and not cmd[3].startswith("gs://")
    if is_upload and (
        self.upload_fails or os.path.basename(cmd[3]) in self.upload_fails_for
    ):
      assert kwargs.get("check"), "the upload relies on check=True"
      raise subprocess.CalledProcessError(1, cmd)
    if is_upload:
      # The file is read here, as gcloud would; a run script is checked by
      # the LM tests.
      self.uploads = getattr(self, "uploads", {})
      self.uploads[os.path.basename(cmd[3])] = pathlib.Path(cmd[3]).read_text()
    if is_cp and cmd[3].startswith("gs://"):
      local_dir = pathlib.Path(cmd[-1])
      if any(name in local_dir.parts for name in self.cp_fails_for):
        return subprocess.CompletedProcess(
            cmd, 1, "", "ERROR: (gcloud.storage.cp) permission denied\n"
        )
      (local_dir / "logcat.txt").write_text(self.logcat)
      if self.record is not None:
        (local_dir / "benchmark_run.txt").write_text(self.record)
    return subprocess.CompletedProcess(cmd, 0, "", "")

  def check_output(self, cmd, **kwargs):
    del kwargs
    self.commands.append(cmd)
    if not self.tokens:
      raise subprocess.CalledProcessError(
          1, cmd, stderr="ERROR: Your default credentials were not found.\n"
      )
    return self.tokens.pop(0) + "\n"

  def urlopen(self, req, **kwargs):
    assert kwargs.get("timeout"), "every API request carries a timeout"
    self.requests.append(req)
    if req.data:
      self.events.append("post")
      payload = self.create_response
      if isinstance(payload, Exception):
        raise payload
    else:
      self.events.append("poll")
      payload = self.poll_results.pop(0)
      if isinstance(payload, Exception):
        raise payload
    response = mock.MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    return response

  def sleep(self, secs):
    self.events.append("sleep")
    self.sleeps.append(secs)


@contextlib.contextmanager
def _patched(fake: _FakeCloud, cache_dir: str):
  with (
      mock.patch.object(ddp.subprocess, "run", side_effect=fake.run),
      mock.patch.object(
          ddp.subprocess, "check_output", side_effect=fake.check_output
      ),
      mock.patch.object(
          ddp.urllib.request, "urlopen", side_effect=fake.urlopen
      ),
      mock.patch.object(ddp.time, "sleep", side_effect=fake.sleep),
      mock.patch.object(ddp.uuid, "uuid4", return_value=_UUID),
      mock.patch.object(ddp, "_GCP_BUCKET", None),
      mock.patch.object(constants, "LITERT_CLI_CACHE_DIR", cache_dir),
      # Quiet mode redirects the process's stderr; keep the flag on for the
      # logcat filter but skip the redirect.
      mock.patch.object(constants, "DEFAULT_QUIET", True),
      mock.patch.object(benchmark_cli.utils, "enable_quiet_mode"),
  ):
    yield


class DdpHelpersTest(absltest.TestCase):

  def test_normalize_devices_splits_and_dedupes(self):
    self.assertEqual(
        ddp._normalize_devices(("caiman-35, pa3q-35", "caiman-35")),
        ["caiman-35", "pa3q-35"],
    )
    self.assertEqual(ddp._normalize_devices("caiman-35"), ["caiman-35"])
    self.assertEqual(ddp._normalize_devices(("",)), [])

  def test_display_name_matches_api_rules(self):
    self.assertEqual(ddp._display_name("cpu", "caiman-35"), "cpu-caiman-35")
    self.assertEqual(ddp._display_name("gpu", "a.b/c"), "gpu-a-b-c")
    self.assertLen(ddp._display_name("cpu", "x" * 100), 63)

  def test_url_helpers(self):
    endpoint = "https://devicerun.googleapis.com/v1alpha"
    self.assertEqual(
        ddp._get_sessions_url(endpoint, "my-project", "global"),
        f"{endpoint}/projects/my-project/locations/global/sessions",
    )
    self.assertEqual(
        ddp._get_operation_url(endpoint, "projects/p/operations/op-1"),
        f"{endpoint}/projects/p/operations/op-1",
    )
    self.assertEqual(
        ddp._get_console_url("my-bucket", "litert-cli/sessions", "session-1"),
        "https://console.cloud.google.com/storage/browser/my-bucket/"
        "litert-cli/sessions/session-1",
    )

  def test_benchmark_binary_follows_the_environment_variable(self):
    with mock.patch.dict(os.environ):
      os.environ.pop("DDP_LITERT_VERSION", None)
      self.assertEqual(ddp._benchmark_binary(), _BINARY)
      os.environ["DDP_LITERT_VERSION"] = "nightly"
      self.assertEqual(
          ddp._benchmark_binary(),
          "gs://litert/binaries/nightly/android_arm64/benchmark_model",
      )

  def test_bucket_is_missing_on_measured_gcloud_lines(self):
    self.assertTrue(
        ddp._bucket_is_missing(
            "ERROR: (gcloud.storage.ls) gs://no-such-bucket not found: 404.\n"
        )
    )
    self.assertFalse(
        ddp._bucket_is_missing(
            "ERROR: (gcloud.storage.ls) [me] does not have permission to"
            " access b instance [test] (or it may not exist): Permission"
            " 'storage.objects.list' denied on resource"
            " '//storage.googleapis.com/projects/_/buckets/test'.\n"
        )
    )
    self.assertFalse(ddp._bucket_is_missing("ERROR: bucket [x404y] denied"))

  def test_stderr_tail_keeps_the_last_non_empty_lines(self):
    self.assertEqual(ddp._stderr_tail(None), "")
    self.assertEqual(ddp._stderr_tail("a\n\nb\n  c  \nd\n\n"), "b\nc\nd")

  def test_build_benchmark_args_always_emits_the_numeric_flags(self):
    args = ddp._build_benchmark_args(
        model_name="m.tflite",
        accelerator="cpu",
        num_runs=50,
        warmup_runs=1,
        min_secs=1.0,
        max_secs=150.0,
        warmup_min_secs=0.5,
        input_layer_value_range=None,
        signature_key=None,
    )
    self.assertEqual(
        args,
        [
            f"--graph={_ROOT}/m.tflite",
            "--num_runs=50",
            "--warmup_runs=1",
            "--min_secs=1.0",
            "--max_secs=150.0",
            "--warmup_min_secs=0.5",
            f"--result_file_path={_ROOT}/results.pb",
            f"--model_runtime_info_output_file={_ROOT}/runtime_info.pb",
        ],
    )

  def test_build_benchmark_args_gpu_and_optional_flags(self):
    args = ddp._build_benchmark_args(
        model_name="m.tflite",
        accelerator="gpu",
        num_runs=10,
        warmup_runs=1,
        min_secs=1.0,
        max_secs=150.0,
        warmup_min_secs=0.5,
        input_layer_value_range="input1,1.0,2.0",
        signature_key="serving_default",
    )
    self.assertIn("--use_gpu=true", args)
    self.assertIn("--num_runs=10", args)
    self.assertIn("--warmup_runs=1", args)
    self.assertIn("--input_layer_value_range=input1,1.0,2.0", args)
    self.assertIn("--signature_to_run_for=serving_default", args)

  def test_tflite_execution_timeout_follows_max_secs_within_the_range(self):
    self.assertEqual(ddp._tflite_execution_timeout_secs(150.0), 420)
    # Never below the platform's default of 300 s, which a job without a
    # timeout got before.
    self.assertEqual(ddp._tflite_execution_timeout_secs(90.0), 300)
    self.assertEqual(ddp._tflite_execution_timeout_secs(1.0), 300)
    self.assertEqual(ddp._tflite_execution_timeout_secs(10000.0), 3600)

  def test_build_session_request_one_job_per_device(self):
    body = ddp._build_session_request(
        session_name=_SESSION_NAME,
        model_gcs_path=f"gs://b/litert-cli/inputs/{_SESSION_NAME}/m.tflite",
        model_name="m.tflite",
        accelerator="cpu",
        device_list=["caiman-35", "pa3q-35"],
        output_dir="gs://b/litert-cli/sessions",
        benchmark_binary=_BINARY,
        bench_args=["--graph=x"],
    )
    config = body["sessionConfig"]
    self.assertEqual(config["displayName"], _SESSION_NAME)
    self.assertEqual(
        config["outputDirectoryConfig"]["gcsOutputDirectory"]["path"],
        "gs://b/litert-cli/sessions",
    )
    self.assertLen(config["jobConfigs"], 2)
    job = config["jobConfigs"][1]
    self.assertEqual(job["displayName"], "cpu-pa3q-35")
    binary = job["action"]["androidNativeBinary"]
    self.assertEqual(
        binary["androidNativeBinary"]["gcsInputFile"]["path"], _BINARY
    )
    self.assertEqual(binary["args"], ["--graph=x"])
    device_config = job["allocationConfig"]["deviceConfigs"][0]
    self.assertEqual(device_config["requirement"]["deviceId"], "pa3q-35")
    push, pull, logcat = device_config["actions"]
    file_config = push["androidPushFiles"]["fileConfigs"][0]
    self.assertEqual(
        file_config["sourceFile"]["gcsInputFile"]["path"],
        f"gs://b/litert-cli/inputs/{_SESSION_NAME}/m.tflite",
    )
    self.assertEqual(file_config["destinationPath"], f"{_ROOT}/m.tflite")
    self.assertEqual(
        pull["androidPullFiles"]["paths"],
        [f"{_ROOT}/results.pb", f"{_ROOT}/runtime_info.pb"],
    )
    self.assertEqual(logcat, {"androidLogcat": {}})
    self.assertEqual(
        job["labels"],
        {"tool": "litert-cli", "accelerator": "cpu", "model": "m.tflite"},
    )


class RunDdpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # tempfile rather than absltest's create_tempdir(): the latter reads an
    # absl flag, which is not parsed when the file runs under pytest.
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
    self.model = pathlib.Path(self.tmp_dir) / "m.tflite"
    self.model.write_bytes(b"\0")

  def _invoke(self, fake: _FakeCloud, *extra_args: str, model=None, env=None):
    """Runs the benchmark command through click and returns the result.

    The model caches are on unless `env` sets the variable.
    """
    args = [
        str(model or self.model),
        "--ddp",
        "--device",
        "caiman-35",
        "--gcp-project",
        "p",
        *extra_args,
    ]
    with _patched(fake, self.tmp_dir), mock.patch.dict(os.environ):
      os.environ.pop(constants.ENV_LITERT_DISABLE_MODEL_CACHES, None)
      os.environ.update(env or {})
      return testing.CliRunner().invoke(benchmark_cli.benchmark_cmd, args)

  def _push_path(self, fake: _FakeCloud, job_index: int = 0) -> str:
    body = json.loads(fake.requests[0].data.decode())
    job = body["sessionConfig"]["jobConfigs"][job_index]
    push = job["allocationConfig"]["deviceConfigs"][0]["actions"][0]
    return push["androidPushFiles"]["fileConfigs"][0]["sourceFile"][
        "gcsInputFile"
    ]["path"]

  def test_npu_is_rejected(self):
    with self.assertRaisesRegex(click.ClickException, "NPU on --ddp"):
      ddp.run_ddp(
          "m.tflite", "npu", ["caiman-35"], gcp_project="p", **_BENCH_KWARGS
      )

  def test_missing_device_is_rejected(self):
    with self.assertRaisesRegex(click.ClickException, "--device is required"):
      ddp.run_ddp("m.tflite", "cpu", [""], gcp_project="p", **_BENCH_KWARGS)

  @mock.patch.object(ddp, "_DEFAULT_GCP_PROJECT", None)
  def test_missing_project_is_rejected(self):
    with self.assertRaisesRegex(click.ClickException, "Missing GCP project"):
      ddp.run_ddp("m.tflite", "cpu", ["caiman-35"], **_BENCH_KWARGS)

  def test_cli_uses_the_default_device_for_ddp(self):
    fake = _FakeCloud([_done_response(jobs=(_job_report("cpu-pa3q-35"),))])
    with _patched(fake, self.tmp_dir), mock.patch.dict(os.environ):
      os.environ.pop(constants.ENV_LITERT_DISABLE_MODEL_CACHES, None)
      result = testing.CliRunner().invoke(
          benchmark_cli.benchmark_cmd,
          [str(self.model), "--ddp", "--gcp-project", "p"],
      )
    self.assertEqual(result.exit_code, 0, result.output)
    body = json.loads(fake.requests[0].data.decode())
    job = body["sessionConfig"]["jobConfigs"][0]
    self.assertEqual(job["displayName"], "cpu-pa3q-35")
    self.assertEqual(
        job["allocationConfig"]["deviceConfigs"][0]["requirement"]["deviceId"],
        "pa3q-35",
    )

  def test_cli_rejects_empty_device_for_ddp(self):
    runner = testing.CliRunner()
    with mock.patch.object(constants, "DEFAULT_QUIET", False):
      result = runner.invoke(
          benchmark_cli.benchmark_cmd,
          [str(self.model), "--ddp", "--device", "", "--gcp-project", "p"],
      )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("--device is required", result.output)

  def test_cli_exits_1_when_the_model_is_missing(self):
    fake = _FakeCloud()
    result = self._invoke(fake, model="/nonexistent/m.tflite")
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Error: Local model file not found", result.output)
    self.assertEmpty(fake.commands)

  def test_cli_exits_1_when_the_token_cannot_be_fetched(self):
    fake = _FakeCloud(tokens=())
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("gcloud auth application-default login", result.output)
    self.assertIn("default credentials were not found", result.output)
    self.assertEmpty(fake.requests)

  def test_submits_session_and_fetches_outputs(self):
    fake = _FakeCloud([_RUNNING_RESPONSE, _done_response()])
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 0, result.output)
    # The first poll runs before the first sleep.
    self.assertEqual(fake.events, ["post", "poll", "sleep", "poll"])
    post = fake.requests[0]
    self.assertEqual(
        post.full_url,
        "https://devicerun.googleapis.com/v1alpha/projects/p/locations/"
        "global/sessions",
    )
    self.assertEqual(post.get_header("X-goog-user-project"), "p")
    self.assertEqual(post.get_header("Authorization"), "Bearer token-1")
    body = json.loads(post.data.decode())
    self.assertEqual(
        body["sessionConfig"]["jobConfigs"][0]["displayName"], "cpu-caiman-35"
    )
    self.assertEqual(
        body["sessionConfig"]["outputDirectoryConfig"]["gcsOutputDirectory"][
            "path"
        ],
        "gs://p-devicerun/litert-cli/sessions",
    )
    self.assertEqual(
        fake.requests[1].full_url,
        "https://devicerun.googleapis.com/v1alpha/projects/p/locations/"
        "global/operations/op-1",
    )
    self.assertIn("timeout: 1020 s", result.output)
    self.assertIn("Session 's-1' finished: PASSED", result.output)
    self.assertIn(
        str(pathlib.Path(self.tmp_dir) / "ddp" / "s-1" / "cpu-caiman-35"),
        result.output,
    )
    self.assertIn("benchmark_litert_model", result.output)
    self.assertNotIn("noise line", result.output)
    self.assertNotIn("token-1", result.output)

  def test_the_job_runs_the_script_with_benchmark_model_pushed(self):
    for accelerator in ("cpu", "gpu"):
      fake = _FakeCloud([_done_response()])
      result = self._invoke(fake, f"--{accelerator}")
      self.assertEqual(result.exit_code, 0, result.output)
      self.assertIn(f"Binary: {_BINARY})", result.output)
      inputs_dir = f"gs://p-devicerun/litert-cli/inputs/{_SESSION_NAME}"
      uploads = [
          c[3:]
          for c in fake.commands
          if c[:3] == ["gcloud", "storage", "cp"]
          and not c[3].startswith("gs://")
      ]
      self.assertEqual(
          [os.path.basename(u[0]) for u in uploads],
          ["m.tflite", "benchmark_model_run.sh"],
      )
      body = json.loads(fake.requests[0].data.decode())
      job = body["sessionConfig"]["jobConfigs"][0]
      binary = job["action"]["androidNativeBinary"]
      self.assertEqual(
          binary["androidNativeBinary"]["gcsInputFile"]["path"],
          f"{inputs_dir}/benchmark_model_run.sh",
      )
      cache = f"{_ROOT}/benchmark_cache"
      expected_cache_args = [
          f"--xnnpack_weight_cache_file_path={cache}/m.xnnpack_cache"
      ]
      if accelerator == "gpu":
        expected_cache_args += [
            f"--gpu_serialization_dir={cache}",
            "--gpu_model_cache_key=m",
        ]
      # The job's arguments are the plain ones, as without the caches; the
      # script adds the cache flags.
      self.assertEqual(
          binary["args"][-2:],
          [
              f"--result_file_path={_ROOT}/results.pb",
              f"--model_runtime_info_output_file={_ROOT}/runtime_info.pb",
          ],
      )
      self.assertFalse(any("cache" in arg for arg in binary["args"]))
      self.assertEqual(binary["executionTimeout"], "420s")
      self.assertEqual(job["labels"]["model_caches"], "on")
      actions = job["allocationConfig"]["deviceConfigs"][0]["actions"]
      pushes = [
          (f["sourceFile"]["gcsInputFile"]["path"], f["destinationPath"])
          for f in actions[0]["androidPushFiles"]["fileConfigs"]
      ]
      self.assertEqual(
          pushes,
          [
              (f"{inputs_dir}/m.tflite", f"{_ROOT}/m.tflite"),
              (_BINARY, f"{_ROOT}/benchmark_model"),
          ],
      )
      self.assertEqual(
          actions[1]["androidPullFiles"]["paths"],
          [
              f"{_ROOT}/results.pb",
              f"{_ROOT}/runtime_info.pb",
              f"{_ROOT}/warmup_results.pb",
              f"{_ROOT}/benchmark_run.txt",
          ],
      )
      # The script carries the first process's arguments, the flags the
      # measured process gets on top of the job's, and the run without them.
      script = fake.uploads["benchmark_model_run.sh"]
      self.assertTrue(script.startswith("#!/system/bin/sh\n"))
      self.assertIn(
          "run first "
          + " ".join(
              ddp._tflite_warmup_args(binary["args"], expected_cache_args)
          )
          + "; then\n",
          script,
      )
      self.assertIn(
          'run measured "$@" '
          + " ".join(expected_cache_args)
          + " --report_peak_memory_footprint=true; then\n",
          script,
      )
      self.assertIn('run fallback "$@" || stop', script)
      self.assertIn(f'ROOT="{_ROOT}"', script)

  def test_the_report_follows_the_accelerator(self):
    logcat = _tflite_process("4000", "745.16") + _tflite_process(
        "4100", "307.03", (_TFLITE_LOADED,)
    )
    fake = _FakeCloud(
        [_done_response(jobs=(_job_report("gpu-caiman-35"),))],
        logcat=logcat,
        record=_TFLITE_RECORD,
    )
    result = self._invoke(fake, "--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn(
        "First process (compiles the model and writes the caches, no"
        " inference): init 745.16 ms",
        result.output,
    )
    self.assertIn(
        "The measured process logged: Initialized InferenceContext from"
        " serialized data",
        result.output,
    )

  def test_upload_is_namespaced_by_the_session_name(self):
    fake = _FakeCloud([_done_response()])
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 0, result.output)
    inputs_dir = f"gs://p-devicerun/litert-cli/inputs/{_SESSION_NAME}"
    upload = [c for c in fake.commands if c[:3] == ["gcloud", "storage", "cp"]]
    self.assertEqual(upload[0][3:], [str(self.model), f"{inputs_dir}/"])
    self.assertEqual(self._push_path(fake), f"{inputs_dir}/m.tflite")

  def test_gs_model_is_used_without_upload(self):
    fake = _FakeCloud([_done_response()])
    with (
        _patched(fake, self.tmp_dir),
        mock.patch.dict(os.environ),
        contextlib.redirect_stdout(io.StringIO()),
    ):
      os.environ.pop(constants.ENV_LITERT_DISABLE_MODEL_CACHES, None)
      ddp.run_ddp(
          "gs://b/dir/m.tflite",
          "gpu",
          ["caiman-35"],
          gcp_project="p",
          **_BENCH_KWARGS,
      )
    uploads = [
        c
        for c in fake.commands
        if c[:3] == ["gcloud", "storage", "cp"] and not c[3].startswith("gs://")
    ]
    # Only the run script is uploaded; the model is pushed from its gs:// path.
    self.assertEqual(
        [os.path.basename(c[3]) for c in uploads], ["benchmark_model_run.sh"]
    )
    self.assertEqual(self._push_path(fake), "gs://b/dir/m.tflite")
    body = json.loads(fake.requests[0].data.decode())
    job = body["sessionConfig"]["jobConfigs"][0]
    self.assertEqual(
        job["action"]["androidNativeBinary"]["args"][:2],
        [f"--graph={_ROOT}/m.tflite", "--use_gpu=true"],
    )

  def test_timeout_option_and_default(self):
    fake = _FakeCloud([_done_response()])
    result = self._invoke(fake, "--timeout", "5")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("timeout: 5 s", result.output)
    fake = _FakeCloud([_done_response()])
    result = self._invoke(
        fake, "--devices", "caiman-35, pa3q-35", "--max-secs", "10"
    )
    self.assertEqual(result.exit_code, 0, result.output)
    # Two jobs of max(300, 2 x 10 + 120) s, and the slack.
    self.assertIn("timeout: 1200 s", result.output)
    self.assertEqual(
        json.loads(fake.requests[0].data.decode())["sessionConfig"][
            "jobConfigs"
        ][0]["action"]["androidNativeBinary"]["executionTimeout"],
        "300s",
    )

  def test_the_environment_variable_runs_benchmark_model_itself_as_before(
      self,
  ):
    fake = _FakeCloud([_done_response()])
    result = self._invoke(
        fake, "--gpu", env={constants.ENV_LITERT_DISABLE_MODEL_CACHES: "1"}
    )
    self.assertEqual(result.exit_code, 0, result.output)
    uploads = [
        os.path.basename(c[3])
        for c in fake.commands
        if c[:3] == ["gcloud", "storage", "cp"] and not c[3].startswith("gs://")
    ]
    self.assertEqual(uploads, ["m.tflite"])
    body = json.loads(fake.requests[0].data.decode())
    job = body["sessionConfig"]["jobConfigs"][0]
    binary = job["action"]["androidNativeBinary"]
    self.assertEqual(
        binary["androidNativeBinary"]["gcsInputFile"]["path"], _BINARY
    )
    self.assertEqual(
        binary["args"][:2], [f"--graph={_ROOT}/m.tflite", "--use_gpu=true"]
    )
    self.assertFalse(any("cache" in arg for arg in binary["args"]))
    self.assertNotIn("executionTimeout", binary)
    self.assertEqual(job["labels"]["model_caches"], "off")
    actions = job["allocationConfig"]["deviceConfigs"][0]["actions"]
    self.assertLen(actions[0]["androidPushFiles"]["fileConfigs"], 1)
    self.assertEqual(
        actions[1]["androidPullFiles"]["paths"],
        [f"{_ROOT}/results.pb", f"{_ROOT}/runtime_info.pb"],
    )
    self.assertIn("timeout: 750 s", result.output)
    self.assertIn(
        "Model caches: off (LITERT_DISABLE_MODEL_CACHES=1): benchmark_model"
        " ran once without the cache flags",
        result.output,
    )

  def test_missing_bucket_is_created(self):
    fake = _FakeCloud(
        [_done_response()],
        ls_stderr=(
            "ERROR: (gcloud.storage.ls) gs://p-devicerun not found: 404.\n"
        ),
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 0, result.output)
    creates = [c for c in fake.commands if c[:4] == _CREATE_BUCKET]
    self.assertLen(creates, 1)
    self.assertIn("gs://p-devicerun", creates[0])
    self.assertIn("--project=p", creates[0])

  def test_bucket_permission_error_exits_1_without_creating(self):
    fake = _FakeCloud(
        [_done_response()],
        ls_stderr=(
            "ERROR: (gcloud.storage.ls) [me@example.com] does not have"
            " permission to access b instance [p-devicerun] (or it may not"
            " exist): Permission 'storage.objects.list' denied.\n"
        ),
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Could not list GCS bucket 'gs://p-devicerun'", result.output)
    self.assertIn("does not have permission", result.output)
    self.assertFalse([c for c in fake.commands if c[:4] == _CREATE_BUCKET])
    self.assertEmpty(fake.requests)

  def test_bucket_create_failure_exits_1(self):
    fake = _FakeCloud(
        [_done_response()],
        ls_stderr=(
            "ERROR: (gcloud.storage.ls) gs://p-devicerun not found: 404.\n"
        ),
        create_stderr=(
            "ERROR: (gcloud.storage.buckets.create) HTTPError 409: The"
            " requested bucket name is not available.\n"
        ),
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn(
        "Failed to create GCS bucket 'gs://p-devicerun'", result.output
    )
    self.assertIn("HTTPError 409", result.output)
    self.assertEmpty(fake.requests)

  def test_upload_failure_exits_1(self):
    fake = _FakeCloud([_done_response()], upload_fails=True)
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Error: Failed to upload", result.output)
    self.assertIn("exited with 1", result.output)
    self.assertEmpty(fake.requests)

  def test_submit_connection_error_exits_1(self):
    fake = _FakeCloud(create_response=TimeoutError("timed out"))
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Error: Failed to submit benchmark: timed out", result.output)

  def test_poll_exits_1_after_consecutive_errors(self):
    errors = [
        _http_error(500),
        urllib.error.URLError("connection dropped"),
        TimeoutError("timed out"),
        http.client.RemoteDisconnected("remote end closed"),
        ConnectionResetError("reset by peer"),
    ]
    self.assertLen(errors, ddp._MAX_POLL_FAILURES)
    fake = _FakeCloud(errors)
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertLen(fake.requests, 1 + ddp._MAX_POLL_FAILURES)
    self.assertIn("Error polling operation: HTTP Error 500", result.output)
    for text in ("connection dropped", "timed out", "remote end closed"):
      self.assertIn(text, result.output)
    self.assertIn("reset by peer", result.output)
    self.assertIn(
        f"Polling failed {ddp._MAX_POLL_FAILURES} times in a row", result.output
    )
    self.assertIn("console.cloud.google.com/storage/browser", result.output)

  def test_poll_errors_that_are_not_consecutive_do_not_abort(self):
    polls = []
    for _ in range(ddp._MAX_POLL_FAILURES):
      polls += [_http_error(500), _RUNNING_RESPONSE]
    fake = _FakeCloud(polls + [_done_response()])
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(fake.requests, 2 + 2 * ddp._MAX_POLL_FAILURES)

  def test_poll_exits_1_when_the_token_cannot_be_refreshed(self):
    fake = _FakeCloud([_http_error(401)], tokens=("token-1",))
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Refreshing the access token", result.output)
    self.assertIn("gcloud auth application-default login", result.output)
    self.assertIn("console.cloud.google.com/storage/browser", result.output)

  def test_poll_refreshes_the_token_on_401(self):
    fake = _FakeCloud(
        [_http_error(401), _done_response()], tokens=("token-1", "token-2")
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Refreshing the access token", result.output)
    self.assertEqual(
        fake.requests[1].get_header("Authorization"), "Bearer token-1"
    )
    self.assertEqual(
        fake.requests[2].get_header("Authorization"), "Bearer token-2"
    )
    self.assertEmpty(fake.tokens)

  def test_rejected_submit_exits_1_with_the_device_hint(self):
    fake = _FakeCloud(
        create_response=_http_error(
            400, b'{"error": {"message": "Device no-such-device not found"}}'
        )
    )
    result = self._invoke(fake, "--device", "no-such-device")
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Error: Failed to submit benchmark: 400", result.output)
    self.assertIn("Device no-such-device not found", result.output)
    self.assertIn("gcloud beta device-run devices list", result.output)
    self.assertEqual(fake.events, ["post"])

  def test_operation_error_exits_1(self):
    fake = _FakeCloud([{
        "name": "op-1",
        "done": True,
        "error": {"code": 9, "message": "no device available"},
    }])
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Benchmark failed", result.output)
    self.assertIn("no device available", result.output)

  def test_failed_job_exits_1_after_printing_its_logcat_tail(self):
    fake = _FakeCloud(
        [_done_response("FAILED", (_job_report("cpu-caiman-35", "FAILED"),))]
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Session 's-1' finished: FAILED", result.output)
    self.assertIn("Last 20 lines of logcat.txt", result.output)
    self.assertIn("Error: Job 'cpu-caiman-35': FAILED", result.output)
    self.assertIn(
        "Session outputs on the Cloud Console: https://console.cloud.google"
        ".com/storage/browser/p-devicerun/litert-cli/sessions/s-1",
        result.output,
    )

  def test_download_failure_exits_1_after_the_other_job_is_printed(self):
    fake = _FakeCloud(
        [
            _done_response(
                "PASSED",
                (_job_report("cpu-caiman-35"), _job_report("cpu-pa3q-35")),
            )
        ],
        cp_fails_for=("cpu-caiman-35",),
    )
    result = self._invoke(fake, "--devices", "caiman-35, pa3q-35")
    self.assertEqual(result.exit_code, 1)
    self.assertIn(
        "Failed to download the output files of job 'cpu-caiman-35'",
        result.output,
    )
    self.assertIn("permission denied", result.output)
    self.assertIn(
        str(pathlib.Path(self.tmp_dir) / "ddp" / "s-1" / "cpu-pa3q-35"),
        result.output,
    )
    self.assertIn("benchmark_litert_model", result.output)
    self.assertIn(
        "Error: Job 'cpu-caiman-35': output files not downloaded",
        result.output,
    )

  def test_session_that_did_not_pass_exits_1_even_if_its_jobs_passed(self):
    fake = _FakeCloud([_done_response("ERROR")])
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Job 'cpu-caiman-35': PASSED", result.output)
    self.assertIn("Error: Session 's-1' finished: ERROR", result.output)

  def test_job_without_output_files_exits_1(self):
    job = {"displayName": "cpu-caiman-35", "result": {"resultType": "PASSED"}}
    fake = _FakeCloud([_done_response("PASSED", (job,))])
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("No output files were reported for this job.", result.output)
    self.assertIn(
        "Error: Job 'cpu-caiman-35': no output files reported", result.output
    )

  def test_timeout_exits_1_while_the_session_keeps_running(self):
    fake = _FakeCloud([_RUNNING_RESPONSE, _RUNNING_RESPONSE])
    clock = mock.patch.object(
        ddp.time, "monotonic", side_effect=[0.0, 20.0, 40.0]
    )
    with clock:
      result = self._invoke(fake, "--timeout", "30")
    self.assertEqual(result.exit_code, 1)
    self.assertEqual(fake.events, ["post", "poll", "sleep", "poll"])
    # The last sleep is cut to the remaining 10 s of the 30 s budget.
    self.assertEqual(fake.sleeps, [10.0])
    self.assertIn("did not finish within 30 s", result.output)
    self.assertIn("console.cloud.google.com/storage/browser", result.output)

  def test_session_id_falls_back_to_the_display_name(self):
    fake = _FakeCloud(
        [_done_response()],
        create_response={
            "name": "projects/p/locations/global/operations/op-1",
            "done": False,
        },
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("did not name the session", result.output)
    self.assertIn(f"Waiting for session '{_SESSION_NAME}'", result.output)
    self.assertIn(
        "https://console.cloud.google.com/storage/browser/p-devicerun/"
        "litert-cli/sessions/\n",
        result.output,
    )
    self.assertIn(
        str(
            pathlib.Path(self.tmp_dir) / "ddp" / _SESSION_NAME / "cpu-caiman-35"
        ),
        result.output,
    )

  def test_response_without_an_operation_exits_1(self):
    fake = _FakeCloud(create_response={"name": "my-operations-run"})
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("returned no operation", result.output)
    self.assertEqual(fake.events, ["post"])


# Logs its arguments to calls.txt, one call per "--" line; exits 7 on call N
# when fail<N> exists, and sleeps on call N when slow<N> exists; writes the
# cache files its flags name on the first call, and on the second too when
# rewrite exists; writes its result file. Like benchmark_model, it keeps the
# first copy of a repeated flag.
# Logs its arguments to calls.txt, one call per "--" line. Exits 7 on call N
# when fail<N> exists (leaving a partial result file, as a crash might, unless
# silent<N> exists too), sleeps on call N when slow<N> exists, and exits 1 on a
# call with --dry_run=true when reject exists. Writes the cache files its flags
# name on the first call, and on the second too when rewrite exists, and writes
# the call's number into its result file. Like benchmark_model, it keeps the
# first copy of a repeated flag.
_FAKE_BENCHMARK_MODEL = """#!/bin/sh
dir=$(dirname "$0")
printf '%s\\n' "$@" -- >> "$dir/calls.txt"
n=$(grep -c -- '^--$' "$dir/calls.txt")
for arg in "$@"; do
  case "$arg" in
    --dry_run=true) [ -f "$dir/reject" ] && exit 1 ;;
    --xnnpack_weight_cache_file_path=*) [ -n "$x" ] || x="${arg#*=}" ;;
    --gpu_serialization_dir=*) [ -n "$g" ] || g="${arg#*=}" ;;
    --gpu_model_cache_key=*) [ -n "$k" ] || k="${arg#*=}" ;;
    --result_file_path=*) [ -n "$r" ] || r="${arg#*=}" ;;
  esac
done
if [ -f "$dir/fail$n" ]; then
  [ -n "$r" ] && [ ! -f "$dir/silent$n" ] && printf "partial$n" > "$r"
  exit 7
fi
if [ -f "$dir/slow$n" ]; then echo $$ > "$dir/fake.pid"; exec sleep 30; fi
if [ "$n" = 1 ] || [ -f "$dir/rewrite" ]; then
  [ -n "$x" ] && printf x > "$x"
  [ -n "$g" ] && printf gg > "$g/${k}_mldrift_program_cache.bin"
fi
[ -n "$r" ] && printf "r$n" > "$r"
exit 0
"""


def _tflite_line(pid: str, tag: str, message: str) -> str:
  return f"09-28 22:30:00.100  {pid}  {pid} I {tag}  : {message}\n"


def _tflite_process(pid: str, init: str, extra: tuple[str, ...] = ()) -> str:
  """One benchmark_model process in a logcat, as the GPU run logs it."""
  return (
      _tflite_line(pid, "tflite", "STARTING!")
      + "".join(_tflite_line(pid, "native", line) for line in extra)
      + _tflite_line(
          pid,
          "litert",
          "[benchmark_litert_model.h:94] Model initialization: " + init + " ms",
      )
      + _tflite_line(
          pid,
          "litert",
          "[benchmark_litert_model.h:132] Peak memory:          427.64 MB",
      )
  )


_TFLITE_LOADED = (
    "I0000 00:00:1790597656.477218 82251411 delegate_kernel.cc:866]"
    " Initialized InferenceContext from serialized data."
)
_TFLITE_RECORD = (
    "f3729eaf  ./benchmark_model\n"
    "first process: pid 4000\n"
    "first process: exit 0\n"
    "caches the first process wrote:\n"
    "     160 /data/local/tmp/litert-cli/benchmark_cache/m.xnnpack_cache\n"
    "57801904 /data/local/tmp/litert-cli/benchmark_cache/"
    "m_mldrift_program_cache.bin\n"
    "57802064 total\n"
    "measured process: pid 4100\n"
    "measured process: exit 0\n"
    "caches the measured process wrote again:\n"
)


class TfliteRunScriptTest(absltest.TestCase):
  """The .tflite run script on a POSIX sh with a fake benchmark_model.

  The device-only command of the script (log) fails quietly here; the control
  flow under test is the same.
  """

  def setUp(self):
    super().setUp()
    if shutil.which("sh") is None:
      self.skipTest("needs a POSIX shell")
    self.root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.fake = self.root / "benchmark_model"
    self.fake.write_text(_FAKE_BENCHMARK_MODEL)
    (self.root / "m.tflite").write_bytes(b"\0")
    cache = self.root / "benchmark_cache"
    root = mock.patch.object(
        constants, "LITERT_CLI_ANDROID_ROOT", str(self.root)
    )
    with root:
      self.plain = ddp._build_benchmark_args(
          model_name="m.tflite",
          accelerator="gpu",
          num_runs=50,
          warmup_runs=1,
          min_secs=1.0,
          max_secs=150.0,
          warmup_min_secs=0.5,
          input_layer_value_range=None,
          signature_key=None,
      )
      cache_flags = model_caches.cache_args("gpu", str(cache), "m.tflite")
      self.warmup = ddp._tflite_warmup_args(self.plain, cache_flags)
      self.measured = self.plain + cache_flags + [model_caches.PEAK_MEMORY_ARG]
      # The job's arguments are the plain ones; the script adds the rest.
      script, self.args = ddp._tflite_job("m.tflite", "gpu", self.plain)
      self.assertEqual(self.args, self.plain)
      self.script = self.root / "benchmark_model_run.sh"
      self.script.write_text(script)

  def _run(self) -> subprocess.CompletedProcess[str]:
    calls = self.root / "calls.txt"
    if calls.exists():
      calls.unlink()
    return subprocess.run(
        ["sh", str(self.script), *self.args],
        capture_output=True,
        text=True,
        check=False,
    )

  def _calls(self) -> list[list[str]]:
    text = (self.root / "calls.txt").read_text()
    return [call.split("\n")[:-1] for call in text.split("--\n")[:-1]]

  def _record(self) -> str:
    return (self.root / "benchmark_run.txt").read_text()

  def test_the_first_process_writes_the_caches_the_measured_one_reads(self):
    # What an earlier job left on the device.
    (self.root / "benchmark_cache").mkdir()
    (self.root / "benchmark_cache" / "stale.bin").write_text("old")
    (self.root / "benchmark_run.txt").write_text("measured process: pid 1\n")
    (self.root / "runtime_info.pb").write_text("stale")
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    first, measured = self._calls()
    self.assertEqual(first, self.warmup)
    self.assertIn("--dry_run=true", first)
    self.assertIn(f"--result_file_path={self.root}/warmup_results.pb", first)
    self.assertNotIn(model_caches.PEAK_MEMORY_ARG, first)
    self.assertFalse(any("runtime_info" in arg for arg in first))
    self.assertEqual(measured, self.measured)
    self.assertEqual((self.root / "warmup_results.pb").read_text(), "r1")
    self.assertEqual((self.root / "results.pb").read_text(), "r2")
    # The earlier run's files are gone; the script removes its cache
    # directory after.
    self.assertFalse((self.root / "runtime_info.pb").exists())
    self.assertFalse((self.root / "benchmark_cache").exists())
    run = ddp._tflite_run(self.root / "benchmark_run.txt")
    self.assertEqual(set(run["pids"]), {"first", "measured"})
    self.assertEqual(
        run["written"], {"m.xnnpack_cache": 1, "m_mldrift_program_cache.bin": 2}
    )
    self.assertEqual(run["rewritten"], [])
    self.assertIsNone(run["fallback"])
    record = self._record()
    self.assertIn("first process: pid", record)
    self.assertIn(
        "first process: command ./benchmark_model " + " ".join(self.warmup),
        record,
    )
    self.assertIn(
        "measured process: command ./benchmark_model "
        + " ".join(self.measured),
        record,
    )
    self.assertNotIn("pid 1\n", record)
    self.assertNotIn("without the cache flags", record)

  def test_caches_the_measured_process_writes_again_are_recorded(self):
    (self.root / "rewrite").touch()
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    run = ddp._tflite_run(self.root / "benchmark_run.txt")
    self.assertEqual(
        sorted(run["rewritten"]),
        ["m.xnnpack_cache", "m_mldrift_program_cache.bin"],
    )

  def test_a_failing_first_process_runs_one_process_without_the_flags(self):
    (self.root / "fail1").touch()
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    first, fallback = self._calls()
    self.assertEqual(first, self.warmup)
    self.assertEqual(fallback, self.plain)
    record = self._record()
    self.assertIn("first process: exit 7\n", record)
    self.assertIn(
        "running once without the cache flags: the first process exited"
        " with 7\n",
        record,
    )
    self.assertIn("fallback process: pid", record)
    self.assertIn("fallback process: exit 0\n", record)
    self.assertNotIn("measured process", record)
    run = ddp._tflite_run(self.root / "benchmark_run.txt")
    self.assertEqual(set(run["pids"]), {"first", "fallback"})
    self.assertEqual(run["fallback"], "the first process exited with 7")
    self.assertEqual((self.root / "results.pb").read_text(), "r2")
    self.assertFalse((self.root / "benchmark_cache").exists())

  def test_a_binary_that_rejects_dry_run_runs_one_process_without_it(self):
    (self.root / "reject").touch()
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(self._calls()[1], self.plain)
    self.assertIn(
        "running once without the cache flags: the first process exited"
        " with 1\n",
        self._record(),
    )

  def test_a_failing_measured_process_runs_one_process_without_the_flags(
      self,
  ):
    (self.root / "fail2").touch()
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    first, measured, fallback = self._calls()
    self.assertEqual(measured, self.measured)
    self.assertEqual(fallback, self.plain)
    self.assertIn("measured process: exit 7\n", self._record())
    self.assertIn(
        "running once without the cache flags: the measured process exited"
        " with 7\n",
        self._record(),
    )
    # The failing measured process's partial result file is gone; the
    # first process's stays.
    self.assertEqual((self.root / "results.pb").read_text(), "r3")
    self.assertEqual((self.root / "warmup_results.pb").read_text(), "r1")
    self.assertFalse((self.root / "benchmark_cache").exists())

  def test_a_failing_process_without_the_flags_is_the_jobs_exit_code(self):
    (self.root / "fail1").touch()
    (self.root / "fail2").touch()
    result = self._run()
    self.assertEqual(result.returncode, 7)
    self.assertLen(self._calls(), 2)
    self.assertIn("fallback process: exit 7\n", self._record())
    self.assertIn("the fallback process exited with 7\n", self._record())
    self.assertFalse((self.root / "benchmark_cache").exists())

  def test_a_failed_measured_process_leaves_no_result_file_behind(self):
    # The measured process fails after writing part of results.pb; the
    # process without the flags then fails before writing anything.
    for marker in ("fail2", "fail3", "silent3"):
      (self.root / marker).touch()
    result = self._run()
    self.assertEqual(result.returncode, 7)
    self.assertLen(self._calls(), 3)
    self.assertFalse((self.root / "results.pb").exists())
    self.assertEqual((self.root / "warmup_results.pb").read_text(), "r1")

  def test_no_binary_exits_1_before_any_process(self):
    self.fake.unlink()
    result = self._run()
    self.assertEqual(result.returncode, 1)
    self.assertFalse((self.root / "calls.txt").exists())
    self.assertIn("chmod ./benchmark_model failed", self._record())

  @absltest.skipIf(os.geteuid() == 0, "root can write a read-only directory")
  def test_a_cache_directory_that_cannot_be_made_runs_one_process(self):
    (self.root / "benchmark_run.txt").touch()
    (self.root / "calls.txt").touch()
    self.root.chmod(0o555)
    self.addCleanup(self.root.chmod, 0o755)
    result = subprocess.run(
        ["sh", str(self.script), *self.args],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(self._calls(), [self.plain])
    self.assertIn(
        "running once without the cache flags: could not create"
        f" {self.root}/benchmark_cache\n",
        self._record(),
    )
    self.assertIn("fallback process: exit 0\n", self._record())

  def test_a_missing_cli_directory_exits_1(self):
    with mock.patch.object(
        constants, "LITERT_CLI_ANDROID_ROOT", str(self.root / "missing")
    ):
      script, _ = ddp._tflite_job("m.tflite", "gpu", self.plain)
      self.script.write_text(script)
    result = self._run()
    self.assertEqual(result.returncode, 1)
    self.assertFalse((self.root / "calls.txt").exists())

  def test_a_stopped_job_stops_its_benchmark_process_too(self):
    (self.root / "slow1").touch()
    pid_file = self.root / "fake.pid"
    for sig, name, code in (
        (signal.SIGTERM, "TERM", 143),
        (signal.SIGINT, "INT", 130),
        (signal.SIGHUP, "HUP", 129),
    ):
      if pid_file.exists():
        pid_file.unlink()
      calls = self.root / "calls.txt"
      if calls.exists():
        calls.unlink()
      script = subprocess.Popen(
          ["sh", str(self.script), *self.args],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
          break
        time.sleep(0.05)
      fake_pid = int(pid_file.read_text())
      script.send_signal(sig)
      self.assertEqual(script.wait(timeout=10), code, name)
      for _ in range(60):
        try:
          os.kill(fake_pid, 0)
        except ProcessLookupError:
          break
        time.sleep(0.05)
      else:
        os.kill(fake_pid, signal.SIGKILL)
        self.fail(f"the first process kept running after {name}")
      self.assertIn(f"first process: pid {fake_pid}\n", self._record())
      self.assertIn(f"stopped by {name}\n", self._record())
      self.assertFalse((self.root / "benchmark_cache").exists())

  def test_a_job_stopped_in_the_measured_process_stops_it_too(self):
    (self.root / "slow2").touch()
    script = subprocess.Popen(
        ["sh", str(self.script), *self.args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_file = self.root / "fake.pid"
    for _ in range(100):
      if pid_file.exists() and pid_file.read_text().strip():
        break
      time.sleep(0.05)
    fake_pid = int(pid_file.read_text())
    script.send_signal(signal.SIGTERM)
    self.assertEqual(script.wait(timeout=10), 143)
    for _ in range(60):
      try:
        os.kill(fake_pid, 0)
      except ProcessLookupError:
        break
      time.sleep(0.05)
    else:
      os.kill(fake_pid, signal.SIGKILL)
      self.fail("the measured process kept running after the job was stopped")
    self.assertIn("first process: exit 0\n", self._record())
    self.assertIn(f"measured process: pid {fake_pid}\n", self._record())
    self.assertIn("stopped by TERM\n", self._record())
    self.assertFalse((self.root / "benchmark_cache").exists())

  def test_a_job_stopped_in_the_process_without_the_flags_stops_it_too(self):
    (self.root / "fail1").touch()
    (self.root / "slow2").touch()
    script = subprocess.Popen(
        ["sh", str(self.script), *self.args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_file = self.root / "fake.pid"
    for _ in range(100):
      if pid_file.exists() and pid_file.read_text().strip():
        break
      time.sleep(0.05)
    fake_pid = int(pid_file.read_text())
    script.send_signal(signal.SIGTERM)
    self.assertEqual(script.wait(timeout=10), 143)
    for _ in range(60):
      try:
        os.kill(fake_pid, 0)
      except ProcessLookupError:
        break
      time.sleep(0.05)
    else:
      os.kill(fake_pid, signal.SIGKILL)
      self.fail("the process without the flags kept running after TERM")
    self.assertIn(f"fallback process: pid {fake_pid}\n", self._record())
    self.assertIn("stopped by TERM\n", self._record())

  def test_a_find_that_fails_is_recorded(self):
    shims = self.root / "shims"
    shims.mkdir()
    (shims / "find").write_text("#!/bin/sh\nexit 1\n")
    (shims / "find").chmod(0o755)
    calls = self.root / "calls.txt"
    result = subprocess.run(
        ["sh", str(self.script), *self.args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{shims}:{os.environ['PATH']}"},
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertTrue(calls.exists())
    self.assertTrue(self._record().endswith("(find failed)\n"))
    run = ddp._tflite_run(self.root / "benchmark_run.txt")
    self.assertIsNone(run["rewritten"])

  def test_a_record_that_cannot_be_written_runs_benchmark_model_once(self):
    (self.root / "benchmark_run.txt").mkdir()
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(self._calls(), [self.plain])
    self.assertEqual((self.root / "results.pb").read_text(), "r1")
    # That process's exit code is the job's.
    (self.root / "fail1").touch()
    result = self._run()
    self.assertEqual(result.returncode, 7)


class TflitePrintTest(absltest.TestCase):
  """The measured process's lines and the report, from the script's record."""

  def setUp(self):
    super().setUp()
    self.dir = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

  def _print(self, logcat: str, record: str | None, passed=True) -> str:
    (self.dir / "logcat.txt").write_text(logcat)
    if record is not None:
      (self.dir / "benchmark_run.txt").write_text(record)
    out = io.StringIO()
    with (
        mock.patch.object(constants, "DEFAULT_QUIET", True),
        contextlib.redirect_stdout(out),
    ):
      ddp._print_logcat_results(
          self.dir / "logcat.txt", passed, accelerator="gpu"
      )
    return out.getvalue()

  def test_prints_the_measured_process_and_what_it_read(self):
    logcat = (
        _tflite_process("3000", "999.00")  # An earlier run's process.
        + _tflite_process("4000", "360.37")
        + _tflite_process("4100", "68.77", (_TFLITE_LOADED,))
    )
    out = self._print(logcat, _TFLITE_RECORD)
    self.assertIn("Model initialization: 68.77 ms", out)
    self.assertNotIn("Model initialization: 360.37 ms", out)
    self.assertNotIn("999.00", out)
    self.assertIn("Model caches: on\n", out)
    self.assertIn(
        "First process (compiles the model and writes the caches, no"
        " inference): init 360.37 ms",
        out,
    )
    self.assertIn(
        "Caches it wrote: m.xnnpack_cache (160 B),"
        " m_mldrift_program_cache.bin (57801904 B)",
        out,
    )
    self.assertIn("The measured process left them unchanged", out)
    self.assertIn(
        "The measured process logged: Initialized InferenceContext from"
        " serialized data",
        out,
    )

  def test_a_cpu_job_as_a_device_logs_it(self):
    # Lines of session-280b5f55 (caiman-35, CPU), and its record in the
    # script's current format.
    logcat = (
        "09-28 06:21:39.135 19410 19410 I tflite  : STARTING!\n"
        "09-28 06:21:39.138 19410 19410 I tflite  : XNNPack weight cache:"
        " written to '/data/local/tmp/litert-cli/benchmark_cache/"
        "convnext_tiny.xnnpack_cache'.\n"
        "09-28 06:21:39.234 19410 19410 I litert  :"
        " [benchmark_litert_model.h:94] Model initialization: 97.45 ms\n"
        "09-28 06:21:39.335 19416 19416 I tflite  : STARTING!\n"
        "09-28 06:21:39.336 19416 19416 I litert  :"
        " [benchmark_litert_model.cc:271] Loading model from:"
        " /data/local/tmp/litert-cli/convnext_tiny.tflite\n"
        "09-28 06:21:39.338 19416 19416 I tflite  : XNNPack weight cache"
        " loaded from '/data/local/tmp/litert-cli/benchmark_cache/"
        "convnext_tiny.xnnpack_cache'.\n"
        "09-28 06:21:49.075 19416 19416 I litert  :"
        " [benchmark_litert_model.h:94] Model initialization: 3.65 ms\n"
        "09-28 06:21:49.076 19416 19416 I litert  :"
        " [benchmark_litert_model.h:132] Peak memory:          166.43 MB\n"
    )
    record = (
        "f3729eaf38738c864a6c859ab169385df7ef1fba4914847dddd2593ffb9e2e44 "
        " ./benchmark_model\n"
        "first process: pid 19410\n"
        "first process: exit 0\n"
        "caches the first process wrote:\n"
        "115592936 /data/local/tmp/litert-cli/benchmark_cache/"
        "convnext_tiny.xnnpack_cache\n"
        "measured process: pid 19416\n"
        "measured process: exit 0\n"
        "caches the measured process wrote again:\n"
    )
    (self.dir / "logcat.txt").write_text(logcat)
    (self.dir / "benchmark_run.txt").write_text(record)
    out = io.StringIO()
    with (
        mock.patch.object(constants, "DEFAULT_QUIET", True),
        contextlib.redirect_stdout(out),
    ):
      ddp._print_logcat_results(
          self.dir / "logcat.txt", True, accelerator="cpu"
      )
    self.assertNotIn("97.45 ms\n09", out.getvalue())
    self.assertEqual(
        out.getvalue().splitlines()[-5:],
        [
            "Model caches: on",
            (
                "First process (compiles the model and writes the caches, no"
                " inference): init 97.45 ms"
            ),
            "Caches it wrote: convnext_tiny.xnnpack_cache (115592936 B)",
            "The measured process left them unchanged",
            (
                "The measured process logged: XNNPack weight cache loaded from"
                " '/data/local/tmp/litert-cli/benchmark_cache/"
                "convnext_tiny.xnnpack_cache'."
            ),
        ],
    )
    self.assertIn("Model initialization: 3.65 ms", out.getvalue())

  def test_a_binary_without_the_gpu_flags_is_named(self):
    ignored = (
        "Unconsumed cmdline flags:"
        " --gpu_serialization_dir=/data/local/tmp/litert-cli/benchmark_cache"
        " --gpu_model_cache_key=m"
    )
    logcat = _tflite_process("4000", "590.73").replace(
        "STARTING!",
        "STARTING!\n" + _tflite_line("4000", "tflite", ignored).rstrip("\n"),
    ) + _tflite_process("4100", "585.12")
    record = _TFLITE_RECORD.replace(
        "57801904 /data/local/tmp/litert-cli/benchmark_cache/"
        "m_mldrift_program_cache.bin\n57802064 total\n",
        "",
    )
    out = self._print(logcat, record)
    self.assertIn("Caches it wrote: m.xnnpack_cache (160 B)\n", out)
    self.assertIn(
        "Flags this benchmark_model ignored (logged as unconsumed):"
        " --gpu_serialization_dir=/data/local/tmp/litert-cli/benchmark_cache"
        " --gpu_model_cache_key=m\n",
        out,
    )
    self.assertIn(
        "This benchmark_model has no --gpu_serialization_dir: neither process"
        " wrote or read a GPU cache",
        out,
    )

  def test_a_cache_written_again_or_not_read_is_named(self):
    logcat = _tflite_process("4000", "360.37") + _tflite_process(
        "4100", "355.10"
    )
    record = _TFLITE_RECORD + (
        "/data/local/tmp/litert-cli/benchmark_cache/m.xnnpack_cache\n"
    )
    out = self._print(logcat, record)
    self.assertIn("The measured process wrote again: m.xnnpack_cache", out)
    self.assertIn(
        "The measured process did not log: Initialized InferenceContext from"
        " serialized data",
        out,
    )

  def test_a_record_that_matches_no_logcat_line_prints_every_line(self):
    out = self._print(_tflite_process("3000", "999.00"), _TFLITE_RECORD)
    self.assertIn("Model initialization: 999.00 ms", out)
    self.assertIn(
        "First process: no 'Model initialization' line in its output", out
    )
    self.assertIn("No output lines of the measured process", out)
    self.assertIn(
        "Could not tell the measured process's lines apart in the logcat;"
        " the benchmark lines above are every process's\n",
        out,
    )

  def test_a_process_without_the_flags_is_printed_with_its_reason(self):
    parse_error = (
        "ERROR: Failed to parse flag 'dry_run' against argv '--dry_run=true'"
    )
    logcat = (
        _tflite_line("4000", "tflite", "STARTING!")
        + _tflite_line("4000", "tflite", parse_error)
        + _tflite_process("4200", "590.73")
    )
    record = (
        "first process: pid 4000\nfirst process: exit 1\n"
        "running once without the cache flags: the first process exited"
        " with 1\n"
        "fallback process: pid 4200\nfallback process: exit 0\n"
    )
    out = self._print(logcat, record)
    self.assertIn("Model initialization: 590.73 ms", out)
    self.assertIn(
        "Model caches: off (the first process exited with 1): benchmark_model"
        " ran once without the cache flags, so the results above are a cold"
        " start with no peak memory\n",
        out,
    )
    self.assertIn(
        "The first process logged: Failed to parse flag 'dry_run' against"
        " argv '--dry_run=true'\n",
        out,
    )
    self.assertNotIn("First process (compiles", out)

  def test_a_failed_rewrite_check_is_named(self):
    logcat = _tflite_process("4000", "360.37") + _tflite_process(
        "4100", "68.77", (_TFLITE_LOADED,)
    )
    out = self._print(logcat, _TFLITE_RECORD + "(find failed)\n")
    self.assertIn(
        "Could not check whether the measured process wrote them again", out
    )
    self.assertNotIn("left them unchanged", out)

  def test_a_process_without_the_flags_that_matches_no_line_is_named(self):
    logcat = _tflite_process("4000", "590.73") + _tflite_process(
        "4100", "68.77"
    )
    record = (
        "first process: pid 4000\nfirst process: exit 0\n"
        "measured process: pid 4100\nmeasured process: exit 139\n"
        "running once without the cache flags: the measured process exited"
        " with 139\n"
        "fallback process: pid 4200\nfallback process: exit 0\n"
    )
    out = self._print(logcat, record)
    lines = out.splitlines()
    self.assertIn("Model initialization: 590.73 ms", out)
    self.assertIn("Model initialization: 68.77 ms", out)
    # The note comes before the report.
    self.assertLess(
        lines.index(
            "Could not tell the fallback process's lines apart in the logcat;"
            " the benchmark lines above are every process's"
        ),
        lines.index(
            "Model caches: off (the measured process exited with 139):"
            " benchmark_model ran once without the cache flags, so the results"
            " above are a cold start with no peak memory"
        ),
    )

  def test_a_fallback_pid_without_a_reason_line_still_counts(self):
    record = (
        "first process: pid 4000\nfirst process: exit 7\n"
        "fallback process: pid 4200\nfallback process: exit 0\n"
    )
    out = self._print(_tflite_process("4200", "590.73"), record)
    self.assertIn("Model initialization: 590.73 ms", out)
    self.assertIn("Model caches: off (reason not recorded)", out)

  def test_verbose_mode_heads_each_process(self):
    logcat = (
        _tflite_process("4000", "360.37")
        + _tflite_process("4100", "68.77")
        + _tflite_process("4200", "590.73")
    )
    record = (
        _TFLITE_RECORD
        + "running once without the cache flags: the measured process exited"
        " with 139\n"
        "fallback process: pid 4200\nfallback process: exit 0\n"
    )
    (self.dir / "logcat.txt").write_text(logcat)
    (self.dir / "benchmark_run.txt").write_text(record)
    out = io.StringIO()
    with (
        mock.patch.object(constants, "DEFAULT_QUIET", False),
        contextlib.redirect_stdout(out),
    ):
      ddp._print_logcat_results(
          self.dir / "logcat.txt", True, accelerator="gpu"
      )
    lines = out.getvalue().splitlines()
    self.assertLess(
        lines.index("First process (writes the caches, no inference):"),
        lines.index("Measured process:"),
    )
    self.assertLess(
        lines.index("Measured process:"),
        lines.index("Process without the cache flags:"),
    )
    self.assertIn("Model initialization: 590.73 ms", out.getvalue())

  def test_without_a_record_every_benchmark_line_is_printed(self):
    out = self._print(_tflite_process("4000", "590.73"), None)
    self.assertIn("Model initialization: 590.73 ms", out)
    self.assertNotIn("First process", out)
    self.assertIn(
        "Model caches: no record (the job pulled no benchmark_run.txt); the"
        " benchmark lines above are every process's\n",
        out,
    )

  def test_with_the_caches_off_the_environment_variable_is_named(self):
    (self.dir / "logcat.txt").write_text(_tflite_process("4000", "590.73"))
    out = io.StringIO()
    with (
        mock.patch.object(constants, "DEFAULT_QUIET", True),
        contextlib.redirect_stdout(out),
    ):
      ddp._print_logcat_results(
          self.dir / "logcat.txt", True, accelerator="cpu", caches=False
      )
    self.assertIn("Model initialization: 590.73 ms", out.getvalue())
    self.assertIn(
        "Model caches: off (LITERT_DISABLE_MODEL_CACHES=1): benchmark_model"
        " ran once without the cache flags\n",
        out.getvalue(),
    )

  def test_a_failed_job_prints_the_record_after_the_logcat_tail(self):
    record = (
        "first process: pid 4000\nfirst process: exit 1\n"
        "the first process exited with 1\n"
    )
    out = self._print(_tflite_process("4000", "590.73"), record, passed=False)
    self.assertIn("Last 20 lines of logcat.txt:", out)
    self.assertIn("benchmark_run.txt:\nfirst process: pid 4000\n", out)


def _lm_job_report(name: str, result: str = "PASSED") -> dict:
  prefix = f"gs://p-devicerun/litert-cli/sessions/s-1/{name}/e-1"
  return {
      "displayName": name,
      "result": {"resultType": result},
      "executionReports": [{
          "outputFiles": [
              {"gcsOutputFile": {"path": f"{prefix}/{f}"}}
              for f in (
                  "artifacts/data/local/tmp/litert-cli/metrics.pb",
                  "artifacts/data/local/tmp/litert-cli/provenance.txt",
                  "logcat.txt",
              )
          ]
      }],
  }


class LmHelpersTest(absltest.TestCase):
  """The .litertlm bundle path: binaries, arguments, request and summary."""

  def test_is_lm_bundle(self):
    self.assertTrue(ddp.is_lm_bundle("m.litertlm"))
    self.assertTrue(ddp.is_lm_bundle("gs://b/dir/Model.LiteRTLM"))
    self.assertFalse(ddp.is_lm_bundle("m.tflite"))
    self.assertFalse(ddp.is_lm_bundle("litertlm"))

  def test_lm_binary_dir_follows_the_environment_variable(self):
    with mock.patch.dict(os.environ):
      os.environ.pop("DDP_LITERT_LM_VERSION", None)
      os.environ.pop("DDP_LITERT_LM_DIR", None)
      self.assertEqual(ddp._lm_binary_dir(), _LM_BINARY_DIR)
      os.environ["DDP_LITERT_LM_VERSION"] = "0.18.0"
      self.assertEqual(
          ddp._lm_binary_dir(),
          "gs://litert/binaries/0.18.0/android_arm64/litert_lm",
      )
      os.environ["DDP_LITERT_LM_DIR"] = "gs://my-bucket/litert_lm/"
      self.assertEqual(ddp._lm_binary_dir(), "gs://my-bucket/litert_lm")

  def test_lm_pushes_are_the_binary_and_every_library_listed(self):
    fake = _FakeCloud()
    with mock.patch.object(ddp.subprocess, "run", side_effect=fake.run):
      pushes = ddp._lm_pushes(_LM_BINARY_DIR)
    self.assertEqual(
        pushes,
        [(
            f"{_LM_BINARY_DIR}/litert_lm_advanced_main",
            "litert_lm_advanced_main",
        )]
        + [(f"{_LM_BINARY_DIR}/{lib}", lib) for lib in _LM_LIBS],
    )
    self.assertEqual(
        fake.commands, [["gcloud", "storage", "ls", f"{_LM_BINARY_DIR}/"]]
    )

  def test_lm_pushes_exit_1_when_the_listing_fails_or_lacks_the_binary(self):
    fake = _FakeCloud(lm_ls_stderr="ERROR: (gcloud.storage.ls) not found\n")
    with mock.patch.object(ddp.subprocess, "run", side_effect=fake.run):
      with self.assertRaisesRegex(click.ClickException, "Could not list"):
        ddp._lm_pushes(_LM_BINARY_DIR)
    fake = _FakeCloud(lm_listing=f"{_LM_BINARY_DIR}/litert_lm_main\n")
    with mock.patch.object(ddp.subprocess, "run", side_effect=fake.run):
      with self.assertRaisesRegex(
          click.ClickException, "litert_lm_advanced_main is not under"
      ):
        ddp._lm_pushes(_LM_BINARY_DIR)

  def test_build_lm_benchmark_args(self):
    args = ddp._build_lm_benchmark_args(
        model_name="m.litertlm",
        accelerator="gpu",
        prefill_tokens=1024,
        decode_tokens=256,
        max_num_tokens=1280,
        num_iterations=5,
    )
    self.assertEqual(
        args,
        [
            "--backend=gpu",
            f"--model_path={_ROOT}/m.litertlm",
            "--benchmark=true",
            "--benchmark_prefill_tokens=1024",
            "--benchmark_decode_tokens=256",
            "--max_num_tokens=1280",
            "--num_iterations=5",
            "--report_peak_memory_footprint=true",
            f"--metric_proto_file_path={_ROOT}/metrics.pb",
        ],
    )

  def test_lm_run_script_runs_the_binary_from_the_cli_directory(self):
    script = ddp._lm_run_script()
    self.assertTrue(script.startswith("#!/system/bin/sh\n"))
    self.assertIn(f'ROOT="{_ROOT}"', script)
    self.assertIn(
        'CACHES="./*.xnnpack_cache* ./*_mldrift_*cache*.bin ./*.mtp_drafter*"',
        script,
    )
    self.assertIn("rm -f $CACHES\n", script)
    self.assertIn("> provenance.txt", script)
    self.assertIn("chmod 755 litert_lm_advanced_main || exit 1", script)
    self.assertIn('[ -f "$MODEL" ] && touch "$MODEL"\n', script)
    # The warm-up process, then the measured one with the arguments unchanged.
    warmup = script.index(
        'LD_LIBRARY_PATH="$ROOT" ./litert_lm_advanced_main "$@"'
        " --num_iterations=1 --metric_proto_file_path=\n"
    )
    self.assertIn('[ "$rc" -eq 0 ] || exit "$rc"', script)
    measured = script.index(
        'LD_LIBRARY_PATH="$ROOT" ./litert_lm_advanced_main "$@"\n'
    )
    self.assertLess(warmup, measured)
    self.assertTrue(script.endswith('exit "$rc"\n'))

  @absltest.skipIf(shutil.which("sh") is None, "needs a POSIX shell")
  def test_lm_run_script_runs_a_warmup_process_then_the_measured_one(self):
    """The rendered script on a POSIX sh with a fake binary that logs its calls.

    The device-only commands of the provenance block (getprop, stat -c,
    sha256sum) fail quietly inside it; the control flow under test is the same.
    """
    root = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fake = root / ddp._LM_BINARY
    calls = root / f"{ddp._LM_BINARY}.calls"
    # Logs its arguments; exits 7 on call N when <binary>.failN exists; writes
    # a cache file unless <binary>.nocache exists.
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" >> "$0.calls"\n'
        'printf "%s\\n" -- >> "$0.calls"\n'
        'n=$(grep -c -- "^--$" "$0.calls")\n'
        '[ -f "$0.fail$n" ] && exit 7\n'
        '[ -f "$0.nocache" ] || touch ./m.litertlm_1_2.xnnpack_cache\n'
        "exit 0\n"
    )
    (root / "m.litertlm").write_bytes(b"\0")
    with mock.patch.object(constants, "LITERT_CLI_ANDROID_ROOT", str(root)):
      script = root / ddp._LM_RUN_SCRIPT_NAME
      script.write_text(ddp._lm_run_script())
    args = [
        "--backend=cpu",
        f"--model_path={root}/m.litertlm",
        "--num_iterations=5",
        f"--metric_proto_file_path={root}/metrics.pb",
    ]

    def run() -> subprocess.CompletedProcess[str]:
      if calls.exists():
        calls.unlink()
      return subprocess.run(
          ["sh", str(script), *args],
          capture_output=True,
          text=True,
          check=False,
      )

    def provenance() -> str:
      return (root / ddp._LM_PROVENANCE_FILE).read_text()

    # Both processes run: the warm-up one with the two extra flags, the
    # measured one with the arguments unchanged; both exit codes recorded.
    result = run()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
        calls.read_text().split("--\n"),
        [
            "\n".join(
                args + ["--num_iterations=1", "--metric_proto_file_path="]
            )
            + "\n",
            "\n".join(args) + "\n",
            "",
        ],
    )
    self.assertIn(f"args: {' '.join(args)}\n", provenance())
    self.assertRegex(
        provenance(),
        r"warm-up process: exit 0\ncaches after the warm-up process:\n"
        r".*m.litertlm_1_2.xnnpack_cache\n"
        r"measured process: exit 0\ncaches after the measured process:\n"
        r".*m.litertlm_1_2.xnnpack_cache\n$",
    )
    # A failing warm-up process ends the job with its exit code, unmeasured.
    (root / f"{ddp._LM_BINARY}.fail1").touch()
    result = run()
    self.assertEqual(result.returncode, 7, result.stderr)
    self.assertEqual(calls.read_text().count("--\n"), 1)
    self.assertIn("warm-up process: exit 7\n", provenance())
    self.assertNotIn("measured process", provenance())
    (root / f"{ddp._LM_BINARY}.fail1").unlink()
    # A failing measured process is the job's exit code too.
    (root / f"{ddp._LM_BINARY}.fail2").touch()
    result = run()
    self.assertEqual(result.returncode, 7, result.stderr)
    self.assertEqual(calls.read_text().count("--\n"), 2)
    self.assertIn("measured process: exit 7\n", provenance())
    (root / f"{ddp._LM_BINARY}.fail2").unlink()
    # A warm-up process that wrote no cache is visible in provenance.txt.
    (root / f"{ddp._LM_BINARY}.nocache").touch()
    for cache in root.glob("*.xnnpack_cache"):
      cache.unlink()
    result = run()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("caches after the warm-up process:\n  (none)\n", provenance())
    # No binary: chmod fails, exit 1 before any process.
    fake.unlink()
    result = run()
    self.assertEqual(result.returncode, 1, result.stderr)
    self.assertFalse(calls.exists())

  def test_build_session_request_for_a_bundle(self):
    pushes = [
        (f"{_LM_BINARY_DIR}/litert_lm_advanced_main", "litert_lm_advanced_main")
    ]
    pushes += [(f"{_LM_BINARY_DIR}/{lib}", lib) for lib in _LM_LIBS]
    body = ddp._build_session_request(
        session_name=_SESSION_NAME,
        model_gcs_path=f"gs://b/litert-cli/inputs/{_SESSION_NAME}/m.litertlm",
        model_name="m.litertlm",
        accelerator="gpu",
        device_list=["caiman-35"],
        output_dir="gs://b/litert-cli/sessions",
        benchmark_binary=(
            f"gs://b/litert-cli/inputs/{_SESSION_NAME}/litert_lm_run.sh"
        ),
        bench_args=["--backend=gpu"],
        extra_pushes=pushes,
        result_files=ddp._LM_RESULT_FILES,
        execution_timeout_secs=1800,
        runtime="litert-lm",
    )
    job = body["sessionConfig"]["jobConfigs"][0]
    self.assertEqual(job["displayName"], "gpu-caiman-35")
    binary = job["action"]["androidNativeBinary"]
    self.assertEqual(
        binary["androidNativeBinary"]["gcsInputFile"]["path"],
        f"gs://b/litert-cli/inputs/{_SESSION_NAME}/litert_lm_run.sh",
    )
    self.assertEqual(binary["executionTimeout"], "1800s")
    push, pull, _ = job["allocationConfig"]["deviceConfigs"][0]["actions"]
    file_configs = push["androidPushFiles"]["fileConfigs"]
    self.assertEqual(
        [f["destinationPath"] for f in file_configs],
        [f"{_ROOT}/m.litertlm", f"{_ROOT}/litert_lm_advanced_main"]
        + [f"{_ROOT}/{lib}" for lib in _LM_LIBS],
    )
    self.assertEqual(
        file_configs[1]["sourceFile"]["gcsInputFile"]["path"],
        f"{_LM_BINARY_DIR}/litert_lm_advanced_main",
    )
    self.assertEqual(
        pull["androidPullFiles"]["paths"],
        [f"{_ROOT}/metrics.pb", f"{_ROOT}/provenance.txt"],
    )
    self.assertEqual(job["labels"]["runtime"], "litert-lm")

  def test_lm_summary_leaves_out_the_warmup_iterations(self):
    lines = _LM_LOGCAT.splitlines()
    self.assertEqual([len(p) for p in ddp._lm_processes(lines)], [3])
    self.assertEqual(
        ddp._lm_summary(lines, 1),
        [
            (
                "LiteRT-LM benchmark: 3 iteration(s), 1 warm-up; median of the"
                " other 2:"
            ),
            (
                "  prefill 508.2 tokens/s, decode 63.0 tokens/s, time to first"
                " token 0.13 s; init 2812 ms (once per run)"
            ),
        ],
    )
    self.assertIn(
        "prefill 500.0 tokens/s, decode 60.0", ddp._lm_summary(lines, 0)[1]
    )
    self.assertEqual(ddp._lm_summary(lines, 3), [])
    self.assertEqual(ddp._lm_summary(_LOGCAT.splitlines(), 1), [])

  def test_lm_summary_reads_the_measured_process_after_the_warmup_one(self):
    lines = _LM_LOGCAT_TWO_PROCESSES.splitlines()
    processes = ddp._lm_processes(lines)
    self.assertEqual([len(p) for p in processes], [1, 3])
    self.assertEqual(processes[0][0]["init_ms"], 2811.82)
    self.assertEqual(processes[0][0]["peak_mem_mb"], 1643.35)
    self.assertEqual(processes[1][0]["init_ms"], 900.5)
    self.assertEqual(processes[1][2]["peak_mem_mb"], 1568.74)
    summary = [
        (
            "LiteRT-LM benchmark: second process, 3 iteration(s), 1 warm-up;"
            " median of the other 2:"
        ),
        (
            "  prefill 508.2 tokens/s, decode 63.0 tokens/s, time to first"
            " token 0.13 s; init 900 ms (cold 2812 ms in the first process);"
            " peak memory 1569 MB (RSS)"
        ),
    ]
    self.assertEqual(ddp._lm_summary(lines, 1), summary)
    self.assertEqual(ddp._lm_summary(lines, 3), [])
    # A stray block from an earlier process: the last two processes count.
    stray = _lm_block("0.50", "100.00", "10.00", pid="100", init="5000.00")
    self.assertEqual(
        ddp._lm_summary((stray + _LM_LOGCAT_TWO_PROCESSES).splitlines(), 1),
        summary,
    )
    # Blocks without a logcat prefix are one process.
    bare = [l.split(" : ", 1)[-1] for l in lines]
    self.assertEqual([len(p) for p in ddp._lm_processes(bare)], [4])
    self.assertIn("4 iteration(s), 1 warm-up", ddp._lm_summary(bare, 1)[0])

  def test_lm_summary_without_a_cold_init_or_peak_lines(self):
    # The warm-up process's block lacks an Init line.
    lines = (
        _lm_block("0.20", "480.00", "30.00", pid="8400", init="")
        + _lm_block("0.16", "491.23", "32.09", init="900.50")
        + _lm_block("0.14", "516.38", "65.98", init="900.50")
    ).splitlines()
    self.assertEqual(
        ddp._lm_summary(lines, 1),
        [
            (
                "LiteRT-LM benchmark: second process, 2 iteration(s), 1"
                " warm-up; median of the other 1:"
            ),
            (
                "  prefill 516.4 tokens/s, decode 66.0 tokens/s, time to first"
                " token 0.14 s; init 900 ms"
            ),
        ],
    )

  def test_lm_log_filter_keeps_the_benchmark_lines(self):
    from litert_cli.core.log_filters import LmBenchmarkLogFilter

    log_filter = LmBenchmarkLogFilter(default_quiet=True)
    shown = [l for l in _LM_LOGCAT.splitlines() if log_filter.should_show(l)]
    self.assertIn(
        "09-19 11:22:12.427  8446  8446 I native  :       Decode Speed: 32.09"
        " tokens/sec.",
        shown,
    )
    self.assertTrue(any("gpu_registry.cc" in l for l in shown))
    self.assertFalse(any("noise line" in l for l in shown))
    shown = [
        l
        for l in _LM_LOGCAT_TWO_PROCESSES.splitlines()
        if log_filter.should_show(l)
    ]
    self.assertTrue(
        any("Peak system ram usage: 1568.74MB." in l for l in shown)
    )
    self.assertTrue(
        log_filter.should_show(
            "09-28 15:14:22.295  3484  3484 I tflite  : XNNPack weight cache"
            " loaded from '/data/local/tmp/litert-cli/m.litertlm_1_2"
            ".xnnpack_cache'."
        )
    )
    self.assertTrue(
        log_filter.should_show(
            "09-28 15:23:36.118  9750  9750 I native  : I0000 00:00:1790576616"
            ".118 9750 inference_context.cc:2097] Initialized InferenceContext"
            " from serialized data."
        )
    )
    self.assertTrue(
        LmBenchmarkLogFilter(default_quiet=False).should_show("noise line")
    )


class LmRunDdpTest(absltest.TestCase):
  """`litert benchmark m.litertlm --ddp` through click, cloud mocked."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
    self.bundle = pathlib.Path(self.tmp_dir) / "m.litertlm"
    self.bundle.write_bytes(b"\0")

  def _invoke(self, fake: _FakeCloud, *extra_args: str, model=None):
    args = [
        str(model or self.bundle),
        "--ddp",
        "--device",
        "caiman-35",
        "--gcp-project",
        "p",
        *extra_args,
    ]
    with _patched(fake, self.tmp_dir):
      return testing.CliRunner().invoke(benchmark_cli.benchmark_cmd, args)

  def test_bundle_session_uploads_the_bundle_and_the_run_script(self):
    fake = _FakeCloud(
        [_done_response("PASSED", (_lm_job_report("gpu-caiman-35"),))],
        logcat=_LM_LOGCAT,
    )
    result = self._invoke(fake, "--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    inputs_dir = f"gs://p-devicerun/litert-cli/inputs/{_SESSION_NAME}"
    uploads = [
        c[3:]
        for c in fake.commands
        if c[:3] == ["gcloud", "storage", "cp"] and not c[3].startswith("gs://")
    ]
    self.assertEqual(uploads[0], [str(self.bundle), f"{inputs_dir}/"])
    self.assertEqual(os.path.basename(uploads[1][0]), "litert_lm_run.sh")
    self.assertEqual(fake.uploads["litert_lm_run.sh"], ddp._lm_run_script())
    body = json.loads(fake.requests[0].data.decode())
    job = body["sessionConfig"]["jobConfigs"][0]
    binary = job["action"]["androidNativeBinary"]
    self.assertEqual(
        binary["androidNativeBinary"]["gcsInputFile"]["path"],
        f"{inputs_dir}/litert_lm_run.sh",
    )
    self.assertEqual(
        binary["args"][:3],
        [
            "--backend=gpu",
            f"--model_path={_ROOT}/m.litertlm",
            "--benchmark=true",
        ],
    )
    self.assertIn("--num_iterations=5", binary["args"])
    self.assertIn("--report_peak_memory_footprint=true", binary["args"])
    push = job["allocationConfig"]["deviceConfigs"][0]["actions"][0]
    self.assertLen(push["androidPushFiles"]["fileConfigs"], 2 + len(_LM_LIBS))
    self.assertIn(
        f"Binary: {_LM_BINARY_DIR}/litert_lm_advanced_main", result.output
    )
    # A bundle's session waits for the device's execution timeout.
    self.assertIn("timeout: 2400 s", result.output)
    self.assertIn("Job 'gpu-caiman-35': PASSED", result.output)
    self.assertIn("Decode Speed: 32.09 tokens/sec.", result.output)
    self.assertIn("1 warm-up; median of the other 2", result.output)
    self.assertIn("prefill 508.2 tokens/s, decode 63.0 tokens/s", result.output)
    self.assertNotIn("noise line", result.output)
    self.assertNotIn("999.00", result.output.split("LiteRT-LM benchmark")[1])

  def test_bundle_options_reach_the_binary(self):
    fake = _FakeCloud(
        [_done_response("PASSED", (_lm_job_report("cpu-caiman-35"),))]
    )
    result = self._invoke(
        fake,
        "--prefill-tokens",
        "128",
        "--decode-tokens",
        "32",
        "--max-num-tokens",
        "512",
        "--num-iterations",
        "3",
        "--warmup-runs",
        "0",
    )
    self.assertEqual(result.exit_code, 0, result.output)
    body = json.loads(fake.requests[0].data.decode())
    args = body["sessionConfig"]["jobConfigs"][0]["action"][
        "androidNativeBinary"
    ]["args"]
    self.assertIn("--backend=cpu", args)
    self.assertIn("--benchmark_prefill_tokens=128", args)
    self.assertIn("--benchmark_decode_tokens=32", args)
    self.assertIn("--max_num_tokens=512", args)
    self.assertIn("--num_iterations=3", args)

  def test_bundle_rejects_warmup_runs_not_smaller_than_the_iterations(self):
    fake = _FakeCloud()
    result = self._invoke(fake, "--num-iterations", "2", "--warmup-runs", "2")
    self.assertEqual(result.exit_code, 1)
    self.assertIn("--warmup-runs (2) must be smaller", result.output)
    self.assertEmpty(fake.commands)

  def test_bundle_rejects_benchmark_model_only_options(self):
    fake = _FakeCloud()
    result = self._invoke(fake, "--signature-key", "serving_default")
    self.assertEqual(result.exit_code, 1)
    self.assertIn("benchmark_model options", result.output)
    self.assertEmpty(fake.commands)

  def test_bundle_is_rejected_on_the_other_targets(self):
    for target in ("--android", "--desktop", "--gcp"):
      with mock.patch.object(constants, "DEFAULT_QUIET", False):
        result = testing.CliRunner().invoke(
            benchmark_cli.benchmark_cmd, [str(self.bundle), target]
        )
      self.assertEqual(result.exit_code, 1, target)
      self.assertIn("--ddp target only", result.output)

  def test_bundle_run_script_upload_failure_exits_1(self):
    fake = _FakeCloud(
        [_done_response()], upload_fails_for=("litert_lm_run.sh",)
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Error: Failed to upload", result.output)
    self.assertIn("litert_lm_run.sh", result.output)
    self.assertEmpty(fake.requests)

  def test_bundle_listing_failure_exits_1_before_the_submit(self):
    fake = _FakeCloud(
        [_done_response()],
        lm_ls_stderr="ERROR: (gcloud.storage.ls) gs://litert not found\n",
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Could not list the LiteRT-LM binaries", result.output)
    self.assertEmpty(fake.requests)

  def test_failed_bundle_job_exits_1_with_the_logcat_tail(self):
    fake = _FakeCloud(
        [
            _done_response(
                "FAILED", (_lm_job_report("cpu-caiman-35", "FAILED"),)
            )
        ],
        logcat="F linker  : CANNOT LINK EXECUTABLE\n",
    )
    result = self._invoke(fake)
    self.assertEqual(result.exit_code, 1)
    self.assertIn("Last 20 lines of logcat.txt", result.output)
    self.assertIn("CANNOT LINK EXECUTABLE", result.output)
    self.assertIn("Error: Job 'cpu-caiman-35': FAILED", result.output)


if __name__ == "__main__":
  absltest.main()
