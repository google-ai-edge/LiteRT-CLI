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
import io
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from unittest import mock
import http.client
import urllib.error
import uuid

from absl.testing import absltest
import click
from click import testing
from litert_cli.commands.benchmark import cli as benchmark_cli
from litert_cli.commands.benchmark import ddp
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
    "noise line\n"
    "I benchmark_litert_model: Inference timings in us: Init: 1\n"
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
      cp_fails_for=(),
      tokens=("token-1",),
  ):
    self.poll_results = list(poll_results)
    self.create_response = create_response or _CREATE_RESPONSE
    self.ls_stderr = ls_stderr
    self.create_stderr = create_stderr
    self.upload_fails = upload_fails
    self.cp_fails_for = cp_fails_for
    self.tokens = list(tokens)
    self.commands = []
    self.requests = []
    self.events = []
    self.sleeps = []

  def run(self, cmd, **kwargs):
    self.commands.append(cmd)
    if cmd[:3] == ["gcloud", "storage", "ls"] and self.ls_stderr:
      return subprocess.CompletedProcess(cmd, 1, "", self.ls_stderr)
    if cmd[:4] == _CREATE_BUCKET and self.create_stderr:
      return subprocess.CompletedProcess(cmd, 1, "", self.create_stderr)
    is_cp = cmd[:3] == ["gcloud", "storage", "cp"]
    if is_cp and not cmd[3].startswith("gs://") and self.upload_fails:
      assert kwargs.get("check"), "the upload relies on check=True"
      raise subprocess.CalledProcessError(1, cmd)
    if is_cp and cmd[3].startswith("gs://"):
      local_dir = pathlib.Path(cmd[-1])
      if any(name in local_dir.parts for name in self.cp_fails_for):
        return subprocess.CompletedProcess(
            cmd, 1, "", "ERROR: (gcloud.storage.cp) permission denied\n"
        )
      (local_dir / "logcat.txt").write_text(_LOGCAT)
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

  def _invoke(self, fake: _FakeCloud, *extra_args: str, model=None):
    """Runs the benchmark command through click and returns the result."""
    args = [
        str(model or self.model),
        "--ddp",
        "--device",
        "caiman-35",
        "--gcp-project",
        "p",
        *extra_args,
    ]
    with _patched(fake, self.tmp_dir):
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

  def test_cli_rejects_the_default_device_for_ddp(self):
    runner = testing.CliRunner()
    with mock.patch.object(constants, "DEFAULT_QUIET", False):
      result = runner.invoke(
          benchmark_cli.benchmark_cmd,
          ["m.tflite", "--ddp", "--gcp-project", "p"],
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
    self.assertIn("timeout: 750 s", result.output)
    self.assertIn("Session 's-1' finished: PASSED", result.output)
    self.assertIn(
        str(pathlib.Path(self.tmp_dir) / "ddp" / "s-1" / "cpu-caiman-35"),
        result.output,
    )
    self.assertIn("benchmark_litert_model", result.output)
    self.assertNotIn("noise line", result.output)
    self.assertNotIn("token-1", result.output)

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
        contextlib.redirect_stdout(io.StringIO()),
    ):
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
    self.assertEmpty(uploads)
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
    self.assertIn("timeout: 620 s", result.output)

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
        [_done_response(
            "PASSED",
            (_job_report("cpu-caiman-35"), _job_report("cpu-pa3q-35")),
        )],
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


if __name__ == "__main__":
  absltest.main()
