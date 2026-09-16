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
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
import click
from click import testing
from litert_cli.commands.benchmark import cli as benchmark_cli
from litert_cli.commands.benchmark import ddp
from litert_cli.core import constants

_ROOT = constants.LITERT_CLI_ANDROID_ROOT


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

  def test_build_benchmark_args_cpu_defaults(self):
    self.assertEqual(
        ddp._build_benchmark_args(model_name="m.tflite", accelerator="cpu"),
        [
            f"--graph={_ROOT}/m.tflite",
            f"--result_file_path={_ROOT}/results.pb",
            f"--model_runtime_info_output_file={_ROOT}/runtime_info.pb",
        ],
    )

  def test_build_benchmark_args_gpu_and_non_default_options(self):
    args = ddp._build_benchmark_args(
        model_name="m.tflite",
        accelerator="gpu",
        num_runs=10,
        warmup_runs=1,
        signature_key="serving_default",
    )
    self.assertIn("--use_gpu=true", args)
    self.assertIn("--num_runs=10", args)
    self.assertIn("--signature_to_run_for=serving_default", args)
    self.assertNotIn("--warmup_runs=1", args)

  def test_build_session_request_one_job_per_device(self):
    body = ddp._build_session_request(
        session_name="litert-cli-benchmark-abcd1234",
        model_gcs_path="gs://b/litert-cli/inputs/m.tflite",
        model_name="m.tflite",
        accelerator="cpu",
        device_list=["caiman-35", "pa3q-35"],
        output_dir="gs://b/litert-cli/sessions",
        bench_args=["--graph=x"],
    )
    config = body["sessionConfig"]
    self.assertEqual(config["displayName"], "litert-cli-benchmark-abcd1234")
    self.assertEqual(
        config["outputDirectoryConfig"]["gcsOutputDirectory"]["path"],
        "gs://b/litert-cli/sessions",
    )
    self.assertLen(config["jobConfigs"], 2)
    job = config["jobConfigs"][1]
    self.assertEqual(job["displayName"], "cpu-pa3q-35")
    binary = job["action"]["androidNativeBinary"]
    self.assertEqual(
        binary["androidNativeBinary"]["gcsInputFile"]["path"],
        "gs://litert/binaries/2.2.0/android_arm64/benchmark_model",
    )
    self.assertEqual(binary["args"], ["--graph=x"])
    device_config = job["allocationConfig"]["deviceConfigs"][0]
    self.assertEqual(device_config["requirement"]["deviceId"], "pa3q-35")
    push, pull, logcat = device_config["actions"]
    file_config = push["androidPushFiles"]["fileConfigs"][0]
    self.assertEqual(
        file_config["sourceFile"]["gcsInputFile"]["path"],
        "gs://b/litert-cli/inputs/m.tflite",
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

  def test_npu_is_rejected(self):
    with self.assertRaisesRegex(click.ClickException, "NPU on --ddp"):
      ddp.run_ddp("m.tflite", "npu", ["caiman-35"], gcp_project="p")

  def test_missing_device_is_rejected(self):
    with self.assertRaisesRegex(click.ClickException, "--device is required"):
      ddp.run_ddp("m.tflite", "cpu", [""], gcp_project="p")

  @mock.patch.object(ddp, "_DEFAULT_GCP_PROJECT", None)
  def test_missing_project_is_rejected(self):
    with self.assertRaisesRegex(click.ClickException, "Missing GCP project"):
      ddp.run_ddp("m.tflite", "cpu", ["caiman-35"])

  def test_cli_rejects_the_default_device_for_ddp(self):
    runner = testing.CliRunner()
    with mock.patch.object(constants, "DEFAULT_QUIET", False):
      result = runner.invoke(
          benchmark_cli.benchmark_cmd,
          ["m.tflite", "--ddp", "--gcp-project", "p"],
      )
    self.assertEqual(result.exit_code, 1)
    self.assertIn("--device is required", result.output)

  def test_submits_session_and_fetches_outputs(self):
    create_response = {
        "name": "projects/p/locations/global/operations/op-1",
        "metadata": {"target": "projects/p/locations/global/sessions/s-1"},
        "done": False,
    }
    output_prefix = "gs://p-devicerun/litert-cli/sessions/s-1/cpu-caiman-35/e-1"
    done_response = {
        "name": "projects/p/locations/global/operations/op-1",
        "done": True,
        "response": {
            "sessionReport": {
                "result": {"resultType": "PASSED"},
                "jobReports": [{
                    "displayName": "cpu-caiman-35",
                    "result": {"resultType": "PASSED"},
                    "executionReports": [{
                        "outputFiles": [
                            {"gcsOutputFile": {"path": f"{output_prefix}/{f}"}}
                            for f in (
                                "artifacts/data/local/tmp/results.pb",
                                "artifacts/data/local/tmp/runtime_info.pb",
                                "logcat.txt",
                            )
                        ]
                    }],
                }],
            }
        },
    }
    requests = []

    def fake_urlopen(req):
      requests.append(req)
      payload = create_response if req.data else done_response
      response = mock.MagicMock()
      response.read.return_value = json.dumps(payload).encode()
      response.__enter__.return_value = response
      return response

    def fake_run(cmd, **kwargs):
      del kwargs
      if cmd[:3] == ["gcloud", "storage", "cp"] and cmd[-1].endswith("/"):
        local_dir = pathlib.Path(cmd[-1])
        if local_dir.is_dir():
          (local_dir / "logcat.txt").write_text(
              "noise line\n"
              "I benchmark_litert_model: Inference timings in us: Init: 1\n"
          )
      return mock.MagicMock(returncode=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
      model = pathlib.Path(tmp_dir) / "m.tflite"
      model.write_bytes(b"\0")
      stdout = io.StringIO()
      with (
          mock.patch.object(ddp.subprocess, "run", side_effect=fake_run),
          mock.patch.object(
              ddp.subprocess, "check_output", return_value="secret-token\n"
          ),
          mock.patch.object(
              ddp.urllib.request, "urlopen", side_effect=fake_urlopen
          ),
          mock.patch.object(ddp.time, "sleep"),
          mock.patch.object(ddp, "_GCP_BUCKET", None),
          mock.patch.object(constants, "LITERT_CLI_CACHE_DIR", tmp_dir),
          contextlib.redirect_stdout(stdout),
      ):
        ddp.run_ddp(str(model), "cpu", ["caiman-35"], gcp_project="p")

      self.assertLen(requests, 2)
      post = requests[0]
      self.assertEqual(
          post.full_url,
          "https://devicerun.googleapis.com/v1alpha/projects/p/locations/"
          "global/sessions",
      )
      self.assertEqual(post.get_header("X-goog-user-project"), "p")
      body = json.loads(post.data.decode())
      job = body["sessionConfig"]["jobConfigs"][0]
      self.assertEqual(job["displayName"], "cpu-caiman-35")
      self.assertEqual(
          job["allocationConfig"]["deviceConfigs"][0]["actions"][0][
              "androidPushFiles"
          ]["fileConfigs"][0]["sourceFile"]["gcsInputFile"]["path"],
          "gs://p-devicerun/litert-cli/inputs/m.tflite",
      )
      self.assertEqual(
          body["sessionConfig"]["outputDirectoryConfig"]["gcsOutputDirectory"][
              "path"
          ],
          "gs://p-devicerun/litert-cli/sessions",
      )
      self.assertEqual(
          requests[1].full_url,
          "https://devicerun.googleapis.com/v1alpha/projects/p/locations/"
          "global/operations/op-1",
      )
      output = stdout.getvalue()
      self.assertIn("Session 's-1' finished: PASSED", output)
      self.assertIn(
          str(pathlib.Path(tmp_dir) / "ddp" / "s-1" / "cpu-caiman-35"), output
      )
      self.assertIn("benchmark_litert_model", output)
      self.assertNotIn("noise line", output)
      self.assertNotIn("secret-token", output)

  def test_gs_model_is_used_without_upload(self):
    create_response = {
        "name": "projects/p/locations/global/operations/op-1",
        "metadata": {"target": "projects/p/locations/global/sessions/s-1"},
        "done": False,
    }
    done_response = {"name": "op-1", "done": True, "response": {}}
    requests = []
    run_calls = []

    def fake_urlopen(req):
      requests.append(req)
      payload = create_response if req.data else done_response
      response = mock.MagicMock()
      response.read.return_value = json.dumps(payload).encode()
      response.__enter__.return_value = response
      return response

    def fake_run(cmd, **kwargs):
      del kwargs
      run_calls.append(cmd)
      return mock.MagicMock(returncode=0)

    with (
        mock.patch.object(ddp.subprocess, "run", side_effect=fake_run),
        mock.patch.object(ddp.subprocess, "check_output", return_value="t\n"),
        mock.patch.object(
            ddp.urllib.request, "urlopen", side_effect=fake_urlopen
        ),
        mock.patch.object(ddp.time, "sleep"),
        mock.patch.object(ddp, "_GCP_BUCKET", None),
        contextlib.redirect_stdout(io.StringIO()),
    ):
      ddp.run_ddp("gs://b/dir/m.tflite", "gpu", ["caiman-35"], gcp_project="p")

    self.assertFalse(
        [c for c in run_calls if c[:3] == ["gcloud", "storage", "cp"]]
    )
    body = json.loads(requests[0].data.decode())
    job = body["sessionConfig"]["jobConfigs"][0]
    self.assertEqual(
        job["allocationConfig"]["deviceConfigs"][0]["actions"][0][
            "androidPushFiles"
        ]["fileConfigs"][0]["sourceFile"]["gcsInputFile"]["path"],
        "gs://b/dir/m.tflite",
    )
    self.assertEqual(
        job["action"]["androidNativeBinary"]["args"][:2],
        [f"--graph={_ROOT}/m.tflite", "--use_gpu=true"],
    )


if __name__ == "__main__":
  absltest.main()
