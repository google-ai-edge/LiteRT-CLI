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

"""Benchmarking on Developer Device Platform (DDP) devices in Google Cloud."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

import click
from litert_cli.core import constants
from litert_cli.core.log_filters import BenchmarkLogFilter

_DEFAULT_GCP_PROJECT = os.environ.get("LITERT_GCP_PROJECT")
_GCP_BUCKET = os.environ.get("LITERT_GCP_BUCKET")
_DEFAULT_DDP_LOCATION = "global"
_DEFAULT_BUCKET_LOCATION = "US"
_DEFAULT_DEVICE_RUN_ENDPOINT = "https://devicerun.googleapis.com/v1alpha"
# NOTE: Keep in sync with LiteRT stable releases.
_DEFAULT_DDP_LITERT_VERSION = "2.2.0"
_DDP_BENCHMARK_BINARY = (
    "gs://litert/binaries/"
    f"{_DEFAULT_DDP_LITERT_VERSION}/android_arm64/benchmark_model"
)
_GCS_INPUTS_PREFIX = "litert-cli/inputs"
_GCS_SESSIONS_PREFIX = "litert-cli/sessions"
_RESULT_FILES = ("results.pb", "runtime_info.pb")
_POLL_INTERVAL_SECS = 15
_LOGCAT_TAIL_LINES = 20
# Session and job display names must match ^[A-Za-z0-9][A-Za-z0-9-_ ]*$ and
# be at most 63 bytes.
_DISPLAY_NAME_ALLOWED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_ "
)
_DISPLAY_NAME_MAX_LEN = 63
_DEVICE_LIST_HINT = (
    "List DDP device ids with: gcloud beta device-run devices list"
    " --project <PROJECT_ID>"
)


def _get_sessions_url(endpoint: str, gcp_project: str, location: str) -> str:
  """Returns the URL for creating a Device Run session."""
  return f"{endpoint}/projects/{gcp_project}/locations/{location}/sessions"


def _get_operation_url(endpoint: str, op_name: str) -> str:
  """Returns the URL for polling operation status."""
  return f"{endpoint}/{op_name}"


def _get_console_url(bucket: str, prefix: str, session_id: str) -> str:
  """Returns the Google Cloud Console URL of the session output directory."""
  return (
      "https://console.cloud.google.com/storage/browser/"
      f"{bucket}/{prefix}/{session_id}"
  )


def _display_name(*parts: str) -> str:
  """Joins parts into a display name accepted by the Device Run API."""
  name = "-".join(parts)
  name = "".join(c if c in _DISPLAY_NAME_ALLOWED else "-" for c in name)
  return name.lstrip("-_ ")[:_DISPLAY_NAME_MAX_LEN] or "job"


def _normalize_devices(
    devices: str | list[str] | tuple[str, ...],
) -> list[str]:
  """Splits comma-separated device ids and drops blanks and duplicates."""
  items = [devices] if isinstance(devices, str) else devices
  device_list: list[str] = []
  for item in items:
    if item:
      for part in item.split(","):
        part = part.strip()
        if part and part not in device_list:
          device_list.append(part)
  return device_list


def _build_benchmark_args(
    *,
    model_name: str,
    accelerator: str,
    num_runs: int = 50,
    warmup_runs: int = 1,
    min_secs: float = 1.0,
    max_secs: float = 150.0,
    warmup_min_secs: float = 0.5,
    input_layer_value_range: str | None = None,
    signature_key: str | None = None,
) -> list[str]:
  """Builds benchmark_model arguments, mirroring the Android target."""
  root = constants.LITERT_CLI_ANDROID_ROOT
  bench_args = [f"--graph={root}/{model_name}"]
  if accelerator == "gpu":
    bench_args.append("--use_gpu=true")
  if num_runs != 50:
    bench_args.append(f"--num_runs={num_runs}")
  if warmup_runs != 1:
    bench_args.append(f"--warmup_runs={warmup_runs}")
  if min_secs != 1.0:
    bench_args.append(f"--min_secs={min_secs}")
  if max_secs != 150.0:
    bench_args.append(f"--max_secs={max_secs}")
  if warmup_min_secs != 0.5:
    bench_args.append(f"--warmup_min_secs={warmup_min_secs}")
  if input_layer_value_range:
    bench_args.append(f"--input_layer_value_range={input_layer_value_range}")
  if signature_key:
    bench_args.append(f"--signature_to_run_for={signature_key}")
  bench_args.append(f"--result_file_path={root}/{_RESULT_FILES[0]}")
  bench_args.append(
      f"--model_runtime_info_output_file={root}/{_RESULT_FILES[1]}"
  )
  return bench_args


def _build_session_request(
    *,
    session_name: str,
    model_gcs_path: str,
    model_name: str,
    accelerator: str,
    device_list: list[str],
    output_dir: str,
    bench_args: list[str],
) -> dict[str, Any]:
  """Builds the Device Run session request: one job per device."""
  root = constants.LITERT_CLI_ANDROID_ROOT
  job_configs = []
  for device_id in device_list:
    job_configs.append({
        "displayName": _display_name(accelerator, device_id),
        "action": {
            "androidNativeBinary": {
                "androidNativeBinary": {
                    "gcsInputFile": {"path": _DDP_BENCHMARK_BINARY}
                },
                "args": bench_args,
            }
        },
        "allocationConfig": {
            "deviceConfigs": [{
                "requirement": {"deviceId": device_id},
                "actions": [
                    {
                        "androidPushFiles": {
                            "fileConfigs": [{
                                "sourceFile": {
                                    "gcsInputFile": {"path": model_gcs_path}
                                },
                                "destinationPath": f"{root}/{model_name}",
                            }]
                        }
                    },
                    {
                        "androidPullFiles": {
                            "paths": [f"{root}/{f}" for f in _RESULT_FILES]
                        }
                    },
                    {"androidLogcat": {}},
                ],
            }]
        },
        "labels": {
            "tool": "litert-cli",
            "accelerator": accelerator,
            "model": model_name,
        },
    })
  return {
      "sessionConfig": {
          "displayName": session_name,
          "outputDirectoryConfig": {
              "gcsOutputDirectory": {"path": output_dir}
          },
          "jobConfigs": job_configs,
      }
  }


def _print_logcat_results(logcat_path: pathlib.Path, passed: bool) -> None:
  """Prints the benchmark lines of a logcat, or its tail when the job failed."""
  lines = logcat_path.read_text(errors="replace").splitlines()
  if passed:
    log_filter = BenchmarkLogFilter(constants.DEFAULT_QUIET)
    for line in lines:
      if log_filter.should_show(line):
        click.echo(line)
  else:
    click.echo(f"Last {_LOGCAT_TAIL_LINES} lines of {logcat_path.name}:")
    for line in lines[-_LOGCAT_TAIL_LINES:]:
      click.echo(line)


def _fetch_session_outputs(
    session_report: dict[str, Any], session_id: str
) -> None:
  """Downloads each job's output files and prints the benchmark results."""
  local_root = (
      pathlib.Path(constants.LITERT_CLI_CACHE_DIR) / "ddp" / session_id
  )
  for job in session_report.get("jobReports", []):
    job_name = job.get("displayName", job.get("id", "job"))
    job_result = job.get("result", {}).get("resultType", "UNKNOWN")
    click.secho(
        f"\nJob '{job_name}': {job_result}",
        fg="green" if job_result == "PASSED" else "red",
    )
    gcs_paths = [
        f["gcsOutputFile"]["path"]
        for execution in job.get("executionReports", [])
        for f in execution.get("outputFiles", [])
        if "gcsOutputFile" in f
    ]
    if not gcs_paths:
      click.echo("No output files were reported for this job.")
      continue
    local_dir = local_root / job_name
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
      subprocess.run(
          ["gcloud", "storage", "cp", *gcs_paths, f"{local_dir}/"],
          check=True,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
    except subprocess.CalledProcessError as e:
      click.secho(f"Error: Failed to download output files: {e}", fg="red")
      continue
    click.echo(f"Output files saved to: {local_dir}")
    logcat_path = local_dir / "logcat.txt"
    if logcat_path.exists():
      _print_logcat_results(logcat_path, passed=(job_result == "PASSED"))


def run_ddp(
    model_path_str: str,
    accelerator: str,
    devices: list[str],
    gcp_project: str | None = None,
    gcp_bucket: str | None = None,
    num_runs: int = 50,
    warmup_runs: int = 1,
    min_secs: float = 1.0,
    max_secs: float = 150.0,
    warmup_min_secs: float = 0.5,
    input_layer_value_range: str | None = None,
    signature_key: str | None = None,
) -> None:
  """Runs the model on DDP devices via the Device Run API.

  Uploads model to GCS if it's not already there.
  Submits a Device Run session that runs benchmark_model on each device.
  Polls the session operation, then downloads and prints the results.

  Args:
    model_path_str: Path to the LiteRT model file (local or gs://).
    accelerator: Hardware accelerator to use (cpu, gpu).
    devices: Target DDP device id(s) (e.g., 'caiman-35').
    gcp_project: GCP project ID for benchmarking.
    gcp_bucket: GCS bucket name for the model and the session outputs.
    num_runs: Target number of benchmark iterations.
    warmup_runs: Number of warmup iterations before benchmarking.
    min_secs: Minimum seconds to run.
    max_secs: Maximum seconds to run.
    warmup_min_secs: Minimum warmup duration in seconds.
    input_layer_value_range: Value range for input layers.
    signature_key: The signature key to benchmark.
  """
  if accelerator == "npu":
    raise click.ClickException("NPU on --ddp is not supported yet.")

  device_list = _normalize_devices(devices)
  if not device_list:
    raise click.ClickException(
        "--device is required for running DDP benchmark tests."
        f" {_DEVICE_LIST_HINT}"
    )

  if not gcp_project:
    gcp_project = _DEFAULT_GCP_PROJECT

  if not gcp_project:
    raise click.ClickException(
        "Missing GCP project. You must specify a GCP project by passing"
        " '--gcp-project <PROJECT_ID>' or by setting the 'LITERT_GCP_PROJECT'"
        " environment variable."
    )

  model_path = model_path_str
  local_model = None
  if not model_path.startswith("gs://"):
    local_model = pathlib.Path(model_path)
    if not local_model.exists():
      click.secho(f"Error: Local model file not found: {model_path}", fg="red")
      return
  model_name = model_path.rsplit("/", 1)[-1]

  target_bucket = gcp_bucket or _GCP_BUCKET
  if not target_bucket:
    target_bucket = f"{gcp_project}-devicerun"
    click.secho(
        "Note: GCS bucket not specified via '--gcp-bucket' or"
        " 'LITERT_GCP_BUCKET' environment variable. Using default"
        f" project-bound bucket 'gs://{target_bucket}'.",
        fg="yellow",
    )
  else:
    click.echo(f"Using specified GCS bucket 'gs://{target_bucket}'.")

  # Check if bucket exists, create if not
  click.echo(
      f"Ensuring GCS bucket 'gs://{target_bucket}' exists for project"
      f" '{gcp_project}'..."
  )
  try:
    check_res = subprocess.run(
        ["gcloud", "storage", "ls", f"gs://{target_bucket}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check_res.returncode != 0:
      click.secho(
          f"Creating GCS bucket 'gs://{target_bucket}' in location"
          f" '{_DEFAULT_BUCKET_LOCATION}'...",
          fg="cyan",
      )
      subprocess.run(
          [
              "gcloud",
              "storage",
              "buckets",
              "create",
              f"gs://{target_bucket}",
              f"--project={gcp_project}",
              f"--location={_DEFAULT_BUCKET_LOCATION}",
          ],
          check=True,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
  except subprocess.CalledProcessError as e:
    click.secho(
        f"Error: Failed to ensure GCS bucket 'gs://{target_bucket}': {e}",
        fg="red",
    )
    return

  # Upload model to GCS if it's not already there.
  if local_model is not None:
    model_gcs_dir = f"gs://{target_bucket}/{_GCS_INPUTS_PREFIX}"
    click.secho(
        f"Uploading local model '{model_path}' to {model_gcs_dir}/...",
        fg="cyan",
    )
    try:
      subprocess.run(
          ["gcloud", "storage", "cp", str(local_model), f"{model_gcs_dir}/"],
          check=True,
      )
      model_path = f"{model_gcs_dir}/{model_name}"
    except subprocess.CalledProcessError as e:
      click.secho(
          f"Error: Failed to upload '{model_path}' to Google Cloud Storage:"
          f" {e}",
          fg="red",
      )
      return

  session_name = f"litert-cli-benchmark-{uuid.uuid4().hex[:8]}"
  output_dir = f"gs://{target_bucket}/{_GCS_SESSIONS_PREFIX}"

  click.echo("Fetching GCP access token...")
  try:
    token = subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        text=True,
    ).strip()
  except subprocess.CalledProcessError as e:
    click.secho(
        "Error: Failed to get gcloud access token. Please run 'gcloud auth"
        f" application-default login' first. Details: {e}",
        fg="red",
    )
    return

  endpoint = os.environ.get(
      "DEVICE_RUN_ENDPOINT", _DEFAULT_DEVICE_RUN_ENDPOINT
  ).rstrip("/")
  url = _get_sessions_url(endpoint, gcp_project, _DEFAULT_DDP_LOCATION)
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
      "X-Goog-User-Project": gcp_project,
  }

  bench_args = _build_benchmark_args(
      model_name=model_name,
      accelerator=accelerator,
      num_runs=num_runs,
      warmup_runs=warmup_runs,
      min_secs=min_secs,
      max_secs=max_secs,
      warmup_min_secs=warmup_min_secs,
      input_layer_value_range=input_layer_value_range,
      signature_key=signature_key,
  )
  body = _build_session_request(
      session_name=session_name,
      model_gcs_path=model_path,
      model_name=model_name,
      accelerator=accelerator,
      device_list=device_list,
      output_dir=output_dir,
      bench_args=bench_args,
  )

  # Submit the session via http requests to the Device Run API.
  req = urllib.request.Request(
      url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
  )
  click.echo(
      f"Submitting '{accelerator}' benchmark session '{session_name}' to DDP"
      f" (Project: {gcp_project}, Devices: {', '.join(device_list)},"
      f" Binary: {_DDP_BENCHMARK_BINARY})..."
  )

  try:
    with urllib.request.urlopen(req) as response:
      resp_data = json.loads(response.read().decode())
      click.secho("Benchmark session submitted successfully!", fg="green")

      op_name = resp_data.get("name", "")
      if op_name and "operations" in op_name:
        op_url = _get_operation_url(endpoint, op_name)
        session_id = (
            resp_data.get("metadata", {})
            .get("target", "")
            .rsplit("/", 1)[-1]
        )
        console_url = _get_console_url(
            target_bucket, _GCS_SESSIONS_PREFIX, session_id
        )
        click.echo(
            f"Waiting for session '{session_id}' to complete (Operation:"
            f" {op_name}). This may take a few minutes..."
        )
        click.secho(
            f"View the outputs on the Cloud Console: {console_url}", fg="cyan"
        )

        try:
          while True:
            time.sleep(_POLL_INTERVAL_SECS)
            click.echo(".", nl=False)
            req_op = urllib.request.Request(op_url, headers=headers)
            try:
              with urllib.request.urlopen(req_op) as res_op:
                op_data = json.loads(res_op.read().decode())
            except urllib.error.HTTPError as e:
              with e:
                click.secho(f"\nError polling operation: {e}", fg="yellow")
              continue

            if op_data.get("done"):
              click.echo("")  # Print a newline after the dots
              if "error" in op_data:
                click.secho(
                    "Benchmark failed:"
                    f" {json.dumps(op_data['error'], indent=2)}",
                    fg="red",
                )
              else:
                session_report = op_data.get("response", {}).get(
                    "sessionReport", {}
                )
                result = session_report.get("result", {}).get(
                    "resultType", "UNKNOWN"
                )
                click.secho(
                    f"Session '{session_id}' finished: {result}",
                    fg="green" if result == "PASSED" else "red",
                )
                _fetch_session_outputs(session_report, session_id)
              break
        except KeyboardInterrupt:
          click.echo("")
          click.secho(
              "\nPolling interrupted. The benchmark session is still running.",
              fg="yellow",
          )
          click.echo(
              "You can check its status later by viewing it in the console:"
              f" {console_url}"
          )
      else:
        click.echo(json.dumps(resp_data, indent=2))
  except urllib.error.HTTPError as e:
    err_body = e.read().decode()
    click.secho(f"Failed to submit benchmark: {e.code} {e.reason}", fg="red")
    click.secho(f"Details: {err_body}", fg="red")
    if e.code in (400, 404) or "device" in err_body.lower():
      click.echo(_DEVICE_LIST_HINT)
