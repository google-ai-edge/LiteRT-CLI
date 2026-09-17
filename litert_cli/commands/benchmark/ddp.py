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

import http.client
import json
import os
import pathlib
import re
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
# The LiteRT release whose prebuilt benchmark_model runs on the devices.
# NOTE: Keep in sync with LiteRT stable releases.
_DEFAULT_DDP_LITERT_VERSION = "2.2.0"
_ENV_DDP_LITERT_VERSION = "DDP_LITERT_VERSION"
_GCS_INPUTS_PREFIX = "litert-cli/inputs"
_GCS_SESSIONS_PREFIX = "litert-cli/sessions"
_RESULT_FILE = "results.pb"
_RUNTIME_INFO_FILE = "runtime_info.pb"
_RESULT_FILES = (_RESULT_FILE, _RUNTIME_INFO_FILE)
_POLL_INTERVAL_SECS = 15
# Socket timeout of one Device Run API request.
_HTTP_TIMEOUT_SECS = 60
# Added to max_secs x number of devices to get the default --timeout.
_POLL_TIMEOUT_SLACK_SECS = 600
# Consecutive poll errors after which the CLI stops waiting.
_MAX_POLL_FAILURES = 5
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


def _benchmark_binary() -> str:
  """Returns the GCS path of the prebuilt benchmark_model to run."""
  version = os.environ.get(_ENV_DDP_LITERT_VERSION, _DEFAULT_DDP_LITERT_VERSION)
  return f"gs://litert/binaries/{version}/android_arm64/benchmark_model"


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


def _gcloud(*args: str) -> subprocess.CompletedProcess[str]:
  """Runs a gcloud command with stdout discarded and stderr captured."""
  return subprocess.run(
      ["gcloud", *args],
      check=False,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.PIPE,
      text=True,
  )


def _stderr_tail(stderr: str | None, max_lines: int = 3) -> str:
  """Returns the last non-empty lines of a captured stderr."""
  lines = [line.strip() for line in (stderr or "").splitlines()]
  lines = [line for line in lines if line]
  return "\n".join(lines[-max_lines:])


def _bucket_is_missing(stderr: str) -> bool:
  """Whether a failed `gcloud storage ls` says the bucket does not exist.

  gcloud prints "ERROR: (gcloud.storage.ls) gs://<name> not found: 404." for
  a missing bucket; a permission, project or network problem is worded
  differently and gets no create attempt.
  """
  if "not found" in stderr.lower():
    return True
  return re.search(r"\b404\b", stderr) is not None


def _access_token() -> str:
  """Returns an access token from Application Default Credentials."""
  try:
    return subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        stderr=subprocess.PIPE,
        text=True,
    ).strip()
  except subprocess.CalledProcessError as e:
    raise click.ClickException(
        "Failed to get a gcloud access token. Run 'gcloud auth"
        f" application-default login' first.\n{_stderr_tail(e.stderr)}"
    ) from e


def _ensure_bucket(target_bucket: str, gcp_project: str) -> None:
  """Creates the GCS bucket when it does not exist."""
  click.echo(
      f"Ensuring GCS bucket 'gs://{target_bucket}' exists for project"
      f" '{gcp_project}'..."
  )
  check_res = _gcloud("storage", "ls", f"gs://{target_bucket}")
  if check_res.returncode == 0:
    return
  stderr = check_res.stderr or ""
  if not _bucket_is_missing(stderr):
    raise click.ClickException(
        f"Could not list GCS bucket 'gs://{target_bucket}':\n"
        f"{_stderr_tail(stderr)}"
    )
  click.secho(
      f"Creating GCS bucket 'gs://{target_bucket}' in location"
      f" '{_DEFAULT_BUCKET_LOCATION}'...",
      fg="cyan",
  )
  create_res = _gcloud(
      "storage",
      "buckets",
      "create",
      f"gs://{target_bucket}",
      f"--project={gcp_project}",
      f"--location={_DEFAULT_BUCKET_LOCATION}",
  )
  if create_res.returncode != 0:
    raise click.ClickException(
        f"Failed to create GCS bucket 'gs://{target_bucket}':\n"
        f"{_stderr_tail(create_res.stderr)}"
    )


def _build_benchmark_args(
    *,
    model_name: str,
    accelerator: str,
    num_runs: int,
    warmup_runs: int,
    min_secs: float,
    max_secs: float,
    warmup_min_secs: float,
    input_layer_value_range: str | None,
    signature_key: str | None,
) -> list[str]:
  """Builds benchmark_model arguments.

  The numeric flags are always emitted, so their defaults live only in the
  click options of cli.py.
  """
  root = constants.LITERT_CLI_ANDROID_ROOT
  bench_args = [f"--graph={root}/{model_name}"]
  if accelerator == "gpu":
    bench_args.append("--use_gpu=true")
  bench_args += [
      f"--num_runs={num_runs}",
      f"--warmup_runs={warmup_runs}",
      f"--min_secs={min_secs}",
      f"--max_secs={max_secs}",
      f"--warmup_min_secs={warmup_min_secs}",
  ]
  if input_layer_value_range:
    bench_args.append(f"--input_layer_value_range={input_layer_value_range}")
  if signature_key:
    bench_args.append(f"--signature_to_run_for={signature_key}")
  bench_args.append(f"--result_file_path={root}/{_RESULT_FILE}")
  bench_args.append(
      f"--model_runtime_info_output_file={root}/{_RUNTIME_INFO_FILE}"
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
    benchmark_binary: str,
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
                    "gcsInputFile": {"path": benchmark_binary}
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


def _wait_for_operation(
    op_url: str,
    headers: dict[str, str],
    timeout_secs: float,
    console_url: str,
) -> dict[str, Any]:
  """Polls the operation until it is done and returns its final state.

  The first poll runs right away. A 401 refreshes the access token in
  `headers`; other HTTP errors, connection and socket errors (OSError) and
  truncated or non-JSON responses are retried.

  Args:
    op_url: URL of the long-running operation.
    headers: Request headers; the Authorization entry is replaced on a 401.
    timeout_secs: Seconds to wait before giving up.
    console_url: Cloud Console URL of the session outputs, for messages.

  Returns:
    The operation as returned by the API once `done` is true.

  Raises:
    click.ClickException: after _MAX_POLL_FAILURES consecutive errors, when
      the token cannot be refreshed, or once timeout_secs elapse.
  """
  still_running = (
      "The session may still be running; view it on the Cloud Console:"
      f" {console_url}"
  )
  deadline = time.monotonic() + timeout_secs
  failures = 0
  while True:
    click.echo(".", nl=False)
    req_op = urllib.request.Request(op_url, headers=headers)
    try:
      with urllib.request.urlopen(req_op, timeout=_HTTP_TIMEOUT_SECS) as res_op:
        op_data = json.loads(res_op.read().decode())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
      # urllib wraps only the request in URLError (an OSError); a drop while
      # reading the response raises ConnectionResetError, RemoteDisconnected,
      # IncompleteRead or an SSL error, and a gateway page is not JSON.
      failures += 1
      click.secho(f"\nError polling operation: {e}", fg="yellow")
      if isinstance(e, urllib.error.HTTPError):
        e.close()
      if failures >= _MAX_POLL_FAILURES:
        raise click.ClickException(
            f"Polling failed {failures} times in a row. {still_running}"
        ) from e
      if getattr(e, "code", None) == 401:
        click.echo("Refreshing the access token...")
        try:
          headers["Authorization"] = f"Bearer {_access_token()}"
        except click.ClickException as token_error:
          raise click.ClickException(
              f"{token_error.message}\n{still_running}"
          ) from token_error
    else:
      failures = 0
      if op_data.get("done"):
        click.echo("")  # Print a newline after the dots
        return op_data
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      click.echo("")
      raise click.ClickException(
          f"The session did not finish within {timeout_secs:.0f} s."
          f" {still_running}"
      )
    time.sleep(min(_POLL_INTERVAL_SECS, remaining))


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
) -> list[str]:
  """Downloads each job's output files and prints the benchmark results.

  Returns one line per job that did not pass or whose output files could not
  be downloaded; an empty list when every job passed and downloaded.
  """
  local_root = pathlib.Path(constants.LITERT_CLI_CACHE_DIR) / "ddp" / session_id
  problems: list[str] = []
  for job in session_report.get("jobReports", []):
    job_name = job.get("displayName", job.get("id", "job"))
    job_result = job.get("result", {}).get("resultType", "UNKNOWN")
    passed = job_result == "PASSED"
    click.secho(
        f"\nJob '{job_name}': {job_result}", fg="green" if passed else "red"
    )
    if not passed:
      problems.append(f"Job '{job_name}': {job_result}")
    gcs_paths = [
        f["gcsOutputFile"]["path"]
        for execution in job.get("executionReports", [])
        for f in execution.get("outputFiles", [])
        if "gcsOutputFile" in f
    ]
    if not gcs_paths:
      click.echo("No output files were reported for this job.")
      problems.append(f"Job '{job_name}': no output files reported")
      continue
    local_dir = local_root / job_name
    local_dir.mkdir(parents=True, exist_ok=True)
    download = _gcloud("storage", "cp", *gcs_paths, f"{local_dir}/")
    if download.returncode != 0:
      click.secho(
          f"Failed to download the output files of job '{job_name}':\n"
          f"{_stderr_tail(download.stderr)}",
          fg="red",
      )
      problems.append(f"Job '{job_name}': output files not downloaded")
      continue
    click.echo(f"Output files saved to: {local_dir}")
    logcat_path = local_dir / "logcat.txt"
    if logcat_path.exists():
      _print_logcat_results(logcat_path, passed=passed)
  return problems


def run_ddp(
    model_path_str: str,
    accelerator: str,
    devices: list[str],
    gcp_project: str | None = None,
    gcp_bucket: str | None = None,
    *,
    num_runs: int,
    warmup_runs: int,
    min_secs: float,
    max_secs: float,
    warmup_min_secs: float,
    input_layer_value_range: str | None,
    signature_key: str | None,
    timeout: int | None,
) -> None:
  """Runs the model on DDP devices via the Device Run API.

  Uploads model to GCS if it's not already there.
  Submits a Device Run session that runs benchmark_model on each device.
  Polls the session operation, then downloads and prints the results.
  Raises click.ClickException (exit code 1) when any step fails or when a
  job does not pass.

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
    timeout: Seconds to wait for the session; None means max_secs times the
      number of devices plus _POLL_TIMEOUT_SLACK_SECS.
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
      raise click.ClickException(f"Local model file not found: {model_path}")
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

  _ensure_bucket(target_bucket, gcp_project)

  # The session name namespaces the uploaded model, so concurrent runs and
  # same-named models from different directories never share an object.
  session_name = f"litert-cli-benchmark-{uuid.uuid4().hex[:8]}"
  if local_model is not None:
    model_gcs_dir = f"gs://{target_bucket}/{_GCS_INPUTS_PREFIX}/{session_name}"
    click.secho(
        f"Uploading local model '{model_path}' to {model_gcs_dir}/...",
        fg="cyan",
    )
    try:
      subprocess.run(
          ["gcloud", "storage", "cp", str(local_model), f"{model_gcs_dir}/"],
          check=True,
      )
    except subprocess.CalledProcessError as e:
      raise click.ClickException(
          f"Failed to upload '{model_path}' to {model_gcs_dir}/ (gcloud"
          f" storage cp exited with {e.returncode})."
      ) from e
    model_path = f"{model_gcs_dir}/{model_name}"

  output_dir = f"gs://{target_bucket}/{_GCS_SESSIONS_PREFIX}"
  if timeout is None:
    timeout = int(max_secs * len(device_list)) + _POLL_TIMEOUT_SLACK_SECS

  click.echo("Fetching GCP access token...")
  token = _access_token()

  endpoint = os.environ.get(
      "DEVICE_RUN_ENDPOINT", _DEFAULT_DEVICE_RUN_ENDPOINT
  ).rstrip("/")
  url = _get_sessions_url(endpoint, gcp_project, _DEFAULT_DDP_LOCATION)
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
      "X-Goog-User-Project": gcp_project,
  }

  benchmark_binary = _benchmark_binary()
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
      benchmark_binary=benchmark_binary,
      bench_args=bench_args,
  )

  # Submit the session via http requests to the Device Run API.
  req = urllib.request.Request(
      url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
  )
  click.echo(
      f"Submitting '{accelerator}' benchmark session '{session_name}' to DDP"
      f" (Project: {gcp_project}, Devices: {', '.join(device_list)},"
      f" Binary: {benchmark_binary})..."
  )
  try:
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECS) as response:
      resp_data = json.loads(response.read().decode())
  except urllib.error.HTTPError as e:
    try:
      err_body = e.read().decode(errors="replace")
    except (OSError, http.client.HTTPException):
      err_body = ""
    message = f"Failed to submit benchmark: {e.code} {e.reason}\n{err_body}"
    if e.code in (400, 404) or "device" in err_body.lower():
      message += f"\n{_DEVICE_LIST_HINT}"
    raise click.ClickException(message) from e
  except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
    raise click.ClickException(f"Failed to submit benchmark: {e}") from e
  click.secho("Benchmark session submitted successfully!", fg="green")

  op_name = resp_data.get("name", "")
  if "/operations/" not in op_name:
    click.echo(json.dumps(resp_data, indent=2))
    raise click.ClickException(
        "The Device Run API returned no operation to wait for (response"
        " above)."
    )
  op_url = _get_operation_url(endpoint, op_name)
  session_id = (
      resp_data.get("metadata", {}).get("target", "").rsplit("/", 1)[-1]
  )
  if session_id:
    console_url = _get_console_url(
        target_bucket, _GCS_SESSIONS_PREFIX, session_id
    )
  else:
    # Without the session id, point at the sessions folder and keep the
    # local output directory under the CLI's own session name.
    session_id = session_name
    console_url = _get_console_url(target_bucket, _GCS_SESSIONS_PREFIX, "")
    click.secho(
        "The operation metadata did not name the session; using the display"
        f" name '{session_name}' for the local output directory.",
        fg="yellow",
    )
  click.echo(
      f"Waiting for session '{session_id}' to complete (Operation: {op_name},"
      f" timeout: {timeout} s). This may take a few minutes..."
  )
  click.secho(
      f"View the outputs on the Cloud Console: {console_url}", fg="cyan"
  )

  try:
    op_data = _wait_for_operation(op_url, headers, timeout, console_url)
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
    raise click.Abort() from None

  if "error" in op_data:
    raise click.ClickException(
        f"Benchmark failed: {json.dumps(op_data['error'], indent=2)}"
    )
  session_report = op_data.get("response", {}).get("sessionReport", {})
  result = session_report.get("result", {}).get("resultType", "UNKNOWN")
  click.secho(
      f"Session '{session_id}' finished: {result}",
      fg="green" if result == "PASSED" else "red",
  )
  problems = _fetch_session_outputs(session_report, session_id)
  if result != "PASSED" and not problems:
    problems.append(f"Session '{session_id}' finished: {result}")
  if problems:
    problems.append(f"Session outputs on the Cloud Console: {console_url}")
    raise click.ClickException("\n".join(problems))
