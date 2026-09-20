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
import statistics
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

import click
from litert_cli.core import constants
from litert_cli.core.log_filters import BenchmarkLogFilter
from litert_cli.core.log_filters import LmBenchmarkLogFilter

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
# A .litertlm bundle runs LiteRT-LM's prebuilt benchmark binary instead of
# benchmark_model. The binaries are published under
# gs://litert/binaries/<version>/android_arm64/litert_lm/; `latest` is the only
# version there so far.
_LM_BUNDLE_SUFFIX = ".litertlm"
_DEFAULT_DDP_LITERT_LM_VERSION = "latest"
_ENV_DDP_LITERT_LM_VERSION = "DDP_LITERT_LM_VERSION"
# A whole gs:// directory laid out like the published one, to try a
# directory before it is published.
_ENV_DDP_LITERT_LM_DIR = "DDP_LITERT_LM_DIR"
_LM_BINARY = "litert_lm_advanced_main"
_LM_METRICS_FILE = "metrics.pb"
_LM_PROVENANCE_FILE = "provenance.txt"
_LM_RESULT_FILES = (_LM_METRICS_FILE, _LM_PROVENANCE_FILE)
# Init plus --num-iterations prefill and decode cycles; the device stops the
# job after this long.
_LM_EXECUTION_TIMEOUT_SECS = 1800
_LM_RUN_SCRIPT_NAME = "litert_lm_run.sh"
# Runs on the device as the job's binary. Runs LiteRT-LM's benchmark binary
# from the CLI's directory with that directory on LD_LIBRARY_PATH (the
# accelerator libraries pushed beside the binary load from there), removes the
# caches an earlier session left beside the bundle so Init is a cold start, and
# writes provenance.txt. Every argument goes to the binary unchanged, and the
# binary's exit code is the job's.
_LM_RUN_SCRIPT = """#!/system/bin/sh
ROOT="{root}"
cd "$ROOT" || exit 1
chmod 755 {binary} || exit 1
rm -f ./*.xnnpack_cache* ./*_mldrift_*cache*.bin ./*.mtp_drafter*
MODEL=""
for arg in "$@"; do
  case "$arg" in --model_path=*) MODEL="${{arg#*=}}" ;; esac
done
{{
  echo "date: $(date)"
  echo "product: $(getprop ro.product.model) ($(getprop ro.product.device))"
  echo "build: $(getprop ro.build.fingerprint)"
  echo "args: $*"
  echo "sha256:"
  sha256sum {binary} "$MODEL" ./*.so 2>/dev/null
}} > {provenance}
LD_LIBRARY_PATH="$ROOT" exec ./{binary} "$@"
"""
# One BenchmarkInfo block per iteration in the logcat; the values the summary
# reads from each block (the first prefill and decode turn).
_LM_BLOCK_START = "BenchmarkInfo:"
_LM_AGGREGATED_BLOCK = "Aggregated BenchmarkInfo"
_LM_METRIC_PATTERNS = {
    "init_ms": re.compile(r"Init Total: ([\d.]+) ms"),
    "ttft_s": re.compile(r"Time to first token: ([\d.]+) s"),
    "prefill_tok_s": re.compile(r"Prefill Speed: ([\d.]+) tokens/sec"),
    "decode_tok_s": re.compile(r"Decode Speed: ([\d.]+) tokens/sec"),
}
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


def is_lm_bundle(model_path: str) -> bool:
  """Whether the model is a LiteRT-LM .litertlm bundle."""
  return model_path.lower().endswith(_LM_BUNDLE_SUFFIX)


def _lm_binary_dir() -> str:
  """Returns the GCS directory of the prebuilt LiteRT-LM binaries.

  DDP_LITERT_LM_DIR names the whole directory; otherwise
  DDP_LITERT_LM_VERSION names the version under gs://litert/binaries/.
  """
  directory = os.environ.get(_ENV_DDP_LITERT_LM_DIR)
  if directory:
    return directory.rstrip("/")
  version = os.environ.get(
      _ENV_DDP_LITERT_LM_VERSION, _DEFAULT_DDP_LITERT_LM_VERSION
  )
  return f"gs://litert/binaries/{version}/android_arm64/litert_lm"


def _lm_run_script() -> str:
  """The device-side script that runs LiteRT-LM's benchmark binary."""
  return _LM_RUN_SCRIPT.format(
      root=constants.LITERT_CLI_ANDROID_ROOT,
      binary=_LM_BINARY,
      provenance=_LM_PROVENANCE_FILE,
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


def _build_lm_benchmark_args(
    *,
    model_name: str,
    accelerator: str,
    prefill_tokens: int,
    decode_tokens: int,
    max_num_tokens: int,
    num_iterations: int,
) -> list[str]:
  """Builds the LiteRT-LM benchmark binary's arguments for a bundle.

  The token counts and the iteration count are always emitted, so their
  defaults live only in the click options of cli.py.
  """
  root = constants.LITERT_CLI_ANDROID_ROOT
  return [
      f"--backend={accelerator}",
      f"--model_path={root}/{model_name}",
      "--benchmark=true",
      f"--benchmark_prefill_tokens={prefill_tokens}",
      f"--benchmark_decode_tokens={decode_tokens}",
      f"--max_num_tokens={max_num_tokens}",
      f"--num_iterations={num_iterations}",
      f"--metric_proto_file_path={root}/{_LM_METRICS_FILE}",
  ]


def _lm_pushes(binary_dir: str) -> list[tuple[str, str]]:
  """The prebuilt files a bundle's job pushes beside the bundle.

  Lists the LiteRT-LM binaries directory and returns (GCS path, file name)
  pairs: the benchmark binary first, then every shared library there. The
  binary loads its libraries (the OpenCL accelerator and sampler, the NPU
  dispatch libraries, and the library it links at start) from its own
  directory on the device, so the directory is pushed as published.
  """
  listing = subprocess.run(
      ["gcloud", "storage", "ls", f"{binary_dir}/"],
      check=False,
      capture_output=True,
      text=True,
  )
  if listing.returncode != 0:
    raise click.ClickException(
        f"Could not list the LiteRT-LM binaries at {binary_dir}/:\n"
        f"{_stderr_tail(listing.stderr)}"
    )
  names = [
      line.strip().rsplit("/", 1)[-1]
      for line in listing.stdout.splitlines()
      if line.strip() and not line.strip().endswith("/")
  ]
  if _LM_BINARY not in names:
    raise click.ClickException(
        f"{_LM_BINARY} is not under {binary_dir}/ ({_ENV_DDP_LITERT_LM_VERSION}"
        f" picks the version, {_ENV_DDP_LITERT_LM_DIR} the directory)."
    )
  libraries = sorted(name for name in names if name.endswith(".so"))
  return [(f"{binary_dir}/{name}", name) for name in [_LM_BINARY] + libraries]


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
    extra_pushes: list[tuple[str, str]] | None = None,
    result_files: tuple[str, ...] = _RESULT_FILES,
    execution_timeout_secs: int | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
  """Builds the Device Run session request: one job per device.

  Args:
    session_name: The session's display name.
    model_gcs_path: GCS path of the model, pushed to the CLI's directory.
    model_name: The model's file name on the device.
    accelerator: cpu or gpu.
    device_list: DDP device ids, one job each.
    output_dir: GCS directory the session's outputs go to.
    benchmark_binary: GCS path of the job's binary.
    bench_args: Arguments of the job's binary.
    extra_pushes: (GCS path, file name) pairs pushed beside the model.
    result_files: File names pulled from the CLI's directory after the run.
    execution_timeout_secs: Seconds after which the device stops the job; None
      leaves the platform's default.
    runtime: A runtime label for the job, when not benchmark_model.
  """
  root = constants.LITERT_CLI_ANDROID_ROOT
  pushes = [(model_gcs_path, model_name)] + list(extra_pushes or [])
  native_binary: dict[str, Any] = {
      "androidNativeBinary": {"gcsInputFile": {"path": benchmark_binary}},
      "args": bench_args,
  }
  if execution_timeout_secs is not None:
    native_binary["executionTimeout"] = f"{execution_timeout_secs}s"
  labels = {
      "tool": "litert-cli",
      "accelerator": accelerator,
      "model": model_name,
  }
  if runtime:
    labels["runtime"] = runtime
  job_configs = []
  for device_id in device_list:
    job_configs.append({
        "displayName": _display_name(accelerator, device_id),
        "action": {"androidNativeBinary": native_binary},
        "allocationConfig": {
            "deviceConfigs": [{
                "requirement": {"deviceId": device_id},
                "actions": [
                    {
                        "androidPushFiles": {
                            "fileConfigs": [
                                {
                                    "sourceFile": {
                                        "gcsInputFile": {"path": gcs_path}
                                    },
                                    "destinationPath": f"{root}/{name}",
                                }
                                for gcs_path, name in pushes
                            ]
                        }
                    },
                    {
                        "androidPullFiles": {
                            "paths": [f"{root}/{f}" for f in result_files]
                        }
                    },
                    {"androidLogcat": {}},
                ],
            }]
        },
        "labels": labels,
    })
  return {
      "sessionConfig": {
          "displayName": session_name,
          "outputDirectoryConfig": {"gcsOutputDirectory": {"path": output_dir}},
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


def _lm_iterations(lines: list[str]) -> list[dict[str, float]]:
  """One dict per BenchmarkInfo block of a LiteRT-LM logcat.

  The binary logs one block per iteration and, after them, an aggregated
  block over every iteration, which is skipped. A block without a prefill
  and a decode speed is dropped.
  """
  iterations: list[dict[str, float]] = []
  current: dict[str, float] | None = None
  for line in lines:
    if _LM_AGGREGATED_BLOCK in line:
      current = None
      continue
    if line.rstrip().endswith(_LM_BLOCK_START):
      current = {}
      iterations.append(current)
      continue
    if current is None:
      continue
    for key, pattern in _LM_METRIC_PATTERNS.items():
      match = pattern.search(line)
      if match and key not in current:
        current[key] = float(match.group(1))
  return [i for i in iterations if "prefill_tok_s" in i and "decode_tok_s" in i]


def _lm_summary(lines: list[str], warmup_iterations: int) -> list[str]:
  """The medians over the iterations after the warm-up ones, as text lines.

  Empty when the logcat holds no measured iteration beyond the warm-up.
  """
  iterations = _lm_iterations(lines)
  measured = iterations[warmup_iterations:]
  if not measured:
    return []

  def median(key: str) -> float | None:
    values = [i[key] for i in measured if key in i]
    return statistics.median(values) if values else None

  prefill, decode, ttft = (
      median("prefill_tok_s"),
      median("decode_tok_s"),
      median("ttft_s"),
  )
  init = iterations[0].get("init_ms")
  summary = [
      (
          f"LiteRT-LM benchmark: {len(iterations)} iteration(s),"
          f" {warmup_iterations} warm-up; median of the other {len(measured)}:"
      ),
      f"  prefill {prefill:.1f} tokens/s, decode {decode:.1f} tokens/s"
      + (f", time to first token {ttft:.2f} s" if ttft is not None else "")
      + (f"; init {init:.0f} ms (once per run)" if init is not None else ""),
  ]
  return summary


def _print_logcat_results(
    logcat_path: pathlib.Path,
    passed: bool,
    *,
    lm: bool = False,
    warmup_iterations: int = 0,
) -> None:
  """Prints the benchmark lines of a logcat, or its tail when the job failed.

  For a LiteRT-LM run (`lm`), the benchmark lines are the binary's
  BenchmarkInfo blocks, followed by the medians over the iterations after the
  first `warmup_iterations`.
  """
  lines = logcat_path.read_text(errors="replace").splitlines()
  if passed:
    log_filter = (
        LmBenchmarkLogFilter(constants.DEFAULT_QUIET)
        if lm
        else BenchmarkLogFilter(constants.DEFAULT_QUIET)
    )
    for line in lines:
      if log_filter.should_show(line):
        click.echo(line)
    if lm:
      for line in _lm_summary(lines, warmup_iterations):
        click.secho(line, fg="green")
  else:
    click.echo(f"Last {_LOGCAT_TAIL_LINES} lines of {logcat_path.name}:")
    for line in lines[-_LOGCAT_TAIL_LINES:]:
      click.echo(line)


def _upload(local_path: str, gcs_dir: str, what: str) -> None:
  """Uploads a local file into a GCS directory with gcloud."""
  click.secho(f"Uploading {what} '{local_path}' to {gcs_dir}/...", fg="cyan")
  try:
    subprocess.run(
        ["gcloud", "storage", "cp", local_path, f"{gcs_dir}/"], check=True
    )
  except subprocess.CalledProcessError as e:
    raise click.ClickException(
        f"Failed to upload '{local_path}' to {gcs_dir}/ (gcloud storage cp"
        f" exited with {e.returncode})."
    ) from e


def _fetch_session_outputs(
    session_report: dict[str, Any],
    session_id: str,
    *,
    lm: bool = False,
    warmup_iterations: int = 0,
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
      _print_logcat_results(
          logcat_path,
          passed=passed,
          lm=lm,
          warmup_iterations=warmup_iterations,
      )
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
    prefill_tokens: int = 1024,
    decode_tokens: int = 256,
    max_num_tokens: int = 1280,
    num_iterations: int = 5,
) -> None:
  """Runs the model on DDP devices via the Device Run API.

  Uploads model to GCS if it's not already there.
  Submits a Device Run session that runs benchmark_model on each device, or
  LiteRT-LM's benchmark binary for a .litertlm bundle.
  Polls the session operation, then downloads and prints the results.
  Raises click.ClickException (exit code 1) when any step fails or when a
  job does not pass.

  Args:
    model_path_str: Path to the LiteRT model file or .litertlm bundle (local or
      gs://).
    accelerator: Hardware accelerator to use (cpu, gpu).
    devices: Target DDP device id(s) (e.g., 'caiman-35').
    gcp_project: GCP project ID for benchmarking.
    gcp_bucket: GCS bucket name for the model and the session outputs.
    num_runs: Target number of benchmark iterations.
    warmup_runs: Number of warmup iterations before benchmarking. For a bundle,
      the leading iterations left out of the printed medians.
    min_secs: Minimum seconds to run.
    max_secs: Maximum seconds to run.
    warmup_min_secs: Minimum warmup duration in seconds.
    input_layer_value_range: Value range for input layers.
    signature_key: The signature key to benchmark.
    timeout: Seconds to wait for the session; None means max_secs (the execution
      timeout for a bundle) times the number of devices plus
      _POLL_TIMEOUT_SLACK_SECS.
    prefill_tokens: Prefill tokens of a bundle's benchmark.
    decode_tokens: Decode tokens of a bundle's benchmark.
    max_num_tokens: Context length of a bundle's benchmark.
    num_iterations: Prefill and decode cycles of a bundle's benchmark, in one
      process.
  """
  if accelerator == "npu":
    raise click.ClickException("NPU on --ddp is not supported yet.")

  lm = is_lm_bundle(model_path_str)
  if lm and (input_layer_value_range or signature_key):
    raise click.ClickException(
        "--input-layer-value-range and --signature-key are benchmark_model"
        " options; a .litertlm bundle runs LiteRT-LM's benchmark binary."
    )
  if lm and warmup_runs >= num_iterations:
    raise click.ClickException(
        f"--warmup-runs ({warmup_runs}) must be smaller than --num-iterations"
        f" ({num_iterations}) for a .litertlm bundle."
    )

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
  inputs_gcs_dir = f"gs://{target_bucket}/{_GCS_INPUTS_PREFIX}/{session_name}"
  if local_model is not None:
    _upload(str(local_model), inputs_gcs_dir, "local model")
    model_path = f"{inputs_gcs_dir}/{model_name}"

  output_dir = f"gs://{target_bucket}/{_GCS_SESSIONS_PREFIX}"
  if timeout is None:
    job_secs = _LM_EXECUTION_TIMEOUT_SECS if lm else max_secs
    timeout = int(job_secs * len(device_list)) + _POLL_TIMEOUT_SLACK_SECS

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

  if lm:
    # The job's binary is the run script; it starts LiteRT-LM's benchmark
    # binary, pushed beside the bundle with the libraries of its directory.
    lm_pushes = _lm_pushes(_lm_binary_dir())
    benchmark_binary = lm_pushes[0][0]
    with tempfile.TemporaryDirectory() as tmp_dir:
      script_path = os.path.join(tmp_dir, _LM_RUN_SCRIPT_NAME)
      with open(script_path, "w") as f:
        f.write(_lm_run_script())
      _upload(script_path, inputs_gcs_dir, "run script")
    bench_args = _build_lm_benchmark_args(
        model_name=model_name,
        accelerator=accelerator,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        max_num_tokens=max_num_tokens,
        num_iterations=num_iterations,
    )
    body = _build_session_request(
        session_name=session_name,
        model_gcs_path=model_path,
        model_name=model_name,
        accelerator=accelerator,
        device_list=device_list,
        output_dir=output_dir,
        benchmark_binary=f"{inputs_gcs_dir}/{_LM_RUN_SCRIPT_NAME}",
        bench_args=bench_args,
        extra_pushes=lm_pushes,
        result_files=_LM_RESULT_FILES,
        execution_timeout_secs=_LM_EXECUTION_TIMEOUT_SECS,
        runtime="litert-lm",
    )
  else:
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
        "The Device Run API returned no operation to wait for (response above)."
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
  problems = _fetch_session_outputs(
      session_report, session_id, lm=lm, warmup_iterations=warmup_runs
  )
  if result != "PASSED" and not problems:
    problems.append(f"Session '{session_id}' finished: {result}")
  if problems:
    problems.append(f"Session outputs on the Cloud Console: {console_url}")
    raise click.ClickException("\n".join(problems))
