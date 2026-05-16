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

"""Benchmarking on Google AI Edge Portal in GCP."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import click

_DEFAULT_GCP_PROJECT = os.environ.get("LITERT_GCP_PROJECT")
_DEFAULT_GCP_LOCATION = "us-central1"
_GCP_BUCKET = os.environ.get("LITERT_GCP_BUCKET")
_DEFAULT_PORTAL_ENDPOINT = "https://aiedgeportal.googleapis.com/v1alpha"


def _get_submission_url(
    portal_endpoint: str, gcp_project: str, gcp_location: str, job_id: str
) -> str:
  """Returns the URL for submitting a benchmark job."""
  return (
      f"{portal_endpoint}/projects/{gcp_project}/locations/{gcp_location}"
      f"/benchmarks?benchmarkId={job_id}"
  )


def _get_operation_url(portal_endpoint: str, op_name: str) -> str:
  """Returns the URL for polling operation status."""
  return f"{portal_endpoint}/{op_name}"


def _get_console_url(
    gcp_project: str, gcp_location: str, benchmark_id: str
) -> str:
  """Returns the Google Cloud Console URL for viewing progress."""
  return (
      "https://console.cloud.google.com/ai-edge-portal/benchmarks/details/"
      f"{gcp_location}/{benchmark_id}?project={gcp_project}"
  )


def run_gcp(
    model_path_str: str,
    accelerator: str,
    devices: list[str],
    gcp_project: str | None = None,
    gcp_bucket: str | None = None,
    compilation_mode: str | None = None,
    soc_model: str | None = None,
) -> None:
  """Runs the model on GCP via AI Edge Portal Cloud API.

  Uploads model to GCS if it's not already there.
  Submits benchmark job to AI Edge Portal Cloud API.
  Polls benchmark job status.

  Args:
    model_path_str: Path to the LiteRT model file (local or gs://).
    accelerator: Hardware accelerator to use (cpu, gpu, npu).
    devices: Target device model(s) (e.g., 'pixel 7', 'sm-s931u1').
    gcp_project: GCP project ID for benchmarking.
    gcp_bucket: GCS bucket name for uploading model.
    compilation_mode: Compilation mode for NPU (jit, aot).
    soc_model: Target SoC model for NPU AOT mode.
  """

  device_list = []
  if isinstance(devices, str):
    items = [devices]
  else:
    items = devices

  for item in items:
    if item:
      parts = [p.strip() for p in item.split(",") if p.strip()]
      device_list.extend(parts)

  if not device_list:
    raise click.ClickException(
        "Error: --device is required for running GCP benchmark tests."
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
  # Upload model to GCS if it's not already there.
  if not model_path.startswith("gs://"):
    local_model = pathlib.Path(model_path)
    if not local_model.exists():
      click.secho(f"Error: Local model file not found: {model_path}", fg="red")
      return

    target_bucket = gcp_bucket or _GCP_BUCKET
    if not target_bucket:
      target_bucket = f"{gcp_project}-litert-models"
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
            f" '{_DEFAULT_GCP_LOCATION}'...",
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
                f"--location={_DEFAULT_GCP_LOCATION}",
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

    click.secho(
        f"Uploading local model '{model_path}' to gs://{target_bucket}/...",
        fg="cyan",
    )
    try:
      subprocess.run(
          [
              "gcloud",
              "storage",
              "cp",
              str(local_model),
              f"gs://{target_bucket}/",
          ],
          check=True,
      )
      model_path = f"gs://{target_bucket}/{local_model.name}"
    except subprocess.CalledProcessError as e:
      click.secho(
          f"Error: Failed to upload '{model_path}' to Google Cloud Storage:"
          f" {e}",
          fg="red",
      )
      return

  job_id = f"litert-cli-benchmark-{uuid.uuid4().hex[:8]}"

  click.echo("Fetching GCP access token...")
  try:
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
  except subprocess.CalledProcessError as e:
    click.secho(
        "Error: Failed to get gcloud access token. Please run 'gcloud auth"
        f" login' first. Details: {e}",
        fg="red",
    )
    return

  portal_endpoint = os.environ.get(
      "AI_EDGE_PORTAL_ENDPOINT", _DEFAULT_PORTAL_ENDPOINT
  ).rstrip("/")
  url = _get_submission_url(
      portal_endpoint, gcp_project, _DEFAULT_GCP_LOCATION, job_id
  )
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
      "X-Goog-User-Project": gcp_project,
  }

  accel_name = accelerator.upper()

  run_spec: dict[str, any] = {
      "accelerator": accel_name,
      "id": accelerator.lower(),
      "displayName": f"{accelerator.lower()}_test",
      "runtimeVersion": "litert-v2.0.3",
  }

  if accel_name == "NPU":
    comp_mode = (compilation_mode or "jit").upper()
    if comp_mode == "AOT":
      if not soc_model:
        raise click.ClickException(
            "Error: --soc-model is required when using NPU AOT compilation mode."
        )
      run_spec["npuConfig"] = {
          "npuCompilationMode": "AOT",
          "socConfigs": [{
              "socModel": soc_model,
              "aotModelPath": model_path.replace("gs://", ""),
          }],
          "cpuFallbackConfig": {"threadCount": 4},
      }
    else:
      run_spec["modelPath"] = model_path.replace("gs://", "")
      run_spec["npuConfig"] = {
          "npuCompilationMode": "JIT",
          "cpuFallbackConfig": {"threadCount": 4},
      }
  else:
    run_spec["modelPath"] = model_path.replace("gs://", "")

  body = {
      "displayName": job_id,
      "modelPaths": [],
      "deviceConfigs": [{"deviceModel": d} for d in device_list],
      "runSpecs": [run_spec],
  }

  # Submit the benchmark job via http requests to AI Edge Portal Cloud API.
  req = urllib.request.Request(
      url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
  )
  click.echo(
      f"Submitting '{accelerator}' benchmark job '{job_id}' to AI Edge Portal"
      f" (Project: {gcp_project}, Location:"
      f" {_DEFAULT_GCP_LOCATION})..."
  )

  try:
    with urllib.request.urlopen(req) as response:
      resp_data = json.loads(response.read().decode())
      click.secho("Benchmark job submitted successfully!", fg="green")

      op_name = resp_data.get("name", "")
      if op_name and "operations" in op_name:
        op_url = _get_operation_url(portal_endpoint, op_name)
        console_url = _get_console_url(
            gcp_project, _DEFAULT_GCP_LOCATION, job_id
        )
        click.echo(
            f"Waiting for benchmark to complete (Operation: {op_name}). This"
            " may take a few minutes..."
        )
        click.secho(
            f"View progress on the Cloud Console: {console_url}", fg="cyan"
        )

        try:
          while True:
            time.sleep(15)
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
                click.secho(
                    "Benchmark operation completed successfully!", fg="green"
                )
                click.echo(json.dumps(op_data.get("response", {}), indent=2))
              break
        except KeyboardInterrupt:
          click.echo("")
          click.secho(
              "\nPolling interrupted. The benchmark job is still running.",
              fg="yellow",
          )
          click.echo(
              f"You can check its status later by viewing it in the console:"
              " {console_url}"
          )
      else:
        click.echo(json.dumps(resp_data, indent=2))
  except urllib.error.HTTPError as e:
    err_body = e.read().decode()
    click.secho(f"Failed to submit benchmark: {e.code} {e.reason}", fg="red")
    click.secho(f"Details: {err_body}", fg="red")
