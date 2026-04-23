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

# This is internal experimental code, not ready for general public.
# TODO: b/493604945 - change this as public APIs later.
_DEFAULT_GCP_PROJECT = "aep-e2e-test"
_DEFAULT_GCP_LOCATION = "us-central1"
_GCP_BUCKET = os.environ.get("LITERT_GCP_BUCKET", "litert-cli-test")
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
    device: str,
) -> None:
  """Runs the model on GCP via AI Edge Portal Cloud API.

  Uploads model to GCS if it's not already there.
  Submits benchmark job to AI Edge Portal Cloud API.
  Polls benchmark job status.

  Args:
    model_path_str: Path to the LiteRT model file (local or gs://).
    accelerator: Hardware accelerator to use (cpu, gpu, npu).
    device: Target device model (e.g., 'pixel 7').
  """
  model_path = model_path_str
  # Upload model to GCS if it's not already there.
  if not model_path.startswith("gs://"):
    local_model = pathlib.Path(model_path)
    if not local_model.exists():
      click.secho(f"Error: Local model file not found: {model_path}", fg="red")
      return

    click.secho(
        f"Uploading local model '{model_path}' to"
        f" gs://{_GCP_BUCKET}/...",
        fg="cyan",
    )
    try:
      subprocess.run(
          [
              "gcloud",
              "storage",
              "cp",
              str(local_model),
              f"gs://{_GCP_BUCKET}/",
          ],
          check=True,
      )
      model_path = f"gs://{_GCP_BUCKET}/{local_model.name}"
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
      portal_endpoint, _DEFAULT_GCP_PROJECT, _DEFAULT_GCP_LOCATION, job_id
  )
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
  }

  accel_name = accelerator.upper()

  body = {
      "display_name": job_id,
      "device_configs": [{
          "device_model": device,
      }],
      "run_specs": [{
          "accelerator": accel_name,
          "model_path": model_path.replace("gs://", ""),
          "id": accelerator.lower(),
          "display_name": f"{accelerator.lower()}_test",
      }],
  }

  # Submit the benchmark job via http requests to AI Edge Portal Cloud API.
  req = urllib.request.Request(
      url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
  )
  click.echo(
      f"Submitting '{accelerator}' benchmark job '{job_id}' to AI Edge Portal"
      f" (Project: {_DEFAULT_GCP_PROJECT}, Location:"
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
            _DEFAULT_GCP_PROJECT, _DEFAULT_GCP_LOCATION, job_id
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
