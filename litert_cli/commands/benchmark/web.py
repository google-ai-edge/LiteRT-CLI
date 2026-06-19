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

"""Web benchmarking orchestration for LiteRT CLI.

Starts a local HTTP server to serve LiteRT.js Model Tester and the model,
and opens a browser tab to run the benchmark.
"""

from __future__ import annotations

import http.server
from importlib import resources
import os
import socket
import sys
import webbrowser

import click


class LiteRtWebBenchmarkHandler(http.server.SimpleHTTPRequestHandler):
  """Custom HTTP handler to serve Model Tester static assets and the model."""

  def __init__(
      self,
      *args,
      model_path: str | None = None,
      **kwargs,
  ):
    self.model_path = model_path
    super().__init__(*args, **kwargs)

  def end_headers(self):
    # Inject crucial headers for WebAssembly SharedArrayBuffer support (multi-threading)
    self.send_header("Cross-Origin-Opener-Policy", "same-origin")
    self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
    self.send_header("Access-Control-Allow-Origin", "*")
    super().end_headers()

  def do_GET(self):
    # 1. Virtual route to serve the target model file
    if self.path.startswith("/model.tflite"):
      if not self.model_path or not os.path.exists(self.model_path):
        self.send_error(404, "Model file not found")
        return
      self.send_response(200)
      self.send_header("Content-Type", "application/octet-stream")
      self.send_header("Content-Length", str(os.path.getsize(self.model_path)))
      self.end_headers()
      with open(self.model_path, "rb") as f:
        self.wfile.write(f.read())
      return

    # 2. Route for the main entry point (serve index.html from package resources)
    if self.path == "/" or self.path.startswith("/?"):
      try:
        index_content = (
            resources.files("litert_cli.assets")
            .joinpath("index.html")
            .read_bytes()
        )
      except Exception as e:
        self.send_error(
            500, f"Static index.html not found in package resources: {e}"
        )
        return
      self.send_response(200)
      self.send_header("Content-Type", "text/html")
      self.send_header("Content-Length", str(len(index_content)))
      self.end_headers()
      self.wfile.write(index_content)
      return

    # 3. Route for the compiled JS bundle (serve from package resources)
    if self.path.startswith("/bundle.js"):
      try:
        bundle_content = (
            resources.files("litert_cli.assets")
            .joinpath("bundle.js")
            .read_bytes()
        )
      except Exception as e:
        self.send_error(404, f"JS bundle not found in package resources: {e}")
        return
      self.send_response(200)
      self.send_header("Content-Type", "application/javascript")
      self.send_header("Content-Length", str(len(bundle_content)))
      self.end_headers()
      self.wfile.write(bundle_content)
      return

    # 4. Route for WASM helper files (serve from package resources)
    if self.path.startswith("/wasm/"):
      wasm_file = self.path.split("/wasm/", 1)[1]
      wasm_file = wasm_file.split("?", 1)[0]
      wasm_file = wasm_file.split("#", 1)[0]

      try:
        wasm_content = (
            resources.files("litert_cli.assets")
            .joinpath("wasm", wasm_file)
            .read_bytes()
        )
      except Exception as e:
        self.send_error(404, f"WASM file not found in package resources: {e}")
        return

      self.send_response(200)
      if wasm_file.endswith(".js"):
        self.send_header("Content-Type", "application/javascript")
      elif wasm_file.endswith(".wasm"):
        self.send_header("Content-Type", "application/wasm")
      self.send_header("Content-Length", str(len(wasm_content)))
      self.end_headers()
      self.wfile.write(wasm_content)
      return

    # 5. Default fallback: Send 404 (More secure than serving arbitrary files)
    self.send_error(404, f"File not found: {self.path}")


def run_web_benchmark(model_path: str) -> None:
  """Starts local HTTP server and opens browser to benchmark the model."""
  abs_model_path = os.path.abspath(model_path)
  if not os.path.exists(abs_model_path):
    raise click.ClickException(f"Model file not found: {model_path}")

  # Find an available ephemeral port
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(("localhost", 0))
  port = s.getsockname()[1]
  s.close()

  # Create custom handler class closed over variables
  class ThreadSafeHandler(LiteRtWebBenchmarkHandler):

    def __init__(self, *args, **kwargs):
      super().__init__(
          *args,
          model_path=abs_model_path,
          **kwargs,
      )

  server = http.server.HTTPServer(("localhost", port), ThreadSafeHandler)

  url = f"http://localhost:{port}/?modelUrl=/model.tflite"
  click.secho(f"Starting local benchmark server at {url}", fg="green")
  click.secho("Press Ctrl+C to stop the server.", fg="cyan")

  # Open default browser in a new tab
  webbrowser.open(url)

  try:
    server.serve_forever()
  except KeyboardInterrupt:
    click.echo("\nStopping web benchmark server...")
    server.server_close()
    sys.exit(0)
