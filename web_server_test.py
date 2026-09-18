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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import http.server
import socket
import threading
import urllib.error
import urllib.request
from absl.testing import absltest
from litert_cli.commands.benchmark.web import LiteRtWebBenchmarkHandler


class WebServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Find an available port
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.bind(("localhost", 0))
    self.port = self.sock.getsockname()[1]
    self.sock.close()

    # We need a dummy model file path for the handler.
    # We use a real test model from the runfiles.
    self.model_path = "third_party/odml/litert/litert/test/testdata/mobilenet_v2_1.0_224.tflite"

    # Create a subclass of the handler to close over the model_path
    model_path = self.model_path

    class TestHandler(LiteRtWebBenchmarkHandler):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, model_path=model_path, **kwargs)

    self.server = http.server.HTTPServer(("localhost", self.port), TestHandler)
    self.server_thread = threading.Thread(target=self.server.serve_forever)
    self.server_thread.daemon = True
    self.server_thread.start()

  def tearDown(self):
    self.server.shutdown()
    self.server.server_close()
    self.server_thread.join()
    super().tearDown()

  def test_serve_index_html(self):
    url = f"http://localhost:{self.port}/"
    response = urllib.request.urlopen(url, timeout=5)
    self.assertEqual(response.status, 200)
    self.assertIn(b"model-tester", response.read())
    self.assertEqual(response.headers.get("Content-Type"), "text/html")

  def test_serve_bundle_js(self):
    url = f"http://localhost:{self.port}/bundle.js"
    response = urllib.request.urlopen(url, timeout=5)
    self.assertEqual(response.status, 200)
    self.assertEqual(
        response.headers.get("Content-Type"), "application/javascript"
    )
    body = response.read()
    self.assertIn(b"run-result-visualizer", body)

  def test_serve_wasm_files(self):
    # Test one JS and one WASM file
    url_js = f"http://localhost:{self.port}/wasm/litert_wasm_jspi_internal.js"
    response_js = urllib.request.urlopen(url_js, timeout=5)
    self.assertEqual(response_js.status, 200)
    self.assertEqual(
        response_js.headers.get("Content-Type"), "application/javascript"
    )
    self.assertIn(b"Module", response_js.read())

    url_wasm = (
        f"http://localhost:{self.port}/wasm/litert_wasm_jspi_internal.wasm"
    )
    response_wasm = urllib.request.urlopen(url_wasm, timeout=5)
    self.assertEqual(response_wasm.status, 200)
    self.assertEqual(
        response_wasm.headers.get("Content-Type"), "application/wasm"
    )
    self.assertTrue(response_wasm.read().startswith(b"\x00asm"))

  def test_serve_model(self):
    url = f"http://localhost:{self.port}/model.tflite"
    response = urllib.request.urlopen(url, timeout=5)
    self.assertEqual(response.status, 200)
    self.assertEqual(
        response.headers.get("Content-Type"), "application/octet-stream"
    )

  def test_serve_invalid_path_returns_404(self):
    url = f"http://localhost:{self.port}/invalid_file.txt"
    with self.assertRaises(urllib.error.HTTPError) as cm:
      urllib.request.urlopen(url, timeout=5)
    self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
  absltest.main()
