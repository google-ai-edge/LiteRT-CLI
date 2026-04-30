"""Shared runner for ASR models in litert-cli."""

from __future__ import annotations

import math
import os
import queue
import sys
import time
from typing import Any, TYPE_CHECKING

import click
from litert_cli.core import deps
import numpy as np

if TYPE_CHECKING:
  from ai_edge_litert.compiled_model import CompiledModel  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.tensor_buffer import TensorBuffer  # pylint: disable=g-import-not-at-top


class ASRRunner:
  """A runner that handles LiteRT execution for both CTC and Encoder-Decoder models.

  Centralizes the logic of path resolution, loading CompiledModel,
  running the encoder, detecting the 'decode' signature, and running
  the auto-regressive loop if present.
  """

  def __init__(self, model_id: str, **kwargs: Any):
    """Initializes the instance.

    Extracts paths from intent parameters.

    Args:
      model_id: The model identifier or path passed from CLI.
      **kwargs: Dynamic arguments containing paths and options.
    """
    # Auto-install ASR dependencies if missing
    deps.ensure_extra("asr")

    # Extract model_path and wav_path
    self.model_path = kwargs.get("model_path") or model_id
    self.wav_path = kwargs.get("wav_path")

    # Parse inputs tuple for wav_path if not directly in kwargs
    inputs = kwargs.get("inputs", [])
    if not self.wav_path and inputs:
      for inp in inputs:
        if inp.startswith("wav_path="):
          self.wav_path = inp.split("=", 1)[1]
          break
        elif "=" not in inp and len(inputs) == 1:
          self.wav_path = inp

    # Default to speaker if still not set
    if not self.wav_path:
      self.wav_path = "speaker"

    if not self.model_path:
      raise ValueError("model_path must be provided via parameter or model_id.")

    from ai_edge_litert.compiled_model import CompiledModel  # pylint: disable=g-import-not-at-top

    click.echo(f"Loading LiteRT model from {self.model_path}...")
    self.litert_model = CompiledModel.from_file(self.model_path)
    self.signatures = self.litert_model.get_signature_list()

  def _get_librosa(self) -> Any:
    """Lazy loads librosa module."""
    import librosa  # pylint: disable=g-import-not-at-top

    return librosa

  def run_inference(self, inputs: dict[str, Any], model: Any) -> np.ndarray:
    """Executes the model signatures to produce logits."""
    # 1. Run Encoder
    encoder_index = self.litert_model.get_signature_index("encode")
    if encoder_index == -1:
      encoder_index = 0

    enc_inputs = self.litert_model.create_input_buffers(encoder_index)
    enc_outputs = self.litert_model.create_output_buffers(encoder_index)

    # Write input features to the first input buffer with padding/truncation
    # Support both 'input_features' and 'input_values' (used by Moonshine)
    feat = inputs.get("input_features")
    if feat is None:
      feat = inputs.get("input_values")
    if feat is None:
      raise KeyError(
          "Neither 'input_features' nor 'input_values' found in inputs."
      )

    feat_np = feat.numpy() if hasattr(feat, "numpy") else feat
    enc_inputs[0].write(feat_np)

    if "attention_mask" in inputs and len(enc_inputs) > 1:
      mask = inputs["attention_mask"]
      mask_np = mask.numpy() if hasattr(mask, "numpy") else mask
      if mask_np.dtype == bool or mask_np.dtype == np.bool_:
        mask_np = mask_np.astype(np.int8)
      enc_inputs[1].write(mask_np)

    self.litert_model.run_by_index(encoder_index, enc_inputs, enc_outputs)

    # 2. Check for Decode Signature
    decoder_index = self.litert_model.get_signature_index("decode")

    if decoder_index != -1:
      # Detected 'decode' signature. Running auto-regressive loop...
      return self._decode_loop(decoder_index, enc_outputs, model)
    else:
      # No 'decode' signature found. Returning encoder output directly.
      return self.read_buffer(
          enc_outputs[0], encoder_index, output_index=0, unused_model=model
      )

  def run_full_pipeline(self, model_handler: Any, **kwargs: Any) -> str:
    """Handles full pipeline: audio loading, preprocessing, and inference."""
    librosa = self._get_librosa()

    model_params = kwargs.get("model_params", {})

    # Check for speaker input
    if self.wav_path == "speaker":
      return self._run_speaker_mode(model_handler, **kwargs)

    # Check if we should use chunked file mode by default for long files
    try:
      total_duration = librosa.get_duration(path=self.wav_path)
    except Exception:  # pylint: disable=broad-except
      total_duration = 0.0

    chunk_duration = float(model_params.get("chunk_duration", 5.0))
    stride = model_params.get("stride")
    if stride:
      stride = float(stride)
    else:
      stride = chunk_duration

    if total_duration > chunk_duration or total_duration == 0.0:
      click.secho(
          f"File duration: {total_duration:.2f}s. Using chunked mode...",
          fg="cyan",
      )
      return self._run_chunked_file_mode(
          model_handler, chunk_duration, stride, **kwargs
      )

    # Fallback to single-chunk processing for short files
    duration = float(model_params.get("duration", kwargs.get("duration", 5.0)))
    offset = float(model_params.get("offset", kwargs.get("offset", 0.0)))
    return self._run_single_chunk(
        model_handler, duration=duration, offset=offset, **kwargs
    )

  def _run_single_chunk(
      self,
      model_handler: Any,
      *,
      duration: float,
      offset: float,
      show_box: bool = True,
      **unused_kwargs: Any,
  ) -> str:
    """Runs pipeline for a single audio chunk."""
    librosa = self._get_librosa()

    processor = model_handler.get_processor()

    sample, rate = librosa.load(
        self.wav_path,
        sr=processor.feature_extractor.sampling_rate,
        duration=duration,
        offset=offset,
    )
    processed_inputs = processor(
        audio=sample, sampling_rate=rate, return_tensors="pt"
    )

    # Hook for model-specific preprocessing (e.g. truncation)
    if hasattr(model_handler, "preprocess_inputs"):
      processed_inputs = model_handler.preprocess_inputs(processed_inputs)

    logits = self.run_inference(processed_inputs, model_handler)
    tokens = np.argmax(logits, axis=-1)

    # Hook for model-specific token postprocessing (e.g. CTC collapse)
    if hasattr(model_handler, "postprocess_tokens"):
      tokens = model_handler.postprocess_tokens(tokens)

    text = processor.batch_decode(tokens, skip_special_tokens=True)

    if show_box:
      click.secho("\n" + "=" * 60, fg="blue")
      click.secho(f"Recognized audio: {text[0]}", fg="green", bold=True)
      click.secho("=" * 60 + "\n", fg="blue")

    return text[0]

  def _run_chunked_file_mode(
      self,
      model_handler: Any,
      chunk_duration: float,
      stride: float,
      **unused_kwargs: Any,
  ) -> str:
    """Processes file in chunks and prints results continuously."""
    librosa = self._get_librosa()

    click.secho(f"Starting chunked processing on: {self.wav_path}", fg="cyan")

    try:
      total_duration = librosa.get_duration(path=self.wav_path)
    except Exception:  # pylint: disable=broad-except
      total_duration = 10.0  # Fallback

    # Calculate number of steps
    if total_duration <= chunk_duration:
      num_chunks = 1
    else:
      num_chunks = int((total_duration - chunk_duration) / stride) + 1

    full_text = []

    click.secho("\n=== Recognized audio ===", fg="green")
    for i in range(num_chunks):
      offset = float(i) * stride
      try:
        # Run without box for continuous effect
        text = self._run_single_chunk(
            model_handler,
            duration=chunk_duration,
            offset=offset,
            show_box=False,
        )
        if text:
          click.echo(f" >> {text}")
          full_text.append(text)
        time.sleep(0.1)  # Small delay for smoother output
      except Exception as e:  # pylint: disable=broad-except
        click.secho(
            f"\nError processing chunk at {offset}s: {e}", fg="red", err=True
        )

    click.secho("=============================\n", fg="magenta", bold=True)
    return " ".join(full_text)

  def _run_speaker_mode(self, model_handler: Any, **unused_kwargs: Any) -> str:
    """Runs real-time microphone streaming."""
    try:
      import sounddevice as sd  # pylint: disable=g-import-not-at-top
    except ImportError:
      click.secho(
          "Error: 'sounddevice' library not found. Please install it.", fg="red"
      )
      return "Failed: sounddevice not installed."

    processor = model_handler.get_processor()
    q = queue.Queue()

    def callback(indata, unused_frames, unused_time_info, status):
      if status:
        click.echo(status, err=True)
      q.put(indata.copy())

    sampling_rate = 16000
    window_duration = 5.0
    slide_duration = 1.0

    window_samples = int(window_duration * sampling_rate)
    slide_samples = int(slide_duration * sampling_rate)

    audio_buffer = np.zeros(window_samples, dtype=np.float32)
    samples_accumulated = 0

    click.echo("\n=== Live Mic Streaming Started ===")
    click.echo(f"Window: {window_duration}s, Slide: {slide_duration}s")
    click.echo("Speak into your microphone. Press Ctrl+C to stop.")
    click.echo("===================================\n")

    try:
      with sd.InputStream(
          samplerate=sampling_rate,
          channels=1,
          blocksize=1600,
          callback=callback,
      ):
        while True:
          try:
            chunk = q.get(timeout=2.0)
          except queue.Empty:
            print("\r\033[K\033[33m[Listening...]\033[0m", end="", flush=True)
            continue

          chunk = chunk.flatten()
          audio_buffer = np.roll(audio_buffer, -len(chunk))
          audio_buffer[-len(chunk) :] = chunk

          samples_accumulated += len(chunk)

          if samples_accumulated >= slide_samples:
            samples_accumulated = 0

            try:
              # Preprocess audio using the model's processor
              inputs = processor(
                  audio=audio_buffer,
                  sampling_rate=sampling_rate,
                  return_tensors="pt",
              )

              if hasattr(model_handler, "preprocess_inputs"):
                inputs = model_handler.preprocess_inputs(inputs)

              logits = self.run_inference(inputs, model_handler)
              tokens = np.argmax(logits, axis=-1)

              if hasattr(model_handler, "postprocess_tokens"):
                tokens = model_handler.postprocess_tokens(tokens)

              text = processor.batch_decode(tokens, skip_special_tokens=True)

              if text and text[0]:
                curr_text = text[0]
                terminal_width = 80
                try:
                  terminal_width = os.get_terminal_size().columns
                except OSError:
                  pass

                max_text_len = terminal_width - 15
                if len(curr_text) > max_text_len:
                  curr_text = "..." + curr_text[-(max_text_len - 3) :]

                print(
                    f"\r\033[K\033[32m[Live]\033[0m >> {curr_text}",
                    end="",
                    flush=True,
                )

            except Exception as e:  # pylint: disable=broad-except
              click.echo(f"\nError during inference: {e}", err=True)

    except KeyboardInterrupt:
      print("\n\n=== Live Mic Streaming Stopped ===")
    except Exception as e:  # pylint: disable=broad-except
      print(f"\nUnexpected error: {e}", file=sys.stderr)

    return "Streaming finished."

  def _decode_loop(
      self, decoder_index: int, encoder_output_buffers: list[Any], model: Any
  ) -> np.ndarray:
    """Implements the auto-regressive decoding loop."""
    dec_inputs = self.litert_model.create_input_buffers(decoder_index)
    dec_outputs = self.litert_model.create_output_buffers(decoder_index)

    encoder_index = self.litert_model.get_signature_index("encode")
    if encoder_index == -1:
      encoder_index = 0

    for i, buffer in enumerate(encoder_output_buffers):
      dec_inputs[i].write(
          self.read_buffer(
              buffer, encoder_index, output_index=i, unused_model=model
          )
      )

    tflite_logits = []

    try:
      start_token = model.get_decode_start_token_id()
      stop_token = model.get_decode_stop_token_id()
    except AttributeError:
      start_token = 1
      stop_token = 2

    if isinstance(start_token, list):
      tflite_tokens = list(start_token)
    else:
      tflite_tokens = [start_token]

    try:
      # Determine token size from model metadata
      details = self.litert_model.get_input_details(decoder_index)
      # Assuming the last few inputs are tokens and masks
      token_input_idx = len(dec_inputs) - 2
      decode_token_size = math.prod(details[token_input_idx]["shape"])
    except AttributeError:
      # Work around for litert-lite runtime on some platforms where
      # TensorBuffer has no get_tensor_details
      if model and hasattr(model, "get_decode_token_size"):
        decode_token_size = model.get_decode_token_size()
      else:
        # Fallback to a smaller size to avoid buffer overflow on mobile models!
        decode_token_size = 64

    for i in range(decode_token_size - 1):
      input_tokens = np.zeros((1, decode_token_size), dtype=np.int32)
      input_tokens[0, : len(tflite_tokens)] = tflite_tokens

      input_mask = np.zeros((1, decode_token_size), dtype=np.float32)
      input_mask[0, : len(tflite_tokens)] = 1.0

      dec_inputs[-2].write(input_tokens)
      dec_inputs[-1].write(input_mask)

      self.litert_model.run_by_index(decoder_index, dec_inputs, dec_outputs)

      dec_output = self.read_buffer(
          dec_outputs[0], decoder_index, output_index=0, unused_model=model
      )
      current_logits = dec_output[0, i, :]

      # Repetition Penalty
      current_logits = self._apply_repetition_penalty(
          current_logits, tflite_tokens
      )

      token_id = int(np.argmax(current_logits))
      tflite_logits.append(current_logits)
      tflite_tokens.append(token_id)

      if token_id == stop_token:
        break

    return np.expand_dims(np.stack(tflite_logits, axis=0), axis=0)

  def read_buffer(
      self,
      buffer: Any,
      signature_index: int,
      output_index: int,
      unused_model: Any = None,
  ) -> np.ndarray:
    """Reads a TensorBuffer by checking CompiledModel details."""
    try:
      # Try standard API first
      details = buffer.get_tensor_details()
      shape = details["shape"]
      dtype = details["dtype"]
    except AttributeError:
      # Fallback to CompiledModel details if buffer.get_tensor_details is
      # missing
      details_list = self.litert_model.get_output_details(signature_index)
      shape = details_list[output_index]["shape"]
      dtype = details_list[output_index]["dtype"]

    num_elements = math.prod(shape)
    return buffer.read(num_elements, dtype).reshape(shape)

  def _apply_repetition_penalty(
      self,
      current_logits: np.ndarray,
      tflite_tokens: list[int],
      penalty: float = 1.1,
  ) -> np.ndarray:
    """Applies repetition penalty to current logits."""
    for past_token in tflite_tokens:
      if past_token < len(current_logits):
        original_logit = current_logits[past_token]
        if original_logit > 0:
          current_logits[past_token] = original_logit / penalty
        else:
          current_logits[past_token] = original_logit * penalty
    return current_logits
