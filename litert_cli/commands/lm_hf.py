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

"""Hugging Face repo ids as model references of `litert lm run|benchmark`.

`litert lm` forwards its arguments to `litert-lm`. When the model reference
(the first argument after `run` or `benchmark`) is a Hugging Face repo id
(`org/name`) that is neither a local path nor a model imported into litert-lm,
it is resolved before forwarding:

- A repo with a `.litertlm` file is passed to litert-lm with
  `--from-huggingface-repo org/name`.
- A repo with a transformers checkpoint is converted once with
  `litert convert` into the LiteRT CLI model cache
  (`~/.cache/litert-cli/models/<org>__<name>/<variant>/model.litertlm`), and
  the bundle directory is passed to litert-lm. Later runs reuse the bundle.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import datetime
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time

import click
from litert_cli.core import constants
from litert_cli.core import deps

_SUBCOMMANDS = ("run", "benchmark")
_HELP_ARGS = ("-h", "--help")
_HF_REPO_ARG = "--from-huggingface-repo"
_HF_TOKEN_ARG = "--huggingface-token"
# Hugging Face repo id: `namespace/name`, each starting with a letter, digit
# or underscore. Rules out `./x`, `../x`, `~/x` and absolute paths.
_REPO_ID = re.compile(r"^\w[\w.-]*/\w[\w.-]*$", re.ASCII)

DEFAULT_RECIPE = "dynamic_wi8_afp32"
DEFAULT_PREFILL_LENGTHS = (256,)
DEFAULT_CACHE_LENGTH = 4096
# `--quantize none` converts without quantization (fp32).
NO_QUANTIZATION = "none"

# Conversion options read here and not forwarded to litert-lm.
_VALUE_OPTIONS = {
    "--quantize": "recipe",
    "--quantize-recipe": "recipe",
    "--prefill-lengths": "prefill_lengths",
    "--cache-length": "cache_length",
}
_FLAG_OPTIONS = ("--reconvert",)

_BUNDLE_FILE = "model.litertlm"
_LOG_FILE = "convert.log"
_METADATA_FILE = "metadata.json"
_TMP_MARKER = ".tmp-"
_LITERT_TORCH_ISSUES = "https://github.com/google-ai-edge/litert-torch/issues"
_HUB_TIMEOUT_SECS = 30
# Distributions recorded with each conversion.
_RECORDED_DISTRIBUTIONS = (
    "litert-torch-nightly",
    "litert-torch",
    "litert-lm-builder",
    "litert-lm-builder-nightly",
    "litert-lm-nightly",
    "litert-lm",
    "transformers",
    "torch",
)

# Conversions measured on an Apple M4 Max (128 GB) with the default options
# (dynamic_wi8_afp32, prefill 256, cache 4096), checkpoint already downloaded:
# (parameters, peak RSS in GB, seconds) of Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B.
# LFM2-350M and SmolLM2-360M-Instruct took 10.6 and 11.3 GB, 28 and 44 s.
_MEASURED_COSTS = (
    (751_632_384, 10.7, 43),
    (2_031_739_904, 25.1, 94),
    (4_022_468_096, 40.8, 202),
)


@dataclasses.dataclass(frozen=True)
class ConvertOptions:
  """Options of the conversion, which also name its cache variant."""

  recipe: str = DEFAULT_RECIPE
  prefill_lengths: tuple[int, ...] = DEFAULT_PREFILL_LENGTHS
  cache_length: int = DEFAULT_CACHE_LENGTH
  reconvert: bool = False

  @property
  def variant(self) -> str:
    recipe = re.sub(r"[^\w.-]", "_", self.recipe, flags=re.ASCII)
    prefill = "_".join(str(p) for p in self.prefill_lengths)
    return f"litertlm-{recipe}-p{prefill}-c{self.cache_length}"


@dataclasses.dataclass(frozen=True)
class RepoInfo:
  """What `litert lm` needs to know about a Hugging Face repo."""

  repo_id: str
  revision: str
  files: tuple[str, ...]
  gated: bool = False
  parameters: int | None = None
  checkpoint_bytes: int = 0

  @property
  def litertlm_files(self) -> tuple[str, ...]:
    return tuple(f for f in self.files if f.endswith(".litertlm"))

  @property
  def has_config(self) -> bool:
    return "config.json" in self.files


def should_resolve(args: Sequence[str]) -> bool:
  """Returns True if `litert lm <args>` names a Hugging Face repo to resolve."""
  if len(args) < 2 or args[0] not in _SUBCOMMANDS:
    return False
  for arg in args:
    if arg in _HELP_ARGS or arg.partition("=")[0] == _HF_REPO_ARG:
      return False
  return looks_like_hf_repo(args[1])


def looks_like_hf_repo(reference: str) -> bool:
  """Returns True if `reference` is an `org/name` id and not a local model."""
  if not _REPO_ID.match(reference) or reference.endswith(".litertlm"):
    return False
  if os.path.exists(reference):
    return False
  return not os.path.isfile(_imported_litert_lm_model(reference))


def _imported_litert_lm_model(model_id: str) -> str:
  # Where `litert-lm import` puts a model, as litert-lm resolves it first:
  # $LITERT_LM_DIR (default ~/.litert-lm)/models/<org--name>/model.litertlm.
  base = os.environ.get("LITERT_LM_DIR") or os.path.join(
      os.path.expanduser("~"), ".litert-lm"
  )
  return os.path.join(
      os.path.abspath(base),
      "models",
      model_id.replace("/", "--"),
      _BUNDLE_FILE,
  )


def split_convert_options(
    args: Sequence[str],
) -> tuple[ConvertOptions, list[str], list[str]]:
  """Takes the conversion options out of the arguments for litert-lm.

  Args:
    args: Arguments after the model reference, as `--key value`, `--key=value`
      or flags.

  Returns:
    The conversion options, the arguments left for litert-lm, and the names of
    the conversion options found.

  Raises:
    click.UsageError: If an option has no value or an invalid one.
  """
  values = {}
  found = []
  rest = []
  i = 0
  while i < len(args):
    arg = args[i]
    name, has_value, value = arg.partition("=")
    if name in _VALUE_OPTIONS:
      if not has_value:
        # As click does for its own options, a following option is not a
        # value: `--quantize --prompt hi` must not take `--prompt` as the
        # recipe.
        if i + 1 >= len(args) or args[i + 1].startswith("-"):
          raise click.UsageError(f"Option '{name}' requires a value.")
        i += 1
        value = args[i]
      values[_VALUE_OPTIONS[name]] = (name, value)
      found.append(name)
    elif arg in _FLAG_OPTIONS:
      values["reconvert"] = (arg, True)
      found.append(arg)
    else:
      rest.append(arg)
    i += 1

  options = ConvertOptions()
  if "recipe" in values:
    options = dataclasses.replace(options, recipe=values["recipe"][1])
  if "prefill_lengths" in values:
    name, value = values["prefill_lengths"]
    options = dataclasses.replace(
        options, prefill_lengths=_parse_positive_ints(name, value)
    )
  if "cache_length" in values:
    name, value = values["cache_length"]
    (cache_length,) = _parse_positive_ints(name, value, single=True)
    options = dataclasses.replace(options, cache_length=cache_length)
  if "reconvert" in values:
    options = dataclasses.replace(options, reconvert=True)
  return options, rest, found


def _parse_positive_ints(
    name: str, value: str, single: bool = False
) -> tuple[int, ...]:
  parts = [value] if single else value.split(",")
  try:
    numbers = tuple(int(p.strip()) for p in parts)
  except ValueError:
    numbers = ()
  if not numbers or any(n <= 0 for n in numbers):
    expected = (
        "a positive integer" if single else "positive integers, e.g. 128,512"
    )
    raise click.UsageError(
        f"Invalid value for '{name}': '{value}' (expected {expected})."
    )
  return numbers


def _token_from_args(args: Sequence[str]) -> str | None:
  """Returns the value of litert-lm's --huggingface-token, if given."""
  for i, arg in enumerate(args):
    name, has_value, value = arg.partition("=")
    if name == _HF_TOKEN_ARG:
      if has_value:
        return value
      if i + 1 < len(args):
        return args[i + 1]
  return None


class HubUnreachableError(Exception):
  """The Hugging Face Hub could not be reached (offline, network, 5xx, 429)."""


def resolve_args(args: Sequence[str]) -> list[str]:
  """Returns the litert-lm arguments with the Hugging Face repo id resolved.

  Args:
    args: `litert lm` arguments for which `should_resolve` is True.

  Returns:
    The arguments to forward to litert-lm: the repo id is replaced with
    `--from-huggingface-repo org/name` or with a converted bundle directory,
    and the conversion options are removed.

  Raises:
    click.ClickException: If the repo cannot be used or the conversion fails.
  """
  subcommand, repo_id = args[0], args[1]
  options, rest, found = split_convert_options(args[2:])
  token = _token_from_args(rest)

  try:
    repo = inspect_repo(repo_id, token)
  except HubUnreachableError as e:
    cached = None if options.reconvert else cached_bundle(repo_id, options)
    if cached is None:
      if options.reconvert:
        reason = "and --reconvert needs it"
      else:
        variant_dir = _display_path(model_cache_dir(repo_id) / options.variant)
        reason = (
            "and no bundle converted with these options is cached"
            f" ({variant_dir})"
        )
      raise click.ClickException(
          f"Could not reach the Hugging Face Hub to look up '{repo_id}' ({e}),"
          f" {reason}."
      ) from e
    click.echo(f"Could not reach the Hugging Face Hub ({e}).")
    click.echo(_cache_hit_line(*cached))
    return [subcommand, str(cached[0]), *rest]

  if repo.litertlm_files:
    if found:
      click.echo(
          f"Ignoring {', '.join(found)}: {repo_id} has a .litertlm file, so"
          " nothing is converted."
      )
    click.echo(
        f"{repo_id} has a .litertlm file; running it with litert-lm"
        f" {_HF_REPO_ARG} {repo_id}."
    )
    return [subcommand, _HF_REPO_ARG, repo_id, *rest]

  if not repo.has_config:
    raise click.ClickException(_no_checkpoint_message(repo))

  cached = None if options.reconvert else cached_bundle(repo_id, options)
  if cached is not None:
    bundle_dir, record = cached
    click.echo(_cache_hit_line(bundle_dir, record))
    converted_revision = record.get("revision")
    if converted_revision and converted_revision != repo.revision:
      click.echo(
          f"{repo_id} has changed since the bundle was converted (revision"
          f" {converted_revision[:7]}, now {repo.revision[:7]}); using the"
          " cached bundle. Pass --reconvert to rebuild it."
      )
    return [subcommand, str(bundle_dir), *rest]

  if repo.gated:
    _check_gated_access(repo, token)
  bundle_dir = convert_to_cache(repo, options, token)
  return [subcommand, str(bundle_dir), *rest]


def inspect_repo(repo_id: str, token: str | None = None) -> RepoInfo:
  """Reads the file list and revision of a Hugging Face model repo.

  Args:
    repo_id: The Hugging Face repo id.
    token: Hugging Face token; None uses HF_TOKEN or the stored login.

  Returns:
    The repo's revision, files, gating, parameter count and checkpoint size.

  Raises:
    click.ClickException: If the repo does not exist or is private, or the
      Hub refuses the request (HTTP 4xx other than 429, such as 401 or 403).
    HubUnreachableError: If the Hub could not be reached, failed (HTTP 5xx)
      or limited the rate (HTTP 429).
  """
  import huggingface_hub  # pylint: disable=g-import-not-at-top
  import huggingface_hub.utils  # pylint: disable=g-import-not-at-top

  try:
    info = huggingface_hub.HfApi().model_info(
        repo_id, files_metadata=True, token=token, timeout=_HUB_TIMEOUT_SECS
    )
  except huggingface_hub.utils.RepositoryNotFoundError as e:
    raise click.ClickException(
        f"'{repo_id}' is not a local path, and the Hugging Face Hub has no"
        f" model repo '{repo_id}' that this account can read. For a private"
        " repo, set HF_TOKEN or run 'hf auth login'."
    ) from e
  except huggingface_hub.utils.HFValidationError as e:
    raise click.ClickException(
        f"'{repo_id}' is not a local path or a valid Hugging Face repo id:"
        f" {e}"
    ) from e
  except huggingface_hub.utils.OfflineModeIsEnabled as e:
    raise HubUnreachableError("HF_HUB_OFFLINE is set") from e
  except huggingface_hub.utils.HfHubHTTPError as e:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is not None and 400 <= status < 500 and status != 429:
      # The Hub answered and refused (e.g. 401 or 403): falling back to a
      # cached bundle would hide the problem.
      detail = _first_line(getattr(e, "server_message", None) or e)
      hint = (
          f" Check HF_TOKEN or {_HF_TOKEN_ARG}, or run 'hf auth login'."
          if status in (401, 403)
          else ""
      )
      raise click.ClickException(
          f"The Hugging Face Hub refused to look up '{repo_id}' (HTTP"
          f" {status}: {detail}).{hint}"
      ) from e
    # Server errors and rate limits: the caller falls back to a cached bundle.
    raise HubUnreachableError(f"{type(e).__name__}: {_first_line(e)}") from e
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Network errors: the caller falls back to a cached bundle.
    raise HubUnreachableError(f"{type(e).__name__}: {_first_line(e)}") from e

  siblings = info.siblings or []
  files = tuple(s.rfilename for s in siblings)
  weights = [s for s in siblings if s.rfilename.endswith(".safetensors")]
  if not weights:
    weights = [s for s in siblings if s.rfilename.endswith(".bin")]
  safetensors = getattr(info, "safetensors", None)
  return RepoInfo(
      repo_id=repo_id,
      revision=info.sha or "",
      files=files,
      gated=bool(info.gated),
      parameters=getattr(safetensors, "total", None),
      checkpoint_bytes=sum(s.size or 0 for s in weights),
  )


def _check_gated_access(repo: RepoInfo, token: str | None) -> None:
  """Fails early if the account cannot download a gated repo's files."""
  import huggingface_hub  # pylint: disable=g-import-not-at-top
  import huggingface_hub.utils  # pylint: disable=g-import-not-at-top

  url = huggingface_hub.hf_hub_url(
      repo.repo_id, "config.json", revision=repo.revision or None
  )
  try:
    huggingface_hub.get_hf_file_metadata(url, token=token)
  except huggingface_hub.utils.GatedRepoError as e:
    status = getattr(getattr(e, "response", None), "status_code", None)
    raise click.ClickException(
        f"Hugging Face repo '{repo.repo_id}' is gated, and this account has"
        f" no access to its files{f' (HTTP {status})' if status else ''}."
        f" Request access on https://huggingface.co/{repo.repo_id}, then set"
        " HF_TOKEN or run 'hf auth login'."
    ) from e
  except Exception:  # pylint: disable=broad-exception-caught
    # Anything else surfaces in the conversion itself.
    pass


def _no_checkpoint_message(repo: RepoInfo) -> str:
  shown = ", ".join(repo.files[:8]) + (", ..." if len(repo.files) > 8 else "")
  return (
      f"'{repo.repo_id}' has no .litertlm file and no transformers"
      f" config.json (files: {shown or 'none'}). litert lm needs a repo with a"
      " .litertlm file or a transformers checkpoint."
  )


def model_cache_dir(repo_id: str) -> pathlib.Path:
  """Returns the LiteRT CLI cache directory of a Hugging Face repo."""
  # Same flattening as `litert download` and `litert import`.
  return pathlib.Path(constants.LITERT_MODELS_CACHE_DIR) / repo_id.replace(
      "/", "__"
  )


def cached_bundle(
    repo_id: str, options: ConvertOptions
) -> tuple[pathlib.Path, dict] | None:
  """Returns the cached bundle directory and its conversion record, if any."""
  model_dir = model_cache_dir(repo_id)
  bundle_dir = model_dir / options.variant
  if not (bundle_dir / _BUNDLE_FILE).is_file():
    return None
  record = _read_metadata(model_dir).get("conversions", {})
  return bundle_dir, record.get(options.variant, {})


def _cache_hit_line(bundle_dir: pathlib.Path, record: dict) -> str:
  converted_on = str(record.get("created_at", ""))[:10] or "an earlier run"
  details = []
  versions = record.get("versions", {})
  converter = next(
      (
          f"{d} {versions[d]}"
          for d in ("litert-torch-nightly", "litert-torch")
          if d in versions
      ),
      None,
  )
  if converter:
    details.append(converter)
  if "elapsed_s" in record:
    details.append(f"{record['elapsed_s']:.0f} s")
  suffix = f" ({', '.join(details)})" if details else ""
  return (
      f"Using the bundle converted on {converted_on}{suffix}:"
      f" {_display_path(bundle_dir)}"
  )


def _read_metadata(model_dir: pathlib.Path) -> dict:
  try:
    with open(model_dir / _METADATA_FILE, "r") as f:
      metadata = json.load(f)
  except (OSError, json.JSONDecodeError):
    return {}
  return metadata if isinstance(metadata, dict) else {}


def _write_conversion_record(
    model_dir: pathlib.Path, repo_id: str, variant: str, record: dict
) -> None:
  metadata = _read_metadata(model_dir)
  # Keep the keys of `litert download` / `litert import` if they wrote first.
  metadata.setdefault("model_ref", repo_id)
  metadata.setdefault("hf_id", repo_id)
  metadata.setdefault("source", "converted")
  metadata.setdefault("created_at", record["created_at"])
  conversions = metadata.get("conversions")
  if not isinstance(conversions, dict):
    conversions = {}
  conversions[variant] = record
  metadata["conversions"] = conversions
  with open(model_dir / _METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)


def convert_to_cache(
    repo: RepoInfo, options: ConvertOptions, token: str | None = None
) -> pathlib.Path:
  """Converts a Hugging Face checkpoint into the cache with `litert convert`.

  The conversion runs in a child process (`python -m litert_cli.litert
  convert`) that writes into `<variant>.tmp-<pid>/`, with its stderr in
  `convert.log` there. The directory is renamed to `<variant>/` once
  `model.litertlm` exists, so an interrupted conversion is never a cache hit.
  When the conversion fails, its log is moved to `failure_log_path()` and the
  directory is removed.

  Args:
    repo: The repo to convert.
    options: The conversion options.
    token: Hugging Face token for the download; None uses HF_TOKEN or the
      stored login.

  Returns:
    The bundle directory (it contains `model.litertlm`).

  Raises:
    click.ClickException: If the conversion fails.
  """
  model_dir = model_cache_dir(repo.repo_id)
  bundle_dir = model_dir / options.variant
  _print_plan(repo, options, bundle_dir)
  deps.ensure_extra("convert")

  model_dir.mkdir(parents=True, exist_ok=True)
  _remove_stale_tmp_dirs(model_dir)
  tmp_dir = model_dir / f"{options.variant}{_TMP_MARKER}{os.getpid()}"
  shutil.rmtree(tmp_dir, ignore_errors=True)
  tmp_dir.mkdir()
  log_path = tmp_dir / _LOG_FILE

  cmd = [
      sys.executable,
      "-m",
      "litert_cli.litert",
      "convert",
      repo.repo_id,
      "--output",
      str(tmp_dir),
      "--prefill-lengths",
      ",".join(str(p) for p in options.prefill_lengths),
      "--cache-length",
      str(options.cache_length),
  ]
  if options.recipe != NO_QUANTIZATION:
    cmd += ["--quantize", options.recipe]
  env = dict(os.environ)
  if token:
    env["HF_TOKEN"] = token

  start = time.monotonic()
  try:
    with open(log_path, "wb") as log:
      # stdout (litert-torch progress, "Error during conversion: ...") stays
      # on the terminal; stdin is kept for litert-lm (e.g. a piped prompt).
      result = subprocess.run(
          cmd, stdin=subprocess.DEVNULL, stderr=log, env=env, check=False
      )
  except KeyboardInterrupt:
    _remove_tmp_dir(tmp_dir)
    raise
  elapsed = time.monotonic() - start
  peak_rss = _children_peak_rss_bytes()

  if result.returncode != 0 or not (tmp_dir / _BUNDLE_FILE).is_file():
    # Keep the log outside the model cache, so `litert list` does not show
    # the failed conversion as a model.
    kept_log = failure_log_path(repo.repo_id, options.variant)
    kept_log.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(log_path), str(kept_log))
    _remove_tmp_dir(tmp_dir)
    raise click.ClickException(
        failure_message(
            repo.repo_id, result.returncode, kept_log, elapsed, token=token
        )
    )

  shutil.rmtree(bundle_dir, ignore_errors=True)
  os.rename(tmp_dir, bundle_dir)
  bundle_bytes = (bundle_dir / _BUNDLE_FILE).stat().st_size
  record = {
      "revision": repo.revision,
      "recipe": options.recipe,
      "prefill_lengths": list(options.prefill_lengths),
      "cache_length": options.cache_length,
      "versions": _installed_versions(),
      "elapsed_s": round(elapsed, 1),
      "peak_rss_bytes": peak_rss,
      "bundle_bytes": bundle_bytes,
      "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }
  _write_conversion_record(model_dir, repo.repo_id, options.variant, record)
  rss = f" (peak RSS {_gb(peak_rss)} GB)" if peak_rss else ""
  click.echo(
      f"Converted in {elapsed:.0f} s{rss}, bundle {_gb(bundle_bytes)} GB:"
      f" {_display_path(bundle_dir)}"
  )
  return bundle_dir


def _print_plan(
    repo: RepoInfo, options: ConvertOptions, bundle_dir: pathlib.Path
) -> None:
  """Prints what the conversion is about to do and what it may cost."""
  click.echo(
      f"{repo.repo_id} has no .litertlm file; converting the checkpoint once"
      f" (revision {repo.revision[:7]})."
  )
  size = []
  if repo.parameters:
    size.append(f"{repo.parameters / 1e9:.2f} B parameters")
  if repo.checkpoint_bytes:
    size.append(f"{_gb(repo.checkpoint_bytes)} GB checkpoint")
  prefill = ",".join(str(p) for p in options.prefill_lengths)
  settings = (
      f"recipe {options.recipe}, prefill {prefill}, cache"
      f" {options.cache_length}"
  )
  click.echo(f"  {'; '.join(size + [settings])}")
  parameters = _parameter_count(repo.parameters, repo.checkpoint_bytes)
  click.echo(f"  {estimate_line(parameters)}")
  warning = ram_warning(_physical_ram_bytes(), estimate(parameters)[0])
  if warning:
    click.echo(f"  {warning}")
  click.echo(f"  output: {_display_path(bundle_dir)}")
  click.echo(
      "  conversion options: --quantize, --prefill-lengths, --cache-length,"
      " --reconvert"
  )


def _parameter_count(parameters: int | None, checkpoint_bytes: int) -> int:
  if parameters:
    return parameters
  # 16-bit checkpoints are about 2 bytes per parameter.
  return checkpoint_bytes // 2


def estimate(parameters: int) -> tuple[float, float, bool]:
  """Returns (peak RAM in GB, seconds, measured) of a conversion.

  Interpolated between the measured conversions. Below the smallest one it is
  that one's cost; above the largest it is the largest one's cost, which is
  then a lower bound (`measured` is False).

  Args:
    parameters: Parameter count of the checkpoint.
  """
  points = _MEASURED_COSTS
  if parameters <= points[0][0]:
    return points[0][1], points[0][2], True
  if parameters > points[-1][0]:
    return points[-1][1], points[-1][2], False
  for (p0, ram0, s0), (p1, ram1, s1) in zip(points, points[1:]):
    if parameters <= p1:
      t = (parameters - p0) / (p1 - p0)
      return ram0 + t * (ram1 - ram0), s0 + t * (s1 - s0), True
  return points[-1][1], points[-1][2], True


def _duration(seconds: float) -> str:
  if seconds < 60:
    return "under a minute"
  if seconds < 90:
    return "about a minute"
  return f"about {round(seconds / 60)} minutes"


def estimate_line(parameters: int) -> str:
  """Returns the expected RAM and time of a conversion, from measurements."""
  ram, seconds, measured = estimate(parameters)
  if not measured:
    return (
        f"expect more than {ram:.0f} GB of RAM and {round(seconds / 60)}"
        " minutes, plus the download (measured up to"
        f" {_MEASURED_COSTS[-1][0] / 1e9:.0f} B parameters on an Apple M4 Max)"
    )
  return (
      f"expect about {ram:.0f} GB of RAM and {_duration(seconds)}, plus the"
      " download (estimated from conversions on an Apple M4 Max)"
  )


def ram_warning(physical_bytes: int | None, needed_gb: float) -> str | None:
  """Returns a warning if the machine has less RAM than the estimate."""
  if physical_bytes and physical_bytes < needed_gb * 1e9:
    return (
        f"warning: this machine has {_gb(physical_bytes)} GB of RAM; the"
        " conversion may run out of memory."
    )
  return None


def _physical_ram_bytes() -> int | None:
  try:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  except (AttributeError, OSError, ValueError):
    return None


def _children_peak_rss_bytes() -> int | None:
  try:
    import resource  # pylint: disable=g-import-not-at-top
  except ImportError:  # Windows
    return None
  peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
  return peak if sys.platform == "darwin" else peak * 1024


def _installed_versions() -> dict[str, str]:
  versions = {}
  for dist in _RECORDED_DISTRIBUTIONS:
    try:
      versions[dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
      pass
  return versions


def _remove_stale_tmp_dirs(model_dir: pathlib.Path) -> None:
  """Removes `<variant>.tmp-<pid>` dirs whose conversion process is gone."""
  if os.name == "nt":
    return
  for path in model_dir.glob(f"*{_TMP_MARKER}*"):
    pid = path.name.rpartition(_TMP_MARKER)[2]
    if pid.isdigit() and not _pid_alive(int(pid)):
      shutil.rmtree(path, ignore_errors=True)


def _pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _remove_tmp_dir(tmp_dir: pathlib.Path) -> None:
  """Removes a conversion's tmp dir, and the model dir if nothing is left."""
  shutil.rmtree(tmp_dir, ignore_errors=True)
  try:
    tmp_dir.parent.rmdir()
  except OSError:  # Other bundles, downloads or metadata are there.
    pass


def failure_log_path(repo_id: str, variant: str) -> pathlib.Path:
  """Returns where the log of a failed conversion is kept."""
  return (
      pathlib.Path(constants.LITERT_CLI_CACHE_DIR)
      / "logs"
      / f"{repo_id.replace('/', '__')}-{variant}.log"
  )


_FRAME = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$'
)


def summarize_log(text: str) -> tuple[str | None, str | None]:
  """Returns the final exception and its deepest litert_torch frame.

  Args:
    text: The conversion's stderr.

  Returns:
    (exception line, frame) where the exception line is the last line of the
    last traceback (e.g. "AttributeError: ...") and the frame is the deepest
    one inside litert_torch, as "litert_torch/<path>, line N, in <function>".
    Either is None when not found.
  """
  lines = text.splitlines()
  starts = [i for i, l in enumerate(lines) if l.startswith("Traceback (most")]
  if not starts:
    return None, None
  frame = None
  exception = None
  for line in lines[starts[-1] + 1 :]:
    match = _FRAME.match(line)
    if match:
      path = match["file"].replace("\\", "/")
      if "/litert_torch/" in path:
        relative = "litert_torch/" + path.rsplit("/litert_torch/", 1)[1]
        frame = f"{relative}, line {match['line']}, in {match['func']}"
      continue
    if line and not line[0].isspace():
      exception = line.strip()
      break
  return exception, frame


def _last_line(text: str) -> str | None:
  lines = [l.strip() for l in text.splitlines() if l.strip()]
  return lines[-1] if lines else None


def failure_message(
    repo_id: str,
    returncode: int,
    log_path: pathlib.Path,
    elapsed: float,
    token: str | None = None,
) -> str:
  """Returns the message for a failed conversion: error, location, next steps.

  Args:
    repo_id: The converted repo.
    returncode: Exit code of the conversion process (negative = signal).
    log_path: The conversion's stderr log.
    elapsed: Seconds the conversion ran.
    token: Hugging Face token for the litert-community lookup.
  """
  try:
    text = log_path.read_text(errors="replace")
  except OSError:
    text = ""
  exception, frame = summarize_log(text)
  is_import_error = bool(
      exception
      and exception.split(":", 1)[0].endswith(
          ("ModuleNotFoundError", "ImportError")
      )
  )

  if returncode < 0:
    try:
      name = signal.Signals(-returncode).name
    except ValueError:
      name = f"signal {-returncode}"
    headline = (
        f"Conversion of {repo_id} was stopped by {name} after {elapsed:.0f} s"
    )
    if -returncode == getattr(signal, "SIGKILL", 9):
      headline += "; the machine may have run out of memory"
    last = _last_line(text)
    if last:
      headline += f". Last log line: {last}"
  elif exception:
    headline = f"Conversion of {repo_id} failed: {exception}"
  elif returncode == 0:
    headline = (
        f"Conversion of {repo_id} finished without writing {_BUNDLE_FILE}"
    )
  else:
    headline = f"Conversion of {repo_id} failed (exit code {returncode})"

  lines = [headline]
  if frame:
    lines.append(f"  in {frame}")
  if is_import_error:
    lines.append(
        f"  The converter ({_converter_version()}) failed to import; this is an"
        " install problem, not a problem with the model."
    )
  lines.append(f"  full log: {_display_path(log_path)}")
  lines.append("  next steps:")
  if is_import_error:
    lines.append(
        "    - reinstall the 'convert' extra in a fresh virtual environment,"
        " or pin an earlier litert-torch-nightly"
    )
    lines.append(
        f"    - check or report the import error: {_LITERT_TORCH_ISSUES}"
    )
  else:
    community_repo = _community_bundle_repo(repo_id, token)
    if community_repo:
      lines.append(
          f"    - a converted bundle exists: litert lm run {community_repo}"
      )
    lines.append(
        "    - run the conversion by itself for the full output:"
        f" LITERT_VERBOSE=1 litert convert {repo_id} --output DIR"
    )
    if returncode >= 0:
      lines.append(
          "    - export_hf may not support this architecture yet:"
          f" {_LITERT_TORCH_ISSUES}"
      )
  return "\n".join(lines)


def _converter_version() -> str:
  for dist in ("litert-torch-nightly", "litert-torch"):
    try:
      return f"{dist} {importlib.metadata.version(dist)}"
    except importlib.metadata.PackageNotFoundError:
      pass
  return "litert-torch"


def _community_bundle_repo(repo_id: str, token: str | None) -> str | None:
  """Returns litert-community/<name> if it has a .litertlm file."""
  name = repo_id.split("/", 1)[1]
  candidate = f"litert-community/{name}"
  if candidate == repo_id:
    return None
  try:
    import huggingface_hub  # pylint: disable=g-import-not-at-top

    info = huggingface_hub.HfApi().model_info(
        candidate, token=token, timeout=10
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  if any(s.rfilename.endswith(".litertlm") for s in info.siblings or []):
    return candidate
  return None


def _first_line(e: BaseException | str) -> str:
  text = str(e).strip()
  return text.splitlines()[0] if text else ""


def _gb(num_bytes: int | float) -> str:
  value = num_bytes / 1e9
  return f"{value:.2f}" if value < 1 else f"{value:.1f}"


def _display_path(path: pathlib.Path) -> str:
  text = str(path)
  home = os.path.expanduser("~")
  if home and home != "~" and text.startswith(home + os.sep):
    return "~" + text[len(home) :]
  return text
