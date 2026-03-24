"""CLI interface for LiteRT-LM."""

from __future__ import annotations

import subprocess
import sys

import click


@click.command(
    name="lm",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        help_option_names=[],
    ),
    help="""LiteRT-LM related commands.

This command is a transparent proxy to the native `litert-lm-cli` package.
Any wildcard arguments used here are forwarded directly to the actual engine.

Examples:

  Display the native litert-lm help:

    $ litert lm --help

  Run a generative LLM model:

    $ litert lm run model.litertlm
    $ litert lm run gemma3-1b

  Benchmark a generative LLM model:

    $ litert lm benchmark gemma3-1b
""",
)
@click.pass_context
def lm_cmd(ctx: click.Context) -> None:
  """LiteRT-LM related commands.

  This command is a transparent proxy to the native `litert-lm-cli` package.
  Any wildcard arguments used here are forwarded directly to the actual engine.

  Args:
    ctx: click Context object containing forwarded arguments.
  """
  if ctx.args and ctx.args[0] == "run-local":
    _run_with_python_api(ctx.args[1:])
    return

  try:
    result = subprocess.run(["litert-lm"] + ctx.args, check=False)
    sys.exit(result.returncode)
  except FileNotFoundError:
    click.secho("Error: 'litert-lm' executable not found in PATH.", fg="red")
    click.secho(
        "Please install 'litert-lm-cli' to use this command.", fg="yellow"
    )
    sys.exit(1)


def _run_with_python_api(args: list[str]) -> None:
  """Runs the model using litert-lm-nightly Python API.

  Args:
    args: A list of arguments passed to the 'run-local' command. The first argument
      should be the path to the model file, and the second (optional) can be
      the prompt.
  """
  if not args:
    click.secho("Usage: litert lm run-local <model_path> [prompt]", fg="red")
    sys.exit(1)

  model_path = args[0]
  prompt = args[1] if len(args) > 1 else None

  try:
    import litert_lm
  except ImportError:
    click.secho(
        "Error: 'litert-lm-nightly' not installed. Please install it to "
        "use python API fallback.",
        fg="red",
    )
    sys.exit(1)

  click.secho(
      f"Using litert-lm-nightly python API for model: {model_path}", fg="cyan"
  )

  try:
    with litert_lm.Engine(model_path) as engine:
      with engine.create_conversation() as conversation:
        if prompt:
          _send_and_print(conversation, prompt, stream=True)
        else:
          click.secho(
              "Starting interactive chat. Type 'exit' to quit.", fg="green"
          )
          while True:
            try:
              user_input = click.prompt("\nYou")
            except click.Abort:
              print("\nChat session aborted.")
              break
            if user_input.lower() in ["exit", "quit"]:
              break
            _send_and_print(conversation, user_input, stream=True)
  except Exception as e:
    click.secho(f"Error running model with python API: {e}", fg="red")
    sys.exit(1)


def _send_and_print(conversation, message_text: str, stream: bool = True) -> None:
  """Sends message and prints response, optionally streaming.

  Args:
    conversation: The active conversation object from litert_lm.
    message_text: The user input text to send to the model.
    stream: Whether to stream the response (True) or wait for full response.
  """
  if stream:
    try:
      stream_it = conversation.send_message_async(message_text)
      print("\nAssistant:")
      for chunk in stream_it:
        if isinstance(chunk, str):
          print(chunk, end="", flush=True)
        elif isinstance(chunk, dict):
          try:
            text = chunk["content"][0]["text"]
            print(text, end="", flush=True)
          except (KeyError, IndexError, TypeError):
            # Try to print the whole dict or search for text
            text = chunk.get("text", "")
            if text:
              print(text, end="", flush=True)
            else:
              print(chunk, end="", flush=True)
      print()
    except Exception as e:
      click.secho(f"Error during streaming: {e}", fg="red")
      # Fallback to sync if async fails or is not supported
      _send_and_print_sync(conversation, message_text)
  else:
    _send_and_print_sync(conversation, message_text)


def _send_and_print_sync(conversation, message_text: str) -> None:
  """Sends message and prints response synchronously.

  Args:
    conversation: The active conversation object from litert_lm.
    message_text: The user input text to send to the model.
  """
  response = conversation.send_message(message_text)
  try:
    text = response["content"][0]["text"]
    print(f"\nAssistant:\n{text}")
  except (KeyError, IndexError, TypeError):
    print(f"\nAssistant:\n{response}")
