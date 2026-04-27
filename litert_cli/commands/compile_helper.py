"""Helper script for AOT compilation in a child process.

This script reads arguments from stdin as JSON and performs the actual AOT
compilation using the ai_edge_litert library. It isolates the dynamic library
loading issues.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export
from litert_cli.core import npu_utils


def main():
    # Read arguments from stdin
    try:
        args = json.load(sys.stdin)
    except Exception as e:
        print(f"ERROR_IN_CHILD: Failed to parse JSON args: {e!r}", file=sys.stderr)
        sys.exit(1)

    model_path = pathlib.Path(args["model_path"])
    target = args["target"]
    output_dir = pathlib.Path(args["output_dir"])
    export_aipack = (
        pathlib.Path(args["export_aipack"]) if args["export_aipack"] else None
    )
    base_name = model_path.stem

    aot_targets = [npu_utils.get_aot_target(t) for t in target]

    try:
        print(f"Compiling model {model_path} for targets: {', '.join(target)}")
        compiled_models = aot_lib.aot_compile(
            str(model_path),
            target=aot_targets,
            keep_going=False,
        )

        if export_aipack:
            print(f"Exporting AI Pack to: {export_aipack}")
            if export_aipack.exists():
                if export_aipack.is_dir():
                    shutil.rmtree(export_aipack)
                else:
                    export_aipack.unlink()
            export_aipack.mkdir(parents=True, exist_ok=True)
            ai_pack_export.export(
                compiled_models, str(export_aipack), base_name, "model"
            )
        else:
            print(f"Exporting compiled TFLite to: {output_dir}")
            compiled_models.export(str(output_dir), model_name=base_name)
    except Exception as e:
        print(f"ERROR_IN_CHILD: {e!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
