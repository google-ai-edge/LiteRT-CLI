#!/bin/bash
set -ex

# Portable test runner for GitHub
cd "$(dirname "$0")/.."

# Set up venv
python3 -m venv venv_test
source venv_test/bin/activate

# Install package in editable mode with stable subset of optional dependencies
pip install -q absl-py
pip install -q -e ".[download,run,image]"

# Run python tests using discover
python3 -m unittest discover -s litert_cli -p "*_test.py"

# Run a subset of CLI commands to verify they load
litert --help
litert download --help
litert convert --help
litert quantize --help
litert run --help
litert visualize --help
litert benchmark --help
