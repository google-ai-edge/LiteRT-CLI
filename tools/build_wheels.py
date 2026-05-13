import argparse
import datetime
import os
import subprocess
import sys

# Mapping of base package name to its pinned stable version specifier.
# Update these version numbers when preparing a new stable release.
STABLE_DEPENDENCIES = {
    "ai-edge-litert": "==2.1.4",
    "ai-edge-litert-sdk-qualcomm": "==2.1.4",
    "ai-edge-litert-sdk-mediatek": "==2.1.4",
    "litert-torch": "==2.1.4",
    "ai-edge-quantizer": "==2.1.4",
    "litert-lm": "==2.1.4",
}

def configure_build(is_nightly: bool, restore: bool = False):
    pyproject_path = "pyproject.toml"
    version_path = "VERSION"

    if not os.path.exists(pyproject_path) or not os.path.exists(version_path):
        raise FileNotFoundError("Could not find pyproject.toml or VERSION. Please run this script from the repository root.")

    # 1. Handle VERSION file
    with open(version_path, "r") as f:
        original_version = f.read().strip()

    # If original_version already has .dev, strip it to get clean base version
    base_version = original_version.split(".dev")[0]

    if restore or not is_nightly:
        # Restore or stable build uses clean base version
        with open(version_path, "w") as f:
            f.write(f"{base_version}\n")
    else:
        # Nightly build appends .devYYYYMMDD
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        nightly_version = f"{base_version}.dev{date_str}"
        with open(version_path, "w") as f:
            f.write(f"{nightly_version}\n")

    # 2. Handle pyproject.toml
    with open(pyproject_path, "r") as f:
        content = f.read()

    # Normalize project name to base 'litert-cli'
    content = content.replace('name = "litert-cli-nightly"', 'name = "litert-cli"')
    if is_nightly and not restore:
        content = content.replace('name = "litert-cli"', 'name = "litert-cli-nightly"')

    # 3. Configure dependencies
    # If restore is True, we always restore to nightly dependencies for local development
    to_nightly_deps = True if restore else is_nightly

    for pkg, stable_ver in STABLE_DEPENDENCIES.items():
        if to_nightly_deps:
            content = content.replace(f'"{pkg}{stable_ver}"', f'"{pkg}-nightly"')
            content = content.replace(f'"{pkg}"', f'"{pkg}-nightly"')
        else:
            content = content.replace(f'"{pkg}-nightly"', f'"{pkg}{stable_ver}"')

    with open(pyproject_path, "w") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Build stable or nightly wheels using uv.")
    parser.add_argument("--type", choices=["stable", "nightly"], default="nightly", help="Type of wheel to build (stable or nightly).")
    args = parser.parse_args()

    is_nightly = args.type == "nightly"
    print(f"Configuring setup for {args.type} build...")
    configure_build(is_nightly=is_nightly, restore=False)

    try:
        print("Running 'uv build'...")
        subprocess.run(["uv", "build"], check=True)
        print(f"Successfully built {args.type} wheels in ./dist")
    except subprocess.CalledProcessError as e:
        print(f"Error during build: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Always restore files back to development defaults
        print("Restoring files to development defaults...")
        configure_build(is_nightly=is_nightly, restore=True)

if __name__ == "__main__":
    main()
