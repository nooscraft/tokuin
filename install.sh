#!/usr/bin/env bash

set -euo pipefail

REPO="nooscraft/tokuin"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0" >&2
      exit 1
      ;;
  esac
done

for cmd in curl tar; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd is required" >&2
    exit 1
  fi
done

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Error: python3 (or python) is required" >&2
  exit 1
fi

uname_s=$(uname -s)
uname_m=$(uname -m)

case "$uname_s" in
  Darwin)
    case "$uname_m" in
      x86_64) target="x86_64-apple-darwin" ;;
      arm64|aarch64) target="aarch64-apple-darwin" ;;
      *) echo "Unsupported macOS architecture: $uname_m" >&2; exit 1 ;;
    esac
    archive_ext="tar.gz"
    ;;
  Linux)
    case "$uname_m" in
      x86_64) target="x86_64-unknown-linux-gnu" ;;
      arm64|aarch64) target="aarch64-unknown-linux-gnu" ;;
      *) echo "Unsupported Linux architecture: $uname_m" >&2; exit 1 ;;
    esac
    archive_ext="tar.gz"
    ;;
  *)
    echo "Unsupported operating system: $uname_s" >&2
    exit 1
    ;;
esac

install_dir="/usr/local/bin"
if [ ! -w "$install_dir" ]; then
  install_dir="$HOME/.local/bin"
  mkdir -p "$install_dir"
  echo "Installing to $install_dir (make sure this directory is on your PATH)."
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

echo "Fetching latest release metadata for $target..."
release_json=$(curl -fsSL \
  -H "Accept: application/vnd.github+json" \
  -H "User-Agent: tokuin-installer" \
  "$API_URL")

if [ -z "$release_json" ]; then
  echo "Failed to fetch release metadata from GitHub. Please try again later." >&2
  exit 1
fi

asset_url=$(RELEASE_JSON="$release_json" TARGET="$target" "$PYTHON_BIN" <<'PY'
import json
import os

data = json.loads(os.environ["RELEASE_JSON"])
target = os.environ["TARGET"]
for asset in data.get("assets", []):
    name = asset.get("name", "")
    if name.endswith(f"{target}.tar.gz"):
        print(asset["browser_download_url"])
        break
else:
    print("")
PY
)

if [ -z "$asset_url" ]; then
  echo "Unable to find a release artifact for target $target" >&2
  exit 1
fi

checksums_url=$(RELEASE_JSON="$release_json" "$PYTHON_BIN" <<'PY'
import json
import os

data = json.loads(os.environ["RELEASE_JSON"])
for asset in data.get("assets", []):
    if asset.get("name") == "checksums.txt":
        print(asset["browser_download_url"])
        break
PY
)

asset_name=$(basename "$asset_url")
asset_path="$tmp_dir/$asset_name"

echo "Downloading $asset_name..."
curl -fsSL "$asset_url" -o "$asset_path"

if [ -n "$checksums_url" ]; then
  checksum_path="$tmp_dir/checksums.txt"
  echo "Downloading checksums.txt..."
  curl -fsSL "$checksums_url" -o "$checksum_path"

  if command -v sha256sum >/dev/null 2>&1; then
    actual_checksum=$(sha256sum "$asset_path" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    actual_checksum=$(shasum -a 256 "$asset_path" | awk '{print $1}')
  else
    echo "Warning: neither sha256sum nor shasum available; skipping checksum verification." >&2
    actual_checksum=""
  fi

  if [ -n "$actual_checksum" ]; then
    expected_checksum=$(grep " $asset_name" "$checksum_path" | awk '{print $1}')
    if [ -z "$expected_checksum" ]; then
      echo "Warning: checksum entry not found for $asset_name" >&2
    elif [ "$expected_checksum" != "$actual_checksum" ]; then
      echo "Checksum verification failed for $asset_name" >&2
      exit 1
    else
      echo "Checksum verified."
    fi
  fi
else
  echo "Warning: checksums.txt not found; skipping checksum verification." >&2
fi

echo "Extracting archive..."
tar -C "$tmp_dir" -xzf "$asset_path"

if [ ! -f "$tmp_dir/tokuin" ]; then
  echo "Installation failed: tokuin binary not found in archive" >&2
  exit 1
fi

install_path="$install_dir/tokuin"
mv "$tmp_dir/tokuin" "$install_path"
chmod +x "$install_path"

echo "Installed tokuin to $install_path"

case ":$PATH:" in
  *:"$install_dir":*) ;;
  *)
    echo ""
    echo "Note: $install_dir is not on your PATH."
    echo "Add the following to your shell profile (~/.zshrc or ~/.bashrc):"
    echo "  export PATH=\"$install_dir:\$PATH\""
    ;;
esac

echo ""
echo "Done! Run 'tokuin --help' to get started."

