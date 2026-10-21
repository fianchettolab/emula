#!/usr/bin/env bash
# Puts a Maia-3 Core ML model into a directory, for example an app's model folder.
#
#   install_model.sh [--model 23m|5m|79m] --to <dir>
#
# 1. Downloads the model from the release in manifest.json and checks the zip's SHA-256
#    (needs only curl, shasum and unzip; gh while the repo is private). EMULA_RELEASE_URL
#    replaces the releases URL, for testing.
# 2. Otherwise reuses an existing export in out/ when its content hash matches manifest.json.
# 3. Otherwise exports it from the pinned checkpoint (first run: sets up the Python venv, a
#    few minutes).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="fianchettolab/emula"
size="23m"
destination=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) size="$2"; shift 2 ;;
        --to) destination="$2"; shift 2 ;;
        *) echo "Usage: $0 [--model 23m|5m|79m] --to <dir>" >&2; exit 2 ;;
    esac
done
if [[ -z "$destination" || ! "$size" =~ ^(5m|23m|79m)$ ]]; then
    echo "Usage: $0 [--model 23m|5m|79m] --to <dir>" >&2
    exit 2
fi
mkdir -p "$destination"
destination="$(cd "$destination" && pwd)"
model="maia3-$size-fp32.mlpackage"

# Reads a value from manifest.json by key path, empty if it isn't there.
manifest_value() {
    plutil -extract "$1" raw -o - "$here/manifest.json" 2>/dev/null || true
}

# 1. The release download.
tag="$(manifest_value release.tag)"
zip="$(manifest_value "release.assets.maia3-$size.file")"
sha256="$(manifest_value "release.assets.maia3-$size.sha256")"
if [[ -n "$tag" && -n "$zip" && -n "$sha256" ]]; then
    scratch="$(mktemp -d)"
    trap 'rm -rf "$scratch"' EXIT
    url="${EMULA_RELEASE_URL:-https://github.com/$repo/releases/download}/$tag/$zip"
    echo "Downloading $zip ($tag)…"
    if curl -fsSL -o "$scratch/$zip" "$url" \
        || { command -v gh >/dev/null && gh release download "$tag" --repo "$repo" --pattern "$zip" --dir "$scratch" 2>/dev/null; }; then
        if [[ "$(shasum -a 256 "$scratch/$zip" | cut -d' ' -f1)" != "$sha256" ]]; then
            echo "$zip doesn't match the SHA-256 in manifest.json; not installing it." >&2
            exit 1
        fi
        rm -rf "${destination:?}/$model"
        unzip -q "$scratch/$zip" -d "$destination"
        echo "Installed $model in $destination."
        exit 0
    fi
    echo "The download failed; using a local export instead."
fi

# 2 and 3. A local export, checked by its content hash (needs the export's Python venv).
venv="$here/.venv"
if [[ ! -x "$venv/bin/python" ]]; then
    if ! command -v python3.13 >/dev/null; then
        echo "Python 3.13 is needed (for example: brew install python@3.13)." >&2
        exit 1
    fi
    echo "Setting up the export environment in $venv…"
    python3.13 -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet -r "$here/requirements.txt"
fi

# Exits 0 when the package at $1 matches the manifest's content hash for the model.
matches_manifest() {
    "$venv/bin/python" - "$1" "$here" "maia3-$size" <<'EOF' 2>/dev/null
import json, sys
from pathlib import Path
path, here, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(here))
from export_maia3 import describe
manifest = json.loads((here / "manifest.json").read_text())
artifacts = manifest["models"][key]["artifacts"]
expected = next(a["content_sha256"] for a in artifacts if a["file"] == path.name)
sys.exit(0 if path.is_dir() and describe(path)["content_sha256"] == expected else 1)
EOF
}

if [[ -d "$here/out/$model" ]] && matches_manifest "$here/out/$model"; then
    echo "Using the existing export in out/."
else
    echo "Exporting Maia-3 $(echo "$size" | tr a-z A-Z) to Core ML…"
    manifest_scratch="$(mktemp -d)"
    # A scratch manifest, so the committed one isn't rewritten.
    (cd "$here" && "$venv/bin/python" export_maia3.py --models "$size" --formats coreml --skip-fp16 \
        --manifest "$manifest_scratch/manifest.json")
    rm -rf "$manifest_scratch"
    if ! matches_manifest "$here/out/$model"; then
        echo "The export doesn't match manifest.json; not installing it." >&2
        exit 1
    fi
fi

if [[ "$destination" != "$here/out" ]]; then
    rm -rf "${destination:?}/$model"
    cp -R "$here/out/$model" "$destination/"
    matches_manifest "$destination/$model"
fi
echo "Installed $model in $destination."
