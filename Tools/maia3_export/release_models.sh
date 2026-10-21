#!/usr/bin/env bash
# Zips the 23M and 5M Core ML models from out/ and records each zip in manifest.json's
# "release" entry, which install_model.sh downloads from.
#
#   release_models.sh <tag>             # zips and records; commit manifest.json, push, then:
#   release_models.sh <tag> --publish   # creates the GitHub release with the zips
#
# The models must match manifest.json's content hashes (checked with the export's venv).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="fianchettolab/emula"
tag="${1:?Usage: $0 <tag> [--publish]}"
publish="${2:-}"
venv="$here/.venv"
zips="$here/out/release"

if [[ "$publish" == "--publish" ]]; then
    if [[ "$(plutil -extract release.tag raw -o - "$here/manifest.json")" != "$tag" ]]; then
        echo "manifest.json records another release; run $0 $tag first." >&2
        exit 1
    fi
    gh release create "$tag" --repo "$repo" --title "$tag" \
        --notes "Maia-3 Core ML models (fp32), checked against manifest.json." \
        "$zips/maia3-23m-fp32.mlpackage.zip" "$zips/maia3-5m-fp32.mlpackage.zip"
    exit 0
fi

rm -rf "$zips"
mkdir -p "$zips"
for size in 23m 5m; do
    model="maia3-$size-fp32.mlpackage"
    "$venv/bin/python" - "$here/out/$model" "$here" "maia3-$size" <<'EOF'
import json, sys
from pathlib import Path
path, here, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(here))
from export_maia3 import describe
artifacts = json.loads((here / "manifest.json").read_text())["models"][key]["artifacts"]
expected = next(a["content_sha256"] for a in artifacts if a["file"] == path.name)
sys.exit(0 if describe(path)["content_sha256"] == expected else f"{path.name} doesn't match manifest.json")
EOF
    (cd "$here/out" && zip -qrX "$zips/$model.zip" "$model")
done

"$venv/bin/python" - "$here" "$tag" <<'EOF'
import hashlib, json, sys
from pathlib import Path
here, tag = Path(sys.argv[1]), sys.argv[2]
manifest_path = here / "manifest.json"
manifest = json.loads(manifest_path.read_text())
assets = {}
for size in ("23m", "5m"):
    zip_path = here / "out" / "release" / f"maia3-{size}-fp32.mlpackage.zip"
    assets[f"maia3-{size}"] = {
        "file": zip_path.name,
        "bytes": zip_path.stat().st_size,
        "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }
manifest["release"] = {"tag": tag, "assets": assets}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
for key, asset in assets.items():
    print(f"{asset['file']}: {asset['bytes']} bytes, sha256 {asset['sha256']}")
EOF
echo "Recorded $tag in manifest.json. Commit and push it, then run: $0 $tag --publish"
