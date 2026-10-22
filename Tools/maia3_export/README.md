# Maia-3 model export

Converts the official Maia-3 checkpoints (5M, 23M, 79M) into files an iOS app can run:
**Core ML** (`.mlpackage`, the preferred runtime) and **ONNX** (the fallback). Each export
is checked against the upstream PyTorch model, and its size and hash go into
`manifest.json`. The model contract is in [`docs/maia3-spec.md`](../../docs/maia3-spec.md).

## Run

Needs macOS (Core ML conversion and its check run on the Mac) and Python 3.13.

```bash
cd Tools/maia3_export
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python export_maia3.py              # all models, about 10 minutes
.venv/bin/python export_maia3.py --models 5m  # one model
```

The checkpoints are downloaded from Hugging Face at pinned revisions into the usual
Hugging Face cache, and checked against the size and SHA-256 in the script. Outputs go to
`out/`, which isn't committed; `manifest.json` is.

The script exits non-zero if an fp32 export fails its check.

## Installing a model

```bash
Tools/maia3_export/install_model.sh --to <dir>              # 23M
Tools/maia3_export/install_model.sh --model 5m --to <dir>
```

Puts `maia3-<size>-fp32.mlpackage` into `<dir>`, for example an app's model folder, where
Xcode compiles it into the app. In order:

1. **The release download.** When `manifest.json` has a `release` entry, the script
   downloads the model's zip from that GitHub release and checks its SHA-256 against the
   entry. It needs only `curl`, `shasum` and `unzip`. A zip that doesn't match is refused.
2. **An existing export** in `out/`, when its content hash matches `manifest.json`.
3. **A new export** from the pinned checkpoint (Core ML only, using a scratch manifest so
   the committed one isn't touched), setting up `.venv` if needed. It refuses to install a
   model whose hash doesn't match.

Steps 2 and 3 need the Python environment, because the content hash reads the model spec.

## Releasing the models

```bash
Tools/maia3_export/release_models.sh <tag>             # zips 23M and 5M, records them in manifest.json
Tools/maia3_export/release_models.sh <tag> --publish   # after committing and pushing manifest.json
```

The first step checks each model in `out/` against its content hash, zips it into
`out/release/`, and writes the tag and each zip's size and SHA-256 into `manifest.json`'s
`release` entry. The second creates the GitHub release with the zips. Apps that pin a model
copy its zip's SHA-256 from there.

## What it does

For each model:

1. Loads the checkpoint into upstream's `MAIA3Model`, strictly (every key must match).
2. Builds 41 test inputs: fixed positions (both sides to move, castling, en passant,
   promotions) and positions from seeded random games in both history modes, with
   several Elo pairs.
3. Replaces `nn.RMSNorm` with the same formula written out, because the ONNX exporter
   can't export `aten::rms_norm` at opset 17. It checks that this changes nothing.
4. Exports:
   - `maia3-<size>.onnx`: opset 17, dynamic batch, int64 Elo inputs. Checked with
     ONNX Runtime (CPU).
   - `maia3-<size>-fp32.mlpackage`: ML Program, iOS 17+, batch 1, int32 Elo inputs.
     Checked with Core ML on CPU only and on all compute units.
   - `maia3-<size>-fp16.mlpackage`: kept only if it passes its looser check (see below).
5. Compares every output against PyTorch:
   - fp32 must satisfy **|error| ≤ 1e-4 + 1e-5 × |reference|** on every value of all
     three outputs (`numpy.allclose`), and pick the same top move everywhere. The relative
     term is there for 79M: its logits reach about 24, and plain float32 rounding puts its
     errors at about 1.3e-4 (PyTorch alone differs by 5e-5 between batched and single
     inference). 5M and 23M also meet a plain 1e-4.
   - fp16 must be within **1e-3** on legal-move probabilities (the Swift parity tolerance)
     and pick the same top move. If it fails, the file is deleted and the manifest records
     the failure.

## Reproducibility

- The ONNX file is byte-identical from run to run.
- An `.mlpackage` isn't: `Manifest.json` gets fresh UUIDs, and `model.mlmodel` reorders
  protobuf maps. The manifest therefore records `content_sha256`, which skips
  `Manifest.json` and hashes the model spec in deterministic serialization. Weight files
  are hashed as they are on disk.
- Pins: torch 2.7.0 and coremltools 9.0 go together (coremltools 9.0 fails with newer
  torch).

## Golden fixtures

`generate_fixtures.py` writes the test fixtures for the Swift port to `fixtures/` (committed;
about 1 MB per model). `verify_fixtures.py` re-encodes every case using only the rules in
`docs/maia3-spec.md`, as a check that the spec is complete. See
[`fixtures/README.md`](fixtures/README.md) for the format.

```bash
.venv/bin/python generate_fixtures.py                      # 23M and 5M, under a minute
.venv/bin/python verify_fixtures.py fixtures/*.json
```
