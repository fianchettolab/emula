# AGENTS.md

Emula is part of Fianchetto Lab. Handoff notes for anyone (human or agent) picking up
Emula cold: how to work here and the decisions already settled with the owner. What the
package does, its layout and how to build it are in [README.md](README.md); the model
export is in [Tools/maia3_export/README.md](Tools/maia3_export/README.md), and the model
contract in [docs/maia3-spec.md](docs/maia3-spec.md). Read the spec before touching the
engine code.

## Working with the owner

- **Advise before applying** anything non-trivial (API changes, new dependencies, model
  or sampling changes): lay out the options and a recommendation, then wait for a
  go-ahead.
- **Let the owner test before committing.** Don't commit until they say so.
- **Releases and pushes need a go-ahead.** A release is public the moment it's created,
  and apps download its models.
- **Record settled decisions here**, as the current state, not as a history of changes.
- **Commits** use the owner's GitHub noreply address (set in the global git config); never
  their personal email.

## Build and test

- **Models for the tests:** `Tools/maia3_export/install_model.sh --to Tools/maia3_export/out`
  and the same with `--model 5m`. The model tests skip without them.
- **Tests:** `swift test` (25, about 30 s on the Mac) checks the encoder against the
  Python fixtures, Scacchi's legal moves against python-chess, both models' parity, the
  sampler and seeded self-play. Keep builds free of warnings in the Swift 6 language mode.
- **The Python side** (export, fixtures, parity) needs Python 3.13 and macOS; see the
  tools' README.
- **Scacchi** comes from its release by URL. To change it alongside, use a local checkout
  as an override (`swift package edit scacchi --path ../scacchi`, or Xcode); don't commit
  a path dependency.

## Settled decisions

- **Runtime:** Core ML in fp32. fp16 fails the 1e-3 parity target for every model size,
  so it isn't shipped. ONNX Runtime is the fallback, added only if Core ML ever fails on a
  device.
- **Parity:** fp32 exports must match PyTorch within |error| ≤ 1e-4 + 1e-5 × |reference|,
  with the same top move everywhere. The relative term is for 79M's larger logits. The
  Swift port must stay within 1e-3 on the top-10 probabilities and W/D/L.
- **Sizes:** 23M is the one to ship (about 92 MB in fp32, about 5 ms a move on an
  iPhone 17, against a target of under 100 ms warm on a recent iPhone). 5M is for the
  tests. 79M (316 MB) gains little over 23M and isn't released.
- **Sampling** follows upstream's top-p rule exactly: it leaves out the move that crosses
  the threshold (spec 8.5). Temperature 0 is argmax, lowest index on ties.
- **History modes:** both are supported (`MaiaHistoryMode`). The current position eight
  times is the official engine's default; the last eight positions is upstream's
  `--use-uci-history`. They play differently, so callers choose.
- **`maia3Commit` reports the upstream** (apps show it next to Emula's own version):
  Maia-3 has no releases, so the commit is its version. A test checks it against
  `manifest.json` and the models' Core ML metadata; moving to a new upstream commit
  changes it with `MAIA3_COMMIT`, the spec and a re-export.
- **Chess rules** come from Scacchi only; don't add another chess library. Squares are
  rank-first in both (a1 = 0).
- **Concurrency:** `MaiaSession` is an actor and calls Core ML's synchronous
  `prediction(from:)` on its own executor, because `MLModel` isn't `Sendable`. The public
  API uses Emula's own types (`MaiaGame`, `MaiaResult`).
- **Models aren't in git** or package resources (SwiftPM binary targets only take
  XCFrameworks). They're release assets, listed in `manifest.json` with each zip's
  SHA-256; apps pin a tag and that hash.
- **Licence:** AGPL-3.0, like Maia-3, and the weights are treated as AGPL-3.0. Apps that
  ship Emula show Maia-3's attribution and citation (`THIRD_PARTY_NOTICES.md`) and
  Scacchi's notices.

## Releasing

A release tags the code and carries the models, under one version:

1. `Tools/maia3_export/release_models.sh <version>`: zips the 23M and 5M models from
   `out/` (checked against their content hashes) and records them in `manifest.json`.
2. Commit, and tag `<version>`.
3. With the owner's go-ahead: `git push origin main <version>`, then
   `Tools/maia3_export/release_models.sh <version> --publish`.

A new model changes the zips' hashes, so apps update their pin. Versions follow semantic
versioning; while it's 0.x, a minor version may break the API.

## Gotchas

- **Pins:** torch 2.7.0 and coremltools 9.0 go together; coremltools 9.0 fails with newer
  torch.
- **An `.mlpackage` isn't byte-reproducible:** `Manifest.json` gets fresh UUIDs on every
  export. Compare models by `content_sha256` (the spec in deterministic serialization),
  never by file hashes.
- **After changing `Package.swift`,** Xcode may need File → Packages → Reset Package
  Caches.
