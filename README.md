# Emula

[Maia-3](https://github.com/CSSLab/maia3), CSSLab's human-like chess engine, on Core ML in
Swift: it plays the move a human of a given rating would play, fully on the device.

*Emula* is Italian for a rival who competes by imitating, which is what Maia does.

It's built for [Fianchetto Lab](https://github.com/fianchettolab/fianchetto), a chess app
for iOS, and comes as-is. It isn't affiliated with CSSLab or the Maia project.

## What it does

```swift
import Emula

let session = try await MaiaSession(modelURL: modelURL)    // a compiled Maia-3 model
let result = try await session.bestMove(
    in: MaiaGame(moves: ["e2e4", "c7c5"]),                  // UCI, from any start FEN
    selfElo: 1500, opponentElo: 1500,
    temperature: 1, topP: 0.95,
    historyMode: .recentPositions
)
result.move        // in UCI, for example "g1f3"
result.topMoves    // the most likely moves with their probabilities
result.wdl         // win, draw and loss for the side to move
```

The chess rules come from [Scacchi](https://github.com/fianchettolab/scacchi). The model
contract (input encoding, move vocabulary, value head, sampling) is in
[`docs/maia3-spec.md`](docs/maia3-spec.md), and the measurements on the device in
[`docs/maia3-perf.md`](docs/maia3-perf.md).

## Layout

| Path | What it is |
|---|---|
| `Sources/Emula/` | `MaiaSession` (actor: Core ML, `bestMove`), `MaiaGame`, `MaiaEncoder`, `MaiaMoveDecoder`, `MaiaSampler` |
| `Tests/EmulaTests/` | The encoder against the Python fixtures, the legal moves against python-chess, the models' parity, the sampler, seeded self-play |
| `Tools/maia3_export/` | Exports Maia-3 to Core ML and ONNX, checks parity, writes `manifest.json` and the golden fixtures; installs and releases the models (see its README) |
| `docs/` | The model contract and the performance notes |
| `Package.swift` | `Emula`, over Scacchi |

### Notes

- Swift 6 language mode; iOS 17+ and macOS 14+, so `swift test` runs on the Mac.
- Scacchi's and Maia's square indexes are both rank-first (a1 = 0, b1 = 1).
- `MLModel` isn't `Sendable` (Scacchi's `Game` is). The public API uses Emula's own types
  (`MaiaGame`, `MaiaResult`), and the session calls Core ML's synchronous
  `prediction(from:)` on its own executor.
- The sampler follows upstream's top-p rule exactly: it leaves out the move that crosses
  the threshold (spec 8.5).

## The model

The models aren't in git and aren't package resources (SwiftPM binary targets only take
XCFrameworks). Install one into your app's sources, where Xcode compiles it:

```bash
Tools/maia3_export/install_model.sh --to <your app>/Maia
```

The script downloads the model from this repo's release and checks its SHA-256 against
`manifest.json`, or exports it from the pinned checkpoint (Python 3.13). The app uses
23M (about 92 MB in fp32, about 5 ms a move on an iPhone 17); 5M is for the tests.

## Building

During development, Emula and Scacchi are checked out next to each other
(`../scacchi`, built once with its `build-xcframework.sh`). Then:

```bash
Tools/maia3_export/install_model.sh --to Tools/maia3_export/out
Tools/maia3_export/install_model.sh --model 5m --to Tools/maia3_export/out
swift test
```

The tests load the 23M and 5M models from `Tools/maia3_export/out/` (where the export puts
them too) and skip the model tests when they're missing. With both, all 23 run (about 30 s
on the Mac).

## Licence

AGPL-3.0, like Maia-3. See `LICENSE`. Maia-3's attribution and citation, and the other
components, are in `THIRD_PARTY_NOTICES.md`.
