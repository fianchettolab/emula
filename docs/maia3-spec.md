# Maia-3 model specification

The ground truth for porting Maia-3 to Swift: how a game becomes model input, what the
outputs mean, and how the reference engine picks a move. Every claim cites the upstream
source; claims marked **[run]** were also checked by running the code (see
[Verification run](#12-verification-run)).

Citation format: `file.py:lines` refers to **CSSLab/maia3 at
`1e13597c42d4858b7cfd7cfdae01e297263364b2`** (2026-05-25) unless prefixed with `android:`,
which refers to **Dash1971/maia-chess-android at
`9f7eaa51667d9b123f121e396016643a7a71d78c`** (2026-10-04).

## 1. Sources and pins

| Item | Pin |
|---|---|
| Code | `github.com/CSSLab/maia3` @ `1e13597c42d4858b7cfd7cfdae01e297263364b2` |
| Android reference | `github.com/Dash1971/maia-chess-android` @ `9f7eaa51667d9b123f121e396016643a7a71d78c` |
| Paper | Monroe, Eilender, Chalmers, Tang, Anderson. *Chessformer: A Unified Architecture for Chess Modeling.* ICLR 2026. arXiv:2605.19091 |

Checkpoints (Hugging Face, `UofTCSSLab/*`; sizes and SHA-256 from the HF API; the 5M hash
was re-checked locally **[run]**):

| Model | HF revision | File | Bytes | SHA-256 |
|---|---|---|---|---|
| Maia3-5M | `b6559de2398d7140b985f28fd2c19fb5e47ddabe` | `maia3-5m.pt` | 20,968,049 | `ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f` |
| Maia3-23M | `51a0145a8178046f7de23119160b136672deeb2b` | `maia3-23m.pt` | 91,799,307 | `bce6cd1af5f0399ac7eed33fabb7a6a2ef6193662c2740f262bf93af7bfb3569` |
| Maia3-79M | `a107d6ceb7b298cb04ae1da4edffe2939858b894` | `maia3-79m.pt` | 315,651,851 | `3fc6181d5db789b45a15305732148757ae74efa3e0028e81ba335b462dac45c2` |

Architecture per model (`model_registry.py:33-115`): all share `history 8`, `dim_emb 128`,
8 blocks, RMSNorm, GELU, GAB attention bias. They differ in width: 5M is `dim_vit 256`,
8 heads; 23M is 512, 16 heads; 79M is 1024, 32 heads. The 5M checkpoint is a plain
state dict of 5,230,084 parameters whose keys use the old `smolgen` prefix, renamed to
`gab` on load (`uci.py:146-151`); after renaming nothing is missing or unexpected **[run]**.

## 2. Model interface

`MAIA3Model.forward(tokens, self_elos, oppo_elos)` (`models.py:351-400`):

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `tokens` | float32 | `[B, 64, 97]` | Board history, section 3 |
| `self_elo` | int64 | `[B]` | Elo of the side to move, section 4 |
| `opponent_elo` | int64 | `[B]` | Elo of the other side |
| `move_logits` | float32 | `[B, 4352]` | One logit per vocabulary entry, section 5 |
| `value_logits` | float32 | `[B, 3]` | `[loss, draw, win]`, section 6 |
| `ponder` | float32 | `[B]` | Scalar head, section 7 |

The ONNX names above are the Android export's (`android:tool/export_maia3_onnx.py:63-64`).
The Swift side should treat this table as the contract for any runtime (section 11).

## 3. Input encoding (`tokens`)

### 3.1 One frame: 64 × 12

`tokenize_board` (`dataset.py:16-30`):

- **Rows are squares** in python-chess order, iterated over `chess.SQUARES`
  (`dataset.py:23`): a1 = 0, b1 = 1 … h1 = 7, a2 = 8 … h8 = 63. Index = `rank * 8 + file`,
  both zero-based.
- **Black to move → mirror.** If the side to move is Black, the board is replaced by
  `board.mirror()` (`dataset.py:20-21`). python-chess `mirror()` flips ranks (rank r → 9 − r,
  files unchanged; *not* a 180° rotation) and swaps colours. After this, the side to move is
  always "White" on ranks 1–2.
- **Channels** (`dataset.py:6-13,26-28`): `channel = PIECE_MAP[type] + (6 if black else 0) - 1`
  with P=1, N=2, B=3, R=4, Q=5, K=6. After mirroring this means:

  | Channel | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Piece | P | N | B | R | Q | K | p | n | b | r | q | k |
  | Owner | side to move | | | | | | opponent | | | | | |

- **Nothing else is encoded**: no castling rights, en-passant square, side-to-move flag,
  repetition count or move counters. Only piece placement.

Worked examples **[run]**:
- Start position: square 0 (a1) has channel 3 (rook) set; square 4 (e1) has channel 5 (king).
- After 1.e4 (Black to move): the frame looks like a start position for "White" with the
  opponent's pawn moved from e7 to e5 (the real e4 pawn, mirrored to rank 5); Black's
  pieces sit on ranks 1–2 in channels 0–5, and Black's king is on e1 (square 4).

### 3.2 History: 8 frames + 1 column

`get_historical_tokens` (`dataset.py:44-66`):

- Frames are concatenated along the channel axis, **oldest first, current last**
  (`dataset.py:51`). Frame k occupies columns `12k ..< 12k+12`; the current position is
  frame 7, columns 84–95.
- **Padding**: with fewer than 8 frames, the **earliest** frame is repeated on the left
  (`dataset.py:52-54`); the docstring says inference always pads this way
  (`dataset.py:47-48`).
- **Column 96** is `clk_ponder / 100` (`dataset.py:62-64`). The engine always passes 0
  (`uci.py:278-280`), and the model slices it off before use:
  `tokens[:, :, :12 * history]` (`models.py:355-358`). Its value never affects the output;
  write 0.
- History length is 8 for every released model (`model_registry.py:34`, spread into every
  spec at lines 58, 74, 90, 106), and `include_time_info` is False (`model_registry.py:36`),
  so the shape is always `[64, 97]`. Layout in memory (row-major) is
  `tokens[square * 97 + column]`.

### 3.3 The two history modes

The model accepts either; **both are specified, and the app picks one after on-device
testing.**

**(a) Current position × 8**: the official engine's default and the Android app's only
mode. Without `--use-uci-history`, the history deque holds just the current position
(`uci.py:262-266,451-462`), so padding fills all 8 frames with it **[run]**. Android
does the same explicitly (`android:lib/src/maia.dart:74-89`).

**(b) Real history**: with `--use-uci-history` (`uci.py:77-78`), the engine replays
`position … moves …` from the start position or FEN, appending one frame per ply, keeping
the last 8 (`uci.py:436-450`). Frames before the FEN/start position don't exist and are
padded with the earliest one. Each frame is tokenized **from its own side to move's point
of view** (`tokenize_board` mirrors per frame), so consecutive frames alternate between
unmirrored and mirrored, and channels 0–5 alternate between White's and Black's pieces
**[run]**.

Known issue with (b): the Android app used real history at first and reported that Black
copied White's moves ("false copy-the-opponent signal", `android:lib/src/maia.dart:75-78`;
fixed by switching to (a) in commit `49da564dff6896725176d1ba5e6c0d8f1fefecf0`). Upstream's
mode (b) builds frames the same way. Whether training used the same per-frame mirroring is
not knowable from this repository (no training code), so (b) should be tested against
(a) before choosing it.

Effect on the output after 1.e4, 1500/1500 (5M) **[run]**:

| Mode | Top moves for Black |
|---|---|
| (a) current ×8 | e5 42.5%, e6 12.2%, d5 11.0%, c5 9.8%, c6 5.8% |
| (b) real history | e5 48.4%, c5 15.1%, d5 9.4%, e6 8.8%, c6 7.1% |

## 4. Elo

- Raw integers, `torch.long` (`uci.py:312-313`); not bucketed.
- Clamped to [0, 5000], then a linear blend of two learned 128-d vectors:
  `w = elo / 5000; emb = w * emb_low + (1 - w) * emb_high` (`models.py:337-348`). The
  names look swapped (the "low" vector grows with Elo); this has no effect at inference.
- Both embeddings are broadcast to all 64 squares and concatenated after the 96 piece
  columns before the input projection (`models.py:360-366`).
- `self_elo` is the side to move, `opponent_elo` the other side. Defaults: both 1500
  (`uci.py:71,227-228`); UCI options `Elo`, `SelfElo`, `OppoElo`, range 0–5000
  (`uci.py:364-366,383-389`). Android always passes the same value for both and limits it
  to 600–2600.

## 5. Move vocabulary

`get_all_possible_moves` (`utils.py:23-46`) builds 4352 entries, all distinct **[run]**,
in the side-to-move frame (Black's moves are mirrored):

- **0–4095, plain moves**: `index = from * 64 + to`, squares numbered as in 3.1
  (`utils.py:26-34`). Every from/to pair is present, including impossible ones like a1a1.
- **4096–4351, promotions**: `index = 4096 + (fromFile * 8 + toFile) * 4 + piece`, with
  piece q=0, r=1, b=2, n=3, always from rank 7 to rank 8 (`utils.py:36-44`). The model
  builds these logits in the same order (`models.py:384-394`). Underpromotions have their
  own entries. A promotion's plain entry (e.g. a7a8 at index 3128) is never legal, because
  python-chess always writes the piece letter (`dataset.py:36-39`).
- **Castling** is the king's two-square move, `e1g1` / `e1c1`, as python-chess writes UCI
  for standard chess **[run]**. Chess960 isn't supported.
- **En passant** is an ordinary pawn move (e.g. e5d6) **[run]**.
- **Mirroring**: for Black, a legal move is mirrored before lookup (`dataset.py:37`), and a
  chosen index is mirrored back (`uci.py:283-295`). `mirror_move` maps rank r → 9 − r on
  both squares and keeps the promotion letter (`utils.py:49-66`).

Worked examples **[run]** (Android has unit tests for the first four):

| Position | Move | Lookup string | Index |
|---|---|---|---|
| White to move | e2e4 | e2e4 | 796 |
| White to move | g1f3 | g1f3 | 405 |
| Black to move | e7e5 | e2e4 | 796 |
| White to move | a7a8q | a7a8q | 4096 |
| White to move | a7a8n | a7a8n | 4099 |
| White to move | h7h8n | h7h8n | 4351 |
| Black to move | b2b1q | b7b8q | 4132 |
| Black to move | b2b1n | b7b8n | 4135 |
| White, O-O | e1g1 | e1g1 | 262 |
| White, O-O-O | e1c1 | e1c1 | 258 |
| Black, O-O | e8g8 | e1g1 | 262 |
| Black, O-O-O | e8c8 | e1c1 | 258 |
| White, e.p. | e5d6 | e5d6 | 2347 |

Swift note: Scacchi's `Game.legalMoves()` produces this UCI format (lowercase promotion
letter, king-move castling), so the lookup string can come straight from it.

## 6. Value head

- `value_logits` is `[loss, draw, win]` **from the side to move's point of view**
  (`uci.py:196-200`). Apply softmax to get probabilities.
- Start position, 1500/1500 (5M) **[run]**: loss 47.0%, draw 2.6%, win 50.4% for White.
- How the UCI engine reports WDL for a candidate move (`uci.py:336-354`): it plays the
  move, evaluates the new position with the two Elos **swapped** (the opponent is now to
  move), and swaps win and loss back. It prints `wdl` in permille as
  win/draw/loss (`uci.py:186-200`) and `score cp` = win − loss (`uci.py:208-210`). The
  README calls this cp a GUI compatibility value, not a search score.

## 7. Ponder head

A scalar regression head (`models.py:328-329,398-400`). The only hint at its meaning is
the comments calling the clock column a "label" (`models.py:300,353-354`), which suggests
it predicts thinking time. Units and training loss aren't documented, and neither the
engine (`uci.py:316`) nor Android uses it. Start position, 1500 (5M): 0.41 **[run]**.
**Unverified; don't build on it.**

## 8. Sampling (official engine)

`score_moves` and `sample_from_logits` (`uci.py:163-183,297-356`):

1. If the game is over (python-chess `is_game_over()`), return no move (`uci.py:300-301`).
2. Mask illegal moves to −inf (`uci.py:303,318-320`).
3. **T ≤ 0**: argmax (`uci.py:167-168`). Ties go to the lowest vocabulary index.
4. **T > 0**: `p = softmax(logits / T)` (`uci.py:170`).
5. **Top-p < 1**: sort descending; keep moves while the cumulative probability is
   **≤ p**, and always keep the top move; renormalize and sample (`uci.py:172-181`). The
   move that crosses p is *excluded*. (Standard nucleus sampling and the Android app
   include it.)
6. Otherwise sample from `p` with `torch.multinomial` (`uci.py:183`).
7. Map the index back to a move, mirroring for Black (`uci.py:283-295`).

Defaults: T = 1.0, top-p = 1.0, seed 42 (`uci.py:68,72-73,524`). The preset commands
`maia3-5m`, `maia3-23m`, `maia3-79m` force T = 0 (`presets.py:41`). MultiPV reports
`softmax(masked logits)` at T = 1, not the sampling distribution (`uci.py:325-327`).

## 9. Android reference: differences from upstream

The Android app agrees with upstream on encoding, vocabulary and mirroring
(`android:lib/src/maia.dart:91-123,255-278`). It differs in:

| | Upstream | Android |
|---|---|---|
| History | (a) by default, (b) optional | (a) only |
| Top-p | excludes the crossing move | includes it (`maia.dart:167-176`) |
| Temperature | unbounded | clamped to 0.001–1.0 (`maia.dart:144`) |
| Argmax ties | lowest vocab index | first in legal-move order |
| Elo | separate self/opponent | always equal; 600–2600 |
| value / ponder | value used for WDL | both ignored |

Pitfalls it recorded:
- ONNX Runtime 1.23.2 crashed with SIGILL on CPUs with SME instructions; fixed in 1.24.3
  (`android:docs/release-notes/v2.1.3.md:7-10`). Apple M4-class chips also
  have SME, so any ONNX Runtime build must be checked on such devices.
- The export pins torch 2.2.2 and shims `nn.RMSNorm` (`android:tool/export_maia3_onnx.py:16-29`).
  See section 10 for why.

## 10. Model conversion findings **[run]**

First probe, run on a Mac against the 5M checkpoint with five positions (both sides to move,
Elos 800–2200):

| Path | Result |
|---|---|
| ONNX, opset 17 | With torch 2.14, fails on `aten::rms_norm` unless `nn.RMSNorm` is replaced by its explicit formula. With that replacement (torch 2.14 and 2.7): max abs error vs PyTorch 1.6e-5 (logits), 1e-6 (value), 1e-7 (ponder), well under 1e-4. |
| Core ML, coremltools 9.0 + torch 2.14 | Fails (TypeError in the converter). coremltools 9.0 is tested only up to torch 2.7. |
| Core ML fp32, coremltools 9.0 + torch 2.7.0 | Converts with or without the RMSNorm replacement; parity measured with it. Max abs error 2e-5 (logits); legal-move probabilities within 1e-6; top-1 5/5. 20 MB `.mlpackage`. |
| Core ML fp16, same | Converts (10 MB) but probabilities differ by up to 1.9e-2 (CPU) / 2.3e-3 (all units). Top-1 still 5/5. **Fails the 1e-3 parity target.** |

Mac latency for the exported Core ML fp32 models (Apple M3, batch 1, warm, 50 runs). This
says little about iPhones; step 7 measures those.

| Model | CPU only, p50 / p95 | All compute units, p50 / p95 | Load |
|---|---|---|---|
| 5M | 2.4 / 2.5 ms | 7.7 / 9.6 ms | 0.6 s |
| 23M | 6.7 / 7.0 ms | 7.1 / 12.1 ms | 0.8 s |
| 79M | 20.8 / 21.3 ms | 17.4 / 23.9 ms | 1.2 s |

**Model choice (owner):** 23M is bundled and is the app's opponent. 79M as an optional
download is #24. 5M is kept for fast tests only.

### Step 2 export results

`Tools/maia3_export/export_maia3.py` exports all three models (see its README). Each export
is checked on 41 positions: fixed openings, middlegames, endgames, castling, en passant
and promotions for both sides, plus positions from seeded random games in both history
modes. fp32 passes when every output value is within 1e-4 + 1e-5 × |reference| and every
top move matches. fp16 passes when legal-move probabilities are within 1e-3 and every top
move matches. Worst errors over both Core ML compute-unit settings:

| Model | ONNX logits | Core ML fp32 logits | fp32 legal probs | fp16 legal probs | fp16 |
|---|---|---|---|---|---|
| 5M | 1.8e-5 | 1.9e-5 | ≤ 3.8e-6 | 1.8e-2 | fails |
| 23M | 4.1e-5 | 2.9e-5 | ≤ 4.3e-6 | 3.0e-2 | fails |
| 79M | 1.3e-4 | 1.1e-4 | ≤ 7.0e-6 | 2.1e-2 | fails |

All fp32 exports pick the same top move as PyTorch in 41/41 positions. 79M's logits reach
about 24, so its absolute error reflects float32 rounding (relative error about 1e-5;
PyTorch itself differs by 5e-5 between batched and single inference). No fp16 model is
produced. Sizes and hashes are in `Tools/maia3_export/manifest.json`.

Notes for the Swift side:
- Core ML models take batch 1 and **int32** Elo inputs (Core ML has no int64 inputs). The
  ONNX models keep int64 and a dynamic batch.
- The Core ML models need iOS 17 or later (ML Program).

## 11. Runtime contract for Emula

Whatever runtime is chosen (Core ML preferred, ONNX Runtime as fallback), the model file
must satisfy section 2 with batch 1, and the Swift side owns everything else:

| Responsibility | Section |
|---|---|
| Build `tokens` (3), Elo ints (4) | encoder |
| Legal-move mask, index ↔ UCI with mirroring (5) | decoder, using Scacchi's `Game.legalMoves()` |
| Softmax, temperature, top-p, seeded RNG (8) | sampler |
| WDL from `value_logits` (6) | result |

**Settled:** the sampler follows upstream's top-p rule exactly (section 8, step 5: keep
moves while the cumulative probability is ≤ p, always keep the top move, exclude the
crossing move), so the Python fixtures match.

**Settled (app):** both history modes (3.3) are offered as the "Maia Sees" setting. The
app defaults to (b), real history (the last 8 positions); (a), current position ×8, is the
official engine's default and the alternative. `MaiaResult.wdl` is for the current
position, not per candidate as the engine reports it.

## 12. Verification run

Scripts were run outside the repo (Python 3.13, maia3 @ `1e13597`, python-chess 1.11.2,
torch 2.14.1 for the encoding checks, torch 2.7.0 + coremltools 9.0 + onnxruntime 1.30.0
for conversion). They print the tokens, vocabulary indices, state-dict check, top-5 and WDL
quoted above. Step 2 turns them into the reproducible `Tools/maia3_export/` pipeline.

Other results from the run, 5M, start position, current ×8:

| Elo | Top 5 (White) | L / D / W |
|---|---|---|
| 800/800 | e4 67.3%, d4 20.5%, e3 3.0%, Nf3 2.3%, c4 1.2% | 44.9 / 3.0 / 52.2 |
| 1500/1500 | e4 64.3%, d4 24.2%, c4 2.5%, Nf3 2.3%, e3 1.7% | 47.0 / 2.6 / 50.4 |
| 2200/2200 | e4 50.8%, d4 30.9%, Nf3 6.4%, c4 5.8%, b3 1.3% | 46.2 / 5.1 / 48.7 |

## 13. Unverified or open

- Meaning and units of the ponder head (section 7).
- Whether training used the per-frame mirroring that mode (b) produces (section 3.3).
- iPhone latency and memory for any runtime (step 7).
