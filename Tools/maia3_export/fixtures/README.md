# Maia-3 golden fixtures

The source of truth for the Swift port. Generated from the upstream PyTorch models by
`../generate_fixtures.py`; checked against the spec alone by `../verify_fixtures.py`.

| File | Model | Use |
|---|---|---|
| `maia3-23m-fixtures.json` | 23M | parity of the model to ship |
| `maia3-5m-fixtures.json` | 5M | fast tests |

Both files use the same 201 positions, each in both history modes: 402 cases. Regenerating
gives byte-identical files.

## Positions

- **33 curated** (`label` describes each): castling for both sides (available, already
  done, blocked by an attacked square), en passant for both colours, promotions and
  underpromotions for both colours (pushing and capturing), checks, mate in one, basic
  endgames, a few named openings, a full 8-frame history, and a game starting from a FEN.
- **168 from self-play**: 21 seeded games of the 5M model at mixed Elos (800 to 2300), with
  8 positions spread over each game, so openings, middlegames and endgames all appear.

Elo pairs rotate over White/Black 800/800, 1500/1500 and 2200/1800.

## Case format

```jsonc
{
  "id": "042-history",            // position number and history mode
  "label": "self-play game 2, ply 17",
  "tags": ["middlegame", "black_to_move", "in_check", ...],
  "start_fen": "...",             // where the game starts
  "moves": ["e2e4", "e7e5", ...], // UCI moves from start_fen to the position
  "fen": "...",                   // the position to evaluate (after the moves)
  "history_mode": "current",      // "current": current position x8; "history": last 8 positions
  "white_elo": 2200, "black_elo": 1800,
  "self_elo": 1800,               // the side to move's Elo
  "opponent_elo": 2200,
  "tokens_active": [3, 100, ...], // flat indices of the 1s in tokens[64][97], row-major
                                  // (square * 97 + column); every other value is 0
  "legal_move_count": 31,
  "top_moves": [                  // up to 10, by probability; softmax over legal moves at T=1
    {"uci": "e7e5", "index": 796, "logit": 7.123456, "prob": 0.42471234}
  ],
  "near_tie": false,              // top two probabilities within 1e-4
  "value": {"logits": [l, d, w], "loss": 0.47, "draw": 0.03, "win": 0.50},  // side to move
  "ponder": 0.41,
  "sampling": [
    {"temperature": 0.0, "top_p": 1.0, "argmax": "e7e5"},
    {"temperature": 1.0, "top_p": 0.9, "nucleus": [{"uci": "e7e5", "prob": 0.51}, ...]},
    {"temperature": 0.5, "top_p": 0.8, "nucleus": [...]}
  ]
}
```

- `index` is the vocabulary index in the side-to-move frame (Black's moves mirrored).
- `nucleus` is the set of moves the upstream sampler can pick for that temperature and
  top-p, with their renormalized probabilities. It follows upstream's rule: keep moves
  while the cumulative probability is ≤ p, always keep the top move. A seeded Swift sampler
  must only ever pick from this set.
- Values are rounded: logits to 6 decimals, probabilities to 8.
