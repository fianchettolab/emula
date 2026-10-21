#!/usr/bin/env python3
"""Generate golden test fixtures for the Swift port of Maia-3.

    python generate_fixtures.py                 # 23M and 5M
    python generate_fixtures.py --models 5m

Each fixture records, for one position, history mode and Elo pair: the game as a start FEN
plus UCI moves, the exact model input, the top-10 moves, W/D/L, and the expected result of
the upstream sampler for a few temperature/top-p settings. Outputs come from the upstream
PyTorch model. The format is described in fixtures/README.md; the encoding in
docs/maia3-spec.md.
"""

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

import chess
import torch

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.utils import get_all_possible_moves, mirror_move

from export_maia3 import CHECKPOINTS, MAIA3_COMMIT, fetch_checkpoint, load_reference_model

HERE = Path(__file__).resolve().parent
VOCAB = get_all_possible_moves()
VOCAB_INDEX = {uci: i for i, uci in enumerate(VOCAB)}

# (White Elo, Black Elo). self_elo is the side to move's.
ELO_PAIRS = [(800, 800), (1500, 1500), (2200, 1800)]
HISTORY_MODES = ["current", "history"]
# (temperature, top_p) settings whose deterministic part is recorded per case.
SAMPLING = [(1.0, 0.9), (0.5, 0.8)]
TOP_K = 10
NEAR_TIE = 1e-4

SELFPLAY_GAMES = 21
SELFPLAY_SEED = 2026
SNAPSHOTS_PER_GAME = 8

# Positions for features self-play rarely reaches. Each is (label, start FEN, UCI moves);
# moves give the history modes something to work with where it matters.
CURATED = [
    ("start position, empty history", chess.STARTING_FEN, []),
    ("start position, Black to move after 1.e4", chess.STARTING_FEN, ["e2e4"]),
    ("White can castle both sides", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8", []),
    ("Black can castle both sides", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 6 8", []),
    ("White castled kingside earlier", chess.STARTING_FEN,
     ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1", "g8f6", "d2d3", "d7d6"]),
    ("Black castles queenside next", chess.STARTING_FEN,
     ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "c1g5", "c8f5", "e2e3", "b8d7",
      "f1d3", "f5d3", "d1d3", "d8a5", "g1f3", "e7e6", "e1g1"]),
    ("castling blocked by attack on f1", "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1", []),
    ("White en passant available", chess.STARTING_FEN,
     ["e2e4", "a7a6", "e4e5", "d7d5"]),
    ("Black en passant available", chess.STARTING_FEN,
     ["g1f3", "e7e5", "b1c3", "e5e4", "d2d4"]),
    ("Black in check from a pawn, en passant captures it", "8/8/8/2k5/3Pp3/8/8/3KQ3 b - d3 0 1", []),
    ("White promotion, all four pieces", "8/P6k/8/8/8/8/6K1/8 w - - 0 1", []),
    ("Black promotion, all four pieces", "8/6k1/8/8/8/8/p6K/8 b - - 0 1", []),
    ("White capture-promotion either side", "1n1r3k/2P5/8/8/8/8/8/6K1 w - - 0 1", []),
    ("Black capture-promotion", "6k1/8/8/8/8/8/1p6/R1N3K1 b - - 0 1", []),
    ("promotion by capture on either side", "3r1r2/4P1k1/6p1/8/8/8/8/6K1 w - - 0 1", []),
    ("pawn race", "8/8/1P4k1/8/8/8/5p2/2K5 w - - 0 1", []),
    ("White in check after ...Qh4+", chess.STARTING_FEN, ["e2e4", "e7e5", "f2f4", "d8h4"]),
    ("Black in check after Bb5+", chess.STARTING_FEN, ["e2e4", "d7d5", "f1b5"]),
    ("mate in one for White", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", []),
    ("KQ vs K", "8/8/8/4k3/8/8/8/3QK3 w - - 0 1", []),
    ("KR vs K, Black to move", "8/8/8/4k3/8/8/8/R3K3 b - - 0 1", []),
    ("K+P vs K, opposition", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", []),
    ("rook endgame", "8/5pk1/6p1/8/3R4/6P1/r4PK1/8 b - - 0 40", []),
    ("opposite bishops", "8/5k2/3b2p1/8/8/6P1/2B2K2/8 w - - 0 50", []),
    ("queen endgame, Black to move", "8/6k1/6p1/8/3q4/8/5PK1/4Q3 b - - 0 45", []),
    ("near-stalemate trap", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", []),
    ("only king moves", "k7/8/1K6/8/8/8/8/7R b - - 0 1", []),
    ("Sicilian Najdorf", chess.STARTING_FEN,
     ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]),
    ("Queen's Gambit Declined", chess.STARTING_FEN,
     ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]),
    ("Ruy Lopez, Black to move", chess.STARTING_FEN,
     ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),
    ("King's Indian", chess.STARTING_FEN,
     ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8"]),
    ("long game, history window full", chess.STARTING_FEN,
     ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6",
      "e1g1", "e8g8", "b2b4", "c5b6", "a2a4", "a7a6", "b1d2", "c8e6", "c4e6", "f7e6"]),
    ("FEN start with moves after it", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     ["d2d3", "f8c5", "e1g1", "d7d6"]),
]


# --- Encoding and model --------------------------------------------------------------


def boards_for(start_fen, moves):
    board = chess.Board(start_fen)
    boards = [board.copy(stack=False)]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} in {board.fen()}")
        board.push(move)
        boards.append(board.copy(stack=False))
    return boards


def encode(cfg, boards, mode):
    """tokens [64, 97] for the last board. "current": the current position only, padded to 8
    frames (upstream default). "history": the last 8 positions, padded with the earliest."""
    frames = boards[-1:] if mode == "current" else boards[-cfg.history:]
    return get_historical_tokens(deque(tokenize_board(b) for b in frames), cfg, 0.0, 0.0, 0.0, 0.0)


def to_uci(board, index):
    uci = VOCAB[index]
    return mirror_move(uci) if board.turn == chess.BLACK else uci


def run(model, tokens, self_elo, opponent_elo):
    with torch.no_grad():
        move, value, ponder = model(tokens.unsqueeze(0), torch.tensor([self_elo]), torch.tensor([opponent_elo]))
    return move[0], value[0], float(ponder[0])


# --- Upstream sampling, deterministic parts ------------------------------------------


def nucleus(logits, temperature, top_p):
    """The moves upstream sample_from_logits can pick, with their renormalized probabilities
    (uci.py:163-183): keep while cumulative <= top_p, always keep the top move."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, stable=True)
    keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
    keep[0] = True
    kept = sorted_probs[keep]
    return sorted_idx[keep].tolist(), (kept / kept.sum()).tolist()


# --- Positions -----------------------------------------------------------------------


def selfplay_positions(model, cfg):
    """Seeded self-play at mixed Elos with temperature 1, giving realistic games. Snapshots
    are spread over each game so openings, middlegames and endgames all appear."""
    rng = random.Random(SELFPLAY_SEED)
    positions = []
    for game in range(SELFPLAY_GAMES):
        white_elo, black_elo = rng.choice([(800, 800), (1200, 1400), (1500, 1500), (1900, 2100), (2300, 2300)])
        boards, moves = [chess.Board()], []
        while not boards[-1].is_game_over() and len(moves) < 200:
            board = boards[-1]
            elos = (white_elo, black_elo) if board.turn == chess.WHITE else (black_elo, white_elo)
            logits, _, _ = run(model, encode(cfg, boards, "history"), *elos)
            mask = get_legal_moves_mask(board, VOCAB_INDEX)
            probs = torch.softmax(logits.masked_fill(~mask, float("-inf")).double(), -1)
            legal = mask.nonzero().flatten().tolist()
            index = rng.choices(legal, weights=[float(probs[i]) for i in legal])[0]
            moves.append(to_uci(board, index))
            board = board.copy(stack=False)
            board.push_uci(moves[-1])
            boards.append(board)
        playable = [ply for ply in range(len(moves) + 1) if not boards[ply].is_game_over()]
        step = max(1, len(playable) // SNAPSHOTS_PER_GAME)
        for ply in playable[rng.randrange(step)::step][:SNAPSHOTS_PER_GAME]:
            positions.append((f"self-play game {game + 1}, ply {ply}", chess.STARTING_FEN, moves[:ply]))
    return positions


def tags(board, moves):
    legal = list(board.legal_moves)
    pieces = len(board.piece_map())
    phase = "opening" if len(moves) < 16 and pieces > 28 else "endgame" if pieces <= 12 else "middlegame"
    found = [phase, "white_to_move" if board.turn == chess.WHITE else "black_to_move"]
    if board.is_check():
        found.append("in_check")
    if any(board.is_castling(m) for m in legal):
        found.append("castling_legal")
    if any(board.is_en_passant(m) for m in legal):
        found.append("en_passant_legal")
    if any(m.promotion for m in legal):
        found.append("promotion_legal")
    if any(m.promotion and m.promotion != chess.QUEEN for m in legal):
        found.append("underpromotion_legal")
    if len(moves) == 0:
        found.append("no_moves")
    return found


# --- Cases ---------------------------------------------------------------------------


def make_case(model, cfg, number, label, start_fen, moves, mode, white_elo, black_elo):
    boards = boards_for(start_fen, moves)
    board = boards[-1]
    self_elo, opponent_elo = (white_elo, black_elo) if board.turn == chess.WHITE else (black_elo, white_elo)
    tokens = encode(cfg, boards, mode)
    logits, value, ponder = run(model, tokens, self_elo, opponent_elo)

    mask = get_legal_moves_mask(board, VOCAB_INDEX)
    masked = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked.double(), dim=-1)
    order = sorted(mask.nonzero().flatten().tolist(), key=lambda i: (-float(probs[i]), i))
    top = order[:TOP_K]
    loss, draw, win = torch.softmax(value.double(), dim=-1).tolist()

    sampling = [{"temperature": 0.0, "top_p": 1.0, "argmax": to_uci(board, int(torch.argmax(masked)))}]
    for temperature, top_p in SAMPLING:
        kept, kept_probs = nucleus(masked, temperature, top_p)
        sampling.append({"temperature": temperature, "top_p": top_p,
                         "nucleus": [{"uci": to_uci(board, i), "prob": round(p, 8)} for i, p in zip(kept, kept_probs)]})

    gap = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else 1.0
    return {
        "id": f"{number:03d}-{mode}",
        "label": label,
        "tags": tags(board, moves),
        "start_fen": start_fen,
        "moves": moves,
        "fen": board.fen(),
        "history_mode": mode,
        "white_elo": white_elo,
        "black_elo": black_elo,
        "self_elo": self_elo,
        "opponent_elo": opponent_elo,
        "tokens_active": tokens.flatten().nonzero().flatten().tolist(),
        "legal_move_count": int(mask.sum()),
        "top_moves": [{"uci": to_uci(board, i), "index": i, "logit": round(float(logits[i]), 6),
                       "prob": round(float(probs[i]), 8)} for i in top],
        "near_tie": gap < NEAR_TIE,
        "value": {"logits": [round(float(v), 6) for v in value], "loss": round(loss, 8),
                  "draw": round(draw, 8), "win": round(win, 8)},
        "ponder": round(ponder, 6),
        "sampling": sampling,
    }


def generate(key, out_dir, positions):
    spec = CHECKPOINTS[key]
    cfg, model = load_reference_model(spec, fetch_checkpoint(spec))

    cases = []
    for number, (label, fen, moves) in enumerate(positions, start=1):
        white_elo, black_elo = ELO_PAIRS[number % len(ELO_PAIRS)]
        for mode in HISTORY_MODES:
            cases.append(make_case(model, cfg, number, label, fen, moves, mode, white_elo, black_elo))

    fixture = {
        "meta": {
            "model": spec["alias"],
            "checkpoint_sha256": spec["sha256"],
            "maia3_commit": MAIA3_COMMIT,
            "generated_by": "Tools/maia3_export/generate_fixtures.py",
            "torch": torch.__version__,
            "chess": chess.__version__,
            "positions": len(positions),
            "cases": len(cases),
            "tokens_shape": [64, 97],
            "vocabulary_size": len(VOCAB),
            "history_modes": HISTORY_MODES,
            "elo_pairs_white_black": ELO_PAIRS,
            "near_tie_threshold": NEAR_TIE,
        },
        "cases": cases,
    }
    path = out_dir / f"maia3-{key}-fixtures.json"
    path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")

    counts = {}
    for case in cases[::2]:
        for tag in case["tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    print(f"{path.name}: {len(positions)} positions, {len(cases)} cases, {path.stat().st_size / 1e6:.1f} MB")
    print("  positions per tag:", dict(sorted(counts.items())))
    print("  near ties:", sum(c["near_tie"] for c in cases))
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(CHECKPOINTS), default=["23m", "5m"])
    parser.add_argument("--out", type=Path, default=HERE / "fixtures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Self-play positions come from the 5M model so every fixture file shares the same games.
    spec = CHECKPOINTS["5m"]
    cfg, model = load_reference_model(spec, fetch_checkpoint(spec))
    positions = CURATED + selfplay_positions(model, cfg)
    for label, fen, moves in positions:
        board = boards_for(fen, moves)[-1]
        if not board.is_valid() or board.is_game_over():
            sys.exit(f"bad position: {label}")
    for key in args.models:
        generate(key, args.out, positions)


if __name__ == "__main__":
    sys.exit(main())
