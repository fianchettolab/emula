#!/usr/bin/env python3
"""Check the fixtures against docs/maia3-spec.md alone, without any maia3 code.

Re-encodes every case from its FEN and moves using only the rules written in the spec
(sections 3 and 5), and re-derives the sampling expectations from the recorded top moves
where possible. If this passes, the spec is enough to write the Swift encoder and decoder.

    python verify_fixtures.py fixtures/maia3-23m-fixtures.json ...
"""

import json
import math
import sys

import chess

PIECES = "pnbrqk"


def frame(board):
    """Spec 3.1: 64 x 12, rows a1=0..h8=63, mirrored (rank r -> 9-r, colours swapped) when
    Black is to move; channels 0-5 side to move P N B R Q K, 6-11 opponent."""
    black = board.turn == chess.BLACK
    active = set()
    for square, piece in board.piece_map().items():
        file, rank = square % 8, square // 8
        if black:
            rank = 7 - rank
        own = piece.color == board.turn
        channel = PIECES.index(piece.symbol().lower()) + (0 if own else 6)
        active.add((rank * 8 + file, channel))
    return active


def tokens(boards, mode):
    """Spec 3.2-3.3: 8 frames oldest first, left-padded with the earliest; column 96 is 0."""
    frames = boards[-1:] if mode == "current" else boards[-8:]
    frames = [frames[0]] * (8 - len(frames)) + frames
    active = []
    for k, board in enumerate(frames):
        active += [square * 97 + 12 * k + channel for square, channel in frame(board)]
    return sorted(active)


def square_index(name):
    return "abcdefgh".index(name[0]) + (int(name[1]) - 1) * 8


def vocab_index(uci, black_to_move):
    """Spec 5: mirror for Black, then from*64+to, or 4096 + (fromFile*8+toFile)*4 + qrbn."""
    if black_to_move:
        uci = "".join(c if not c.isdigit() else str(9 - int(c)) for c in uci)
    if len(uci) == 5:
        return 4096 + ("abcdefgh".index(uci[0]) * 8 + "abcdefgh".index(uci[2])) * 4 + "qrbn".index(uci[4])
    return square_index(uci[:2]) * 64 + square_index(uci[2:4])


def check(path):
    fixture = json.load(open(path))
    failures = []
    for case in fixture["cases"]:
        board = chess.Board(case["start_fen"])
        boards = [board.copy(stack=False)]
        for uci in case["moves"]:
            board.push_uci(uci)
            boards.append(board.copy(stack=False))
        black = board.turn == chess.BLACK
        problems = []
        if board.fen() != case["fen"]:
            problems.append("fen")
        if tokens(boards, case["history_mode"]) != case["tokens_active"]:
            problems.append("tokens")
        if (case["self_elo"], case["opponent_elo"]) != ((case["black_elo"], case["white_elo"]) if black else (case["white_elo"], case["black_elo"])):
            problems.append("elo")
        legal = {m.uci() for m in board.legal_moves}
        if len(legal) != case["legal_move_count"]:
            problems.append("legal count")
        top = case["top_moves"]
        if any(m["uci"] not in legal or vocab_index(m["uci"], black) != m["index"] for m in top):
            problems.append("vocabulary")
        if any(a["prob"] < b["prob"] for a, b in zip(top, top[1:])):
            problems.append("top order")
        # Softmax over legal moves: logit differences must match probability ratios.
        for m in top[1:]:
            ratio = math.exp(m["logit"] - top[0]["logit"])
            if abs(m["prob"] - top[0]["prob"] * ratio) > 1e-5:
                problems.append("prob vs logit")
                break
        value = case["value"]
        if abs(value["loss"] + value["draw"] + value["win"] - 1) > 1e-6:
            problems.append("wdl sum")
        argmax = case["sampling"][0]
        if argmax["argmax"] != top[0]["uci"]:
            problems.append("argmax")
        for s in case["sampling"][1:]:
            # Spec 8.5: kept moves are a prefix of the sorted list, keep while cumulative
            # T-adjusted probability <= p, top move always kept.
            kept = s["nucleus"]
            if abs(sum(m["prob"] for m in kept) - 1) > 1e-5 or kept[0]["uci"] != top[0]["uci"]:
                problems.append("nucleus")
            prefix = [m["uci"] for m in top[:len(kept)]]
            if len(kept) <= len(top) and [m["uci"] for m in kept] != prefix:
                problems.append("nucleus order")
        if problems:
            failures.append((case["id"], case["label"], problems))
    print(f"{path}: {len(fixture['cases'])} cases, {len(failures)} failing")
    for failure in failures[:20]:
        print("  ", failure)
    return not failures


if __name__ == "__main__":
    sys.exit(0 if all([check(p) for p in sys.argv[1:]]) else 1)
