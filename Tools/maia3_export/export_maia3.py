#!/usr/bin/env python3
"""Export Maia-3 checkpoints to ONNX and Core ML, check them against PyTorch, and record
sizes and hashes in manifest.json.

    python export_maia3.py                  # all models
    python export_maia3.py --models 5m      # one model

The model contract (inputs, outputs, encoding) is in docs/maia3-spec.md. Core ML export and
its parity check need macOS.
"""

import argparse
import copy
import hashlib
import json
import platform
import random
import shutil
import sys
import warnings
from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch

from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.models import MAIA3Model
from maia3.uci import parse_args
from maia3.utils import get_all_possible_moves

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

HERE = Path(__file__).resolve().parent
MAIA3_COMMIT = "1e13597c42d4858b7cfd7cfdae01e297263364b2"

# Hugging Face checkpoints, pinned by revision and verified by size and SHA-256.
CHECKPOINTS = {
    "5m": {
        "alias": "maia3-5m",
        "repo": "UofTCSSLab/Maia3-5M",
        "revision": "b6559de2398d7140b985f28fd2c19fb5e47ddabe",
        "filename": "maia3-5m.pt",
        "bytes": 20968049,
        "sha256": "ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f",
    },
    "23m": {
        "alias": "maia3-23m",
        "repo": "UofTCSSLab/Maia3-23M",
        "revision": "51a0145a8178046f7de23119160b136672deeb2b",
        "filename": "maia3-23m.pt",
        "bytes": 91799307,
        "sha256": "bce6cd1af5f0399ac7eed33fabb7a6a2ef6193662c2740f262bf93af7bfb3569",
    },
    "79m": {
        "alias": "maia3-79m",
        "repo": "UofTCSSLab/Maia3-79M",
        "revision": "a107d6ceb7b298cb04ae1da4edffe2939858b894",
        "filename": "maia3-79m.pt",
        "bytes": 315651851,
        "sha256": "3fc6181d5db789b45a15305732148757ae74efa3e0028e81ba335b462dac45c2",
    },
}

INPUT_NAMES = ["tokens", "self_elo", "opponent_elo"]
OUTPUT_NAMES = ["move_logits", "value_logits", "ponder"]

# fp32 exports must match PyTorch on every output within |error| <= ATOL + RTOL * |reference|
# (numpy.allclose). The relative term covers the larger logits of the 79M model. fp16 is kept
# only if legal-move probabilities stay within the Swift parity tolerance and the top move
# never changes.
FP32_ATOL = 1e-4
FP32_RTOL = 1e-5
FP16_MAX_PROB = 1e-3

ELO_PAIRS = [(800, 800), (1500, 1500), (2200, 1800), (1200, 1600)]

# Fixed positions covering both sides to move, castling, en passant and promotions.
FIXED_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    "r3k2r/ppp2ppp/2n5/3pP3/8/8/PPP2PPP/R3K2R w KQkq d6 0 12",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/P6k/8/8/8/8/1p5K/8 w - - 0 50",
    "8/P6k/8/8/8/8/1p5K/8 b - - 0 50",
    "8/5k2/8/3K4/8/8/1p6/8 b - - 0 60",
]


# --- Model loading -------------------------------------------------------------------


class ExplicitRMSNorm(torch.nn.Module):
    """Same math as torch.nn.RMSNorm. The ONNX exporter has no aten::rms_norm at opset 17."""

    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def replace_rms_norms(module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.RMSNorm):
            setattr(module, name, ExplicitRMSNorm(child))
            count += 1
        else:
            count += replace_rms_norms(child)
    return count


def fetch_checkpoint(spec):
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(spec["repo"], spec["filename"], revision=spec["revision"]))
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != spec["bytes"] or digest != spec["sha256"]:
        sys.exit(f"{path}: expected {spec['bytes']} bytes / {spec['sha256']}, got {size} / {digest}")
    return path


def load_reference_model(spec, checkpoint):
    """The upstream model, loaded strictly (upstream's own loader only warns on bad keys)."""
    cfg = parse_args(["--model", spec["alias"], "--checkpoint-path", str(checkpoint), "--device", "cpu"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    state = {key.replace("smolgen", "gab"): value for key, value in state.items()}
    model = MAIA3Model(cfg)
    model.load_state_dict(state, strict=True)
    return cfg, model.eval()


# --- Parity inputs -------------------------------------------------------------------


def parity_samples():
    """(history, self_elo, opponent_elo) triples; history is a list of boards, current last."""
    samples = []
    for i, fen in enumerate(FIXED_FENS):
        samples.append(([chess.Board(fen)], *ELO_PAIRS[i % len(ELO_PAIRS)]))

    rng = random.Random(7)
    for game in range(4):
        board, history = chess.Board(), [chess.Board()]
        for ply in range(1, 61):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            history.append(board.copy(stack=False))
            if ply in (9, 22, 41, 60):
                elos = ELO_PAIRS[(game + ply) % len(ELO_PAIRS)]
                samples.append(([history[-1]], *elos))  # current position x8
                samples.append((history[-8:], *elos))  # real history
    return [s for s in samples if not s[0][-1].is_game_over()]


def encode(cfg, samples):
    tokens = torch.stack([
        get_historical_tokens(deque(tokenize_board(b) for b in history), cfg, 0.0, 0.0, 0.0, 0.0)
        for history, _, _ in samples
    ])
    self_elo = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opponent_elo = torch.tensor([s[2] for s in samples], dtype=torch.long)
    return tokens, self_elo, opponent_elo


def legal_masks(samples):
    index = {uci: i for i, uci in enumerate(get_all_possible_moves())}
    return np.stack([get_legal_moves_mask(history[-1], index).numpy() for history, _, _ in samples])


# --- Comparison ----------------------------------------------------------------------


def legal_probs(logits, masks):
    masked = np.where(masks, logits.astype(np.float64), -np.inf)
    exp = np.exp(masked - masked.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def compare(outputs, reference, masks):
    outputs = [np.asarray(out).reshape(ref.shape) for out, ref in zip(outputs, reference)]
    errors = {name: float(np.abs(out - ref).max()) for name, out, ref in zip(OUTPUT_NAMES, outputs, reference)}
    errors["fp32_within"] = all(np.allclose(out, ref, atol=FP32_ATOL, rtol=FP32_RTOL)
                                for out, ref in zip(outputs, reference))
    probs, ref_probs = legal_probs(outputs[0], masks), legal_probs(reference[0], masks)
    errors["legal_probs"] = float(np.abs(probs - ref_probs).max())
    errors["top1_agree"] = f"{int((probs.argmax(1) == ref_probs.argmax(1)).sum())}/{len(masks)}"
    errors["top1_all"] = bool((probs.argmax(1) == ref_probs.argmax(1)).all())
    return errors


def fp32_ok(errors):
    return errors["fp32_within"] and errors["top1_all"]


# --- Exporters -----------------------------------------------------------------------


def export_onnx(model, inputs, path):
    import onnx
    import onnxruntime as ort

    torch.onnx.export(
        model, tuple(x[:1] for x in inputs), str(path), dynamo=False,
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES},
        opset_version=17, do_constant_folding=True,
    )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {name: x.numpy() for name, x in zip(INPUT_NAMES, inputs)}
    return {"onnxruntime_cpu": session.run(OUTPUT_NAMES, feed)}


def export_coreml(model, inputs, path, spec, precision):
    import coremltools as ct

    traced = torch.jit.trace(model, tuple(x[:1] for x in inputs))
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType("tokens", shape=(1, 64, 97), dtype=np.float32),
            ct.TensorType("self_elo", shape=(1,), dtype=np.int32),
            ct.TensorType("opponent_elo", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name) for name in OUTPUT_NAMES],
        compute_precision=ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.author = "CSSLab, University of Toronto (Maia-3)"
    mlmodel.license = "AGPL-3.0"
    mlmodel.short_description = f"{spec['alias']} ({precision}), converted for Fianchetto. See docs/maia3-spec.md."
    mlmodel.user_defined_metadata.update({
        "maia3_commit": MAIA3_COMMIT,
        "checkpoint_sha256": spec["sha256"],
        "history": "8",
        "vocabulary": "4352",
    })
    descriptions = {
        "tokens": "Board history, [1, 64, 97]: 8 frames x 12 piece planes + 1 unused column",
        "self_elo": "Elo of the side to move",
        "opponent_elo": "Elo of the other side",
        "move_logits": "Logit per vocabulary entry (4352), side-to-move frame",
        "value_logits": "[loss, draw, win] logits for the side to move",
        "ponder": "Scalar head, meaning undocumented",
    }
    for name, text in descriptions.items():
        if name in INPUT_NAMES:
            mlmodel.input_description[name] = text
        else:
            mlmodel.output_description[name] = text
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))

    results = {}
    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL):
        loaded = ct.models.MLModel(str(path), compute_units=units)
        rows = []
        for i in range(len(inputs[0])):
            out = loaded.predict({
                "tokens": inputs[0][i:i + 1].numpy(),
                "self_elo": inputs[1][i:i + 1].numpy().astype(np.int32),
                "opponent_elo": inputs[2][i:i + 1].numpy().astype(np.int32),
            })
            rows.append([np.asarray(out[name]).reshape(-1) for name in OUTPUT_NAMES])
        results[f"coreml_{units.name.lower()}"] = [np.stack([row[k] for row in rows]) for k in range(3)]
    return results


# --- Hashing and manifest ------------------------------------------------------------


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mlmodel_spec_sha256(path):
    """Hash of a model.mlmodel spec in deterministic protobuf serialization. The saved file
    itself varies between runs only in map ordering."""
    from coremltools.proto import Model_pb2

    spec = Model_pb2.Model()
    spec.ParseFromString(path.read_bytes())
    return hashlib.sha256(spec.SerializeToString(deterministic=True)).hexdigest()


def describe(path):
    """Size and hash of a file, or of an .mlpackage directory.

    For an .mlpackage, `content_sha256` is reproducible across exports: it covers every file
    except Manifest.json (random package UUIDs), with model.mlmodel hashed by its
    deterministic serialization. Per-file hashes of the bytes on disk are listed too.
    """
    if path.is_file():
        return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    content = hashlib.sha256()
    entries = {}
    for p in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = p.relative_to(path).as_posix()
        entries[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        if rel == "Manifest.json":
            continue
        digest = mlmodel_spec_sha256(p) if p.name == "model.mlmodel" else entries[rel]["sha256"]
        content.update(f"{rel}\0{digest}\n".encode())
    return {"bytes": sum(e["bytes"] for e in entries.values()), "content_sha256": content.hexdigest(),
            "files": entries}


def tool_versions():
    import coremltools
    import onnx
    import onnxruntime

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "coremltools": coremltools.__version__,
        "chess": chess.__version__,
    }


# --- Main ----------------------------------------------------------------------------


def export_model(key, out_dir, formats, fp16=True):
    spec = CHECKPOINTS[key]
    print(f"== {spec['alias']}")
    checkpoint = fetch_checkpoint(spec)
    cfg, reference_model = load_reference_model(spec, checkpoint)

    samples = parity_samples()
    inputs = encode(cfg, samples)
    masks = legal_masks(samples)
    with torch.no_grad():
        reference = [x.numpy() for x in reference_model(*inputs)]

    export_model_ = copy.deepcopy(reference_model)
    replaced = replace_rms_norms(export_model_)
    with torch.no_grad():
        swapped = [x.numpy() for x in export_model_(*inputs)]
    swap_errors = compare(swapped, reference, masks)
    print(f"   RMSNorm replaced: {replaced}, max abs change {swap_errors['move_logits']:.2e}")
    if not fp32_ok(swap_errors):
        sys.exit("   explicit RMSNorm changed the outputs")

    artifacts = []

    def record(path, fmt, precision, results, gate):
        checks = {runtime: compare(outs, reference, masks) for runtime, outs in results.items()}
        passed = all(gate(e) for e in checks.values())
        for runtime, e in checks.items():
            print(f"   {path.name} [{runtime}]: logits {e['move_logits']:.2e}, value {e['value_logits']:.2e}, "
                  f"ponder {e['ponder']:.2e}, legal probs {e['legal_probs']:.2e}, top-1 {e['top1_agree']}")
        print(f"   {path.name}: {'PASS' if passed else 'FAIL'}")
        entry = {"file": path.name, "format": fmt, "precision": precision, "parity_passed": passed,
                 "parity": {r: {k: v for k, v in e.items() if k not in ("top1_all", "fp32_within")}
                            for r, e in checks.items()}}
        if passed:
            entry.update(describe(path))
        elif path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        artifacts.append(entry)
        return passed

    ok = True
    if "onnx" in formats:
        path = out_dir / f"{spec['alias']}.onnx"
        ok &= record(path, "onnx", "fp32", export_onnx(export_model_, inputs, path), fp32_ok)
    if "coreml" in formats:
        path = out_dir / f"{spec['alias']}-fp32.mlpackage"
        ok &= record(path, "mlpackage", "fp32", export_coreml(export_model_, inputs, path, spec, "fp32"), fp32_ok)
    if "coreml" in formats and fp16:
        # fp16 is optional: a failed parity check drops the file but doesn't fail the run.
        path = out_dir / f"{spec['alias']}-fp16.mlpackage"
        record(path, "mlpackage", "fp16", export_coreml(export_model_, inputs, path, spec, "fp16"),
               lambda e: e["legal_probs"] <= FP16_MAX_PROB and e["top1_all"])

    entry = {
        "checkpoint": {k: spec[k] for k in ("repo", "revision", "filename", "bytes", "sha256")},
        "parity_samples": len(samples),
        "artifacts": artifacts,
    }
    return ok, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--formats", nargs="+", choices=["onnx", "coreml"], default=["onnx", "coreml"])
    parser.add_argument("--skip-fp16", action="store_true", help="don't try the fp16 Core ML export")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    args = parser.parse_args()

    if "coreml" in args.formats and platform.system() != "Darwin":
        sys.exit("Core ML export and its parity check need macOS (use --formats onnx elsewhere)")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    manifest.update({
        "generated_by": "Tools/maia3_export/export_maia3.py",
        "maia3_commit": MAIA3_COMMIT,
        "tools": tool_versions(),
        "tolerances": {"fp32_atol": FP32_ATOL, "fp32_rtol": FP32_RTOL, "fp16_max_legal_prob": FP16_MAX_PROB},
    })
    manifest.setdefault("models", {})

    all_ok = True
    for key in args.models:
        ok, entry = export_model(key, args.out, args.formats, fp16=not args.skip_fp16)
        manifest["models"][CHECKPOINTS[key]["alias"]] = entry
        all_ok &= ok
    manifest["models"] = dict(sorted(manifest["models"].items(), key=lambda kv: int(kv[0].split("-")[1][:-1])))

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {args.manifest}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
