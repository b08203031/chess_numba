"""
One-command entry for Classical HCE tuning.

Examples (from repo root)::

    python -m tuner.run status
    python -m tuner.run sync
    python -m tuner.run data --pgn path/to.pgn --games 500
    python -m tuner.run tune --tune KNIGHT_MOBILITY_BONUS BISHOP_MOBILITY_BONUS --iter 500
    python -m tuner.run apply --dry-run
    python -m tuner.run pipeline --pgn path/to.pgn --tune KNIGHT_MOBILITY_BONUS --iter 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tuner" / "data"
HCE_NPZ = DATA / "hce" / "dataset.npz"
JSONL_MG = DATA / "jsonl" / "training_data_middlegame_cleaned.jsonl"
JSONL_EG = DATA / "jsonl" / "training_data_endgame_cleaned.jsonl"
NNUE_NPZ = DATA / "nnue" / "ultimate_halfka_farseerT75.npz"
NNUE_BIN = DATA / "nnue" / "official_data_farseerT75.bin"


def _run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.call(args, cwd=str(ROOT))


def cmd_status(_args: argparse.Namespace) -> int:
    print("=== HCE tuning data (used by python -m tuner.tuner) ===")
    print(f"  dataset.npz : {'OK ' if HCE_NPZ.exists() else 'MISSING'}  {HCE_NPZ}")
    print(f"  jsonl MG    : {'OK ' if JSONL_MG.exists() else 'MISSING'}  {JSONL_MG}")
    print(f"  jsonl EG    : {'OK ' if JSONL_EG.exists() else 'MISSING'}  {JSONL_EG}")
    pgns = list((DATA / "pgn").glob("*.pgn")) if (DATA / "pgn").exists() else []
    print(f"  pgn files   : {len(pgns)} under {DATA / 'pgn'}")
    for p in pgns[:8]:
        print(f"    - {p.name} ({p.stat().st_size // (1024*1024)} MB)")

    print("\n=== NNUE data (NOT used by HCE SPSA; for nnue training only) ===")
    print(f"  halfka npz  : {'OK ' if NNUE_NPZ.exists() else 'MISSING'}  {NNUE_NPZ}")
    print(f"  bullet bin  : {'OK ' if NNUE_BIN.exists() else 'MISSING'}  {NNUE_BIN}")

    print("\n=== How to build HCE dataset ===")
    print("  A) Reuse NNUE Bullet (recommended if bin exists):")
    print("       python -m tuner.run data --from-bullet --label-mode game --limit 2000000")
    print("  B) From PGN (TWIC etc.):")
    print("       python -m tuner.run data --pgn path/to.pgn --label-mode game")
    print("  label-mode game = {0,0.5,1};  sf = SF score sigmoid")
    if not HCE_NPZ.exists():
        print("\n>>> HCE dataset missing. Fastest path with your NNUE dump:")
        print("    python -m tuner.run data --from-bullet --label-mode game --limit 2000000")
    return 0


def cmd_sync(_args: argparse.Namespace) -> int:
    rc = _run([sys.executable, "-m", "tuner.sync_constants_snapshot"])
    if rc != 0:
        return rc
    rc = _run([sys.executable, "-m", "tuner.codegen.generate_tunable_eval"])
    if rc != 0:
        return rc
    if _args.skip_test:
        return 0
    return _run([sys.executable, "-m", "unittest", "tests.test_tunable_eval_parity", "-v"])


def cmd_data(args: argparse.Namespace) -> int:
    (DATA / "jsonl").mkdir(parents=True, exist_ok=True)
    (DATA / "hce").mkdir(parents=True, exist_ok=True)

    # Fast path: reuse NNUE Bullet dump (has board + score + game result)
    if getattr(args, "from_bullet", False):
        bullet = Path(args.bullet) if args.bullet else NNUE_BIN
        if not bullet.is_file():
            print(f"Error: bullet bin not found: {bullet}", file=sys.stderr)
            return 2
        out = Path(args.output) if getattr(args, "output", None) else HCE_NPZ
        cmd = [
            sys.executable, "-m", "tuner.data_pipeline.convert_bullet_to_hce",
            "--input", str(bullet),
            "--output", str(out),
            "--label-mode", args.label_mode,
            "--limit", str(args.limit),
            "--stride", str(args.stride),
        ]
        if args.label_mode == "mix":
            cmd += ["--sf-weight", str(args.sf_weight)]
        return _run(cmd)

    if not args.pgn:
        print(
            "Error: provide --pgn <file.pgn> or --from-bullet to reuse NNUE data.",
            file=sys.stderr,
        )
        return 2
    pgn = Path(args.pgn)
    if not pgn.is_file():
        print(f"Error: PGN not found: {pgn}", file=sys.stderr)
        return 2

    gen = [
        sys.executable, "-m", "tuner.data_pipeline.generate_training_data",
        "--pgn", str(pgn),
        "--output-mg", str(JSONL_MG),
        "--output-eg", str(JSONL_EG),
        "--label-mode", args.label_mode,
        "--min-elo", str(args.min_elo),
        "--min-move", str(args.min_move),
    ]
    if args.games and args.games > 0:
        gen += ["--games", str(args.games)]
    if args.workers:
        gen += ["--workers", str(args.workers)]
    if args.label_mode == "sf":
        gen += ["--stockfish-depth", str(args.stockfish_depth)]
        if args.stockfish_path:
            gen += ["--stockfish-path", args.stockfish_path]

    rc = _run(gen)
    if rc != 0:
        return rc
    return _run([
        sys.executable, "-m", "tuner.data_pipeline.preprocess_data",
        "--output", str(HCE_NPZ),
    ])


def cmd_tune(args: argparse.Namespace) -> int:
    if not HCE_NPZ.exists():
        print(f"Error: missing {HCE_NPZ}. Run: python -m tuner.run data --pgn ...", file=sys.stderr)
        return 2
    cmd = [
        sys.executable, "-m", "tuner.tuner",
        "--dataset", str(HCE_NPZ),
        "--iter", str(args.iter),
        "--batch-size", str(args.batch_size),
        "--eval-sample", str(args.eval_sample),
        "--metrics-sample", str(args.metrics_sample),
        "--seed", str(args.seed),
        "--report-every", str(args.report_every),
        "--patience", str(args.patience),
        "--alpha", str(args.alpha),
        "--c", str(args.c),
    ]
    if args.tune:
        cmd += ["--tune", *args.tune]
    if args.exclude:
        cmd += ["--exclude", *args.exclude]
    return _run(cmd)


def cmd_apply(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "tuner.apply_tuned"]
    if args.dry_run:
        cmd.append("--dry-run")
    return _run(cmd)


def cmd_pipeline(args: argparse.Namespace) -> int:
    """sync → data → tune → (optional apply)."""
    rc = cmd_sync(argparse.Namespace(skip_test=args.skip_test))
    if rc != 0:
        return rc
    rc = cmd_data(args)
    if rc != 0:
        return rc
    rc = cmd_tune(args)
    if rc != 0:
        return rc
    if args.apply:
        return cmd_apply(argparse.Namespace(dry_run=False))
    print("\nSPSA done. Review tuner/tuned_constants.py then:")
    print("  python -m tuner.run apply --dry-run")
    print("  python -m tuner.run apply")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tuner.run",
        description="Condensed Classical HCE tuning workflow.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Show which datasets exist and what HCE uses")
    p_status.set_defaults(func=cmd_status)

    p_sync = sub.add_parser("sync", help="Sync constants + regenerate tunable_eval + parity test")
    p_sync.add_argument("--skip-test", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    def add_data_args(p):
        p.add_argument("--pgn", type=str, help="Input PGN (master games / TWIC)")
        p.add_argument(
            "--from-bullet",
            action="store_true",
            help="Reuse NNUE Bullet bin (board+score+result) → HCE dataset.npz",
        )
        p.add_argument(
            "--bullet",
            type=str,
            default=None,
            help="Bullet .bin path (default: tuner/data/nnue/official_data_farseerT75.bin)",
        )
        p.add_argument("--limit", type=int, default=2_000_000, help="Max positions when --from-bullet")
        p.add_argument("--stride", type=int, default=1, help="Subsample every N-th bullet record")
        p.add_argument(
            "--label-mode",
            choices=("game", "sf", "mix"),
            default="game",
            help="Bullet labels: game | sf | mix=(1-λ)*game+λ*sf",
        )
        p.add_argument(
            "--sf-weight",
            type=float,
            default=0.3,
            help="λ for --label-mode mix (default 0.3)",
        )
        p.add_argument(
            "--output",
            type=str,
            default=None,
            help="HCE npz path when --from-bullet (default: tuner/data/hce/dataset.npz)",
        )
        p.add_argument("--min-elo", type=int, default=2400)
        p.add_argument("--min-move", type=int, default=15)
        p.add_argument("--games", type=int, default=0, help="0 = all games")
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--stockfish-path", type=str, default=None)
        p.add_argument("--stockfish-depth", type=int, default=10)

    p_data = sub.add_parser("data", help="Generate labels + preprocess → tuner/data/hce/dataset.npz")
    add_data_args(p_data)
    p_data.set_defaults(func=cmd_data)

    def add_tune_args(p):
        p.add_argument("--tune", nargs="+", help="Parameter names to tune (others frozen)")
        p.add_argument("--exclude", nargs="+", help="Parameter names to freeze")
        p.add_argument("--iter", type=int, default=5000)
        p.add_argument("--batch-size", type=int, default=65536)
        p.add_argument("--eval-sample", type=int, default=1_000_000)
        p.add_argument(
            "--metrics-sample",
            type=int,
            default=500_000,
            help="Fixed validation subset for phase/endgame bucket MSE (0=off)",
        )
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--report-every", type=int, default=50)
        p.add_argument(
            "--patience",
            type=int,
            default=30,
            help="Stop after N consecutive reports without new best (0=off)",
        )
        p.add_argument("--alpha", type=float, default=30000.0)
        p.add_argument("--c", type=float, default=3.0)

    p_tune = sub.add_parser("tune", help="Run SPSA on existing dataset.npz")
    add_tune_args(p_tune)
    p_tune.set_defaults(func=cmd_tune)

    p_apply = sub.add_parser("apply", help="Merge tuned_constants.py into classical/constants.py")
    p_apply.add_argument("--dry-run", action="store_true")
    p_apply.set_defaults(func=cmd_apply)

    p_pipe = sub.add_parser("pipeline", help="sync + data + tune (+ optional --apply)")
    add_data_args(p_pipe)
    add_tune_args(p_pipe)
    p_pipe.add_argument("--apply", action="store_true", help="Also write into classical/constants.py")
    p_pipe.add_argument("--skip-test", action="store_true")
    p_pipe.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
