import os
import sys
import subprocess
import re
import numpy as np
import numba

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Classical engine components
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.evaluation import evaluate_position as evaluate_classical

# Import NNUE engine components
from chess_engine.nnue.ml_eval.inference import init_accumulator, nnue_forward_incremental, weights_loaded, warmup_numba_kernels

# Warm up NNUE JIT compiler
if weights_loaded:
    warmup_numba_kernels()

# Path to Stockfish executable (support both Windows and Linux/Colab)
import shutil
if sys.platform == "win32":
    STOCKFISH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/stockfish/stockfish-windows-x86-64-avx2.exe"))
else:
    # Linux/Colab: Use system-installed Stockfish
    STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"

@numba.njit(numba.int32(numba.types.Array(numba.uint64, 1, 'C'), numba.int32), fastmath=True, cache=False)
def evaluate_our_nnue(piece_bbs, side_to_move):
    accumulator_stack = np.zeros((1, 2, 512), dtype=np.int32)
    init_accumulator(piece_bbs, accumulator_stack)
    
    piece_count = 0
    for p_idx in range(12):
        bb = piece_bbs[p_idx]
        while bb:
            piece_count += 1
            bb &= bb - np.uint64(1)
            
    return nnue_forward_incremental(0, side_to_move, accumulator_stack, piece_count)

def get_stockfish_eval(fen):
    """
    Launches Stockfish, loads the FEN, runs the 'eval' command,
    and extracts the NNUE and Final evaluations (relative to White).
    """
    if not os.path.exists(STOCKFISH_PATH):
        return None, None
        
    try:
        p = subprocess.Popen([STOCKFISH_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        commands = f"uci\nisready\nposition fen {fen}\neval\nquit\n"
        out, _ = p.communicate(commands)
        
        sf_nnue = None
        sf_final = None
        for line in out.splitlines():
            if 'NNUE evaluation' in line:
                parts = line.split()
                try:
                    # Stockfish outputs in pawns (e.g. +3.44), convert to centipawns
                    sf_nnue = int(float(parts[2]) * 100)
                except (ValueError, IndexError):
                    pass
            elif 'Final evaluation' in line:
                parts = line.split()
                try:
                    sf_final = int(float(parts[2]) * 100)
                except (ValueError, IndexError):
                    pass
        return sf_nnue, sf_final
    except Exception as e:
        print(f"Error querying Stockfish: {e}")
        return None, None

def run_evaluations():
    # 1. Starting positions with White missing one piece at a time
    # Format: (Description, FEN)
    starting_missing_pieces = [
        ("White missing a2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Black missing a7 pawn", "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing b2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Black missing b7 pawn", "rnbqkbnr/p1pppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing c2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Black missing c7 pawn", "rnbqkbnr/pp1ppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing d2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"),
        ("Black missing d7 pawn", "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing e2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"),
        ("Black missing e7 pawn", "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing f2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1"),
        ("Black missing f7 pawn", "rnbqkbnr/ppppp1pp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing g2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPP1P/RNBQKBNR w KQkq - 0 1"),
        ("Black missing g7 pawn", "rnbqkbnr/pppppp1p/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing h2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1"),
        ("Black missing h7 pawn", "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing b1 knight", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1"),
        ("Black missing b8 knight", "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing g1 knight", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"),
        ("Black missing g8 knight", "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing c1 bishop", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1"),
        ("Black missing c8 bishop", "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing f1 bishop", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1"),
        ("Black missing f8 bishop", "rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing a1 rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1"),
        ("Black missing a8 rook", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing h1 rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1"),
        ("Black missing h8 rook", "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White missing d1 queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
        ("Black missing d8 queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ]

    # 2. Puzzles from test_puzzle (non-mate)
    puzzles = [
        ("Puzzle 1 (Middlegame sac)", "4rrk1/p2p1p1p/1p2p1p1/2nPq2P/2P5/4B3/PbB2PP1/1R1Q2KR w - - 2 20"),
        ("Puzzle 2 (Endgame pawn)", "8/8/2p5/1p1p1k2/3P4/1PP1pK2/8/8 w - - 4 65"),
        ("Puzzle 3 (Endgame minor)", "8/8/1p1k1p1p/3np3/2B2p2/PP1K1PP1/7P/8 w - - 0 37"),
        ("Puzzle 4 (Middlegame exposed king)", "r1r2k2/ppq3bQ/4p2p/4n3/3p4/2P5/PBB2PPP/4R1K1 w - - 3 25"),
        ("Puzzle 5 (Middlegame active defense)", "r2qr2k/1pp2Qp1/1b4np/pP2P3/P4n2/B1N2N1P/5PP1/R3R1K1 b - - 0 20"),
    ]

    all_positions = starting_missing_pieces + puzzles

    print(f"{'Position/Description':<35} | {'STM':<3} | {'Our Classical':<13} | {'Our NNUE':<10} | {'SF NNUE':<10} | {'SF Final':<10}")
    print("-" * 95)

    markdown_rows = []

    for desc, fen in all_positions:
        # Parse FEN
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        side_to_move = int(game_state[0])
        stm_str = "W" if side_to_move == 0 else "B"

        # Evaluate Classical
        cls_val = evaluate_classical(piece_bbs, occupancy_bbs, game_state, lazy=False)
        # Convert to White's perspective
        cls_val_white = cls_val if side_to_move == 0 else -cls_val

        # Evaluate Our NNUE
        our_nnue_val_white = None
        if weights_loaded:
            our_nnue_val = evaluate_our_nnue(piece_bbs, side_to_move)
            # Convert to White's perspective
            our_nnue_val_white = our_nnue_val if side_to_move == 0 else -our_nnue_val

        # Evaluate Stockfish
        sf_nnue, sf_final = get_stockfish_eval(fen)

        # Format values
        our_nnue_str = f"{our_nnue_val_white:+d}" if our_nnue_val_white is not None else "N/A"
        sf_nnue_str = f"{sf_nnue:+d}" if sf_nnue is not None else "N/A"
        sf_final_str = f"{sf_final:+d}" if sf_final is not None else "N/A"
        cls_str = f"{cls_val_white:+d}"

        print(f"{desc:<35} | {stm_str:<3} | {cls_str:>13} | {our_nnue_str:>10} | {sf_nnue_str:>10} | {sf_final_str:>10}")

        # Store for markdown
        markdown_rows.append(
            f"| {desc} | {stm_str} | {cls_str} | {our_nnue_str} | {sf_nnue_str} | {sf_final_str} |"
        )

    # Generate Markdown Report
    report = f"""# Static Evaluation Comparison Report

This report compares the static evaluations of different positions from three perspectives:
1. **Our Classical Evaluation** (Tapered evaluation: material, PST, mobility, king safety, pawn structure, etc.)
2. **Our NNUE Evaluation** (8-bucket network trained on HalfKAv2_hm features)
3. **Stockfish 17.1 NNUE Evaluation** (Pure NNUE network evaluation)
4. **Stockfish 17.1 Final Evaluation** (Stockfish's final output including scale factor and heuristics)

*Note: All scores are presented in centipawns (cp) and converted to **White's perspective** (positive scores favor White, negative scores favor Black).*

## Comparison Table

| Position Description | STM | Our Classical | Our NNUE | Stockfish NNUE | Stockfish Final |
| :--- | :---: | :---: | :---: | :---: | :---: |
""" + "\n".join(markdown_rows) + "\n"

    # Save report to artifacts directory if available, or current directory
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts"))
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
    
    report_path = os.path.join(artifacts_dir, "eval_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    run_evaluations()
