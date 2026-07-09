import sys
import os
import shutil

# Clear Numba cache in chess_engine/classical/__pycache__ before importing
classical_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chess_engine", "classical", "__pycache__"))
if os.path.exists(classical_cache_dir):
    try:
        shutil.rmtree(classical_cache_dir)
        print(f"Cleared Numba cache: {classical_cache_dir}")
    except Exception as e:
        print(f"Warning: Failed to clear Numba cache: {e}")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import time

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move_generator import generate_legal_moves
from chess_engine.classical.board_operations import make_move, unmake_move
from chess_engine.classical.move import get_from_square, get_to_square
from chess_engine.classical.core import perft, _jit_perft_divide, SQUARE_TO_ALGEBRAIC, PERFT_RESULTS

POSITIONS = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "Position 3": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "Position 4": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "Position 5": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
}

def get_test_key(fen: str) -> str:
    """Get expected perft result key for FEN"""
    fen = fen.strip()
    for name, f in POSITIONS.items():
        f_parts = f.split()
        fen_parts = fen.split()
        if len(fen_parts) >= 4 and len(f_parts) >= 4:
            if fen_parts[:4] == f_parts[:4]:
                return name
        elif fen_parts == f_parts:
            return name
    return "custom"

def do_perft(args):
    """Execute perft command"""
    run_perft_test(args.fen, args.depth, divide_on_mismatch=True)

def do_divide(args):
    """Execute divide command"""
    perft_divide(args.fen, args.depth)

def do_test_reversibility(args):
    """Execute test-reversibility command"""
    test_make_unmake_for_fen(args.fen)

def do_run_tests(args):
    """Execute run-tests command"""
    test_perft_suite()

def test_perft_suite():
    """Run a series of perft tests to verify move generator"""
    print("--- Running Perft Test Suite ---")
    
    suite_depths = {
        "startpos": 5,
        "kiwipete": 5,
        "Position 3": 5,
        "Position 4": 5,
        "Position 5": 5
    }
    
    any_failures = False
    
    for name, fen in POSITIONS.items():
        print(f"\nRunning perft for position: {name}")
        print(f"FEN: {fen}")
        print("-" * 40)
        max_depth = suite_depths.get(name, 3)
        
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        
        # Warm-up JIT
        perft(piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy(), 1)
        
        for depth in range(1, max_depth + 1):
            nodes_runs = []
            times_runs = []
            nps_runs = []
            
            for _ in range(5):
                p_bbs_copy = piece_bbs.copy()
                o_bbs_copy = occupancy_bbs.copy()
                g_state_copy = game_state.copy()
                
                start_time = time.perf_counter()
                nodes = perft(p_bbs_copy, o_bbs_copy, g_state_copy, depth)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                nodes_runs.append(nodes)
                times_runs.append(elapsed)
                nps_runs.append(nodes / elapsed if elapsed > 0 else 0.0)
            
            # Sort to remove min and max
            sorted_times = sorted(times_runs)
            sorted_nps = sorted(nps_runs)
            
            avg_time = sum(sorted_times[1:4]) / 3.0
            avg_nps = sum(sorted_nps[1:4]) / 3.0
            
            nodes = nodes_runs[0]
            expected = PERFT_RESULTS.get(name, {}).get(depth, -1)
            
            status = "OK" if nodes == expected or expected == -1 else "MISMATCH!"
            
            print(f"Depth {depth}: Nodes: {nodes:<10} Time: {avg_time:.6f}s, Average NPS: {int(avg_nps):<10} Expected: {expected:<10} -> {status}")
            
            if nodes != expected and expected != -1:
                any_failures = True
                
    if any_failures:
        print("\n--- Test Suite FAILED! ---")
        sys.exit(1)
    else:
        print("\n--- Test Suite PASSED! All nodes matched expected values. ---")

def run_perft_test(fen_string: str, max_depth: int, divide_on_mismatch: bool = False):
    """Run perft test for given FEN string"""
    print("--- Starting Perft Test ---")
    print(f"FEN: {fen_string}")
    print(f"Max Depth: {max_depth}")
    print("---------------------------------")

    total_nodes = 0
    total_time = 0.0
    test_key = get_test_key(fen_string)

    # Initial parse
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen_string)

    # Warm-up JIT compilation
    if max_depth > 0:
        perft(piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy(), 1)

    for depth in range(1, max_depth + 1):
        times = []
        nps_list = []
        nodes = 0

        # Execute 5 times
        for run in range(5):
            p_bbs_copy = piece_bbs.copy()
            o_bbs_copy = occupancy_bbs.copy()
            g_state_copy = game_state.copy()

            start_time = time.perf_counter()
            nodes = perft(p_bbs_copy, o_bbs_copy, g_state_copy, depth)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            times.append(elapsed)
            nps = nodes / elapsed if elapsed > 0 else 0.0
            nps_list.append(nps)

        # Sort and remove highest/lowest
        sorted_times = sorted(times)
        sorted_nps = sorted(nps_list)

        avg_time = sum(sorted_times[1:4]) / 3.0
        avg_nps = sum(sorted_nps[1:4]) / 3.0

        total_nodes += nodes
        total_time += avg_time

        expected = PERFT_RESULTS.get(test_key, {}).get(depth, -1)
        status = "OK" if nodes == expected or expected == -1 else "MISMATCH!"

        print(f"Depth {depth}: Nodes: {nodes:<10} Time: {avg_time:.6f}s, NPS: {int(avg_nps):<10} Expected: {expected:<10} -> {status}")

        if status == "MISMATCH!" and divide_on_mismatch:
            perft_divide(fen_string, depth)
            break

    if total_time > 0:
        avg_nps = int(total_nodes / total_time)
        print("---------------------------------")
        print(f"Total Nodes: {total_nodes}")
        print(f"Total Time: {total_time:.6f}s")
        print(f"Average NPS: {avg_nps}")
    print("--- Perft Test Complete ---")

def perft_divide(fen_string: str, depth: int):
    """Show node count for each move at given depth"""
    print(f"--- Perft Divide for Depth {depth} ---")
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen_string)

    # Pass copies as the function will mutate them
    results = _jit_perft_divide(piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy(), depth)

    total_nodes = 0
    for i in range(len(results)):
        move, nodes = results[i]
        from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(np.uint16(move))]
        to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(np.uint16(move))]
        print(f"{from_sq_alg}{to_sq_alg}: {nodes}")
        total_nodes += nodes

    print(f"\nTotal Nodes: {total_nodes}")

def test_make_unmake_for_fen(fen: str):
    """Test if make_move and unmake_move are completely reversible"""
    print(f"--- Testing make/unmake for FEN: {fen} ---")

    p_bbs, o_bbs, g_state = parse_fen(fen)

    original_p_bbs = p_bbs.copy()
    original_o_bbs = o_bbs.copy()
    original_g_state = g_state.copy()

    legal_moves = generate_legal_moves(p_bbs, o_bbs, g_state)

    failures = []
    for move in legal_moves:
        unmake_info = make_move(p_bbs, o_bbs, g_state, move)
        unmake_move(p_bbs, o_bbs, g_state, move, unmake_info)

        is_equal = (
            np.array_equal(p_bbs, original_p_bbs) and
            np.array_equal(o_bbs, original_o_bbs) and
            np.array_equal(g_state, original_g_state)
        )

        if not is_equal:
            from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(move)]
            to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(move)]
            failures.append(f"{from_sq_alg}{to_sq_alg}")

            p_bbs, o_bbs, g_state = original_p_bbs.copy(), original_o_bbs.copy(), original_g_state.copy()

    if not failures:
        print(f"OK! All {len(legal_moves)} moves were successfully reverted.")
    else:
        print("--- FAILURE ---")
        print(f"The following moves failed to revert the board state: {failures}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line interface for the chess engine.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    parser_perft = subparsers.add_parser("perft", help="Run a performance test to a certain depth.")
    parser_perft.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="FEN string of the position.")
    parser_perft.add_argument("--depth", type=int, required=True, help="The depth to run the test to.")
    parser_perft.set_defaults(func=do_perft)

    parser_divide = subparsers.add_parser("divide", help="Show node count for each move at depth 1 for a given position.")
    parser_divide.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="FEN string of the position.")
    parser_divide.add_argument("--depth", type=int, default=1, help="The depth for the divide test.")
    parser_divide.set_defaults(func=do_divide)

    parser_reversibility = subparsers.add_parser("test-reversibility", help="Test if make_move and unmake_move are perfectly reversible.")
    parser_reversibility.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="FEN string of the position.")
    parser_reversibility.set_defaults(func=do_test_reversibility)

    parser_run_tests = subparsers.add_parser("run-tests", help="Run the full test suite.")
    parser_run_tests.set_defaults(func=do_run_tests)

    args = parser.parse_args()
    args.func(args)
