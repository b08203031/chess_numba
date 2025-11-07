
import argparse
import numpy as np
import sys
import os
import subprocess
import time

from chess_engine.fen_parser import parse_fen
from chess_engine.move_generator import generate_legal_moves, generate_pseudo_legal_moves
from chess_engine.board_operations import make_move, unmake_move
from chess_engine.move import get_from_square, get_to_square
from chess_engine.core import perft, _jit_perft_divide, SQUARE_TO_ALGEBRAIC, PERFT_RESULTS

STOCKFISH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Stockfish', 'src', 'stockfish'))

# --- High-Level Command Functions ---

def do_perft(args):
    """Handler for the 'perft' command."""
    fen = args.fen
    depth = args.depth

    # Determine the test key for comparing with known results
    if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
        test_key = "startpos"
    elif fen == "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -":
        test_key = "kiwipete"
    elif fen == "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1 ":
        test_key = "Position 3"
    elif fen == "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1":
        test_key = "Position 4"
    elif fen == "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8":
        test_key = "Position 5"

    else:
        test_key = "custom"

    run_perft_test(fen, depth, test_key, divide_on_mismatch=True)

def do_divide(args):
    """Handler for the 'divide' command."""
    perft_divide(args.fen, args.depth)

def do_test_reversibility(args):
    """Handler for the 'test-reversibility' command."""
    test_make_unmake_for_fen(args.fen)

def do_run_tests(args):
    """Handler for the 'run-tests' command."""
    test_perft()

# --- Logic Migrated from perft.py and test_board_operations.py ---

def test_perft():
    """Runs a suite of Perft tests to validate move generation."""
    print("--- Running Perft Test Suite ---")

    PERFT_TESTS = [
        # (FEN, depth, expected_nodes, description)
        ("r3kb1Q/p1ppqp2/bn2pnp1/3PN3/4P3/1pN5/PPPBBPPP/R3K2R w KQq - 0 1", 1, 56, "Bug test: Queen diagonal capture (h8f6)"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20, "Start position depth 1"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400, "Start position depth 2"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 1, 48, "Kiwipete depth 1"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 2, 2039, "Kiwipete depth 2"),
    ]

    passed = 0
    failed = 0

    for fen, depth, expected, description in PERFT_TESTS:
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        game_state_typed = (
            np.uint8(game_state[0]), np.uint8(game_state[1]), np.int8(game_state[2]),
            np.uint8(game_state[3]), np.uint64(game_state[4])
        )

        # JIT warm-up
        perft(piece_bbs, occupancy_bbs, game_state_typed, 1)

        result = perft(piece_bbs, occupancy_bbs, game_state_typed, depth)

        if result == expected:
            print(f"  ✓ {description}: got {result}")
            passed += 1
        else:
            print(f"  ✗ {description}: got {result}, expected {expected}")
            failed += 1

    print("---------------------------------")
    print(f"Results: {passed} passed, {failed} failed")
    print("--- Test Suite Complete ---")
    return failed == 0

def run_perft_test(fen_string: str, max_depth: int, test_key: str, divide_on_mismatch: bool = False):
    # This is the logic from the original chess_engine/perft.py
    print("--- Starting Perft Test ---")
    print(f"FEN: {fen_string}")
    print(f"Max Depth: {max_depth}")
    print("---------------------------------")

    total_nodes = 0
    total_time = 0.0

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen_string)
    game_state_typed = (
        np.uint8(game_state[0]), np.uint8(game_state[1]), np.int8(game_state[2]),
        np.uint8(game_state[3]), np.uint64(game_state[4])
    )

    # JIT Compilation Warm-up
    if max_depth > 0:
        perft(piece_bbs, occupancy_bbs, game_state_typed, 1)

    for depth in range(1, max_depth + 1):
        start_time = time.time()
        # Re-parse state for each run to ensure clean state
        p_bbs, o_bbs, g_state = parse_fen(fen_string)
        g_state_typed = (
            np.uint8(g_state[0]), np.uint8(g_state[1]), np.int8(g_state[2]),
            np.uint8(g_state[3]), np.uint64(g_state[4])
        )
        nodes = perft(p_bbs, o_bbs, g_state_typed, depth)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_nodes += nodes
        total_time += elapsed_time
        nps = int(nodes / elapsed_time) if elapsed_time > 0 else 0

        expected_results_for_key = PERFT_RESULTS.get(test_key)
        expected = -1
        if expected_results_for_key:
            expected = expected_results_for_key.get(depth, -1)

        status = "OK" if nodes == expected else "MISMATCH!"

        print(f"Depth {depth}: Nodes: {nodes:<10} Time: {elapsed_time:.3f}s, NPS: {nps:<10} Expected: {expected:<10} -> {status}")

        if status == "MISMATCH!" and divide_on_mismatch:
            perft_divide(fen_string, depth)
            break

    avg_nps = int(total_nodes / total_time) if total_time > 0 else 0
    print("---------------------------------")
    if total_time > 0:
      print(f"Total Nodes: {total_nodes}")
      print(f"Total Time: {total_time:.3f}s")
      print(f"Average NPS: {avg_nps}")
    print("--- Perft Test Complete ---")


def perft_divide(fen_string: str, depth: int):
    # This is the logic from the original chess_engine/perft.py
    print(f"--- Perft Divide for Depth {depth} ---")
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen_string)
    game_state_typed = (
        np.uint8(game_state[0]), np.uint8(game_state[1]), np.int8(game_state[2]),
        np.uint8(game_state[3]), np.uint64(game_state[4])
    )

    results = _jit_perft_divide(piece_bbs, occupancy_bbs, game_state_typed, depth)

    total_nodes = 0
    for i in range(len(results)):
        move, nodes = results[i]
        from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(np.uint16(move))]
        to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(np.uint16(move))]
        print(f"{from_sq_alg}{to_sq_alg}: {nodes}")
        total_nodes += nodes

    print(f"\nTotal Nodes: {total_nodes}")


def test_make_unmake_for_fen(fen: str):
    # This is the logic from the original tests/test_board_operations.py
    print(f"--- Testing make/unmake for FEN: {fen} ---")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    game_state_typed = (
        np.uint8(game_state[0]), np.uint8(game_state[1]), np.int8(game_state[2]),
        np.uint8(game_state[3]), np.uint64(game_state[4])
    )

    board_state_flat = piece_bbs + occupancy_bbs + (game_state_typed[1], game_state_typed[2], game_state_typed[0])
    legal_moves = generate_legal_moves(board_state_flat)

    failures = []
    for move in legal_moves:
        new_piece_bbs, new_occupancy_bbs, new_game_state, unmake_info = make_move(
            piece_bbs, occupancy_bbs, game_state_typed, move
        )
        unmade_piece_bbs, unmade_occupancy_bbs, unmade_game_state = unmake_move(
            new_piece_bbs, new_occupancy_bbs, new_game_state, move, unmake_info
        )

        is_equal = (
            piece_bbs == unmade_piece_bbs and
            occupancy_bbs == unmade_occupancy_bbs and
            game_state_typed[:-1] == unmade_game_state[:-1] # Ignore Zobrist key
        )

        if not is_equal:
            from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(move)]
            to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(move)]
            failures.append(f"{from_sq_alg}{to_sq_alg}")

    if not failures:
        print(f"OK! All {len(legal_moves)} moves successfully reverted the board state.")
    else:
        print("--- FAILURE ---")
        print(f"The following moves failed to revert the board state: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line interface for the chess engine.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Perft Command ---
    parser_perft = subparsers.add_parser("perft", help="Run a performance test to a certain depth.")
    parser_perft.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="FEN string of the position.")
    parser_perft.add_argument("--depth", type=int, required=True, help="The depth to run the test to.")
    parser_perft.set_defaults(func=do_perft)

    # --- Divide Command ---
    parser_divide = subparsers.add_parser("divide", help="Show node count for each move at depth 1 for a given position.")
    parser_divide.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="FEN string of the position.")
    parser_divide.add_argument("--depth", type=int, default=1, help="The depth for the divide test.")
    parser_divide.set_defaults(func=do_divide)

    # --- Test Reversibility Command ---
    parser_reversibility = subparsers.add_parser("test-reversibility", help="Test if make_move and unmake_move are perfectly reversible.")
    parser_reversibility.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", help="FEN string of the position.")
    parser_reversibility.set_defaults(func=do_test_reversibility)

    # --- Run Tests Command ---
    parser_run_tests = subparsers.add_parser("run-tests", help="Run the full test suite.")
    parser_run_tests.set_defaults(func=do_run_tests)

    args = parser.parse_args()
    args.func(args)
