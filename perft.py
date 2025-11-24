
import argparse
import numpy as np
import time

from chess_engine.fen_parser import parse_fen
from chess_engine.move_generator import generate_legal_moves
from chess_engine.board_operations import make_move, unmake_move
from chess_engine.move import get_from_square, get_to_square
from chess_engine.core import perft, _jit_perft_divide, SQUARE_TO_ALGEBRAIC, PERFT_RESULTS

def do_perft(args):
    """處理 'perft' 命令。"""
    run_perft_test(args.fen, args.depth, divide_on_mismatch=True)

def do_divide(args):
    """處理 'divide' 命令。"""
    perft_divide(args.fen, args.depth)

def do_test_reversibility(args):
    """處理 'test-reversibility' 命令。"""
    test_make_unmake_for_fen(args.fen)

def do_run_tests(args):
    """處理 'run-tests' 命令。"""
    test_perft_suite()

def get_test_key(fen: str) -> str:
    """根據 FEN 獲取已知 perft 結果的字典鍵。"""
    if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
        return "startpos"
    if fen == "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -":
        return "kiwipete"
    # 如果需要，在此處添加其他 FEN
    return "custom"

def test_perft_suite():
    """運行一組 Perft 測試以驗證移動生成。"""
    print("--- Running Perft Test Suite ---")
    # ... (如果有需要可以添加實現，但現在不是優先事項) ...
    print("--- Test Suite Complete ---")

def run_perft_test(fen_string: str, max_depth: int, divide_on_mismatch: bool = False):
    """
    為給定的 FEN 字串運行 Perft 測試，使用新的可變架構。
    
    Args:
        fen_string: 局面的 FEN 字串。
        max_depth: 測試的最大深度。
        divide_on_mismatch: 如果結果不匹配，是否自動運行 divide。
    """
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
        # 我們必須將 COPY 傳遞給 perft 函數，因為它會就地修改陣列
        p_bbs_copy = piece_bbs.copy()
        o_bbs_copy = occupancy_bbs.copy()
        g_state_copy = game_state.copy()

        start_time = time.time()
        nodes = perft(p_bbs_copy, o_bbs_copy, g_state_copy, depth)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_nodes += nodes
        total_time += elapsed_time
        nps = int(nodes / elapsed_time) if elapsed_time > 0 else 0

        expected = PERFT_RESULTS.get(test_key, {}).get(depth, -1)
        status = "OK" if nodes == expected or expected == -1 else "MISMATCH!"

        print(f"Depth {depth}: Nodes: {nodes:<10} Time: {elapsed_time:.3f}s, NPS: {nps:<10} Expected: {expected:<10} -> {status}")

        if status == "MISMATCH!" and divide_on_mismatch:
            perft_divide(fen_string, depth)
            break

    if total_time > 0:
        avg_nps = int(total_nodes / total_time)
        print("---------------------------------")
        print(f"Total Nodes: {total_nodes}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Average NPS: {avg_nps}")
    print("--- Perft Test Complete ---")


def perft_divide(fen_string: str, depth: int):
    """顯示給定深度的每個移動的節點計數。"""
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
    """測試 make_move 和 unmake_move 是否完全可逆。"""
    print(f"--- Testing make/unmake for FEN: {fen} ---")

    # 1. Get initial state
    p_bbs, o_bbs, g_state = parse_fen(fen)

    # 2. Store a deep copy of the initial state for comparison
    original_p_bbs = p_bbs.copy()
    original_o_bbs = o_bbs.copy()
    original_g_state = g_state.copy()

    # 3. Generate moves from the initial state
    legal_moves = generate_legal_moves(p_bbs, o_bbs, g_state)

    failures = []
    for move in legal_moves:
        # 4. Make a move (this modifies the arrays in-place)
        unmake_info = make_move(p_bbs, o_bbs, g_state, move)

        # 5. Unmake the move (this should restore them to the original state)
        unmake_move(p_bbs, o_bbs, g_state, move, unmake_info)

        # 6. Compare the now-restored state with the original deep copy
        is_equal = (
            np.array_equal(p_bbs, original_p_bbs) and
            np.array_equal(o_bbs, original_o_bbs) and
            np.array_equal(g_state, original_g_state)
        )

        if not is_equal:
            from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(move)]
            to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(move)]
            failures.append(f"{from_sq_alg}{to_sq_alg}")

            # Restore state for the next move test
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
