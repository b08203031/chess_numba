# chess_engine/perft.py
import numba as nb
import numpy as np
import time

from chess_engine.board_operations import make_move
from chess_engine.move_generator import generate_pseudo_legal_moves, generate_legal_moves, is_square_attacked, get_ls1b_index
from chess_engine.fen_parser import parse_fen
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
import numba.types as nbt
from chess_engine.move import get_from_square, get_to_square

# --- Constants ---
WHITE, BLACK = 0, 1

# --- Standard Perft Results for Verification ---
PERFT_RESULTS = {
    "startpos": [20, 400, 8902, 197281, 4865609, 119060324],
    "kiwipete": [48, 2039, 97862, 4085603, 193690690],
}

SQUARE_TO_ALGEBRAIC = {i: f"{chr(ord('a') + i % 8)}{i // 8 + 1}" for i in range(64)}

@nb.jit(nbt.uint64(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def perft(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    Core recursive Perft function.
    Counts the total number of legal moves to a given depth.
    """

    board_state_flat = piece_bbs + occupancy_bbs + (game_state[1], np.int8(game_state[2]), game_state[0])
    moves = generate_legal_moves(board_state_flat)

    if depth == 1:
        return np.uint64(len(moves))

    nodes = np.uint64(0)

    for i in range(len(moves)):
        move = moves[i]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(piece_bbs, occupancy_bbs, game_state, move)
        nodes += perft(new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1)

    return nodes

@nb.jit(nbt.types.Array(nbt.uint64, 2, "C")(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def _jit_perft_divide(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    JIT-compiled core logic for perft_divide.
    Returns an array of moves and their corresponding node counts.
    """
    if depth == 0:
        return np.zeros((0, 2), dtype=np.uint64)

    side_to_move = game_state[0]
    board_state_flat = piece_bbs + occupancy_bbs + (game_state[1], np.int8(game_state[2]), game_state[0])
    moves = generate_pseudo_legal_moves(board_state_flat)

    results = np.zeros((len(moves), 2), dtype=np.uint64)
    count = 0

    for i in range(len(moves)):
        move = moves[i]

        new_piece_bbs, new_occupancy_bbs, new_game_state, unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

        king_bb = new_piece_bbs[5] if side_to_move == WHITE else new_piece_bbs[11]
        if king_bb == 0:
            continue

        king_sq = get_ls1b_index(king_bb)
        new_board_state_flat = new_piece_bbs + new_occupancy_bbs + (new_game_state[1], np.int8(new_game_state[2]), new_game_state[0])
        opponent_color = 1 - side_to_move

        if not is_square_attacked(king_sq, opponent_color, new_board_state_flat):
            nodes = perft(new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1)
            results[count, 0] = move
            results[count, 1] = nodes
            count += 1

    return results[:count]

def perft_divide(fen_string: str, depth: int):
    """
    High-level wrapper for perft_divide that handles printing.
    """
    print(f"--- Perft Divide for Depth {depth} ---")
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen_string)

    # Explicitly cast game_state to ensure Numba compatibility before passing to JIT functions
    game_state_typed = (
        np.uint8(game_state[0]),
        np.uint8(game_state[1]),
        np.int8(game_state[2]),
        np.uint8(game_state[3]),
        np.uint64(game_state[4])
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

def run_perft_test(fen_string: str, max_depth: int, test_key: str, divide_on_mismatch: bool = False):
    """
    Drives the Perft test, compares results, and prints formatted output.
    """
    print("--- Starting Perft Test ---")
    print(f"FEN: {fen_string}")
    print(f"Max Depth: {max_depth}")
    print("---------------------------------")

    total_nodes = 0
    total_time = 0.0

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen_string)

    # Explicitly cast game_state to ensure Numba compatibility before passing to JIT functions
    game_state_typed = (
        np.uint8(game_state[0]),
        np.uint8(game_state[1]),
        np.int8(game_state[2]),
        np.uint8(game_state[3]),
        np.uint64(game_state[4])
    )

    # JIT Compilation Warm-up
    if max_depth > 0:
        perft(piece_bbs, occupancy_bbs, game_state_typed, 1)

    for depth in range(1, max_depth + 1):
        start_time = time.time()
        nodes = perft(piece_bbs, occupancy_bbs, game_state_typed, depth)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_nodes += nodes
        total_time += elapsed_time
        nps = int(nodes / elapsed_time) if elapsed_time > 0 else 0

        expected = PERFT_RESULTS.get(test_key, [])[depth - 1] if test_key in PERFT_RESULTS and depth <= len(PERFT_RESULTS[test_key]) else -1
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

if __name__ == '__main__':
    # --- Test Case 1: Initial Position ---
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    run_perft_test(startpos_fen, 5, "startpos", divide_on_mismatch=True)

    print("\n") # Separator

    # --- Test Case 2: Kiwipete ---
    kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
    run_perft_test(kiwipete_fen, 4, "kiwipete", divide_on_mismatch=True)
