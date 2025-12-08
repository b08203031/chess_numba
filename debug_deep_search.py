import argparse
import time
import numpy as np
from chess_engine.fen_parser import parse_fen
from chess_engine.search import iterative_deepening_search
from chess_engine.engine_types import SearchContext
from chess_engine.transposition_table import create_transposition_table
from chess_engine.move import move_to_uci
from chess_engine.move_generator import generate_legal_moves
from chess_engine.board_operations import make_move, unmake_move
from chess_engine.constants import MAX_PLY

def find_move_by_uci(piece_bbs, occupancy_bbs, game_state, uci_move_str):
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    for move in moves:
        if move_to_uci(move) == uci_move_str:
            return move
    return None

def debug_search(fen, moves, depth, time_limit_ms):
    print(f"FEN: {fen}")
    print(f"Depth: {depth}")
    print(f"Time Limit: {time_limit_ms}ms")
    print("-" * 60)

    # Setup Search Context
    # Using small TT for debugging
    transposition_table = create_transposition_table(16)
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

    results = []

    for move_str in moves:
        print(f"\nAnalyzing move: {move_str}")

        # Make the move
        move = find_move_by_uci(piece_bbs, occupancy_bbs, game_state, move_str)
        if move is None:
            print(f"Error: Illegal move {move_str}")
            continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

        # Search the resulting position
        # Note: The result of search is from the perspective of the side to move *after* the move.
        # So it's the opponent's score.
        # We negate it to get the score for the side that made the move.

        time_config = {'optimum_time': time_limit_ms, 'maximum_time': time_limit_ms}

        # Reset Search Context stats for each search
        search_context.nodes_searched = np.uint64(0)
        # We might want to clear TT or keep it. Keeping it simulates real game behavior (cache).
        # But for comparison, maybe clearing it is fairer?
        # Let's keep it but reset history/killers?
        # search_context.history_table.fill(0)
        # search_context.killer_moves.fill(0)

        (best_reply, score, nodes, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = iterative_deepening_search(
            piece_bbs, occupancy_bbs, game_state, depth, time_config, search_context
        )

        my_score = -score
        best_reply_uci = move_to_uci(best_reply) if best_reply != 0 else "None"

        print(f"  Score (for me): {my_score} (cp)")
        print(f"  Best Reply: {best_reply_uci}")
        print(f"  Nodes: {nodes}")

        results.append({
            "move": move_str,
            "score": my_score,
            "nodes": nodes,
            "best_reply": best_reply_uci
        })

        # Unmake move
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("-" * 60)
    # Sort results by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    for res in results:
        print(f"Move: {res['move']:<6} | Score: {res['score']:>6} | Best Reply: {res['best_reply']:<6} | Nodes: {res['nodes']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep search analysis of specific moves.")
    parser.add_argument("fen", type=str, help="The FEN string of the position.")
    parser.add_argument("--moves", type=str, nargs='+', required=True, help="List of moves to analyze (UCI format).")
    parser.add_argument("--depth", type=int, default=10, help="Search depth.")
    parser.add_argument("--time", type=int, default=1000, help="Time limit per move in ms.")

    args = parser.parse_args()

    debug_search(args.fen, args.moves, args.depth, args.time)
