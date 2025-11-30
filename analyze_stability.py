import sys
import io
import time
import numpy as np
import chess
import chess.pgn

# Import engine modules
# Ensure the script can find the chess_engine package if run from root
sys.path.append('.')

from chess_engine.fen_parser import parse_fen
from chess_engine.search import iterative_deepening_search, format_score_for_uci
from chess_engine.engine_types import SearchContext
from chess_engine.transposition_table import create_transposition_table
from chess_engine.constants import MAX_PLY, NO_MOVE
from chess_engine.move import move_to_uci

# PGN from the task
PGN_TEXT = """
1. d4 d5 2. c4 e6 3. Nf3 Nf6 4. Nc3 Be7 5. Bf4 O-O 6. e3 b6 7. cxd5 Nxd5 8. Nxd5
Qxd5 9. Be2 Qa5+ 10. Nd2 Ba6 11. O-O c5 12. Nc4 Bxc4 13. Bxc4 cxd4 14. Qf3 Nd7
15. Qb7 Rad8 16. exd4 Nf6 17. Bxe6 fxe6 18. Qxe7 Rfe8 19. Qb7 Rxd4 20. Be3 Rd7
21. Qc6 Qd5 22. Qa4 Rc8 23. h3 Rc4 24. Qb3 Rc6 25. Qa3 Kf7 26. Rac1 Rxc1 27.
Rxc1 Qb5 28. Bf4 Qe2 29. Bg3 Ne4 30. Kh2 Nxg3 31. Qxg3 Qxb2 32. Rc3 Qd2 33. Rf3+
Kg8 34. Qb8+ Rd8 35. Qxa7 Qd6+ 36. Rg3 g6 37. Qa6 Rf8 38. Qe2 Rf5 39. a4 Rf4 40.
Qc2 Rd4 41. Qc8+ Kg7 42. Qb7+ Kg8 1/2-1/2
"""

def main():
    # 1. Parse the game
    pgn_io = io.StringIO(PGN_TEXT)
    game = chess.pgn.read_game(pgn_io)
    if not game:
        print("Error: Could not parse PGN.")
        return

    # 2. Initialize Engine Components
    # 64MB TT
    transposition_table = create_transposition_table(64)
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    # History table: 6 piece types x 64 squares (approximated for simplicity in initialization)
    # The search uses [piece_type, square]
    history_table = np.zeros((6, 64), dtype=np.int32)
    
    search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)

    # 3. Setup Loop
    board = game.board()
    
    # Python list to accumulate history for repetition detection
    game_history_list = []
    
    # To store summary results: (move_number, move_san, depth, score, nodes, nps, time, pv)
    summary_results = []
    
    # We analyze "every move".
    # This implies analyzing the board state *before* each move is played in the PGN?
    # Or analyzing the board state *after* the move?
    # Standard analysis: "After 1. Nf3, engine suggests...".
    # But usually one wants to see if the engine agrees with the move played, or what it would play instead.
    # The user says "Analyze each step".
    # Let's assume we analyze the position *on the board* before applying the next move.
    
    # Add StartPos to history if needed?
    # Standard: StartPos key is usually NOT in history list passed to search, 
    # because search adds current position to path. 
    # But if we want 3-fold of StartPos later, we need it.
    # Let's rely on standard practice: history contains *previous* positions.
    
    # Iterate through moves
    move_count = 0
    tt_generation = 0
    
    for move in game.mainline_moves():
        move_count += 1
        tt_generation += 1 # Increment generation for each move search
        
        # Current board state (before pushing the move)
        fen = board.fen()
        
        # Parse FEN to engine format
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        
        # Update history count in context (copy list to typed array)
        # Note: iterative_deepening_search handles this if we pass the list.
        # But we must update the list ourselves.
        # Wait, iterative_deepening_search takes `game_history_list`.
        # We need to append the *previous* move's resulting position to history.
        # For the first move (StartPos), history is empty.
        
        # Display Move Info
        full_move_num = board.fullmove_number
        is_white = board.turn == chess.WHITE
        move_prefix = f"{full_move_num}. " if is_white else f"{full_move_num}... "
        print(f"Analyzing Move {move_count}: {move_prefix}{board.san(move)}")
        
        # Run Search
        # Fixed depth 8
        time_config = {'maximum_time': 0} # No time limit, just depth
        
        # Important: Pass the history list
        result = iterative_deepening_search(
            piece_bbs, occupancy_bbs, game_state,
            max_depth=8,
            time_config=time_config,
            search_context=search_context,
            game_history_list=game_history_list,
            tt_generation=tt_generation
        )
        
        (best_move, score, nodes, q_nodes, cutoffs, tt_hits,
         depth, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se) = result

        # Store result for summary
        pv_string = " ".join([move_to_uci(search_context.pv_table[0, i]) for i in range(MAX_PLY) if search_context.pv_table[0, i] != NO_MOVE])
        uci_score = format_score_for_uci(score)
        
        # Approximation of time and NPS since iterative_deepening_search prints them but returns total nodes.
        # We can't easily get the exact time of the *last* depth from return values, 
        # but the summary probably just needs the final result data.
        # The prompt asked for "info depth {} ...". This was printed by the function.
        # For the summary, we will reprint the Depth 8 info.
        
        summary_entry = {
            'move_idx': move_count,
            'move_str': f"{move_prefix}{board.san(move)}",
            'depth': depth,
            'score': uci_score,
            'nodes': nodes,
            'pv': pv_string
        }
        summary_results.append(summary_entry)

        # Update History for *next* iteration
        # We must add the Zobrist key of the *current* position (which we just analyzed) 
        # to the history list, because for the next search (after pushing move), 
        # this position is now in the past.
        current_zobrist_key = game_state[4]
        game_history_list.append(current_zobrist_key)
        
        # Push the move to get to the next position
        board.push(move)
        
        print("-" * 40)

    # 4. Final Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY (Depth 8 Results)")
    print("="*50)
    for res in summary_results:
        print(f"Move {res['move_str']:<10} | info depth {res['depth']} score {res['score']} nodes {res['nodes']} pv {res['pv']}")
    print("="*50)

if __name__ == "__main__":
    main()
