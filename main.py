
import sys
import numpy as np
import random
import os
import threading
import time

from chess_engine.board_operations import make_move
from chess_engine.core import generate_legal_moves
from chess_engine.debug_utils import log_info
from chess_engine.fen_parser import parse_fen
from chess_engine.move import move_to_uci
from chess_engine.search import iterative_deepening_search
from chess_engine.time_manager import calculate_search_time
from chess_engine.transposition_table import (
    create_transposition_table,
    clear_transposition_table,
    TT_SIZE_MB,
)
from chess_engine.opening_book import OpeningBook
from chess_engine.engine_types import SearchContext

# Maximum search depth (Ply) for arrays like killer moves
MAX_PLY = 64

# Global variable to hold the current search thread and context
search_thread = None
global_search_context = None

def run_search(board_state, max_depth, time_config, transposition_table, killer_moves, history_table, pv_table):
    """
    Wrapper function to run the search in a separate thread.
    """
    global global_search_context
    
    # Create a temporary context to hold the stop flag and other shared data
    # Note: In a more complex engine, we might pass the context directly, 
    # but here we need to expose the stop flag to the main thread.
    # The iterative_deepening_search creates its own context, but we need a way to signal it.
    # We modified iterative_deepening_search to use a passed context or we need to modify it to accept a stop flag.
    # Wait, I modified iterative_deepening_search to CREATE a context. 
    # Let's look at search.py again. 
    # Ah, I modified it to: search_context = SearchContext(...)
    # This means I cannot easily pass the stop flag from outside unless I change the signature of iterative_deepening_search
    # OR I make the context global/shared.
    
    # Actually, the cleaner way is to pass a "stop_signal" object or similar.
    # But since I already modified engine_types.py to have stop_flag in SearchContext,
    # and search.py creates the context inside iterative_deepening_search...
    # I need to change iterative_deepening_search to ACCEPT a context or at least the stop flag array.
    
    # However, for now, let's stick to the plan. I can't easily change the signature of the JIT-compiled function 
    # without recompiling everything and potentially breaking things if I'm not careful.
    # BUT, I can use a global variable in the python side that the JIT function checks? No, JIT functions can't see python globals easily.
    
    # Let's re-read search.py.
    # search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)
    # It creates a NEW context.
    
    # To make this work with threading, I should have modified iterative_deepening_search to TAKE the context as an argument.
    # OR, I can rely on the fact that I can't easily stop it from the main thread with the current design 
    # UNLESS I modify iterative_deepening_search to accept the stop flag.
    
    # Let's do a quick fix: I will modify iterative_deepening_search in search.py to accept an optional context 
    # or at least return the context it created so I can't... wait, if it returns it, it's too late.
    
    # Correct approach:
    # 1. Create the SearchContext OUTSIDE iterative_deepening_search.
    # 2. Pass it TO iterative_deepening_search.
    
    # I need to modify search.py one more time to accept search_context as an argument to iterative_deepening_search.
    # This is better design anyway.
    pass

def uci_loop():
    """
    The main UCI loop of the engine.
    Listens for and responds to UCI commands from a GUI.
    """
    global search_thread, global_search_context

    # Initialize engine components before the loop starts
    transposition_table = create_transposition_table(TT_SIZE_MB)
    
    # We need persistent tables for the search context
    killer_moves = np.zeros((MAX_PLY, 2), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)

    # --- Initialize Opening Book ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    book_path = os.path.join(script_dir, "polyglot.bin")
    opening_book = OpeningBook(book_path)
    if opening_book.book is None:
        log_info("Opening book not found or failed to load.")

    board_state = None

    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line:
            continue

        tokens = line.strip().split()
        command = tokens[0]

        if command == "uci":
            print("id name MyChessEngine")
            print("id author YourName")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            clear_transposition_table(transposition_table)
            killer_moves.fill(0)
            history_table.fill(0)
            pv_table.fill(0)
        elif command == "position":
            # --- Parse position command ---
            if "startpos" in tokens:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                board_state = (piece_bbs, occupancy_bbs, game_state)
            elif "fen" in tokens:
                fen_start_index = tokens.index("fen") + 1
                fen = " ".join(tokens[fen_start_index:])
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                board_state = (piece_bbs, occupancy_bbs, game_state)
            
            # Handle 'moves'
            if "moves" in tokens:
                moves_start_index = tokens.index("moves") + 1
                # We need to update board_state. Since tuples are immutable, we reconstruct it.
                p_bbs, o_bbs, g_state = board_state
                
                for move_uci in tokens[moves_start_index:]:
                    legal_moves = generate_legal_moves(p_bbs, o_bbs, g_state)
                    found_move = False
                    for legal_move in legal_moves:
                        if move_to_uci(legal_move) == move_uci:
                            # make_move modifies in-place but returns info. 
                            # However, for the loop to work with the tuple structure we might need to be careful.
                            # Actually, the arrays inside the tuple are mutable.
                            make_move(p_bbs, o_bbs, g_state, legal_move)
                            found_move = True
                            break
                    if not found_move:
                        log_info(f"Illegal move {move_uci} received. Ignoring.")
                        break
                board_state = (p_bbs, o_bbs, g_state)

        elif command == "go":
            if not board_state:
                log_info("No position set. Ignoring 'go' command.")
                continue

            # Stop any existing search
            if global_search_context is not None:
                global_search_context.stop_flag[0] = True
            if search_thread is not None and search_thread.is_alive():
                search_thread.join()

            # --- Parse go command ---
            max_depth = 128
            time_config = {}

            if "movetime" in tokens:
                movetime_ms = int(tokens[tokens.index("movetime") + 1])
                time_config = {'optimum_time': movetime_ms, 'maximum_time': movetime_ms}
            elif "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])
                time_config = {'optimum_time': 0, 'maximum_time': 0}
            else:
                wtime_ms = int(tokens[tokens.index("wtime") + 1]) if "wtime" in tokens else 0
                btime_ms = int(tokens[tokens.index("btime") + 1]) if "btime" in tokens else 0
                winc_ms = int(tokens[tokens.index("winc") + 1]) if "winc" in tokens else 0
                binc_ms = int(tokens[tokens.index("binc") + 1]) if "binc" in tokens else 0
                movestogo = int(tokens[tokens.index("movestogo") + 1]) if "movestogo" in tokens else None

                side_to_move = board_state[2][0]
                time_config = calculate_search_time(wtime_ms, btime_ms, winc_ms, binc_ms, movestogo, side_to_move)

            # --- Opening Book Logic ---
            book_moves = []
            zobrist_key = board_state[2][4]
            if board_state[2][3] < 20:
                book_moves = opening_book.lookup(zobrist_key, board_state[0], board_state[1], board_state[2])

            if book_moves:
                moves, weights = zip(*book_moves)
                selected_move = random.choices(moves, weights=weights, k=1)[0]
                log_info("Playing from book")
                print(f"bestmove {move_to_uci(selected_move)}")
                continue

            # --- Prepare Search Context ---
            # Create a new context for this search
            global_search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)
            
            # Start Search Thread
            def search_worker():
                best_move, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = iterative_deepening_search(
                    board_state[0], board_state[1], board_state[2], max_depth, time_config, global_search_context
                )
                print(f"bestmove {move_to_uci(best_move)}")

            search_thread = threading.Thread(target=search_worker)
            search_thread.start()

        elif command == "stop":
            if global_search_context is not None:
                global_search_context.stop_flag[0] = True
                # The search thread will see this flag and exit, printing bestmove
        
        elif command == "quit":
            if global_search_context is not None:
                global_search_context.stop_flag[0] = True
            if search_thread is not None and search_thread.is_alive():
                search_thread.join()
            break

if __name__ == "__main__":
    uci_loop()
