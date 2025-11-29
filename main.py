
import sys
import os
import numpy as np
import random
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
# 殺手步等陣列的最大搜尋深度（層數）
MAX_PLY = 128 # Updated to match constant.py and search.py consistency

# Global variable to hold the current search thread and context
# 用於保存當前搜尋線程和上下文的全局變量
search_thread = None
global_search_context = None

def run_search(board_state, max_depth, time_config, transposition_table, killer_moves, history_table, pv_table):
    """
    Wrapper function to run the search in a separate thread.
    在單獨線程中運行搜尋的包裝函數。（目前未使用，邏輯已整合到 uci_loop 中）
    """
    global global_search_context
    pass

def uci_loop():
    """
    The main UCI loop of the engine.
    Listens for and responds to UCI commands from a GUI.
    
    引擎的主 UCI 循環。
    監聽並回應來自 GUI 的 UCI 命令。
    """
    global search_thread, global_search_context

    # Initialize engine components before the loop starts / 在循環開始前初始化引擎組件
    transposition_table = create_transposition_table(TT_SIZE_MB)
    
    # We need persistent tables for the search context / 我們需要搜尋上下文的持久化表
    # Update killer_moves to be 1D array matching SearchContext definition
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    
    # Track game history for repetition detection
    # List of Zobrist keys since the last irreversible move (or start of game)
    game_history = [] 

    # --- Initialize Opening Book / 初始化開局書 ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Updated path to look in chess_engine subdirectory
    book_path = os.path.join(script_dir, "chess_engine", "polyglot.bin")
    opening_book = OpeningBook(book_path)
    if opening_book.book is None:
        # Fallback to root if not found in subdirectory
        book_path_root = os.path.join(script_dir, "polyglot.bin")
        opening_book = OpeningBook(book_path_root)
        
    if opening_book.book is None:
        log_info("Opening book not found or failed to load.")
    else:
        log_info(f"Opening book loaded from {book_path}")

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
            game_history = []
        elif command == "position":
            game_history = [] # Reset history for new position setup
            
            # --- Parse position command / 解析 position 命令 ---
            if "startpos" in tokens:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                board_state = (piece_bbs, occupancy_bbs, game_state)
            elif "fen" in tokens:
                fen_start_index = tokens.index("fen") + 1
                fen = " ".join(tokens[fen_start_index:])
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                board_state = (piece_bbs, occupancy_bbs, game_state)
            
            # Initial Zobrist key
            game_history.append(board_state[2][4])

            # Handle 'moves' / 處理 'moves'
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
                            # make_move 會就地修改，但返回 info。
                            make_move(p_bbs, o_bbs, g_state, legal_move)
                            
                            # Check for irreversible move to reset history (pawn move or capture)
                            # Actually, simplest is to append to history. 
                            # If halfmove clock resets (handled in make_move), the search uses that to limit lookback.
                            game_history.append(g_state[4])
                            
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

            # Stop any existing search / 停止任何現有的搜尋
            if global_search_context is not None:
                global_search_context.stop_flag[0] = True
            if search_thread is not None and search_thread.is_alive():
                search_thread.join()

            # --- Parse go command / 解析 go 命令 ---
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

            # --- Opening Book Logic / 開局書邏輯 ---
            book_moves = []
            zobrist_key = board_state[2][4]
            # Only use book if halfmove clock is low (early game)
            if board_state[2][3] < 20:
                book_moves = opening_book.lookup(zobrist_key, board_state)

            if book_moves:
                moves, weights = zip(*book_moves)
                selected_move = random.choices(moves, weights=weights, k=1)[0]
                log_info("Playing from book")
                print(f"bestmove {move_to_uci(selected_move)}")
                continue

            # --- Prepare Search Context / 準備搜尋上下文 ---
            # Create a new context for this search / 為此搜尋創建新的上下文
            global_search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)
            
            # Populate Repetition Table from Game History
            # We copy valid keys from history to the context's fixed-size array
            hist_len = len(game_history)
            if hist_len > 0:
                # Be careful not to overflow
                safe_len = min(hist_len, len(global_search_context.repetition_table))
                # Copy the last 'safe_len' moves
                start_idx = hist_len - safe_len
                for i in range(safe_len):
                    global_search_context.repetition_table[i] = game_history[start_idx + i]
                global_search_context.repetition_index = np.uint16(safe_len)

            # Start Search Thread / 啟動搜尋線程
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
                # 搜尋線程將看到此標誌並退出，打印 bestmove
        
        elif command == "quit":
            if global_search_context is not None:
                global_search_context.stop_flag[0] = True
            if search_thread is not None and search_thread.is_alive():
                search_thread.join()
            break

if __name__ == "__main__":
    uci_loop()
