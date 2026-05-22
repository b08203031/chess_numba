
import sys
import os
import numpy as np
import random
import threading
import time

from chess_engine.classical.board_operations import make_move
from chess_engine.classical.core import generate_legal_moves
from chess_engine.classical.debug_utils import log_info
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move import move_to_uci
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.time_manager import calculate_search_time
from chess_engine.classical.transposition_table import (
    create_transposition_table,
    clear_transposition_table,
    TT_SIZE_MB,
)
from chess_engine.classical.opening_book import OpeningBook
from chess_engine.classical.engine_types import SearchContext

# Maximum search depth (Ply) for arrays like killer moves
# 殺手步等陣列的最大搜尋深度（層數）
MAX_PLY = 128 # Updated to match constant.py and search.py consistency

# Global variable to hold the current search thread and context
# 用於保存當前搜尋線程和上下文的全局變量
search_thread = None
global_search_context = None

# Global counter for TT generation
global_tt_generation = 0

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
    global search_thread, global_search_context, global_tt_generation

    # Initialize engine components before the loop starts / 在循環開始前初始化引擎組件
    transposition_table = create_transposition_table(TT_SIZE_MB)
    
    # We need persistent tables for the search context / 我們需要搜尋上下文的持久化表
    # Update killer_moves to be 1D array matching SearchContext definition
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((3, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 12), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16) # CORRECTION_HISTORY_SIZE
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)

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
    game_history = []  # List to track Zobrist keys for repetition detection

    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line:
            continue

        try:
            tokens = line.strip().split()
            if not tokens:
                continue

            command = tokens[0]

            if command == "uci":
                print("id name MyChessEngine")
                print("id author YourName")
                print("uciok", flush=True)
            elif command == "isready":
                print("readyok", flush=True)
            elif command == "ucinewgame":
                clear_transposition_table(transposition_table)
                killer_moves.fill(0)
                history_table.fill(0)
                butterfly_history.fill(0)
                continuation_history.fill(0)
                capture_history.fill(0)
                pawn_history.fill(-1238)
                pawn_correction_history.fill(0)
                minor_correction_history.fill(0)
                non_pawn_correction_history_white.fill(0)
                non_pawn_correction_history_black.fill(0)
                pv_table.fill(0)
                game_history = []
                global_tt_generation = 0 # Reset generation on new game
            elif command == "position":
                # --- Parse position command / 解析 position 命令 ---
                
                # Update history only if a base position is provided
                has_base = "startpos" in tokens or "fen" in tokens or "kiwipete" in tokens
                if has_base:
                    game_history = []
                
                if "startpos" in tokens:
                    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                    board_state = (piece_bbs, occupancy_bbs, game_state)
                    game_history.append(game_state[4])
                
                elif "kiwipete" in tokens:
                    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "
                    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                    board_state = (piece_bbs, occupancy_bbs, game_state)
                    game_history.append(game_state[4])
                    
                elif "fen" in tokens:
                    fen_start_index = tokens.index("fen") + 1
                    # Stop before 'moves' if present
                    if "moves" in tokens:
                        moves_idx = tokens.index("moves")
                        fen = " ".join(tokens[fen_start_index:moves_idx])
                    else:
                        fen = " ".join(tokens[fen_start_index:])
                        
                    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                    board_state = (piece_bbs, occupancy_bbs, game_state)
                    game_history.append(game_state[4])
                
                # Handle 'moves' / 處理 'moves'
                if "moves" in tokens:
                    if board_state is None:
                        log_info("Moves received without board state. Ignoring.")
                        continue

                    moves_start_index = tokens.index("moves") + 1
                    p_bbs, o_bbs, g_state = board_state
                    
                    for move_uci in tokens[moves_start_index:]:
                        legal_moves = generate_legal_moves(p_bbs, o_bbs, g_state)
                        found_move = False
                        for legal_move in legal_moves:
                            if move_to_uci(legal_move) == move_uci:
                                make_move(p_bbs, o_bbs, g_state, legal_move)
                                # Only add to history if not already there (to avoid duplicates if moves are sent incrementally)
                                # But UCI usually sends full list, so we trust it.
                                # Actually, if we didn't reset history, we should only append NEW moves.
                                # But the logic above resets history if 'startpos/fen' is present, which is the standard case.
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
                    print(f"bestmove {move_to_uci(selected_move)}", flush=True)
                    continue

                # --- Prepare Search Context / 準備搜尋上下文 ---
                # Create a new context for this search / 為此搜尋創建新的上下文
                global_search_context = SearchContext(
                    transposition_table, killer_moves, pv_table, history_table,
                    butterfly_history, continuation_history, capture_history, pawn_history,
                    pawn_correction_history, minor_correction_history,
                    non_pawn_correction_history_white, non_pawn_correction_history_black
                )
                
                # Update TT Generation
                global_tt_generation = (global_tt_generation + 1) % 256
                
                # Copy history list to avoid thread safety issues if main thread modifies it (though in UCI it waits)
                current_game_history = list(game_history)

                # Start Search Thread / 啟動搜尋線程
                def search_worker():
                    best_move, _, _, _, _, _ = iterative_deepening_search(
                        board_state[0], board_state[1], board_state[2], max_depth, time_config, global_search_context,
                        game_history_list=current_game_history, tt_generation=global_tt_generation
                    )
                    print(f"bestmove {move_to_uci(best_move)}", flush=True)

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

        except Exception as e:
            log_info(f"Error processing command '{line.strip()}': {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    uci_loop()
