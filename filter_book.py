import os
import sys
import time
import numpy as np
import chess
import chess.polyglot

class SuppressOutput:
    def __init__(self):
        self._original_stdout = sys.stdout
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, call_val, tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# 將 chess_engine_v2 加入路徑以匯入核心模組
sys.path.append(os.path.join(os.path.dirname(__file__), "chess_engine_v2"))

from chess_engine.opening_book import POLYGLOT_DTYPE
from chess_engine.fen_parser import parse_fen
from chess_engine.search import iterative_deepening_search
from chess_engine.time_manager import calculate_search_time
from chess_engine.transposition_table import create_transposition_table, clear_transposition_table
from chess_engine.engine_types import SearchContext
from chess_engine.move import encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION, PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN, get_from_square, get_to_square, get_special_move_flag

# --- 腳本設定參數 ---
# 推薦的深度：
# Depth 10: 速度與品質的最佳平衡。單步評估通常在幾十毫秒內，適合全庫掃描。
# Depth 12: 品質更高，但花費時間會呈指數性增加，適合精簡後的小型核心庫。
# Depth 8: 非常快，但只適合過濾極其明顯的「大漏著」，無法評估長遠陣型。
SEARCH_DEPTH = 12
MAX_CANDIDATES = 2
BOOK_PATH = "chess_engine_v2/chess_engine/polyglot.bin"
OUTPUT_PATH = "polyglot_filtered.bin"
MAX_POSITIONS_TO_EVALUATE = 1000000  # 防止 BFS 爆表

def convert_chess_move_to_engine(chess_move, board):
    """將 python-chess 的 Move 轉換為引擎內部的格式"""
    from_sq = chess_move.from_square
    to_sq = chess_move.to_square
    special_flag = SPECIAL_MOVE_FLAG_NORMAL
    promo_piece = 0
    
    if chess_move.promotion:
        special_flag = SPECIAL_MOVE_FLAG_PROMOTION
        if chess_move.promotion == chess.KNIGHT: promo_piece = PROMO_KNIGHT
        elif chess_move.promotion == chess.BISHOP: promo_piece = PROMO_BISHOP
        elif chess_move.promotion == chess.ROOK: promo_piece = PROMO_ROOK
        elif chess_move.promotion == chess.QUEEN: promo_piece = PROMO_QUEEN

    # 封裝為引擎內部的 16-bit
    return encode_move(from_sq, to_sq, promo_piece, special_flag)

def convert_engine_to_polyglot(engine_move):
    """將引擎的 16-bit move 轉換為 Polyglot 格式"""
    from_sq = get_from_square(engine_move)
    to_sq = get_to_square(engine_move)
    special_flag = get_special_move_flag(engine_move)
    
    from_file = from_sq % 8
    from_rank = from_sq // 8
    to_file = to_sq % 8
    to_rank = to_sq // 8
    
    promo_poly = 0
    if special_flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_piece = (engine_move >> 12) & 0b111
        if promo_piece == PROMO_KNIGHT: promo_poly = 1
        elif promo_piece == PROMO_BISHOP: promo_poly = 2
        elif promo_piece == PROMO_ROOK: promo_poly = 3
        elif promo_piece == PROMO_QUEEN: promo_poly = 4
        
    poly_move = (promo_poly << 12) | (from_rank << 9) | (from_file << 6) | (to_rank << 3) | to_file
    return np.uint16(poly_move)

def evaluate_candidate_move(board, chess_move, search_context, tt, max_depth=10):
    """在給定的盤面上走這一步棋，然後用引擎評估接下來的盤面分數"""
    # 建立一個測試用的棋盤並走子
    test_board = board.copy()
    test_board.push(chess_move)
    
    # 解析新盤面供引擎使用
    fen = test_board.fen()
    try:
        p_bbs, o_bbs, g_state = parse_fen(fen)
    except Exception:
        return -99999
    
    # 設定時間 (實際上我們靠深度限制，但 time_config 大給免得超時)
    time_config = {'optimum_time': 9999999, 'maximum_time': 9999999}
    
    # 重置 TT 以確保沒有歷史干擾 (也可以不重置以利用共享資訊)
    # 這裡我們不重置，以便加速
    global_tt_generation = 1
    recent_history = [g_state[4]]
    
    # 如果走完這步就將死了
    if test_board.is_checkmate():
        return 99000  # 第一步走完對手就死了，代表這步極好(對方視角被將死)

    with SuppressOutput():
        _, score, _, _, _, _ = iterative_deepening_search(
            p_bbs, o_bbs, g_state, max_depth, time_config, search_context,
            game_history_list=recent_history, tt_generation=global_tt_generation
        )
    
    # 引擎搜尋出的 score 是「下一步行動方 (對手)」的優勢分數。
    # 由於我們想評估「我們走的這步」好不好，這步的分數 = -對手的分數
    return -score

def main():
    print(f"Loading Opening Book from: {BOOK_PATH}")
    if not os.path.exists(BOOK_PATH):
        print(f"Error: Could not find {BOOK_PATH}")
        return

    # Load original book
    try:
        raw_book = np.fromfile(BOOK_PATH, dtype=POLYGLOT_DTYPE)
        print(f"Loaded {len(raw_book)} entries from book.")
    except Exception as e:
        print(f"Error loading book: {e}")
        return
        
    # Group by key for O(1) matching
    book_dict = {}
    for entry in raw_book:
        k = int(entry['key'])
        if k not in book_dict:
            book_dict[k] = []
        book_dict[k].append(entry)

    print(f"Total Unique Positions in Book: {len(book_dict)}")
    
    filtered_entries = []
    
    # 準備搜尋模組需要的歷史與陣列
    tt = create_transposition_table(64) # 64MB hash
    killer_moves = np.zeros(256, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((3, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 12), dtype=np.int32)
    p_corr = np.zeros(16384, dtype=np.int16)
    m_corr = np.zeros(16384, dtype=np.int16)
    np_corr_w = np.zeros(16384, dtype=np.int16)
    np_corr_b = np.zeros(16384, dtype=np.int16)
    pv_table = np.zeros((128, 128), dtype=np.uint16)
    
    # Start BFS exploration
    queue = [(chess.Board(), [])]
    visited_hashes = set()
    evaluated_count = 0
    start_time = time.time()
    
    print(f"\n[ 引擎固定搜尋深度設定為: {SEARCH_DEPTH} ]")
    print("開始透過 BFS 模擬找尋與評估開局庫中的盤面 (這可能需要數小時)...\n")
    
    while queue and evaluated_count < MAX_POSITIONS_TO_EVALUATE:
        board, move_hist = queue.pop(0)
        zkey = chess.polyglot.zobrist_hash(board)
        
        if zkey in visited_hashes:
            continue
        visited_hashes.add(zkey)
        
        # Check if the book has this position
        if zkey in book_dict:
            entries = book_dict[zkey]
            candidate_moves = []
            
            # 建立搜尋上下文
            search_context = SearchContext(
                tt, killer_moves, pv_table, history_table,
                butterfly_history, continuation_history, capture_history,
                p_corr, m_corr, np_corr_w, np_corr_b
            )
            search_context.stop_flag[0] = False
            search_context.end_time = 0.0
            
            # 對此盤面的每個推薦步進行評估
            for entry in entries:
                poly_move = int(entry['move'])
                
                # 重建 UCI / python-chess 的 move:
                # polyglot: promo(3) | from_rank(3) | from_file(3) | to_rank(3) | to_file(3)
                to_file = poly_move & 7
                to_rank = (poly_move >> 3) & 7
                from_file = (poly_move >> 6) & 7
                from_rank = (poly_move >> 9) & 7
                promo = (poly_move >> 12) & 7
                
                from_sq = from_rank * 8 + from_file
                to_sq = to_rank * 8 + to_file
                
                promo_char = ''
                if promo == 1: promo_char = 'n'
                elif promo == 2: promo_char = 'b'
                elif promo == 3: promo_char = 'r'
                elif promo == 4: promo_char = 'q'
                
                move_uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq] + promo_char
                
                # 處理 Polyglot 的王車易位特例 (King captures Rook)
                if board.piece_type_at(from_sq) == chess.KING:
                    if from_sq == chess.E1:
                        if to_sq == chess.H1: move_uci = 'e1g1'
                        elif to_sq == chess.A1: move_uci = 'e1c1'
                    elif from_sq == chess.E8:
                        if to_sq == chess.H8: move_uci = 'e8g8'
                        elif to_sq == chess.A8: move_uci = 'e8c8'
                        
                try:
                    chess_move = chess.Move.from_uci(move_uci)
                except ValueError:
                    continue
                
                # 若該步不合法，跳過
                if chess_move not in board.legal_moves:
                    continue
                
                # 取得引擎評估分數
                score = evaluate_candidate_move(board, chess_move, search_context, tt, max_depth=SEARCH_DEPTH)
                candidate_moves.append((score, entry, chess_move))
            
            # 根據引擎評分降冪排序
            candidate_moves.sort(key=lambda x: x[0], reverse=True)
            
            # 只保留前 N 步
            kept_moves = candidate_moves[:MAX_CANDIDATES]
            
            # 格式化輸出有用資訊
            hist_str = " ".join(move_hist) if move_hist else "Startpos"
            cands_str = ", ".join([f"{board.san(m[2])} ({m[0]/100.0:+.2f})" for m in kept_moves])
            print(f"[{evaluated_count:5d} | BFS Queue: {len(queue):5d}] {hist_str}")
            print(f"  -> Retained: {cands_str}")
            
            # 將保留下來的招法加入新字典
            for item in kept_moves:
                filtered_entries.append(item[1])
                # 將產生的子盤面加入佇列，繼續往下探勘
                child_board = board.copy()
                child_board.push(item[2])
                new_hist = move_hist + [board.san(item[2])]
                queue.append((child_board, new_hist))
                
            evaluated_count += 1

    # 封裝並寫出新的 BIN 檔
    print(f"\n===== 過濾完成 =====")
    print(f"總共遍歷了 {evaluated_count} 個真實對局中可抵達的盤面。")
    print(f"原本 {len(raw_book)} 條開局分支，現已精簡為 {len(filtered_entries)} 條。")
    print(f"儲存過濾後的檔案至: {OUTPUT_PATH}")
    
    filtered_array = np.array(filtered_entries, dtype=POLYGLOT_DTYPE)
    # Book must be sorted by key!
    filtered_array.sort(order='key')
    filtered_array.tofile(OUTPUT_PATH)
    print("存檔完成！你可以將此新檔案取代原有的 polyglot.bin。")

if __name__ == "__main__":
    main()
