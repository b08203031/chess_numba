"""
從 PGN 文件生成訓練數據集的腳本。(已整合 Stockfish 數據清洗功能)

這個腳本的目的是為 `tuner.py` 調優腳本創建一個數據集，此版本專為
Texel's Tuning Method 設計。它通過以下步驟工作：
1.  讀取 PGN 文件並解析對局結果。
2.  遍歷棋局中的每個局面。
3.  使用基本過濾器（回合數、子力差距、是否被將軍）篩選候選局面。
4.  **數據清洗 (核心功能)**: 使用 Stockfish 引擎作為裁判，對每個候選局面
    進行淺層分析，並剔除那些最終賽果與 Stockfish 客觀評估嚴重不符的局面。
5.  保存通過所有驗證的數據 (FEN 和對局結果) 到 .jsonl 文件。

這個經過清洗的數據集將更客觀，有助於訓練出一個評估更理性、棋力更強的引擎。
"""
import chess
import chess.pgn
import chess.engine  # 引入引擎通訊模組
import json
import sys
import os
import argparse

# --- 全局變數 ---
STOCKFISH_ENGINE = None # 用於保存 Stockfish 引擎實例

def parse_result(result_str: str) -> float:
    """
    將 PGN 的 Result 字符串轉換為數值。
    '1-0' -> 1.0 (白勝)
    '0-1' -> 0.0 (黑勝)
    '1/2-1/2' -> 0.5 (和棋)
    """
    if result_str == '1-0':
        return 1.0
    elif result_str == '0-1':
        return 0.0
    elif result_str == '1/2-1/2':
        return 0.5
    else:
        return None

def should_skip_position(board):
    """
    根據一些啟發式規則判斷是否應該跳過當前局面。
    - 跳過正在被將軍的局面，因為評分可能不穩定。
    - 跳過子力差距過大的局面。
    """
    if board.is_check():
        return True
    
    # 簡單的子力差距檢查
    material_diff = 0
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for piece_type, value in piece_values.items():
        material_diff += len(board.pieces(piece_type, chess.WHITE)) * value
        material_diff -= len(board.pieces(piece_type, chess.BLACK)) * value
    
    # 如果子力差距大於一車，就認為局面已定，跳過
    if abs(material_diff) > 5:
        return True

    return False

def is_endgame_position(board: chess.Board) -> bool:
    """
    根據子力數量判斷是否為殘局局面。
    規則：如果雙方都沒有皇后，或者盤面上非兵非王的棋子總數不多，則視為殘局。
    """
    # 如果盤面上沒有皇后，通常可以認為是殘局
    no_queens = not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK)
    if no_queens:
        return True

    # 計算雙方重子(Rook)和輕子(Knight, Bishop)的總數
    white_pieces = len(board.pieces(chess.ROOK, chess.WHITE)) * 2 + len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE))
    black_pieces = len(board.pieces(chess.ROOK, chess.BLACK)) * 2 + len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))

    # 如果雙方子力總值（簡化版）小於等於一個車+一個輕子，也視為殘局
    # Stockfish 使用的閾值是 `QUEEN_PHASE = 4`, `ROOK_PHASE = 2`, `MINOR_PHASE = 1`
    # 這裡我們用一個簡化的啟發式規則
    if (white_pieces + black_pieces) <= 7:
        return True

    return False

# --- 新增: Stockfish 相關函式 ---

def setup_stockfish_engine(path: str):
    """
    初始化並返回一個 Stockfish 引擎實例。
    """
    global STOCKFISH_ENGINE
    try:
        STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(path)
        print(f"Stockfish engine initialized successfully from: {path}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Stockfish executable not found at the specified path: {path}", file=sys.stderr)
        print("Please check the --stockfish-path argument.", file=sys.stderr)
        sys.exit(1)

def is_evaluation_consistent(board: chess.Board, result: float, depth: int, threshold_cp: int) -> bool:
    """
    使用 Stockfish 檢查局面評估是否與最終賽果一致。
    """
    if STOCKFISH_ENGINE is None:
        raise RuntimeError("Stockfish engine is not initialized.")

    try:
        info = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white().score(mate_score=30000)

        if result == 1.0: # 白勝
            return score > -threshold_cp
        elif result == 0.0: # 黑勝
            return score < threshold_cp
        elif result == 0.5: # 和棋
            return abs(score) <= threshold_cp
            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
        print(f"\nStockfish engine error during analysis: {e}", file=sys.stderr)
        return False

    return True

# --- 修改: 主生成函式 ---

def generate_data_from_pgn(pgn_file: str, output_file: str, min_move: int, max_move: int, games_limit: int, depth: int, threshold: int):
    """
    從 PGN 文件解析棋局，使用 Stockfish 清洗數據，並將結果保存到 JSON Lines 文件。
    """
    positions_count = 0
    games_processed = 0
    positions_kept = 0
    positions_discarded = 0

    try:
        with open(pgn_file, encoding='utf-8') as pgn:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                    if games_limit and games_processed >= games_limit:
                        print(f"\nReached game limit of {games_limit}.", file=sys.stderr)
                        break
                    
                    result = parse_result(game.headers.get("Result", "*"))
                    if result is None: continue

                    games_processed += 1
                    board = game.board()
                    
                    for i, move in enumerate(game.mainline_moves()):
                        board.push(move)
                        
                        if min_move <= board.fullmove_number <= max_move:
                            positions_count += 1
                            # 應用基礎過濾器
                            if not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material() and not should_skip_position(board) and not is_endgame_position(board):
                                
                                # --- 核心整合: 應用 Stockfish 過濾器 ---
                                if is_evaluation_consistent(board, result, depth, threshold):
                                    fen = board.fen()
                                    data = {"fen": fen, "result": result}
                                    f_out.write(json.dumps(data) + '\n')
                                    positions_kept += 1
                                else:
                                    positions_discarded += 1 # 評估與賽果不符，丟棄
                    
                    if games_processed % 10 == 0:
                         print(f"\rGames: {games_processed} | Total Pos: {positions_count} | Kept: {positions_kept} | Discarded: {positions_discarded}", end="", file=sys.stderr)

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    finally:
        # 確保在程式結束時關閉引擎
        if STOCKFISH_ENGINE:
            STOCKFISH_ENGINE.quit()
            print("\nStockfish engine shut down.", file=sys.stderr)

    print(f"\n--- Data Generation Complete ---", file=sys.stderr)
    print(f"Processed {games_processed} games.", file=sys.stderr)
    print(f"Total positions considered: {positions_count}", file=sys.stderr)
    print(f"Positions kept (consistent): {positions_kept}", file=sys.stderr)
    print(f"Positions discarded (inconsistent): {positions_discarded}", file=sys.stderr)
    print(f"Cleaned data saved to '{output_file}'.", file=sys.stderr)


if __name__ == '__main__':
    # --- 修改: 修正 Stockfish 路徑，並新增相關參數 ---
    default_stockfish_path = r"C:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba\stockfish\stockfish-windows-x86-64-avx2.exe"

    parser = argparse.ArgumentParser(
        description="Generate cleaned chess training data from PGN files using Stockfish as a referee.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pgn", type=str, default="twic1613.pgn", help="Input PGN file with master games.")
    parser.add_argument("--output", type=str, default="training_data_texel_middlegame_cleaned.jsonl", help="Output file in JSON Lines format.")
    parser.add_argument("--min-move", type=int, default=15, help="The first full move number from which to start collecting positions.")
    parser.add_argument("--max-move", type=int, default=40, help="The last full move number at which to stop collecting positions.")
    parser.add_argument("--games", type=int, default=1000, help="Limit the number of games to process from the PGN file (0 for all).")
    
    # --- 新增: 用於數據清洗的命令行參數 ---
    parser.add_argument("--stockfish-path", type=str, default=default_stockfish_path, help="Path to the Stockfish executable.")
    parser.add_argument("--stockfish-depth", type=int, default=10, help="The shallow analysis depth for Stockfish to use for filtering.")
    parser.add_argument("--consistency-threshold", type=int, default=50, help="Evaluation threshold in centipawns to determine if a position is inconsistent with the game result.")

    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"Error: PGN file not found at '{args.pgn}'", file=sys.stderr)
        sys.exit(1)

    # 啟動引擎
    setup_stockfish_engine(args.stockfish_path)
    
    # 執行數據生成與清洗
    generate_data_from_pgn(
        pgn_file=args.pgn,
        output_file=args.output,
        min_move=args.min_move,
        max_move=args.max_move,
        games_limit=args.games if args.games > 0 else None,
        depth=args.stockfish_depth,
        threshold=args.consistency_threshold
    )