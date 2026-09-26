"""
從 PGN 文件生成訓練數據集的腳本。(已整合 Stockfish 數據清洗功能，支援多進程並行處理)

這個腳本的目的是為 `tuner.py` 調優腳本創建一個數據集，此版本專為
Texel's Tuning Method 設計。它通過以下步驟工作：
1.  讀取 PGN 文件並解析對局結果。
2.  將對局分配給多個工作進程 (Worker Process)。
3.  每個 Worker 使用獨立的 Stockfish 引擎實例進行並行分析與清洗。
4.  主進程收集結果並寫入 JSON Lines 文件。

數據清洗邏輯 (Continuous Sigmoid Labeling, NNUE 版本):
使用 Stockfish 的 centipawn 分數轉換為 Sigmoid 值作為連續標籤：
- Sigmoid(score) = 1 / (1 + 10^(-score/400))
- [NNUE 注意] max_cp 預設設為 1000cp（幾乎不過濾非 mate 局面）
  原因：NNUE 標籤來自 SF 深搜分數，對任何 cp 值的局面都是準確的，
  不像 Texel Tuning 需要靠對局結果反推，懸殊局面也有訓練價值。
  過度過濾會造成分佈偏差，讓模型對明確優/劣勢局面評估不準。
- 過濾掉雙方 ELO 低於門檻的對局（預設 2400）
- 跳過前 15 步開局（避開開局書）
- 使用 SF 深度 1（靜態評估）與深度 N 的分數差距超過閾值（100cp）的局面
  視為戰術複雜局面，予以捨棄（避免不穩定標籤污染訓練集）
- 過濾將軍局面、吃子/升變後局面
- Mate 序列局面直接捨棄
"""
import chess
import chess.pgn
import chess.engine
import json
import sys
import os
import argparse
import shutil
import math
import io
import concurrent.futures
import multiprocessing

# --- 全局變數 ---
# 這些變數現在只在 worker 進程中被使用
STOCKFISH_ENGINE = None

def parse_result(result_str: str) -> float:
    """解析 PGN 結果字串。保留以防需要過濾未完成的對局。"""
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
    判斷是否應該跳過此局面。
    僅檢查結構性問題（局面已進入對訓練無用的狀態），
    不過濾材質失衡（NNUE 需要學習各種材質組合下的評估）。
    """
    # 將軍局面：高度強制性，不代表靜態局面
    if board.is_check():
        return True

    return False

def is_endgame_position(board: chess.Board) -> bool:
    no_queens = not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK)
    if no_queens:
        return True

    white_pieces = len(board.pieces(chess.ROOK, chess.WHITE)) * 2 + len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE))
    black_pieces = len(board.pieces(chess.ROOK, chess.BLACK)) * 2 + len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))

    if (white_pieces + black_pieces) <= 7:
        return True

    return False

def setup_stockfish_engine(path: str):
    global STOCKFISH_ENGINE
    if path == 'stockfish' or path == 'stockfish.exe':
        detected = shutil.which('stockfish')
        if detected:
            path = detected
            
    try:
        STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    except FileNotFoundError:
        print(f"Error: Stockfish executable not found at the specified path: {path}", file=sys.stderr)
        sys.exit(1)

def get_stockfish_sigmoid_label(board: chess.Board, depth: int,
                                quiescence_threshold: int = 100) -> tuple:
    """
    使用 Stockfish 分析局面並返回連續 Sigmoid 標籤。

    同時以深度 1（靜態評估）和指定深度進行分析：
    - 若深度 1 與深度 `depth` 的分數差超過 `quiescence_threshold` (預設 100cp)，
      視為戰術複雜局面，予以捨棄。
    - 若偵測到 mate 序列，直接捨棄。
    - 標籤來自 SF 深搜分數，對任何 cp 大小的局面都是準確的，不過濾。
    
    Returns:
        (sigmoid, reject_reason) tuple：
          - sigmoid: 介於 0.0~1.0 之間的連續值，或 None 表示捨棄
          - reject_reason: 'mate' | 'quiescence' | 'engine_error' | None
    """
    if STOCKFISH_ENGINE is None:
        raise RuntimeError("Stockfish engine is not initialized.")

    try:
        # --- 深度 1：靜態評估（用於戰術複雜度過濾）---
        info_d1 = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=1))
        score_d1_obj = info_d1["score"].white()
        if score_d1_obj.is_mate():
            return None, 'mate'
        score_d1 = score_d1_obj.score()

        # --- 深度 N：主要評估標籤來源 ---
        info_dn = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=depth))
        score_dn_obj = info_dn["score"].white()
        if score_dn_obj.is_mate():
            return None, 'mate'
        score_dn = score_dn_obj.score()

        # 過濾戰術複雜局面：靜態評估 vs 深搜評估差距超過閾值
        if abs(score_dn - score_d1) > quiescence_threshold:
            return None, 'quiescence'

        # 轉換為 Sigmoid: 1 / (1 + 10^(-score/400))
        sigmoid = 1.0 / (1.0 + math.pow(10.0, -score_dn / 400.0))
        return sigmoid, None

    except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
        return None, 'engine_error'

def check_elo(headers, min_elo):
    """
    檢查雙方 ELO 是否都高於門檻。
    如果 PGN 沒有 ELO 資訊，則放行（回傳 True）。
    
    Args:
        headers: PGN game headers dict
        min_elo: 最低 ELO 門檻（0 表示不篩選）
    
    Returns:
        True if both players meet the ELO threshold
    """
    if min_elo <= 0:
        return True
    
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except (ValueError, TypeError):
        # 無法解析 ELO，放行
        return True
    
    # 如果兩方都是 0（無 ELO 資訊），放行
    if white_elo == 0 and black_elo == 0:
        return True
    
    return white_elo >= min_elo and black_elo >= min_elo

# --- Worker Function for Parallel Processing ---

def process_game_batch(pgn_strings, stockfish_path, depth, min_move, max_move,
                       quiescence_threshold: int = 100, min_elo: int = 2400,
                       label_mode: str = "game"):
    """
    Worker function to process a batch of raw PGN game strings.

    label_mode:
      - ``game``: classic Texel labels {0, 0.5, 1} from PGN result (recommended for HCE Elo).
      - ``sf``:   continuous sigmoid from Stockfish search score (distillation).
    """
    need_sf = label_mode == "sf"
    if need_sf:
        setup_stockfish_engine(stockfish_path)

    mg_results = []
    eg_results = []
    discarded = 0
    filtered_by_quiescence = 0
    filtered_by_mate = 0
    skipped_non_quiet = 0

    try:
        for pgn_string in pgn_strings:
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            if not game:
                continue

            if not check_elo(game.headers, min_elo):
                continue

            game_label = parse_result(game.headers.get("Result", "*"))
            if label_mode == "game" and game_label is None:
                discarded += 1
                continue

            board = game.board()

            for i, move in enumerate(game.mainline_moves()):
                is_capture_or_promotion = board.is_capture(move) or move.promotion
                board.push(move)

                if is_capture_or_promotion:
                    skipped_non_quiet += 1
                    continue

                if min_move <= board.fullmove_number <= max_move:
                    if (not board.is_checkmate() and not board.is_stalemate()
                            and not board.is_insufficient_material()
                            and not should_skip_position(board)):

                        if label_mode == "game":
                            label = game_label
                            reject_reason = None
                        else:
                            label, reject_reason = get_stockfish_sigmoid_label(
                                board, depth, quiescence_threshold
                            )

                        if label is not None:
                            fen = board.fen()
                            data = json.dumps({"fen": fen, "result": round(float(label), 6)})
                            if is_endgame_position(board):
                                eg_results.append(data)
                            else:
                                mg_results.append(data)
                        else:
                            if reject_reason == "quiescence":
                                filtered_by_quiescence += 1
                            elif reject_reason == "mate":
                                filtered_by_mate += 1
                            else:
                                discarded += 1
                    else:
                        discarded += 1
    finally:
        if STOCKFISH_ENGINE:
            STOCKFISH_ENGINE.quit()

    return mg_results, eg_results, discarded, filtered_by_quiescence, filtered_by_mate, skipped_non_quiet

def pgn_batch_generator(pgn_path, batch_size=50):
    """
    高效的 PGN 分塊生成器。不解析內容，僅透過掃描 [Event 標籤切割原始字串。
    """
    current_batch = []
    current_game = []
    
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('[Event ') and current_game:
                # 收集完一局
                current_batch.append("".join(current_game))
                current_game = []
                
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
            
            current_game.append(line)
            
        if current_game:
            current_batch.append("".join(current_game))
        if current_batch:
            yield current_batch

def generate_data_parallel(pgn_file, output_mg, output_eg, min_move, max_move, games_limit, depth, min_elo,
                           stockfish_path, num_workers=None, quiescence_threshold: int = 100,
                           label_mode: str = "game"):
    """
    Main driver for parallel generation.
    label_mode: ``game`` (Texel results) or ``sf`` (Stockfish sigmoid distillation).
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Starting parallel generation with {num_workers} workers...", file=sys.stderr)
    print(f"Settings: label_mode={label_mode}, min_move={min_move}, max_move={max_move}, "
          f"min_elo={min_elo}, depth={depth}, quiescence_threshold={quiescence_threshold}cp",
          file=sys.stderr)
    if label_mode == "game":
        print("Label: PGN game result {0,0.5,1}; skip checks/captures/promotions.", file=sys.stderr)
    else:
        print(
            f"Label: SF sigmoid; discard if depth1 vs depth{depth} diff > {quiescence_threshold}cp or mate.",
            file=sys.stderr,
        )
    
    games_processed_estimate = 0
    mg_total = 0
    eg_total = 0
    discarded_total = 0
    filtered_by_quiescence_total = 0
    filtered_by_mate_total = 0
    skipped_non_quiet_total = 0
    
    futures = set()
    
    with open(output_mg, 'w', encoding='utf-8') as f_mg, \
         open(output_eg, 'w', encoding='utf-8') as f_eg, \
         concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        # 使用快速分塊生成器
        batch_gen = pgn_batch_generator(pgn_file, batch_size=50)
        
        for batch in batch_gen:
            # 分發任務
            future = executor.submit(
                process_game_batch, batch, stockfish_path, depth,
                min_move, max_move, quiescence_threshold, min_elo, label_mode
            )
            futures.add(future)
            games_processed_estimate += len(batch)
            
            # --- 即時監控已完成的任務 ---
            # 檢查是否有完成的任務，有的話就報進度，不阻塞讀取
            done, _ = concurrent.futures.wait(futures, timeout=0.001, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for f in done:
                futures.remove(f)
                try:
                    mg_res, eg_res, discarded, filtered_by_quiescence, filtered_by_mate, skipped_non_quiet = f.result()
                    
                    for line in mg_res: f_mg.write(line + '\n')
                    for line in eg_res: f_eg.write(line + '\n')
                    
                    mg_total += len(mg_res)
                    eg_total += len(eg_res)
                    discarded_total += discarded
                    filtered_by_quiescence_total += filtered_by_quiescence
                    filtered_by_mate_total += filtered_by_mate
                    skipped_non_quiet_total += skipped_non_quiet
                    
                    print(
                        f"\rProcessing... (~{games_processed_estimate} games scanned) | "
                        f"Kept: {mg_total + eg_total} (MG: {mg_total} | EG: {eg_total}) "
                        f"| Filtered: {filtered_by_quiescence_total + filtered_by_mate_total} "
                        f"| Discarded: {discarded_total}",
                        end="", file=sys.stderr
                    )
                except Exception as e:
                    print(f"\nWorker exception: {e}", file=sys.stderr)
                    
            if games_limit and games_processed_estimate >= games_limit:
                break
        
        # --- 等待剩餘任務 ---
        if futures:
            print(f"\nScanning complete. Waiting for {len(futures)} remaining workers...", file=sys.stderr)
            for f in concurrent.futures.as_completed(futures):
                try:
                    mg_res, eg_res, discarded, filtered_by_quiescence, filtered_by_mate, skipped_non_quiet = f.result()
                    for line in mg_res: f_mg.write(line + '\n')
                    for line in eg_res: f_eg.write(line + '\n')
                    mg_total += len(mg_res)
                    eg_total += len(eg_res)
                    discarded_total += discarded
                    filtered_by_quiescence_total += filtered_by_quiescence
                    filtered_by_mate_total += filtered_by_mate
                    skipped_non_quiet_total += skipped_non_quiet
                    
                    print(
                        f"\rFinalizing... | Kept: {mg_total + eg_total} (MG: {mg_total} | EG: {eg_total})",
                        end="", file=sys.stderr
                    )
                except Exception as e:
                    print(f"\nWorker exception: {e}", file=sys.stderr)

    print(f"\n--- Data Generation Complete ---", file=sys.stderr)
    print(f"Total Games Scanned: ~{games_processed_estimate}", file=sys.stderr)
    print(f"Middlegame positions saved: {mg_total}", file=sys.stderr)
    print(f"Endgame positions saved: {eg_total}", file=sys.stderr)
    print(f"Positions filtered by quiescence: {filtered_by_quiescence_total}", file=sys.stderr)
    print(f"Positions filtered by mate: {filtered_by_mate_total}", file=sys.stderr)
    print(f"Positions discarded (check/mate/error): {discarded_total}", file=sys.stderr)
    print(f"Positions skipped (non-quiet): {skipped_non_quiet_total}", file=sys.stderr)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    default_stockfish = shutil.which('stockfish')
    if not default_stockfish:
        if os.path.exists("./external/stockfish_dir/stockfish-windows-x86-64-avx2.exe"):
             default_stockfish = "./external/stockfish_dir/stockfish-windows-x86-64-avx2.exe"
        elif os.path.exists("./stockfish/stockfish-windows-x86-64-avx2.exe"):
             default_stockfish = "./stockfish/stockfish-windows-x86-64-avx2.exe"
        else:
             default_stockfish = "stockfish"

    default_pgn = "tuner/data/pgn/twic1613.pgn"
    if not os.path.exists(default_pgn) and os.path.exists(os.path.join("tuner", "twic1613.pgn")):
        default_pgn = os.path.join("tuner", "twic1613.pgn")

    parser = argparse.ArgumentParser(
        description="Generate HCE tuning labels from PGN (game-result Texel or SF sigmoid).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pgn", type=str, default=default_pgn, help="Input PGN file.")
    parser.add_argument("--output-mg", type=str, default="tuner/data/jsonl/training_data_middlegame_cleaned.jsonl", help="Output MG file.")
    parser.add_argument("--output-eg", type=str, default="tuner/data/jsonl/training_data_endgame_cleaned.jsonl", help="Output EG file.")
    parser.add_argument("--min-move", type=int, default=15, help="Start move number (skip opening). Default 15 = skip first 14 moves.")
    parser.add_argument("--max-move", type=int, default=200, help="End move number.")
    parser.add_argument("--games", type=int, default=0, help="Games limit (0 for all).")
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=("game", "sf"),
        default="game",
        help="game = Texel {0,0.5,1} from PGN result (recommended for HCE Elo); "
             "sf = continuous sigmoid from Stockfish (distillation).",
    )
    parser.add_argument("--stockfish-path", type=str, default=default_stockfish, help="Stockfish executable path (required for --label-mode sf).")
    parser.add_argument("--stockfish-depth", type=int, default=10, help="Stockfish analysis depth for --label-mode sf.")
    parser.add_argument("--quiescence-threshold", type=int, default=100,
                        help="(sf mode) Max cp gap between depth-1 and depth-N; discard above this.")
    parser.add_argument("--min-elo", type=int, default=2400, help="Minimum ELO for both players (0 to disable).")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count - 1).")

    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"Error: PGN file not found at '{args.pgn}'", file=sys.stderr)
        sys.exit(1)

    if args.label_mode == "sf":
        if not shutil.which(args.stockfish_path) and not os.path.exists(args.stockfish_path):
            print(f"Warning: Stockfish path '{args.stockfish_path}' might be invalid.", file=sys.stderr)

    # Ensure output directories exist
    for path in (args.output_mg, args.output_eg):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    generate_data_parallel(
        pgn_file=args.pgn,
        output_mg=args.output_mg,
        output_eg=args.output_eg,
        min_move=args.min_move,
        max_move=args.max_move,
        games_limit=args.games if args.games > 0 else None,
        depth=args.stockfish_depth,
        min_elo=args.min_elo,
        stockfish_path=args.stockfish_path,
        num_workers=args.workers,
        quiescence_threshold=args.quiescence_threshold,
        label_mode=args.label_mode,
    )
