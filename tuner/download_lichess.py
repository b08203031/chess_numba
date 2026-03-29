import requests
import zstandard as zstd
import json
import numpy as np
import math
import time
import os

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
TARGET_SAMPLES = 10_000_000  # 1千萬筆資料 (約為 144萬 的 7 倍！)
OUTPUT_FILE = "tuner/ultimate_dataset.npz"
BATCH_LOG = 100_000

# Sigmoid常數與Stockfish相仿，將 Centipawn 轉成 [0.0, 1.0] 勝率
K = 0.00368208

# 快速字元對應表
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def cp_to_wdl(cp):
    # 限制以避免 overflow
    cp = max(-10000, min(10000, cp))
    return 1.0 / (1.0 + math.exp(-K * cp))

def parse_fen_fast(fen):
    parts = fen.split(' ')
    board_part = parts[0]
    stm_part = parts[1]
    
    bbs = np.zeros(12, dtype=np.uint64)
    ranks = board_part.split('/')
    for r_idx, rank in enumerate(ranks):
        rank_val = 7 - r_idx
        file_val = 0
        for char in rank:
            if char.isdigit():
                file_val += int(char)
            else:
                sq = rank_val * 8 + file_val
                p_idx = PIECE_MAP[char]
                bbs[p_idx] |= np.uint64(1 << sq)
                file_val += 1
                
    stm = 0 if stm_part == 'w' else 1
    return bbs, stm

def download_and_parse():
    print(f"開始從 Lichess 串流下載百萬盤面... 目標: {TARGET_SAMPLES} 筆")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 預先分配記憶體 (避免 numpy 動態 append 過慢)
    all_bbs = np.zeros((TARGET_SAMPLES, 12), dtype=np.uint64)
    all_stm = np.zeros((TARGET_SAMPLES, 1), dtype=np.uint8)
    all_results = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    
    count = 0
    start_time = time.time()
    
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    # 使用 Zstandard 即時解壓縮流
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    
    # 手動建立一個 Iterator 負責從 stream 讀行
    buffer = b""
    while count < TARGET_SAMPLES:
        chunk = stream_reader.read(65536)
        if not chunk:
            break
        buffer += chunk
        
        lines = buffer.split(b'\n')
        buffer = lines.pop() # 最後一行可能不完整，留給下一次
        
        for line in lines:
            if count >= TARGET_SAMPLES:
                break
            
            try:
                data = json.loads(line.decode('utf-8'))
                fen = data['fen']
                eval_obj = data['evals'][0]['pvs'][0]
                
                bbs, stm = parse_fen_fast(fen)
                
                # 解析分數 (Lichess DB 內建是以白方視角，train.py 會負責基於 STM 翻轉)
                if 'cp' in eval_obj:
                    cp = eval_obj['cp']
                    wdl = cp_to_wdl(cp)
                elif 'mate' in eval_obj:
                    mate = eval_obj['mate']
                    wdl = 1.0 if mate > 0 else 0.0
                else:
                    continue # 略過沒有評分的奇形怪狀
                
                all_bbs[count] = bbs
                all_stm[count, 0] = stm
                all_results[count] = wdl
                
                count += 1
                if count % BATCH_LOG == 0:
                    elapsed = time.time() - start_time
                    fps = count / elapsed
                    print(f"已處理 {count} 筆盤面... (速度: {fps:.0f} 盤面/秒)")
                    
            except Exception as e:
                pass # 忽略異常行

    print("\n✅ 資料收集完成，開始儲存壓縮檔 (需時數十秒)...")
    np.savez_compressed(
        OUTPUT_FILE,
        piece_bbs=all_bbs[:count],
        game_states=all_stm[:count], # fake game_states 只需要保留 index 0 的 stm 即可
        results=all_results[:count]
    )
    print(f"🎉 大功告成！已成功儲存 {count} 筆超級高品質神經網路資料到 {OUTPUT_FILE}！")
    print("立刻執行 `python train.py` 見證 Loss 的直譯下降吧！")

if __name__ == "__main__":
    download_and_parse()
