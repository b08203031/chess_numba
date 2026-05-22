"""
原料檢測站：preview_bullet.py
======================
這個腳本用來「檢驗由 C++ (Primer) 產生的過渡二進制檔案 (.bin)」。
它直接讀取底層由 C++ 所打造的 ChessBoard 結構（固定 32 bytes），
幫助我們在餵給 Python 轉換腳本之前，提早偷看一眼資料的健康度。

主要用途：
驗證 C++ 的戰術過濾是否成功運作、精準檢視底層引擎最原始的評分 (Internal Units)，
並透過組裝 FEN 字串與 ASCII 棋盤，快速確認這批原生物料的完整性，
讓你可以在 1 毫秒內就知道要不要中斷後續漫長的轉換處理流程。
"""
import struct
import math
import numpy as np

K = 0.0025

def board_to_fen(board):
    fen_parts = []
    for rank in range(7, -1, -1):
        empty_count = 0
        rank_str = ""
        for file in range(8):
            sq = rank * 8 + file
            piece = board[sq]
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
        if empty_count > 0:
            rank_str += str(empty_count)
        fen_parts.append(rank_str)
    
    # 永遠是 STM=w, 且 Bullet 省略了 castling/ep 等狀態
    return "/".join(fen_parts) + " w - - 0 1"

def print_bullet_preview(bin_path, num_samples=3):
    struct_fmt = '<Q16shBBB3s'
    struct_len = 32
    
    piece_chars = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',  # White
        8: 'p', 9: 'n', 10: 'b', 11: 'r', 12: 'q', 13: 'k'  # Black
    }
    
    with open(bin_path, 'rb') as f:
        raw = f.read(struct_len * num_samples)
        
    for i in range(num_samples):
        offset = i * struct_len
        s = struct.unpack_from(struct_fmt, raw, offset)
        
        occupancy = s[0]
        pieces_arr = np.frombuffer(s[1], dtype=np.uint8)
        score = s[2]
        wk_sq = s[4]
        bk_sq = s[5]
        
        wdl = 1.0 / (1.0 + math.exp(-K * score))
        wdl = max(0.001, min(0.999, wdl))
        
        # construct board
        board = ['.'] * 64
        
        temp = occupancy
        p_idx = 0
        pc_count = 0
        while temp > 0:
            lsb = temp & (~temp + 1)
            sq = int(math.log2(lsb))
            
            byte_val = pieces_arr[p_idx // 2]
            if (p_idx % 2) == 1:
                p_val = (byte_val >> 4) & 0x0F
            else:
                p_val = byte_val & 0x0F
                
            p_idx += 1
            
            char = piece_chars.get(p_val, '?')
            if char != '?':
                board[sq] = char
                pc_count += 1
                
            temp &= temp - 1
            
        fen = board_to_fen(board)
            
        print(f"\n======== 官方純淨樣本 #{i} ========")
        print(f"✅ 引擎精確評分 (cp): {score}")
        print(f"✅ 勝率估計 (WDL): {wdl:.3f}  (> 0.5 偏白優, < 0.5 偏黑優)")
        print(f"🧩 總棋子數量: {pc_count}")
        print(f"🔗 FEN 格式: {fen}")
        print("♟️ 訓練盤面 (已永遠轉化為 STM 視角):")
        
        print("   +-----------------+")
        for rank in range(7, -1, -1):
            row_str = ""
            for file in range(8):
                sq = rank * 8 + file
                row_str += board[sq] + " "
            print(f" {rank+1} | {row_str}|")
        print("   +-----------------+")
        print("     a b c d e f g h")
        print()

if __name__ == '__main__':
    print_bullet_preview("tuner/official_data.bin")
