"""
成品檢測站：inspect_npz.py
======================
這個腳本用來「檢驗最終輸出的 Machine Learning 矩陣檔案 (.npz)」。
它會讀取已經被轉換為龐大 HalfKA 索引（0~45055）的特徵陣列，並透過「反向工程」
把這些供神經網路學習的抽象特徵，重新還原拼裝成人類可讀的 ASCII 棋盤。

主要用途：
肉眼驗證你的 Python 轉換腳本（convert_bullet_bin.py）的數學邏輯是否正確，
確保真正餵進 NNUE 神經網絡訓練的資料，沒有發生漏掉棋子或座標錯亂的致命問題。
"""
import numpy as np

def print_record(features, targets, piece_counts, i):
    f_stm = features[i]
    wdl = targets[i]
    pc = piece_counts[i]
    
    board = ['.'] * 64
    king_sq = -1
    
    # NNUE Feature -> Piece Mapping (STM perspective, board might be horizontally mirrored)
    # 0:P, 1:N, 2:B, 3:R, 4:Q (White/STM)
    # 5:p, 6:n, 7:b, 8:r, 9:q (Black/NSTM)
    piece_chars = ['P', 'N', 'B', 'R', 'Q', 'p', 'n', 'b', 'r', 'q', 'k']
    
    for idx in f_stm:
        if idx == 22528: # Padding
            continue
            
        # extract features
        bucket = idx // 704
        mapped_type = (idx % 704) // 64
        mapped_sq = idx % 64
        
        # We only need to find the king square once from the bucket.
        # bucket is 0-31 (left half of the board)
        if king_sq == -1:
            # mirror mapping: (bucket // 4)*8 + (bucket % 4)
            king_sq = (bucket // 4) * 8 + (bucket % 4)
            board[king_sq] = 'K' # STM King is ALWAYS White 'K' here
            
        if mapped_type < 11:
            board[mapped_sq] = piece_chars[mapped_type]
            
    # We don't have the black king explicit from HalfKA STM features alone 
    # (unless we cross-reference NSTM, but let's just show what STM sees).
    # Actually, we can just print the mirrored board.
            
    print(f"\n======== 樣本資料 #{i} ========")
    print(f"✅ 勝率估計 (WDL): {wdl:.3f}  (> 0.5 偏白優, < 0.5 偏黑優)")
    print(f"🧩 總棋子數量: {pc}")
    print("♟️ 收錄進 NNUE 的特徵盤面 (可能因對稱性被左右翻轉):")
    
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

def main():
    npz_path = "tuner/ultimate_halfka_farseerT75.npz"
    print("讀取大檔案資料中... (可能需要幾秒鐘)")
    try:
        data = np.load(npz_path)
        f_stm = data['features_stm']
        targets = data['targets']
        piece_counts = data['piece_counts']
        
        total = len(targets)
        print(f"\n📂 成功載入！總共有 {total:,} 筆超乾淨官方樣本。")
        
        # 顯示前面 3 筆資料
        for i in range(3):
            print_record(f_stm, targets, piece_counts, i)
            
        print("======== 特徵陣列原始狀態 ========")
        print(f"輸入層 (Features): {f_stm[0][:10]} ... (非 22528 代表啟動的神經元)")
                
    except Exception as e:
        print(f"請稍後，檔案可能還在產出中... ({e})")

if __name__ == '__main__':
    main()
