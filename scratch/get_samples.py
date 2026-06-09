import os
import struct
import math
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)

WORKSPACE_DIR = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
BIN_PATH = os.path.join(WORKSPACE_DIR, "tuner/official_data_farseerT75.bin")

STRUCT_FMT = '<Q16shBBB3s'
STRUCT_LEN = struct.calcsize(STRUCT_FMT)

K = 0.0025
STOCKFISH_PAWN_VALUE_EG = 208.0

PIECE_CHARS = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',      # White (STM)
    8: 'p', 9: 'n', 10: 'b', 11: 'r', 12: 'q', 13: 'k'    # Black (NSTM)
}

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
    
    # Always STM=w (White to move) from STM perspective in Bullet format
    return "/".join(fen_parts) + " w - - 0 1"

def decode_record(raw_bytes):
    s = struct.unpack(STRUCT_FMT, raw_bytes)
    occupancy = s[0]
    pieces_arr = np.frombuffer(s[1], dtype=np.uint8)
    score = s[2]
    result = s[3]
    wk_sq = s[4]
    bk_sq = s[5]
    
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
        
        char = PIECE_CHARS.get(p_val, '?')
        if char != '?':
            board[sq] = char
            pc_count += 1
            
        temp &= temp - 1
        
    fen = board_to_fen(board)
    cp = score * 100.0 / STOCKFISH_PAWN_VALUE_EG
    wdl = 1.0 / (1.0 + math.exp(-K * cp))
    wdl = max(0.001, min(0.999, wdl))
    
    return {
        'fen': fen,
        'score_raw': score,
        'score_cp': cp,
        'wdl': wdl,
        'result': result,
        'wk_sq': wk_sq,
        'bk_sq': bk_sq,
        'piece_count': pc_count
    }

def main():
    if not os.path.exists(BIN_PATH):
        print(f"Error: {BIN_PATH} not found!")
        return

    file_size = os.path.getsize(BIN_PATH)
    total_records = file_size // STRUCT_LEN
    print(f"Total records in binary file: {total_records:,}")

    # Determine indices to read
    # First 3
    first_indices = [0, 1, 2]
    # Last 3
    last_indices = [total_records - 3, total_records - 2, total_records - 1]
    # Middle random 4 (excluding the first 3 and last 3)
    middle_indices = sorted(random.sample(range(3, total_records - 3), 4))

    all_indices = [
        ('First 3', first_indices),
        ('Middle 4 (Random)', middle_indices),
        ('Last 3', last_indices)
    ]

    with open(BIN_PATH, 'rb') as f:
        for group_name, indices in all_indices:
            print(f"\n==================== {group_name} ====================")
            for idx in indices:
                f.seek(idx * STRUCT_LEN)
                raw = f.read(STRUCT_LEN)
                if not raw:
                    print(f"Index {idx} failed to read!")
                    continue
                info = decode_record(raw)
                print(f"Index: {idx:,}")
                print(f"  FEN:   {info['fen']}")
                print(f"  Score: {info['score_raw']} (raw Stockfish value) | {info['score_cp']:.2f} cp (centipawns)")
                print(f"  WDL:   {info['wdl']:.4f}")
                print(f"  Result: {info['result']} (0=Black Win, 1=Draw, 2=White Win)")
                print()

if __name__ == "__main__":
    main()
