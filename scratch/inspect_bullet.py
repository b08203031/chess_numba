import os
import sys
import struct
import numpy as np

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

# Bullet pieces:
# 0: STM Pawn, 1: STM Knight, 2: Bishop, 3: Rook, 4: Queen, 5: King
# 8: NSTM Pawn, 9: NSTM Knight, 10: Bishop, 11: Rook, 12: Queen, 13: King
BULLET_PIECE_NAMES = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    8: 'p', 9: 'n', 10: 'b', 11: 'r', 12: 'q', 13: 'k'
}

def print_board(occupancy, pieces, king_sq, opp_king_sq):
    board = ['.'] * 64
    
    # Place pieces
    temp_occ = occupancy
    p_idx = 0
    while temp_occ > 0:
        lsb = temp_occ & (~temp_occ + 1)
        sq = 0
        tmp = lsb >> 1
        while tmp > 0:
            sq += 1
            tmp >>= 1
            
        byte_val = pieces[p_idx // 2]
        if (p_idx % 2) == 1:
            bullet_piece = (byte_val >> 4) & 0x0F
        else:
            bullet_piece = byte_val & 0x0F
        p_idx += 1
        
        name = BULLET_PIECE_NAMES.get(bullet_piece, f'{bullet_piece}')
        board[sq] = name
        
        temp_occ &= temp_occ - 1
        
    # Print board 8x8 (from rank 8 to rank 1)
    for r in range(7, -1, -1):
        row = board[r*8 : (r+1)*8]
        print(" ".join(row))

def main():
    bin_path = os.path.join(workspace_dir, "tuner/official_data_farseerT75.bin")
    if not os.path.exists(bin_path):
        print(f"Dataset path {bin_path} does not exist!")
        return
        
    struct_fmt = '<Q16shBBB3s'
    struct_len = struct.calcsize(struct_fmt)
    
    print("Reading first 5 records from binary dataset...")
    with open(bin_path, 'rb') as f:
        for i in range(5):
            raw = f.read(struct_len)
            if len(raw) < struct_len:
                break
            
            occupancy, pieces_raw, score, result, king_sq, opp_king_sq, padding = struct.unpack(struct_fmt, raw)
            print(f"\n--- Record {i+1} ---")
            print(f"Score: {score}, Result: {result}, KingSq: {king_sq}, OppKingSq: {opp_king_sq}")
            print("Board representation:")
            print_board(occupancy, pieces_raw, king_sq, opp_king_sq)

if __name__ == "__main__":
    main()
