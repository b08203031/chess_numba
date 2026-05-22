"""
convert_bullet_bin.py — 將 Primer 輸出的 Bullet .bin 檔案轉換為 NNUE 訓練器的 .npz 格式

用法:
    python tuner/convert_bullet_bin.py input.bin [--output tuner/ultimate_halfka.npz] [--limit 40000000]

Bullet .bin 格式 (32 bytes per struct, C++ 'ChessBoard'):
    uint64_t occupancy;      // 8 bytes (Pieces bitboard)
    uint8_t  pieces[16];     // 16 bytes (Piece types matching the set bits in occupancy)
    int16_t  score;          // 2 bytes (cp score from stm perspective)
    uint8_t  result;         // 1 byte (0=Black Win, 1=Draw, 2=White Win)
    uint8_t  king_square;    // 1 byte (stm king)
    uint8_t  opp_king_square;// 1 byte (nstm king)
    uint8_t  padding[3];     // 3 bytes
"""

import os
import sys
import time
import math
import struct
import numpy as np
import numba

# ==========================================
# ⚙️ 設定區
# ==========================================
DEFAULT_OUTPUT = "tuner/ultimate_halfka_farseerT75.npz"
DEFAULT_LIMIT = 45_000_000
RECORD_SIZE = 32  # bytes per ChessBoard struct

# WDL Scale
K = 0.0025

# ==========================================
# ♟️ Piece Mapping
# ==========================================
# Primer's bulletformatpiece mapping:
# bulletformatpiece[12] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13};
# Stockfish internal: 
# None=0, Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
# White=0..7, Black=8..15 (W_Pawn=1, B_Pawn=9, W_Knight=2, B_Knight=10...)
#
# Let's map Primer's piece enum back to our 0-11 Numba indexing:
# Our train pipeline index:
# 0:W_Pawn, 1:W_Knight, 2:W_Bishop, 3:W_Rook, 4:W_Queen, 5:W_King
# 6:B_Pawn, 7:B_Knight, 8:B_Bishop, 9:B_Rook, 10:B_Queen, 11:B_King

# Mapping from Bullet piece ID to our 0-11 index.
# In bulletformatpiece, Stockfish Piece -> Bullet Piece (bullet = bulletformatpiece[piece]) 
# Note: Since the struct has inverted the board if STM is black, all positions are written
# from White's perspective! Thus, STM is always treating the board as White to move.
# But let's build a safe bitboard array just to feed standard get_halfka_indices.

@numba.njit(cache=True, fastmath=True)
def get_halfka_indices_bullet(occupancy, pieces, wk_sq, bk_sq):
    """
    Extracts HalfKA features from Bullet format — STM perspective.
    Bullet format guarantees the board is stored from STM perspective (STM is treated as White).
    wk_sq is STM king, bk_sq is NSTM king.

    Official HalfKAv2_hm interleaved piece-channel ordering (from STM's perspective):
        0  = PS_W_PAWN   (own pawn)      1  = PS_B_PAWN   (opp pawn)
        2  = PS_W_KNIGHT (own knight)    3  = PS_B_KNIGHT (opp knight)
        4  = PS_W_BISHOP (own bishop)    5  = PS_B_BISHOP (opp bishop)
        6  = PS_W_ROOK   (own rook)      7  = PS_B_ROOK   (opp rook)
        8  = PS_W_QUEEN  (own queen)     9  = PS_B_QUEEN  (opp queen)
        10 = PS_KING     (merged: BOTH kings share this channel)
    Feature index = bucket * 704 + mapped_type * 64 + mapped_sq
    """
    indices = np.full(32, 22528, dtype=np.int32)
    idx_count = 0

    # In Bullet format, board is flipped if STM was Black.
    # Therefore, STM is ALWAYS White here.
    flip_h = (wk_sq % 8) >= 4
    if flip_h:
        king_sq_mapped = wk_sq ^ 7
    else:
        king_sq_mapped = wk_sq

    bucket = (king_sq_mapped // 8) * 4 + (king_sq_mapped % 8)

    # Loop through occupancy bits
    temp_occ = occupancy
    p_idx = 0
    while temp_occ > 0:
        lsb = temp_occ & (~temp_occ + numba.uint64(1))
        sq = 0
        tmp = lsb >> numba.uint64(1)
        while tmp > 0:
            sq += 1
            tmp >>= numba.uint64(1)

        byte_val = pieces[p_idx // 2]
        if (p_idx % 2) == 1:
            bullet_piece = (byte_val >> 4) & 0x0F
        else:
            bullet_piece = byte_val & 0x0F
        p_idx += 1

        # Bullet Piece Decoding (verified against Primer source: nnue_data_binpack_format.h:7553)
        # Bullet value → our_piece (0-5 = STM P/N/B/R/Q/K, 6-11 = NSTM P/N/B/R/Q/K)
        our_piece = -1
        if   bullet_piece == 0:  our_piece = 0   # STM Pawn
        elif bullet_piece == 1:  our_piece = 1   # STM Knight
        elif bullet_piece == 2:  our_piece = 2   # STM Bishop
        elif bullet_piece == 3:  our_piece = 3   # STM Rook
        elif bullet_piece == 4:  our_piece = 4   # STM Queen
        elif bullet_piece == 5:  our_piece = 5   # STM King
        elif bullet_piece == 8:  our_piece = 6   # NSTM Pawn
        elif bullet_piece == 9:  our_piece = 7   # NSTM Knight
        elif bullet_piece == 10: our_piece = 8   # NSTM Bishop
        elif bullet_piece == 11: our_piece = 9   # NSTM Rook
        elif bullet_piece == 12: our_piece = 10  # NSTM Queen
        elif bullet_piece == 13: our_piece = 11  # NSTM King

        if our_piece != -1:
            # Official SF interleaved mapping (STM perspective):
            #   own P/N/B/R/Q → even channels 0,2,4,6,8
            #   opp P/N/B/R/Q → odd channels 1,3,5,7,9
            #   both kings    → merged channel 10 (PS_KING)
            if our_piece == 5 or our_piece == 11:
                mapped_type = 10               # PS_KING (merged)
            elif our_piece < 5:
                mapped_type = our_piece * 2    # own P=0, N=2, B=4, R=6, Q=8
            else:
                mapped_type = (our_piece - 6) * 2 + 1  # opp P=1, N=3, B=5, R=7, Q=9

            mapped_sq = sq
            if flip_h:
                mapped_sq ^= 7

            if idx_count < 32:
                indices[idx_count] = bucket * 704 + mapped_type * 64 + mapped_sq
                idx_count += 1

        temp_occ &= temp_occ - numba.uint64(1)

    return indices
    
@numba.njit(cache=True, fastmath=True)
def get_halfka_indices_bullet_nstm(occupancy, pieces, wk_sq, bk_sq):
    """
    Extracts HalfKA features from Bullet format — NSTM perspective.
    In Bullet format, NSTM is effectively Black.
    Primer already normalizes opp_king_square to NSTM's own perspective (verified:
    nnue_data_binpack_format.h:7577), so bk_sq does NOT need additional ^56.

    Official HalfKAv2_hm interleaved piece-channel ordering (from NSTM's perspective):
        Pieces that are NSTM's own  → even channels (0,2,4,6,8)
        Pieces that are STM (=opp)  → odd channels  (1,3,5,7,9)
        Both kings                  → merged channel 10 (PS_KING)

    NSTM mapped_type formula (our_piece uses STM-relative numbering 0-11):
        our_piece 0-4  (STM P/N/B/R/Q) = opponent → our_piece * 2 + 1
        our_piece 5    (STM King)       = opp king → 10
        our_piece 6-10 (NSTM P/N/B/R/Q) = own     → (our_piece - 6) * 2
        our_piece 11   (NSTM King)      = own king → 10
    """
    indices = np.full(32, 22528, dtype=np.int32)
    idx_count = 0

    # NSTM is effectively Black. Primer already normalizes opp_king_square
    # to the NSTM's perspective — no additional ^56 needed.
    bk_sq_sym = bk_sq
    flip_h = (bk_sq_sym % 8) >= 4
    if flip_h:
        king_sq_mapped = bk_sq_sym ^ 7
    else:
        king_sq_mapped = bk_sq_sym

    bucket = (king_sq_mapped // 8) * 4 + (king_sq_mapped % 8)

    temp_occ = occupancy
    p_idx = 0
    while temp_occ > 0:
        lsb = temp_occ & (~temp_occ + numba.uint64(1))
        sq = 0
        tmp = lsb >> numba.uint64(1)
        while tmp > 0:
            sq += 1
            tmp >>= numba.uint64(1)

        byte_val = pieces[p_idx // 2]
        if (p_idx % 2) == 1:
            bullet_piece = (byte_val >> 4) & 0x0F
        else:
            bullet_piece = byte_val & 0x0F
        p_idx += 1

        our_piece = -1
        if   bullet_piece == 0:  our_piece = 0   # STM Pawn
        elif bullet_piece == 1:  our_piece = 1   # STM Knight
        elif bullet_piece == 2:  our_piece = 2   # STM Bishop
        elif bullet_piece == 3:  our_piece = 3   # STM Rook
        elif bullet_piece == 4:  our_piece = 4   # STM Queen
        elif bullet_piece == 5:  our_piece = 5   # STM King
        elif bullet_piece == 8:  our_piece = 6   # NSTM Pawn
        elif bullet_piece == 9:  our_piece = 7   # NSTM Knight
        elif bullet_piece == 10: our_piece = 8   # NSTM Bishop
        elif bullet_piece == 11: our_piece = 9   # NSTM Rook
        elif bullet_piece == 12: our_piece = 10  # NSTM Queen
        elif bullet_piece == 13: our_piece = 11  # NSTM King

        if our_piece != -1:
            # Official SF interleaved mapping (NSTM perspective):
            #   NSTM's own P/N/B/R/Q → even channels 0,2,4,6,8
            #   STM (opponent) P/N/B/R/Q → odd channels 1,3,5,7,9
            #   both kings → merged channel 10 (PS_KING)
            if our_piece == 5 or our_piece == 11:
                mapped_type = 10                       # PS_KING (merged)
            elif our_piece < 5:
                mapped_type = our_piece * 2 + 1        # STM is opp: P=1, N=3, B=5, R=7, Q=9
            else:
                mapped_type = (our_piece - 6) * 2      # NSTM is own: P=0, N=2, B=4, R=6, Q=8

            # Flip square vertically for NSTM (Black) perspective
            mapped_sq = sq ^ 56
            if flip_h:
                mapped_sq ^= 7

            if idx_count < 32:
                indices[idx_count] = bucket * 704 + mapped_type * 64 + mapped_sq
                idx_count += 1

        temp_occ &= temp_occ - numba.uint64(1)

    return indices

def cp_to_wdl(cp: int) -> float:
    cp = max(-10000, min(10000, cp))
    wdl = 1.0 / (1.0 + math.exp(-K * cp))
    return max(0.001, min(0.999, wdl))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bullet .bin (32 bytes ChessBoard) -> NPZ 訓練資料")
    parser.add_argument("input_bin", nargs="?", default="tuner/official_data.bin", help="輸入的 .bin 檔案路徑 (預設: tuner/official_data.bin)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"輸出的 .npz 檔案路徑")
    parser.add_argument("--limit", "-n", type=int, default=DEFAULT_LIMIT, help=f"最大樣本數")
    args = parser.parse_args()
    
    input_path = args.input_bin
    output_path = args.output
    limit = args.limit
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        sys.exit(1)
    
    file_size = os.path.getsize(input_path)
    if (file_size == 0):
        print(f"輸入檔案是空的。")
        sys.exit(1)
        
    total_records = file_size // RECORD_SIZE
    target = min(limit, total_records)
    
    print(f"解析真正 32-byte 的 ChessBoard Bullet 格式: {input_path}")
    print(f"   總記錄數: {total_records:,}")
    print(f"   目標樣本: {target:,}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_features_stm  = np.zeros((target, 32), dtype=np.uint16)
    all_features_nstm = np.zeros((target, 32), dtype=np.uint16)
    all_targets       = np.zeros(target,       dtype=np.float32)
    all_piece_counts  = np.zeros(target,       dtype=np.int8)
    
    print("JIT 預熱中...")
    dummy_occ = numba.uint64(1)
    dummy_pieces = np.zeros(16, dtype=np.uint8)
    get_halfka_indices_bullet(dummy_occ, dummy_pieces, np.uint8(0), np.uint8(63))
    get_halfka_indices_bullet_nstm(dummy_occ, dummy_pieces, np.uint8(0), np.uint8(63))
    print("JIT 完成！")
    
    count = 0
    skipped = 0
    start_time = time.time()
    
    CHUNK_SIZE = 100_000
    chunk_bytes = CHUNK_SIZE * RECORD_SIZE
    struct_fmt = '<Q16shBBB3s' # uint64, 16 bytes array, int16, uint8, uint8, uint8, 3 padding bytes
    struct_len = struct.calcsize(struct_fmt)
    
    with open(input_path, 'rb') as f:
        while count < target:
            raw = f.read(chunk_bytes)
            if len(raw) < RECORD_SIZE:
                break
                
            n_records = len(raw) // RECORD_SIZE
            for i in range(n_records):
                offset = i * RECORD_SIZE
                s = struct.unpack_from(struct_fmt, raw, offset)
                
                occupancy = s[0]
                pieces_arr = np.frombuffer(s[1], dtype=np.uint8)
                score = s[2]
                # result = s[3]
                wk_sq = s[4]
                bk_sq = s[5]
                
                # Check bounding safely 
                if abs(score) > 10000:
                    skipped += 1
                    continue
                    
                pc_count = 0
                temp = occupancy
                while temp > 0:
                    pc_count += 1
                    temp &= temp - 1
                    
                all_piece_counts[count] = pc_count
                all_targets[count] = cp_to_wdl(score)
                all_features_stm[count] = get_halfka_indices_bullet(np.uint64(occupancy), pieces_arr, np.uint8(wk_sq), np.uint8(bk_sq))
                all_features_nstm[count] = get_halfka_indices_bullet_nstm(np.uint64(occupancy), pieces_arr, np.uint8(wk_sq), np.uint8(bk_sq))
                count += 1
                
                if count >= target:
                    break
                    
            elapsed = time.time() - start_time
            if count % 100_000 == 0:
                speed = count / elapsed if elapsed > 0 else 0
                print(f"  進度: {count:>10,} / {target:,} | 速度: {speed:,.0f} 筆/秒 | 耗時: {elapsed:.0f}s", end='\r')
                
    elapsed = time.time() - start_time
    print(f"\n解析完成！有效樣本: {count:,} | 總耗時: {elapsed/60:.1f} 分鐘")
    
    print(f"正在儲存至 {output_path}...")
    save_start = time.time()
    np.savez_compressed(
        output_path,
        features_stm=all_features_stm[:count],
        features_nstm=all_features_nstm[:count],
        targets=all_targets[:count],
        piece_counts=all_piece_counts[:count]
    )
    print(f"存檔成功！耗時: {time.time()-save_start:.2f}s")


if __name__ == '__main__':
    main()
