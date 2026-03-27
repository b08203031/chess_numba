"""
Convert Stockfish .binpack training data to .npz for train.py.
Parses both stem positions and movetext to extract ALL positions with scores.

Usage:
  python convert_binpack.py -i <path.binpack> -o <path.npz> [--max N] [--verify]
"""
import numpy as np
import chess
import argparse
import time
import os
import math

# ── Nibble → train.py piece_bbs index ──
NIBBLE_TO_TRAINIDX = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]

NIBBLE_TO_PIECE = {
    0: (chess.PAWN, chess.WHITE),   1: (chess.PAWN, chess.BLACK),
    2: (chess.KNIGHT, chess.WHITE), 3: (chess.KNIGHT, chess.BLACK),
    4: (chess.BISHOP, chess.WHITE), 5: (chess.BISHOP, chess.BLACK),
    6: (chess.ROOK, chess.WHITE),   7: (chess.ROOK, chess.BLACK),
    8: (chess.QUEEN, chess.WHITE),  9: (chess.QUEEN, chess.BLACK),
    10: (chess.KING, chess.WHITE),  11: (chess.KING, chess.BLACK),
}


def unsigned_to_signed_16(u):
    r = ((u >> 1) | ((u & 1) << 15)) & 0xFFFF
    if r & 0x8000:
        r ^= 0x7FFF
    if r >= 0x8000:
        return r - 0x10000
    return r


def decompress_to_board(pos_bytes):
    """24-byte CompressedPosition -> chess.Board"""
    occupied = int.from_bytes(pos_bytes[0:8], 'little')
    nibble_data = pos_bytes[8:24]
    board = chess.Board.empty()
    board.castling_rights = chess.BB_EMPTY
    stm = chess.WHITE
    ep_square = None
    nib_idx = 0
    for sq in range(64):
        if occupied & (1 << sq):
            byte_idx = nib_idx >> 1
            nibble = (nibble_data[byte_idx] >> ((nib_idx & 1) * 4)) & 0x0F
            if nibble <= 11:
                pt, color = NIBBLE_TO_PIECE[nibble]
                board.set_piece_at(sq, chess.Piece(pt, color))
            elif nibble == 12:
                if (sq >> 3) == 3:
                    board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
                    ep_square = sq - 8
                else:
                    board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
                    ep_square = sq + 8
            elif nibble == 13:
                board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.WHITE))
                if sq == chess.A1: board.castling_rights |= chess.BB_A1
                elif sq == chess.H1: board.castling_rights |= chess.BB_H1
            elif nibble == 14:
                board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.BLACK))
                if sq == chess.A8: board.castling_rights |= chess.BB_A8
                elif sq == chess.H8: board.castling_rights |= chess.BB_H8
            elif nibble == 15:
                board.set_piece_at(sq, chess.Piece(chess.KING, chess.BLACK))
                stm = chess.BLACK
            nib_idx += 1
    board.turn = stm
    board.ep_square = ep_square
    return board


def board_to_bbs(board):
    """chess.Board -> (piece_bbs[12], stm)"""
    piece_bbs = [0] * 12
    # train.py order: WP=0 WN=1 WB=2 WR=3 WQ=4 WK=5 BP=6 BN=7 BB=8 BR=9 BQ=10 BK=11
    for sq in range(64):
        p = board.piece_at(sq)
        if p is None:
            continue
        # piece_type: PAWN=1 KNIGHT=2 BISHOP=3 ROOK=4 QUEEN=5 KING=6
        # color: WHITE=True BLACK=False  
        idx = (p.piece_type - 1)  # 0-5
        if not p.color:  # BLACK
            idx += 6
        piece_bbs[idx] |= (1 << sq)
    stm = 0 if board.turn == chess.WHITE else 1
    return piece_bbs, stm


# ── Bit Reader ──
class BitReader:
    __slots__ = ['data', 'byte_pos', 'bit_pos']
    def __init__(self, data, offset=0):
        self.data = data
        self.byte_pos = offset
        self.bit_pos = 0
    
    def read_bits(self, n):
        if n == 0:
            return 0
        result = 0
        for _ in range(n):
            byte_val = self.data[self.byte_pos]
            bit = (byte_val >> (7 - self.bit_pos)) & 1
            result = (result << 1) | bit
            self.bit_pos += 1
            if self.bit_pos >= 8:
                self.bit_pos = 0
                self.byte_pos += 1
        return result
    
    def total_bytes(self):
        return self.byte_pos + (1 if self.bit_pos > 0 else 0)


def bits_needed(max_val):
    if max_val <= 0:
        return 0
    return max_val.bit_length()


def nth_set_bit(bb, n):
    for _ in range(n):
        bb &= bb - 1
    return (bb & -bb).bit_length() - 1


def pawn_destinations(board, sq):
    color = board.turn
    occ = board.occupied
    dests = 0
    if color == chess.WHITE:
        up = sq + 8
        if up < 64 and not (occ & (1 << up)):
            dests |= (1 << up)
            if (sq >> 3) == 1:
                up2 = sq + 16
                if not (occ & (1 << up2)):
                    dests |= (1 << up2)
        for csq in (sq + 7, sq + 9):
            if 0 <= csq < 64 and abs((csq & 7) - (sq & 7)) == 1:
                if (board.occupied_co[chess.BLACK] & (1 << csq)) or board.ep_square == csq:
                    dests |= (1 << csq)
    else:
        down = sq - 8
        if down >= 0 and not (occ & (1 << down)):
            dests |= (1 << down)
            if (sq >> 3) == 6:
                down2 = sq - 16
                if not (occ & (1 << down2)):
                    dests |= (1 << down2)
        for csq in (sq - 7, sq - 9):
            if 0 <= csq < 64 and abs((csq & 7) - (sq & 7)) == 1:
                if (board.occupied_co[chess.WHITE] & (1 << csq)) or board.ep_square == csq:
                    dests |= (1 << csq)
    return dests


def decode_move(board, reader):
    """Decode one variable-length encoded move."""
    our = board.occupied_co[board.turn]
    our_count = bin(our).count('1')
    
    piece_bits = bits_needed(our_count - 1)
    piece_idx = reader.read_bits(piece_bits)
    
    bb = our
    for _ in range(piece_idx):
        bb &= bb - 1
    from_sq = (bb & -bb).bit_length() - 1
    
    piece = board.piece_at(from_sq)
    pt = piece.piece_type
    
    if pt == chess.PAWN:
        dests = pawn_destinations(board, from_sq)
        dc = bin(dests).count('1')
        promo_rank = 7 if board.turn == chess.WHITE else 0
        has_promo = any(((dests >> s) & 1) and (s >> 3) == promo_rank for s in range(64) if (dests >> s) & 1)
        if has_promo:
            max_id = dc * 4 - 1
            move_id = reader.read_bits(bits_needed(max_id))
            to_sq = nth_set_bit(dests, move_id // 4)
            promo = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][move_id % 4]
            return chess.Move(from_sq, to_sq, promotion=promo)
        else:
            move_id = reader.read_bits(bits_needed(dc - 1)) if dc > 1 else 0
            to_sq = nth_set_bit(dests, move_id)
            return chess.Move(from_sq, to_sq)
    
    elif pt == chess.KING:
        attacks = chess.BB_KING_ATTACKS[from_sq] & ~our
        dc = bin(attacks).count('1')
        castlings = []
        c = board.turn
        if board.has_queenside_castling_rights(c):
            rook_sq = chess.A1 if c == chess.WHITE else chess.A8
            castlings.append(chess.Move(from_sq, rook_sq))
        if board.has_kingside_castling_rights(c):
            rook_sq = chess.H1 if c == chess.WHITE else chess.H8
            castlings.append(chess.Move(from_sq, rook_sq))
        nc = len(castlings)
        max_id = dc + nc - 1
        move_id = reader.read_bits(bits_needed(max_id))
        if move_id < dc:
            to_sq = nth_set_bit(attacks, move_id)
            return chess.Move(from_sq, to_sq)
        else:
            return castlings[move_id - dc]
    else:
        attacks = board.attacks_mask(from_sq) & ~our
        dc = bin(attacks).count('1')
        move_id = reader.read_bits(bits_needed(dc - 1)) if dc > 1 else 0
        to_sq = nth_set_bit(attacks, move_id)
        return chess.Move(from_sq, to_sq)


def decode_score_varint(reader):
    result = 0
    shift = 0
    while True:
        nibble = reader.read_bits(4)
        ext = reader.read_bits(1)
        result |= (nibble << shift)
        shift += 4
        if ext == 0:
            break
    return result


def find_binp_offsets(path):
    file_size = os.path.getsize(path)
    offsets = []
    CHUNK = 16 * 1024 * 1024
    with open(path, 'rb') as f:
        scanned = 0
        overlap = b''
        while scanned < file_size:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            search = overlap + chunk
            idx = 0
            while True:
                pos = search.find(b'BINP', idx)
                if pos == -1:
                    break
                offsets.append(scanned - len(overlap) + pos)
                idx = pos + 1
            overlap = chunk[-3:] if len(chunk) >= 3 else chunk
            scanned += len(chunk)
    return offsets, file_size


def convert_binpack(input_path, output_path, max_positions=None, verify=False):
    file_size = os.path.getsize(input_path)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")
    
    print("Phase 1: Scanning block boundaries...")
    t0 = time.time()
    offsets, _ = find_binp_offsets(input_path)
    print(f"  Found {len(offsets)} blocks in {time.time()-t0:.1f}s")
    
    BATCH = 500_000
    all_bbs, all_results, all_stm = [], [], []
    batch_bbs = np.zeros((BATCH, 12), dtype=np.uint64)
    batch_results = np.zeros(BATCH, dtype=np.float32)
    batch_stm = np.zeros(BATCH, dtype=np.int8)
    batch_idx = 0
    
    valid = 0
    errors = 0
    chains = 0
    start = time.time()
    
    print("Phase 2: Extracting positions...")
    
    with open(input_path, 'rb') as f:
        for bi in range(len(offsets)):
            if max_positions and valid >= max_positions:
                break
            
            bstart = offsets[bi]
            bend = offsets[bi+1] if bi+1 < len(offsets) else file_size
            f.seek(bstart)
            bdata = f.read(bend - bstart)
            off = 8  # skip header
            
            while off + 34 <= len(bdata):
                if max_positions and valid >= max_positions:
                    break
                
                stem = bdata[off:off+32]
                if stem[0:4] == b'BINP':
                    break
                off += 32
                
                count = int.from_bytes(bdata[off:off+2], 'big')
                off += 2
                
                # Decode stem position
                try:
                    board = decompress_to_board(stem[0:24])
                except Exception:
                    errors += 1
                    break
                
                score_raw = int.from_bytes(stem[26:28], 'big')
                ply_result_raw = int.from_bytes(stem[28:30], 'big')
                stem_score = unsigned_to_signed_16(score_raw)
                result_enc = (ply_result_raw >> 14) & 0x3
                game_result = unsigned_to_signed_16(result_enc)
                
                chains += 1
                
                # Store stem position if has valid score
                def store_pos(b, sc, res):
                    nonlocal batch_idx, valid
                    if abs(sc) >= 32000:
                        return
                    bbs, s = board_to_bbs(b)
                    # Validate
                    if bin(bbs[5]).count('1') != 1 or bin(bbs[11]).count('1') != 1:
                        return
                    # Result: STM perspective -> white perspective
                    if s == 0:
                        wr = (res + 1) / 2.0
                    else:
                        wr = (1 - res) / 2.0
                    for i in range(12):
                        batch_bbs[batch_idx, i] = np.uint64(bbs[i])
                    batch_results[batch_idx] = np.float32(wr)
                    batch_stm[batch_idx] = np.int8(s)
                    batch_idx += 1
                    valid += 1
                    if batch_idx >= BATCH:
                        flush()
                
                def flush():
                    nonlocal batch_idx
                    if batch_idx > 0:
                        all_bbs.append(batch_bbs[:batch_idx].copy())
                        all_results.append(batch_results[:batch_idx].copy())
                        all_stm.append(batch_stm[:batch_idx].copy())
                        batch_idx = 0
                
                store_pos(board, stem_score, game_result)
                
                # Parse movetext
                if count > 0:
                    reader = BitReader(bdata, off)
                    prev_score = stem_score
                    ok = True
                    
                    for mi in range(count):
                        if max_positions and valid >= max_positions:
                            break
                        try:
                            move = decode_move(board, reader)
                            sdelta_enc = decode_score_varint(reader)
                            sdelta = unsigned_to_signed_16(sdelta_enc & 0xFFFF)
                            cur_score = -prev_score - sdelta
                            
                            board.push(move)
                            store_pos(board, cur_score, game_result)
                            prev_score = cur_score
                        except Exception:
                            errors += 1
                            ok = False
                            break
                    
                    if ok:
                        off = reader.byte_pos + (1 if reader.bit_pos > 0 else 0)
                    else:
                        break
            
            # Progress
            if (bi + 1) % 100 == 0 or bi == len(offsets) - 1:
                elapsed = time.time() - start
                rate = valid / elapsed if elapsed > 0 else 0
                pct = (bi + 1) / len(offsets) * 100
                print(f"  {pct:.1f}% | {valid:>10,} pos | {chains:>8,} chains | "
                      f"{errors:>6,} err | {rate:.0f} pos/s")
    
    # Final flush
    if batch_idx > 0:
        all_bbs.append(batch_bbs[:batch_idx].copy())
        all_results.append(batch_results[:batch_idx].copy())
        all_stm.append(batch_stm[:batch_idx].copy())
    
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s: {valid:,} positions from {chains:,} chains ({errors:,} errors)")
    
    if valid == 0:
        print("ERROR: No positions extracted!")
        return
    
    final_bbs = np.concatenate(all_bbs)
    final_results = np.concatenate(all_results)
    final_stm = np.concatenate(all_stm)
    
    print(f"Saving {output_path}...")
    np.savez_compressed(output_path, piece_bbs=final_bbs, results=final_results, stm=final_stm)
    print(f"  Shape: {final_bbs.shape}, Size: {os.path.getsize(output_path)/1e6:.1f} MB")
    
    if verify:
        print("\nVerification samples:")
        for i in range(min(5, valid)):
            bbs = final_bbs[i]
            r = final_results[i]
            s = final_stm[i]
            total_pieces = sum(bin(int(bbs[j])).count('1') for j in range(12))
            wk = bin(int(bbs[5])).count('1')
            bk = bin(int(bbs[11])).count('1')
            print(f"  #{i}: pieces={total_pieces} WK={wk} BK={bk} result={r:.2f} stm={'b' if s else 'w'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    convert_binpack(args.input, args.output, args.max, args.verify)
