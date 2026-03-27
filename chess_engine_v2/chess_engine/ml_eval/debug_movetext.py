"""Deep investigation: try different interpretations of stem + movetext count."""
import chess

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
    if r & 0x8000: r ^= 0x7FFF
    if r >= 0x8000: return r - 0x10000
    return r

def decompress_to_board(pos_bytes):
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

def bits_needed(max_val):
    if max_val <= 0: return 0
    return max_val.bit_length()

def nth_set_bit(bb, n):
    for _ in range(n):
        bb &= bb - 1
    return (bb & -bb).bit_length() - 1

def main():
    path = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba\training_data\test80-2024-01-jan-2tb7p.min-v2.v6.binpack"
    
    with open(path, 'rb') as f:
        f.read(8)  # skip BINP + meta
        block_data = f.read(1048583)
    
    # Starting position
    stem = block_data[0:32]
    print("Stem bytes (hex):")
    for i in range(0, 32, 2):
        print(f"  [{i:2d}-{i+1:2d}]: {stem[i:i+2].hex()}")
    
    board = decompress_to_board(stem[0:24])
    print(f"\nFEN: {board.fen()}")
    
    # The stem layout: pos(24) + move(2) + score(2) + ply_result(2) + rule50(2) = 32 bytes
    stem_move_raw = int.from_bytes(stem[24:26], 'little')
    stem_score_raw = int.from_bytes(stem[26:28], 'little')
    stem_ply_result = int.from_bytes(stem[28:30], 'little')
    stem_rule50 = int.from_bytes(stem[30:32], 'little')
    
    # Decode move: the stem move appears to be a simple from-to encoding
    from_sq = stem_move_raw & 0x3F
    to_sq = (stem_move_raw >> 6) & 0x3F
    type_bits = (stem_move_raw >> 12) & 0x0F
    print(f"\nStem move raw: 0x{stem_move_raw:04x} = {stem_move_raw}")
    print(f"  from={chess.square_name(from_sq)} to={chess.square_name(to_sq)} type={type_bits}")
    print(f"Stem score raw: 0x{stem_score_raw:04x} signed={unsigned_to_signed_16(stem_score_raw)}")
    print(f"Stem ply_result: 0x{stem_ply_result:04x}")
    print(f"Stem rule50: 0x{stem_rule50:04x}")
    
    # The count AFTER the stem
    count_pos = 32
    count_be = int.from_bytes(block_data[count_pos:count_pos+2], 'big')
    count_le = int.from_bytes(block_data[count_pos:count_pos+2], 'little')
    print(f"\nCount bytes: {block_data[count_pos:count_pos+2].hex()}")
    print(f"  BE={count_be}, LE={count_le}")
    
    # Now let's look at the movetext byte stream
    movetext_start = count_pos + 2
    mt_bytes = block_data[movetext_start:movetext_start+20]
    print(f"\nMovetext first 20 bytes: {mt_bytes.hex()}")
    for i, b in enumerate(mt_bytes):
        print(f"  byte {i}: 0x{b:02x} = {b:08b}")
    
    # Try decoding 1. e4: piece_idx=12, 4 bits = 1100
    # Then pawn at e2 destinations: e3(20), e4(28) = 2 destinations
    # e4 = index 1 (e3=0, e4=1), bits_needed(1) = 1 bit
    # So 1. e4 = 4 bits (piece) + 1 bit (dest) = 5 bits
    # Score after e4: varint encoded
    
    # Expected bit pattern for 1. e4: piece_idx=12=1100, dest_idx=1=1
    # = 1100 1 ... = 0xC8... 
    
    # What we see: first byte 0x36 = 00110110
    # This doesn't start with 1100...
    
    # Let me try LE count: count=107=0x006b, but LE interpretation of 00 6b = 27392
    # Unlikely.
    
    # Maybe the count is LE but the bytes are different?
    # The pattern 0x006b is consistent: 00 = high byte, 6b = low byte  
    # Both BE and LE with these bytes: BE=107, LE=27392
    # 107 moves makes sense for a chess game, so BE is correct.
    
    # But what if the stem move bytes (24-25) are part of the movetext count?
    # Let me check: stem might be only 30 bytes, not 32?
    
    # Let me try different stem sizes
    print("\n\n=== Trying different stem sizes ===")
    for stem_size in [24, 26, 28, 30, 32, 34]:
        count_off = stem_size
        if count_off + 2 > len(block_data):
            continue
        ct = int.from_bytes(block_data[count_off:count_off+2], 'big')
        ct_le = int.from_bytes(block_data[count_off:count_off+2], 'little')
        mt_off = count_off + 2
        # Read first 4 bits MSB from movetext
        if mt_off < len(block_data):
            first_byte = block_data[mt_off]
            first_4bits = (first_byte >> 4) & 0x0F
        else:
            first_4bits = -1
        
        print(f"  stem_size={stem_size}: count_BE={ct}, count_LE={ct_le}, "
              f"piece_idx_4bits={first_4bits}, mt_byte={block_data[mt_off]:02x}")
        
        # For starting position with 1. e4: piece_idx should be 12
        if first_4bits == 12:
            print(f"    *** MATCH: piece_idx=12 = e2 pawn! ***")

if __name__ == '__main__':
    main()
