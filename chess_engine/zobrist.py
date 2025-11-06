# chess_engine/zobrist.py
import numpy as np
import numba as nb
from chess_engine.types import piece_bbs_signature, game_state_signature

# --- Constants ---
BLACK = 1

# --- Initialize Zobrist Keys ---
np.random.seed(123456789)
PIECE_SQUARE_KEYS = np.random.randint(1, 2**64 - 1, (12, 64), dtype=np.uint64)
SIDE_TO_MOVE_KEY = np.random.randint(1, 2**64 - 1, dtype=np.uint64)
EN_PASSANT_FILE_KEYS = np.random.randint(1, 2**64 - 1, 8, dtype=np.uint64)
CASTLING_RIGHTS_KEYS = np.random.randint(1, 2**64 - 1, 16, dtype=np.uint64)

from chess_engine.constants import DE_BRUIJN_SEQUENCE, DE_BRUIJN_INDEX

@nb.jit(nb.int8(nb.uint64), nopython=True, inline='always')
def get_lsb_index(bitboard: np.uint64) -> int:
    """
    Finds the index of the LSB using a De Bruijn sequence bitscan.
    """
    bitboard = np.uint64(bitboard)
    if bitboard == 0:
        return -1
    lsb = bitboard & (-bitboard)
    index = (lsb * DE_BRUIJN_SEQUENCE) >> np.uint64(58)
    return DE_BRUIJN_INDEX[index]

@nb.jit(nb.uint64(piece_bbs_signature, game_state_signature), nopython=True)
def compute_initial_hash(piece_bbs: tuple, game_state: tuple) -> np.uint64:
    """
    Computes the Zobrist hash from scratch.
    """
    zobrist_key = np.uint64(0)
    side_to_move, castling_rights, en_passant_square, _, _ = game_state

    # 1. Hash Piece-Square Keys
    for piece_type in range(12):
        bb = piece_bbs[piece_type]
        while bb != 0:
            square = get_lsb_index(bb)
            zobrist_key ^= PIECE_SQUARE_KEYS[piece_type, square]
            bb &= np.uint64(bb - 1)

    # 2. Hash Castling Rights
    zobrist_key ^= CASTLING_RIGHTS_KEYS[castling_rights]

    # 3. Hash En Passant Square
    if en_passant_square != -1:
        ep_file = en_passant_square % 8
        zobrist_key ^= EN_PASSANT_FILE_KEYS[ep_file]

    # 4. Hash Side to Move
    if side_to_move == BLACK:
        zobrist_key ^= SIDE_TO_MOVE_KEY

    return zobrist_key

# # --- Debug Harness Function ---
#
# # Note: This function is intentionally NOT JIT-compiled to allow for print statements.
# def debug_recalculate_hash(piece_bbs: tuple, game_state: tuple) -> np.uint64:
#     """
#     Computes the Zobrist hash from scratch with verbose logging for debugging.
#     """
#     print("\n--- Recalculating Hash (Verbose) ---")
#     key = np.uint64(0)
#     print(f"Initial Key: {key}")
#
#     side_to_move, castling_rights, en_passant_square, _, _ = game_state
#
#     # 1. Hash Piece-Square Keys
#     # Unpack pieces for readable names in logs
#     (wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK) = piece_bbs
#     piece_map = {
#         0: (wP, "wP"), 1: (wN, "wN"), 2: (wB, "wB"), 3: (wR, "wR"), 4: (wQ, "wQ"), 5: (wK, "wK"),
#         6: (bP, "bP"), 7: (bN, "bN"), 8: (bB, "bB"), 9: (bR, "bR"), 10: (bQ, "bQ"), 11: (bK, "bK"),
#     }
#
#     for piece_type_idx in range(12):
#         bb, piece_name = piece_map[piece_type_idx]
#         temp_bb = bb
#         while temp_bb != 0:
#             square = get_lsb_index(temp_bb)
#             zobrist_val = PIECE_SQUARE_KEYS[piece_type_idx, square]
#             print(f" - Hashing Piece: {piece_name} at sq={square} | val={zobrist_val}")
#             key ^= zobrist_val
#             print(f"   -> Running Key: {key}")
#             temp_bb &= np.uint64(temp_bb - 1)
#
#     # 2. Hash Castling Rights
#     zobrist_val = CASTLING_RIGHTS_KEYS[castling_rights]
#     print(f" - Hashing Castling: rights={castling_rights} | val={zobrist_val}")
#     key ^= zobrist_val
#     print(f"   -> Running Key: {key}")
#
#     # 3. Hash En Passant Square
#     if en_passant_square != -1:
#         ep_file = en_passant_square % 8
#         zobrist_val = EN_PASSANT_FILE_KEYS[ep_file]
#         print(f" - Hashing En Passant File: file={ep_file} (from sq={en_passant_square}) | val={zobrist_val}")
#         key ^= zobrist_val
#         print(f"   -> Running Key: {key}")
#     else:
#         print(" - Skipping En Passant: No EP square.")
#
#     # 4. Hash Side to Move
#     if side_to_move == BLACK:
#         zobrist_val = SIDE_TO_MOVE_KEY
#         print(f" - Hashing Side to Move: BLACK | val={zobrist_val}")
#         key ^= zobrist_val
#         print(f"   -> Running Key: {key}")
#     else:
#         print(" - Skipping Side to Move: WHITE.")
#
#
#     print(f"--- Final Recalculated Key: {key} ---")
#     return key
