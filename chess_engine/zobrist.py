# chess_engine/zobrist.py
import numpy as np
import numba as nb
from chess_engine.engine_types import piece_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import get_ls1b_index as get_lsb_index

# --- Constants ---
BLACK = 1

# --- Initialize Zobrist Keys ---
np.random.seed(123456789)
PIECE_SQUARE_KEYS = np.random.randint(1, 2**64 - 1, (12, 64), dtype=np.uint64)
SIDE_TO_MOVE_KEY = np.random.randint(1, 2**64 - 1, dtype=np.uint64)
EN_PASSANT_FILE_KEYS = np.random.randint(1, 2**64 - 1, 8, dtype=np.uint64)
CASTLING_RIGHTS_KEYS = np.random.randint(1, 2**64 - 1, 16, dtype=np.uint64)

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
