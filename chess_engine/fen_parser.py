# chess_engine/fen_parser.py
import numpy as np
from chess_engine.zobrist import compute_initial_hash

# --- Piece Type and Color Constants (for mapping FEN chars) ---
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
WHITE, BLACK = 0, 1

# --- Castling Rights Mapping (for FEN chars) ---
CASTLING_MAP = {'K': 1, 'Q': 2, 'k': 4, 'q': 8}

# --- Algebraic to Square Index Mapping ---
# Example: 'a1' -> 0, 'h8' -> 63
SQUARE_MAP = {
    f"{chr(ord('a') + file)}{rank + 1}": rank * 8 + file
    for rank in range(8) for file in range(8)
}

def parse_fen(fen_string: str):
    """
    Parses a FEN string and returns the board state in the engine's tuple format.

    Args:
        fen_string: The FEN string representing the board position.

    Returns:
        A tuple containing (piece_bbs, occupancy_bbs, game_state).
    """
    parts = fen_string.split()

    # --- 1. Parse Piece Placements ---
    piece_bbs = [np.uint64(0)] * 12
    fen_board = parts[0]
    rank, file = 7, 0
    for char in fen_board:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            square_index = rank * 8 + file
            piece_type_index = PIECE_MAP[char]
            piece_bbs[piece_type_index] |= (np.uint64(1) << square_index)
            file += 1

    final_piece_bbs = tuple(piece_bbs)

    # --- 2. Parse Side to Move ---
    side_to_move = np.uint8(WHITE if parts[1] == 'w' else BLACK)

    # --- 3. Parse Castling Rights ---
    castling_rights = np.uint8(0)
    if len(parts) > 2 and parts[2] != '-':
        for char in parts[2]:
            castling_rights |= CASTLING_MAP.get(char, 0)

    # --- 4. Parse En Passant Square ---
    en_passant_square = np.int8(-1)
    if len(parts) > 3 and parts[3] != '-':
        en_passant_square = np.int8(SQUARE_MAP.get(parts[3], -1))

    # --- 5. Parse Halfmove and Fullmove Clock (with defaults) ---
    halfmove_clock = np.uint8(0)
    if len(parts) > 4:
        try:
            halfmove_clock = np.uint8(int(parts[4]))
        except (ValueError, IndexError):
            pass # Keep default

    # fullmove_number is not used by the engine state, but is part of the FEN spec

    # --- 6. Assemble Final State Tuples ---
    white_occupancy = final_piece_bbs[0] | final_piece_bbs[1] | final_piece_bbs[2] | \
                      final_piece_bbs[3] | final_piece_bbs[4] | final_piece_bbs[5]
    black_occupancy = final_piece_bbs[6] | final_piece_bbs[7] | final_piece_bbs[8] | \
                      final_piece_bbs[9] | final_piece_bbs[10] | final_piece_bbs[11]

    occupancy_bbs = (white_occupancy, black_occupancy, white_occupancy | black_occupancy)

    # Create a temporary game_state without the hash to compute the initial hash
    temp_game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, np.uint64(0))
    zobrist_key = compute_initial_hash(final_piece_bbs, temp_game_state)

    game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key)

    return final_piece_bbs, occupancy_bbs, game_state
