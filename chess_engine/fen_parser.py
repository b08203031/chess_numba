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
    chr(ord('a') + f) + str(r + 1): r * 8 + f
    for r in range(8) for f in range(8)
}

def parse_fen(fen_string: str):
    """
    Parses a FEN string and returns the board state in the engine's NumPy array format.

    Args:
        fen_string: The FEN string representing the board position.

    Returns:
        A tuple containing (piece_bbs, occupancy_bbs, game_state) as NumPy arrays.
    """
    parts = fen_string.split()

    # --- 1. Parse Piece Placements ---
    piece_bbs = np.zeros(12, dtype=np.uint64)
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

    # --- 2. Parse Side to Move ---
    side_to_move = np.uint64(WHITE if parts[1] == 'w' else BLACK)

    # --- 3. Parse Castling Rights ---
    castling_rights = np.uint64(0)
    if len(parts) > 2 and parts[2] != '-':
        for char in parts[2]:
            castling_rights |= np.uint64(CASTLING_MAP.get(char, 0))

    # --- 4. Parse En Passant Square (using 64 as sentinel for 'no EP square') ---
    en_passant_square = np.uint64(64)
    if len(parts) > 3 and parts[3] != '-':
        en_passant_square = np.uint64(SQUARE_MAP.get(parts[3], 64))

    # --- 5. Parse Halfmove Clock ---
    halfmove_clock = np.uint64(0)
    if len(parts) > 4:
        try:
            halfmove_clock = np.uint64(int(parts[4]))
        except (ValueError, IndexError):
            pass  # Keep default

    # --- 6. Assemble Final State Arrays ---
    white_occupancy = piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | \
                      piece_bbs[3] | piece_bbs[4] | piece_bbs[5]
    black_occupancy = piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | \
                      piece_bbs[9] | piece_bbs[10] | piece_bbs[11]

    occupancy_bbs = np.array([
        white_occupancy,
        black_occupancy,
        white_occupancy | black_occupancy
    ], dtype=np.uint64)

    # Create a temporary game_state array to compute the initial hash
    temp_game_state_arr = np.array([
        side_to_move, castling_rights, en_passant_square, halfmove_clock, np.uint64(0)
    ], dtype=np.uint64)

    # Pass NumPy arrays directly to the JIT'd function
    zobrist_key = compute_initial_hash(piece_bbs, temp_game_state_arr)

    game_state = np.array([
        side_to_move,
        castling_rights,
        en_passant_square,
        halfmove_clock,
        zobrist_key
    ], dtype=np.uint64)

    # --- Sanity Checks ---
    assert (occupancy_bbs[0] | occupancy_bbs[1]) == occupancy_bbs[2], \
        "FEN Parser Error: Occupancy calculation is incorrect."
    assert np.count_nonzero(occupancy_bbs[0] & occupancy_bbs[1]) == 0, \
        "FEN Parser Error: White and black pieces overlap on the same square."

    return piece_bbs, occupancy_bbs, game_state
