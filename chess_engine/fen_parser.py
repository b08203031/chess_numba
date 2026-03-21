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
    解析 FEN (Forsyth-Edwards Notation) 字串並返回引擎 NumPy 陣列格式的棋盤狀態。

    Args:
        fen_string (str): 代表棋盤局面的 FEN 字串。

    Returns:
        tuple: 包含三個 NumPy 陣列的元組 (piece_bbs, occupancy_bbs, game_state)。
            - piece_bbs: 12 個位元棋盤，分別代表每種棋子的位置。
            - occupancy_bbs: 3 個位元棋盤，分別代表白方、黑方和所有棋子的佔用情況。
            - game_state: 包含遊戲狀態資訊（行棋方、易位權、吃過路兵方格、半步鐘、Zobrist 鍵值）。
    """
    parts = fen_string.split()

    # --- 1. Parse Piece Placements / 解析棋子位置 ---
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

    # --- 2. Parse Side to Move / 解析行棋方 ---
    side_to_move = np.uint64(WHITE if parts[1] == 'w' else BLACK)

    # --- 3. Parse Castling Rights / 解析王車易位權限 ---
    castling_rights = np.uint64(0)
    if len(parts) > 2 and parts[2] != '-':
        for char in parts[2]:
            castling_rights |= np.uint64(CASTLING_MAP.get(char, 0))

    # --- 4. Parse En Passant Square / 解析吃過路兵方格 ---
    # using 64 as sentinel for 'no EP square' / 使用 64 作為「無吃過路兵方格」的標記
    en_passant_square = np.uint64(64)
    if len(parts) > 3 and parts[3] != '-':
        en_passant_square = np.uint64(SQUARE_MAP.get(parts[3], 64))

    # --- 5. Parse Halfmove Clock / 解析半步鐘 ---
    halfmove_clock = np.uint64(0)
    if len(parts) > 4:
        try:
            halfmove_clock = np.uint64(int(parts[4]))
        except (ValueError, IndexError):
            pass  # Keep default / 保持預設值

    # --- 6. Assemble Final State Arrays / 組裝最終狀態陣列 ---
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
    # 創建一個臨時的 game_state 陣列來計算初始哈希
    temp_game_state_arr = np.array([
        side_to_move, castling_rights, en_passant_square, halfmove_clock, np.uint64(0)
    ], dtype=np.uint64)

    # Pass NumPy arrays directly to the JIT'd function / 直接將 NumPy 陣列傳遞給 JIT 函數
    zobrist_key = compute_initial_hash(piece_bbs, temp_game_state_arr)

    # Compute initial key sets
    pawn_key = np.uint64(0)
    minor_key = np.uint64(0)
    non_pawn_white_key = np.uint64(0)
    non_pawn_black_key = np.uint64(0)
    
    from chess_engine.zobrist import PIECE_SQUARE_KEYS, get_lsb_index
    
    # helper for keys
    def _add_pieces(pieces_bb, p_idx, is_pawn=False, is_minor=False, is_white=False):
        nonlocal pawn_key, minor_key, non_pawn_white_key, non_pawn_black_key
        temp_bb = pieces_bb
        while temp_bb:
            sq = get_lsb_index(temp_bb)
            key = PIECE_SQUARE_KEYS[p_idx, sq]
            if is_pawn:
                pawn_key ^= key
            elif is_minor:
                minor_key ^= key
            if not is_pawn:
                if is_white:
                    non_pawn_white_key ^= key
                else:
                    non_pawn_black_key ^= key
            temp_bb &= temp_bb - np.uint64(1)
    
    # White Pawns (Index 0)
    # Add all pieces systematically
    _add_pieces(piece_bbs[0], 0, is_pawn=True, is_white=True)          # P
    _add_pieces(piece_bbs[1], 1, is_minor=True, is_white=True)         # N
    _add_pieces(piece_bbs[2], 2, is_minor=True, is_white=True)         # B
    _add_pieces(piece_bbs[3], 3, is_white=True)                        # R
    _add_pieces(piece_bbs[4], 4, is_white=True)                        # Q
    _add_pieces(piece_bbs[5], 5, is_white=True)                        # K
    
    _add_pieces(piece_bbs[6], 6, is_pawn=True, is_white=False)         # p
    _add_pieces(piece_bbs[7], 7, is_minor=True, is_white=False)        # n
    _add_pieces(piece_bbs[8], 8, is_minor=True, is_white=False)        # b
    _add_pieces(piece_bbs[9], 9, is_white=False)                       # r
    _add_pieces(piece_bbs[10], 10, is_white=False)                     # q
    _add_pieces(piece_bbs[11], 11, is_white=False)                     # k

    game_state = np.array([
        side_to_move,
        castling_rights,
        en_passant_square,
        halfmove_clock,
        zobrist_key,
        pawn_key,
        minor_key,
        non_pawn_white_key,
        non_pawn_black_key
    ], dtype=np.uint64)

    # --- Sanity Checks / 健全性檢查 ---
    if (occupancy_bbs[0] | occupancy_bbs[1]) != occupancy_bbs[2]:
        raise ValueError("FEN Parser Error: Occupancy calculation is incorrect.")
    if np.count_nonzero(occupancy_bbs[0] & occupancy_bbs[1]) != 0:
        raise ValueError("FEN Parser Error: White and black pieces overlap on the same square.")

    return piece_bbs, occupancy_bbs, game_state
