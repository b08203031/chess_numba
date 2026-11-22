# chess_engine/fen_parser.py
import numpy as np
from chess_engine.classical.zobrist import compute_initial_hash

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

    # Bug #18: drop castling rights whose king/rook are not on their home squares. A stale right
    # would make make_move XOR a rook out of thin air (breaking the occupancy invariants).
    # 移除王/車不在原位的殘留易位權，避免 make_move 憑空生出車而破壞佔用不變量。
    def _has_piece(piece_index, sq):
        return bool((int(piece_bbs[piece_index]) >> sq) & 1)

    _castling_requirements = (
        (1, 5, 4, 3, 7),     # K: white king e1, white rook h1
        (2, 5, 4, 3, 0),     # Q: white king e1, white rook a1
        (4, 11, 60, 9, 63),  # k: black king e8, black rook h8
        (8, 11, 60, 9, 56),  # q: black king e8, black rook a8
    )
    valid_castling = 0
    for right, king_idx, king_sq, rook_idx, rook_sq in _castling_requirements:
        if (int(castling_rights) & right) and _has_piece(king_idx, king_sq) and _has_piece(rook_idx, rook_sq):
            valid_castling |= right
    castling_rights = np.uint64(valid_castling)

    # --- 4. Parse En Passant Square / 解析吃過路兵方格 ---
    # using 64 as sentinel for 'no EP square' / 使用 64 作為「無吃過路兵方格」的標記
    en_passant_square = np.uint64(64)
    if len(parts) > 3 and parts[3] != '-':
        ep_idx = SQUARE_MAP.get(parts[3], 64)
        # Bug #18: a phantom EP square (wrong rank, no double-pushed pawn behind it, or occupied
        # squares) would corrupt occupancy in make_move; ignore it like SF's position validation.
        # 虛構的 EP 格（錯誤橫排／後方無剛雙步之兵／格子被佔）會破壞佔用，故忽略。
        if ep_idx != 64:
            occ_all_int = 0
            for bb in piece_bbs:
                occ_all_int |= int(bb)
            if side_to_move == WHITE:
                # Black just double-pushed: ep on rank 6, black pawn on rank 5, origin (rank 7) empty.
                ok = (
                    ep_idx // 8 == 5
                    and _has_piece(6, ep_idx - 8)
                    and not (occ_all_int >> ep_idx) & 1
                    and not (occ_all_int >> (ep_idx + 8)) & 1
                )
            else:
                # White just double-pushed: ep on rank 3, white pawn on rank 4, origin (rank 2) empty.
                ok = (
                    ep_idx // 8 == 2
                    and _has_piece(0, ep_idx + 8)
                    and not (occ_all_int >> ep_idx) & 1
                    and not (occ_all_int >> (ep_idx - 8)) & 1
                )
            if ok:
                en_passant_square = np.uint64(ep_idx)

    # --- 5. Parse Halfmove Clock / 解析半步鐘 ---
    halfmove_clock = np.uint64(0)
    if len(parts) > 4:
        try:
            # Bug #18: negative values raised OverflowError under NumPy 2.x; values > 255 would wrap
            # the uint8 old_halfmove_clock in make_move. Anything >= 100 is already a draw anyway.
            halfmove_clock = np.uint64(min(max(0, int(parts[4])), 255))
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
    
    from chess_engine.classical.zobrist import PIECE_SQUARE_KEYS, get_lsb_index
    
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
    assert (occupancy_bbs[0] | occupancy_bbs[1]) == occupancy_bbs[2], \
        "FEN Parser Error: Occupancy calculation is incorrect."
    assert np.count_nonzero(occupancy_bbs[0] & occupancy_bbs[1]) == 0, \
        "FEN Parser Error: White and black pieces overlap on the same square."

    return piece_bbs, occupancy_bbs, game_state
