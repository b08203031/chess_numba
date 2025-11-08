# chess_engine/board_operations.py

import numpy as np
import numba as numba
from chess_engine.move import (
    get_from_square, get_to_square, get_special_move_flag, get_promotion_piece,
    SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING
)
from chess_engine.zobrist import (
    PIECE_SQUARE_KEYS, SIDE_TO_MOVE_KEY, EN_PASSANT_FILE_KEYS, CASTLING_RIGHTS_KEYS,
    compute_initial_hash
)
from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, unmake_info_signature
)

# --- Piece Type Constants ---
PAWN, ROOK, KING = 0, 3, 5
WHITE, BLACK = 0, 1

# --- Castling Rights Constants ---
WK_CASTLE, WQ_CASTLE, BK_CASTLE, BQ_CASTLE = 1, 2, 4, 8
ALL_CASTLING_RIGHTS = 15

# --- Pre-computed Castling Update Masks ---
CASTLING_UPDATE_MASK = np.full(64, ALL_CASTLING_RIGHTS, dtype=np.uint8)
CASTLING_UPDATE_MASK[0]  -= WQ_CASTLE
CASTLING_UPDATE_MASK[4]  -= (WK_CASTLE | WQ_CASTLE)
CASTLING_UPDATE_MASK[7]  -= WK_CASTLE
CASTLING_UPDATE_MASK[56] -= BQ_CASTLE
CASTLING_UPDATE_MASK[60] -= (BK_CASTLE | BQ_CASTLE)
CASTLING_UPDATE_MASK[63] -= BK_CASTLE

@numba.jit(numba.int8(piece_bbs_signature, numba.int8, numba.uint8), nopython=True, inline='always')
def find_piece_type_for_square(piece_bbs: tuple, square: int, color: int) -> int:
    """Finds which piece type occupies a given square.

    Args:
        piece_bbs: A tuple of 12 bitboards representing the pieces.
        square: The square to check.
        color: The color of the piece.

    Returns:
        The piece type of the piece on the square, or -1 if no piece is on the square.
    """
    bit = np.uint64(1) << square
    start_index = color * 6
    for i in range(6):
        if piece_bbs[start_index + i] & bit:
            return i
    return -1

@numba.jit(numba.types.Tuple((piece_bbs_signature, occupancy_bbs_signature, game_state_signature, unmake_info_signature))(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.uint16), nopython=True)
def make_move(piece_bbs: tuple, occupancy_bbs: tuple, game_state: tuple, move: np.uint16):
    """
    Applies a move and returns new state tuples and unmake_info.

    Args:
        piece_bbs: A tuple of 12 bitboards representing the pieces.
        occupancy_bbs: A tuple of 3 bitboards representing the occupancy of the board.
        game_state: A tuple representing the current game state.
        move: The move to make.

    Returns:
        A tuple containing the new piece_bbs, new_occupancy_bbs, new_game_state, and unmake_info.
    """
    new_piece_bbs = list(piece_bbs)
    side, current_castling_rights, current_ep_square, current_halfmove_clock, zobrist_key = game_state
    
    # ✅ 立即保存 - 在任何 XOR 操作之前
    original_zobrist_key = zobrist_key
    
    key = zobrist_key

    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    flag = get_special_move_flag(move)

    moving_piece_type = find_piece_type_for_square(piece_bbs, from_sq, side)
    moving_piece_bb_idx = side * 6 + moving_piece_type

    captured_piece_type = np.int8(-1)

    # --- Zobrist & Bitboard Updates for Piece Movement ---
    key ^= PIECE_SQUARE_KEYS[moving_piece_bb_idx, from_sq]
    key ^= PIECE_SQUARE_KEYS[moving_piece_bb_idx, to_sq]
    new_piece_bbs[moving_piece_bb_idx] ^= (np.uint64(1) << from_sq) | (np.uint64(1) << to_sq)

    is_capture = False
    if flag != SPECIAL_MOVE_FLAG_EN_PASSANT and (occupancy_bbs[2] & (np.uint64(1) << to_sq)):
        is_capture = True
        opponent_color = 1 - side
        captured_piece_type = find_piece_type_for_square(piece_bbs, to_sq, opponent_color)
        captured_piece_bb_idx = opponent_color * 6 + captured_piece_type

        new_piece_bbs[captured_piece_bb_idx] &= ~(np.uint64(1) << to_sq)
        key ^= PIECE_SQUARE_KEYS[captured_piece_bb_idx, to_sq]

    # --- Handle Special Moves ---
    if flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_piece_type = get_promotion_piece(move) + 1 # PROMO_KNIGHT is 0, maps to piece type 1
        promo_piece_bb_idx = side * 6 + promo_piece_type

        new_piece_bbs[moving_piece_bb_idx] &= ~(np.uint64(1) << to_sq)
        new_piece_bbs[promo_piece_bb_idx] |= (np.uint64(1) << to_sq)

        key ^= PIECE_SQUARE_KEYS[moving_piece_bb_idx, to_sq] # XOR out pawn
        key ^= PIECE_SQUARE_KEYS[promo_piece_bb_idx, to_sq]   # XOR in promoted piece

    elif flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
        is_capture = True
        captured_piece_type = np.int8(PAWN)
        opponent_color = 1 - side
        captured_pawn_bb_idx = opponent_color * 6 + PAWN

        captured_pawn_sq = to_sq + (8 if side == BLACK else -8)

        new_piece_bbs[captured_pawn_bb_idx] &= ~(np.uint64(1) << captured_pawn_sq)
        key ^= PIECE_SQUARE_KEYS[captured_pawn_bb_idx, captured_pawn_sq]

    elif flag == SPECIAL_MOVE_FLAG_CASTLING:
        king_side_castle = to_sq > from_sq
        rook_from_sq, rook_to_sq = ((7, 5) if king_side_castle else (0, 3)) if side == WHITE else ((63, 61) if king_side_castle else (56, 59))
        rook_bb_idx = side * 6 + ROOK

        new_piece_bbs[rook_bb_idx] ^= (np.uint64(1) << rook_from_sq) | (np.uint64(1) << rook_to_sq)
        key ^= PIECE_SQUARE_KEYS[rook_bb_idx, rook_from_sq]
        key ^= PIECE_SQUARE_KEYS[rook_bb_idx, rook_to_sq]

    # --- Update Castling, En Passant, Side to Move Keys ---
    new_castling_rights = current_castling_rights & CASTLING_UPDATE_MASK[from_sq] & CASTLING_UPDATE_MASK[to_sq]

    if new_castling_rights != current_castling_rights:
        key ^= CASTLING_RIGHTS_KEYS[current_castling_rights]
        key ^= CASTLING_RIGHTS_KEYS[new_castling_rights]

    # --- En Passant Square Calculation (Rewritten for Numba compatibility) ---
    new_ep_square = np.int8(-1)
    if moving_piece_type == PAWN:
        # White pawn double push
        if side == WHITE and (to_sq - from_sq) == 16:
            new_ep_square = np.int8(from_sq + 8)
        # Black pawn double push
        elif side == BLACK and (from_sq - to_sq) == 16:
            new_ep_square = np.int8(from_sq - 8)

    if new_ep_square != current_ep_square:
        if current_ep_square != -1:
            key ^= EN_PASSANT_FILE_KEYS[current_ep_square % 8]
        if new_ep_square != -1:
            key ^= EN_PASSANT_FILE_KEYS[new_ep_square % 8]

    key ^= SIDE_TO_MOVE_KEY

    # --- Update Clocks & Finalize State ---
    new_halfmove_clock = np.uint8(0) if (moving_piece_type == PAWN or is_capture) else np.uint8(current_halfmove_clock + 1)

    final_piece_bbs = (
        new_piece_bbs[0], new_piece_bbs[1], new_piece_bbs[2], new_piece_bbs[3],
        new_piece_bbs[4], new_piece_bbs[5], new_piece_bbs[6], new_piece_bbs[7],
        new_piece_bbs[8], new_piece_bbs[9], new_piece_bbs[10], new_piece_bbs[11]
    )
    white_occupancy = final_piece_bbs[0]|final_piece_bbs[1]|final_piece_bbs[2]|final_piece_bbs[3]|final_piece_bbs[4]|final_piece_bbs[5]
    black_occupancy = final_piece_bbs[6]|final_piece_bbs[7]|final_piece_bbs[8]|final_piece_bbs[9]|final_piece_bbs[10]|final_piece_bbs[11]

    new_occupancy_bbs = (white_occupancy, black_occupancy, white_occupancy | black_occupancy)

    new_game_state = (
        np.uint8(1 - side),
        np.uint8(new_castling_rights),
        np.int8(new_ep_square),
        np.uint8(new_halfmove_clock),
        np.uint64(key)
    )
    unmake_info = (
        np.int8(captured_piece_type),
        np.uint8(current_castling_rights),
        np.int8(current_ep_square),
        np.uint8(current_halfmove_clock),
        np.uint64(original_zobrist_key)
    )

    return final_piece_bbs, new_occupancy_bbs, new_game_state, unmake_info

@numba.jit(numba.types.Tuple((piece_bbs_signature, occupancy_bbs_signature, game_state_signature))(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.uint16, unmake_info_signature
), nopython=True)
def unmake_move(piece_bbs: tuple, occupancy_bbs: tuple, game_state: tuple, move: np.uint16, unmake_info: tuple):
    """
    Reverts a move incrementally using stored Zobrist key.

    Time Complexity: O(1) - Only bitboard restoration and direct key assignment
    """
    new_piece_bbs = list(piece_bbs)

    # ========== 解析狀態 ==========
    side = np.uint8(1 - game_state[0])  # game_state[0] 存儲對手顏色
    captured_piece_type, old_castling_rights, old_ep_square, old_halfmove_clock, old_zobrist_key = unmake_info

    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    flag = get_special_move_flag(move)

    # ========== 找到移動的棋子類型 ==========
    moving_piece_type = find_piece_type_for_square(piece_bbs, to_sq, side)

    # 升變時，目的地的是升變後的棋子，但移動方是兵
    if flag == SPECIAL_MOVE_FLAG_PROMOTION:
        moving_piece_type = PAWN

    moving_piece_bb_idx = side * 6 + moving_piece_type

    # ========== 步驟 1: 恢復基本移動 ==========
    # 將棋子從目的地 XOR 回來源地
    new_piece_bbs[moving_piece_bb_idx] ^= (np.uint64(1) << from_sq) | (np.uint64(1) << to_sq)

    # ========== 步驟 2: 恢復特殊移動 ==========
    if flag == SPECIAL_MOVE_FLAG_PROMOTION:
        # 移除升變後的棋子
        promo_piece_type = get_promotion_piece(move) + 1
        promo_piece_bb_idx = side * 6 + promo_piece_type
        new_piece_bbs[promo_piece_bb_idx] &= ~(np.uint64(1) << to_sq)

    elif flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
        # 恢復被吃掉的對方兵
        opponent_color = 1 - side
        captured_pawn_bb_idx = opponent_color * 6 + PAWN
        captured_pawn_sq = to_sq + (8 if side == BLACK else -8)
        new_piece_bbs[captured_pawn_bb_idx] |= (np.uint64(1) << captured_pawn_sq)

    elif flag == SPECIAL_MOVE_FLAG_CASTLING:
        # 恢復車的位置
        king_side_castle = to_sq > from_sq
        if side == WHITE:
            rook_from_sq = 7 if king_side_castle else 0
            rook_to_sq = 5 if king_side_castle else 3
        else:
            rook_from_sq = 63 if king_side_castle else 56
            rook_to_sq = 61 if king_side_castle else 59
        
        rook_bb_idx = side * 6 + ROOK
        new_piece_bbs[rook_bb_idx] ^= (np.uint64(1) << rook_from_sq) | (np.uint64(1) << rook_to_sq)

    # ========== 步驟 3: 恢復捕獲的棋子 ==========
    if captured_piece_type != -1 and flag != SPECIAL_MOVE_FLAG_EN_PASSANT:
        opponent_color = 1 - side
        captured_piece_bb_idx = opponent_color * 6 + captured_piece_type
        new_piece_bbs[captured_piece_bb_idx] |= (np.uint64(1) << to_sq)

    # ========== 步驟 4: 重建佔用位元板 ==========
    final_piece_bbs = (
        new_piece_bbs[0], new_piece_bbs[1], new_piece_bbs[2], new_piece_bbs[3],
        new_piece_bbs[4], new_piece_bbs[5], new_piece_bbs[6], new_piece_bbs[7],
        new_piece_bbs[8], new_piece_bbs[9], new_piece_bbs[10], new_piece_bbs[11]
    )

    white_occupancy = (final_piece_bbs[0] | final_piece_bbs[1] | final_piece_bbs[2] | 
                     final_piece_bbs[3] | final_piece_bbs[4] | final_piece_bbs[5])
    black_occupancy = (final_piece_bbs[6] | final_piece_bbs[7] | final_piece_bbs[8] | 
                     final_piece_bbs[9] | final_piece_bbs[10] | final_piece_bbs[11])

    restored_occupancy_bbs = (white_occupancy, black_occupancy, white_occupancy | black_occupancy)

    # ========== 步驟 5: 恢復遊戲狀態 ==========
    # ✅ 直接使用存儲的舊鍵 - O(1) 無需計算！
    restored_game_state = (
        np.uint8(side),
        np.uint8(old_castling_rights),
        np.int8(old_ep_square),
        np.uint8(old_halfmove_clock),
        np.uint64(old_zobrist_key)  # 直接賦值，完成
    )

    return final_piece_bbs, restored_occupancy_bbs, restored_game_state


@numba.jit(numba.types.Tuple((piece_bbs_signature, occupancy_bbs_signature, game_state_signature))(piece_bbs_signature, occupancy_bbs_signature, game_state_signature), nopython=True)
def make_null_move(piece_bbs: tuple, occupancy_bbs: tuple, game_state: tuple):
    """
    Performs a null move: flips the side to move and clears the en passant square.
    """
    side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key = game_state

    new_key = zobrist_key ^ SIDE_TO_MOVE_KEY
    if en_passant_square != -1:
        new_key ^= EN_PASSANT_FILE_KEYS[en_passant_square % 8]

    new_game_state = (
        np.uint8(1 - side_to_move),
        castling_rights,
        np.int8(-1),
        np.uint8(halfmove_clock + 1), # halfmove clock increments
        new_key
    )
    
    return piece_bbs, occupancy_bbs, new_game_state
