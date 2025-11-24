# chess_engine/see.py
import numba
import numpy as np

from chess_engine.engine_types import (
    piece_bbs_signature,
    occupancy_bbs_signature,
    WHITE,
    BLACK,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
)
from chess_engine.constants import BB_SQUARES, MG_MATERIAL_VALUES
from chess_engine.move_generator import (
    get_bishop_attacks,
    get_rook_attacks,
    get_queen_attacks,
    KNIGHT_ATTACKS,
    KING_ATTACKS,
    PAWN_ATTACKS,
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.bitboard_utils import find_piece_type_on_square


@numba.njit(
    numba.types.Tuple((numba.uint64, numba.uint64))(
        numba.uint8,
        piece_bbs_signature,
        numba.uint64,
    ),
    cache=True,
)
def _get_attackers_to_square(square, piece_bbs, occupancy):
    """
    獲取所有攻擊給定方格的位元棋盤。
    此函數設計用於 SEE（靜態交換評估），並在動態佔用位元棋盤上操作。

    Args:
        square (int): 目標方格。
        piece_bbs (np.ndarray): 棋子位元棋盤。
        occupancy (np.uint64): 當前模擬的佔用位元棋盤。

    Returns:
        tuple: (white_attackers, black_attackers) 兩個位元棋盤。
    """
    # Note: Pawn attacks don't depend on occupancy
    # 注意：兵的攻擊不依賴於佔用位元棋盤（因為是固定的偏移量）
    white_attackers = PAWN_ATTACKS[BLACK][square] & piece_bbs[PAWN]
    black_attackers = PAWN_ATTACKS[WHITE][square] & piece_bbs[PAWN + 6]

    # Knights / 騎士
    knight_attack_bb = KNIGHT_ATTACKS[square]
    white_attackers |= knight_attack_bb & piece_bbs[KNIGHT]
    black_attackers |= knight_attack_bb & piece_bbs[KNIGHT + 6]

    # Bishops and Queens (Diagonals) / 主教和后（斜線）
    bishop_queen_bb = piece_bbs[BISHOP] | piece_bbs[QUEEN]
    black_bishop_queen_bb = piece_bbs[BISHOP + 6] | piece_bbs[QUEEN + 6]
    bishop_attacks_bb = get_bishop_attacks(square, occupancy)
    white_attackers |= bishop_attacks_bb & bishop_queen_bb
    black_attackers |= bishop_attacks_bb & black_bishop_queen_bb

    # Rooks and Queens (Files/Ranks) / 車和后（直線/橫線）
    rook_queen_bb = piece_bbs[ROOK] | piece_bbs[QUEEN]
    black_rook_queen_bb = piece_bbs[ROOK + 6] | piece_bbs[QUEEN + 6]
    rook_attacks_bb = get_rook_attacks(square, occupancy)
    white_attackers |= rook_attacks_bb & rook_queen_bb
    black_attackers |= rook_attacks_bb & black_rook_queen_bb

    # Kings / 王
    king_attack_bb = KING_ATTACKS[square]
    white_attackers |= king_attack_bb & piece_bbs[KING]
    black_attackers |= king_attack_bb & piece_bbs[KING + 6]

    return white_attackers, black_attackers


@numba.njit(
    numba.int32(
        piece_bbs_signature,
        occupancy_bbs_signature,
        numba.uint8,
        numba.uint8,
        numba.uint8,
    ),
    cache=True,
)
def see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq):
    """
    靜態交換評估 (SEE)。
    這是一個輕量級、基於位元棋盤的實現，避免了複製棋盤狀態和昂貴的 make_move 調用。
    它在臨時的佔用位元棋盤上模擬吃子，並動態發現 X-Ray 攻擊。

    Args:
        piece_bbs (np.ndarray): 棋子位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        side_to_move (int): 當前行棋方。
        from_sq (int): 起始方格。
        to_sq (int): 目標方格。

    Returns:
        int: 交換後的材質增益（分）。
    """
    gain = np.zeros(32, dtype=np.int32)
    depth = 0

    # Initial victim piece type and value / 初始受害者棋子類型和價值
    victim_piece_type = find_piece_type_on_square(piece_bbs, to_sq)
    # Handle en-passant, where the victim is on a different square
    # 處理吃過路兵，受害者在不同的方格
    if victim_piece_type == -1:
        victim_piece_type = PAWN + (6 if side_to_move == WHITE else 0)
    gain[depth] = MG_MATERIAL_VALUES[victim_piece_type % 6]

    # --- Lightweight Simulation Setup / 輕量級模擬設置 ---
    current_side = side_to_move
    occupancy = occupancy_bbs[2]
    from_sq_bb = BB_SQUARES[from_sq]

    # Get all initial attackers to the target square / 獲取所有初始攻擊者
    white_attackers, black_attackers = _get_attackers_to_square(
        to_sq, piece_bbs, occupancy
    )
    all_attackers = white_attackers | black_attackers

    # Remove the initial attacker from the occupancy and attacker sets
    # 從佔用和攻擊者集合中移除初始攻擊者（即移動的棋子）
    occupancy ^= from_sq_bb
    all_attackers &= ~from_sq_bb

    while True:
        depth += 1
        # Safety break for very long capture sequences / 防止過長的吃子序列
        if depth >= 32:
            break
            
        current_side = 1 - current_side  # Switch sides / 交換方

        attackers_for_side = black_attackers if current_side == BLACK else white_attackers
        attackers_for_side &= all_attackers

        if not attackers_for_side:
            break

        # --- Find the least valuable attacker / 尋找價值最低的攻擊者 ---
        aggressor_sq = -1
        aggressor_piece_type = -1

        # Find the piece type with the lowest value among the attackers
        # 在攻擊者中尋找價值最低的棋子類型（P -> N -> B -> R -> Q -> K）
        for piece_type in range(PAWN, KING + 1):
            pt_idx = piece_type + (6 if current_side == BLACK else 0)
            piece_attackers = piece_bbs[pt_idx] & attackers_for_side
            if piece_attackers:
                aggressor_sq = get_lsb_index(piece_attackers)
                aggressor_piece_type = piece_type
                break
        
        if aggressor_sq == -1:
            break
            
        aggressor_sq_bb = BB_SQUARES[aggressor_sq]

        # --- Update Gain Array / 更新增益陣列 ---
        gain[depth] = MG_MATERIAL_VALUES[aggressor_piece_type] - gain[depth - 1]

        # --- Update Simulation State / 更新模擬狀態 ---
        occupancy ^= aggressor_sq_bb
        all_attackers &= ~aggressor_sq_bb

        # --- Handle X-Ray Attacks / 處理 X-Ray 攻擊 ---
        # If the removed piece was a slider, we need to check if its removal
        # revealed a new attack from another slider behind it.
        # 如果被移除的棋子是滑動棋子，我們需要檢查它的移除是否暴露了背後另一個滑動棋子的攻擊。
        if aggressor_piece_type == BISHOP or aggressor_piece_type == QUEEN:
            bishop_attacks_bb = get_bishop_attacks(to_sq, occupancy)
            new_w_bishops = bishop_attacks_bb & (piece_bbs[BISHOP] | piece_bbs[QUEEN])
            new_b_bishops = bishop_attacks_bb & (piece_bbs[BISHOP + 6] | piece_bbs[QUEEN + 6])
            
            revealed_attackers = (new_w_bishops | new_b_bishops) & ~all_attackers
            if revealed_attackers:
                all_attackers |= revealed_attackers
                white_attackers |= new_w_bishops
                black_attackers |= new_b_bishops

        if aggressor_piece_type == ROOK or aggressor_piece_type == QUEEN:
            rook_attacks_bb = get_rook_attacks(to_sq, occupancy)
            new_w_rooks = rook_attacks_bb & (piece_bbs[ROOK] | piece_bbs[QUEEN])
            new_b_rooks = rook_attacks_bb & (piece_bbs[ROOK + 6] | piece_bbs[QUEEN + 6])
            
            revealed_attackers = (new_w_rooks | new_b_rooks) & ~all_attackers
            if revealed_attackers:
                all_attackers |= revealed_attackers
                white_attackers |= new_w_rooks
                black_attackers |= new_b_rooks


    # --- Minimax Calculation / 極大極小值計算 ---
    while depth > 1:
        depth -= 1
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])

    return gain[0]
