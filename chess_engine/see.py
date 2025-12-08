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
from chess_engine.bitboard_utils import find_piece_type_on_square, get_between_bb, count_bits

# 預計算棋子價值，包含 King 作為最大值以防止被吃
PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)

@numba.njit(numba.uint64(numba.uint64, numba.uint64), cache=True, inline='always')
def _get_least_significant_bit(bb, _):
    """
    Returns the least significant bit (LSB) as a bitboard.
    Uses (bb & -bb) trick but since bb is unsigned, we use (~bb + 1) logic implicitly.
    Numba requires some care with signed/unsigned.
    """
    if bb == 0:
        return np.uint64(0)
    return bb & ((~bb) + np.uint64(1))

@numba.njit(
    numba.types.Tuple((numba.uint64, numba.uint64))(
        numba.uint8,
        piece_bbs_signature,
        numba.uint64,
        numba.uint64
    ),
    cache=True,
    inline='always'
)
def _get_pinners_and_blockers(side, piece_bbs, occupancy, king_sq_bb):
    """
    計算牽制者 (Pinners) 和 阻擋者 (Blockers)。
    """
    pinners = np.uint64(0)
    blockers = np.uint64(0)

    if king_sq_bb == 0:
        return pinners, blockers

    king_sq = get_lsb_index(king_sq_bb)
    enemy = 1 - side

    enemy_rooks_queens = (piece_bbs[ROOK + enemy * 6] | piece_bbs[QUEEN + enemy * 6])
    enemy_bishops_queens = (piece_bbs[BISHOP + enemy * 6] | piece_bbs[QUEEN + enemy * 6])

    if (enemy_rooks_queens | enemy_bishops_queens) == 0:
        return pinners, blockers

    snipers = get_rook_attacks(king_sq, np.uint64(0)) & enemy_rooks_queens
    snipers |= get_bishop_attacks(king_sq, np.uint64(0)) & enemy_bishops_queens

    while snipers:
        sniper_sq_bb = _get_least_significant_bit(snipers, 0)
        sniper_sq = get_lsb_index(sniper_sq_bb)
        snipers ^= sniper_sq_bb

        between = get_between_bb(king_sq, sniper_sq) & occupancy

        if between != 0 and count_bits(between) == 1:
            blockers |= between
            if (between & (piece_bbs[PAWN + side*6] | piece_bbs[KNIGHT + side*6] |
                           piece_bbs[BISHOP + side*6] | piece_bbs[ROOK + side*6] |
                           piece_bbs[QUEEN + side*6])):
                 pinners |= sniper_sq_bb

    return pinners, blockers

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
    靜態交換評估 (SEE) - 基於 Stockfish 算法重寫。
    """
    victim_type = find_piece_type_on_square(piece_bbs, to_sq)
    if victim_type == -1:
        return 0

    victim_val = PIECE_VALUES[victim_type % 6]
    attacker_type = find_piece_type_on_square(piece_bbs, from_sq)
    attacker_val = PIECE_VALUES[attacker_type % 6]

    gain = np.zeros(32, dtype=np.int32)
    d = 0
    gain[d] = victim_val

    occupancy = occupancy_bbs[2]
    from_bb = BB_SQUARES[from_sq]

    stm = side_to_move
    king_sq_bb_stm = piece_bbs[KING + stm * 6]
    king_sq_bb_opp = piece_bbs[KING + (1-stm) * 6]
    
    pinners_stm, blockers_stm = _get_pinners_and_blockers(stm, piece_bbs, occupancy, king_sq_bb_stm)
    pinners_opp, blockers_opp = _get_pinners_and_blockers(1-stm, piece_bbs, occupancy, king_sq_bb_opp)
    
    occupancy ^= from_bb
    
    attackers = (
        (get_bishop_attacks(to_sq, occupancy) & (piece_bbs[BISHOP] | piece_bbs[QUEEN])) |
        (get_rook_attacks(to_sq, occupancy) & (piece_bbs[ROOK] | piece_bbs[QUEEN])) |
        (KNIGHT_ATTACKS[to_sq] & piece_bbs[KNIGHT]) |
        (PAWN_ATTACKS[BLACK, to_sq] & piece_bbs[PAWN]) |
        (KING_ATTACKS[to_sq] & piece_bbs[KING])
    )
    attackers |= (
        (get_bishop_attacks(to_sq, occupancy) & (piece_bbs[BISHOP + 6] | piece_bbs[QUEEN + 6])) |
        (get_rook_attacks(to_sq, occupancy) & (piece_bbs[ROOK + 6] | piece_bbs[QUEEN + 6])) |
        (KNIGHT_ATTACKS[to_sq] & piece_bbs[KNIGHT + 6]) |
        (PAWN_ATTACKS[WHITE, to_sq] & piece_bbs[PAWN + 6]) |
        (KING_ATTACKS[to_sq] & piece_bbs[KING + 6])
    )

    attackers &= ~from_bb
    
    curr_stm = 1 - stm
    
    while True:
        d += 1

        stm_attackers = attackers & (
            (piece_bbs[PAWN + curr_stm*6] | piece_bbs[KNIGHT + curr_stm*6] |
             piece_bbs[BISHOP + curr_stm*6] | piece_bbs[ROOK + curr_stm*6] |
             piece_bbs[QUEEN + curr_stm*6] | piece_bbs[KING + curr_stm*6])
        )

        curr_pinners = pinners_stm if curr_stm == side_to_move else pinners_opp
        curr_blockers = blockers_stm if curr_stm == side_to_move else blockers_opp

        if (curr_pinners & occupancy) != 0:
            stm_attackers &= ~curr_blockers

        if stm_attackers == 0:
            break
            
        lva_bb = np.uint64(0)
        lva_val = 0
        lva_type = -1
        
        if (stm_attackers & piece_bbs[PAWN + curr_stm*6]):
            bb = stm_attackers & piece_bbs[PAWN + curr_stm*6]
            lva_bb = _get_least_significant_bit(bb, 0)
            lva_val = PIECE_VALUES[PAWN]
            lva_type = PAWN
        elif (stm_attackers & piece_bbs[KNIGHT + curr_stm*6]):
            bb = stm_attackers & piece_bbs[KNIGHT + curr_stm*6]
            lva_bb = _get_least_significant_bit(bb, 0)
            lva_val = PIECE_VALUES[KNIGHT]
            lva_type = KNIGHT
        elif (stm_attackers & piece_bbs[BISHOP + curr_stm*6]):
            bb = stm_attackers & piece_bbs[BISHOP + curr_stm*6]
            lva_bb = _get_least_significant_bit(bb, 0)
            lva_val = PIECE_VALUES[BISHOP]
            lva_type = BISHOP
        elif (stm_attackers & piece_bbs[ROOK + curr_stm*6]):
            bb = stm_attackers & piece_bbs[ROOK + curr_stm*6]
            lva_bb = _get_least_significant_bit(bb, 0)
            lva_val = PIECE_VALUES[ROOK]
            lva_type = ROOK
        elif (stm_attackers & piece_bbs[QUEEN + curr_stm*6]):
            bb = stm_attackers & piece_bbs[QUEEN + curr_stm*6]
            lva_bb = _get_least_significant_bit(bb, 0)
            lva_val = PIECE_VALUES[QUEEN]
            lva_type = QUEEN
        elif (stm_attackers & piece_bbs[KING + curr_stm*6]):
            bb = stm_attackers & piece_bbs[KING + curr_stm*6]
            lva_bb = _get_least_significant_bit(bb, 0)
            lva_val = PIECE_VALUES[KING]
            lva_type = KING
        else:
            break

        if lva_type == KING:
            opp = 1 - curr_stm
            opp_attackers = attackers & (
                (piece_bbs[PAWN + opp*6] | piece_bbs[KNIGHT + opp*6] |
                 piece_bbs[BISHOP + opp*6] | piece_bbs[ROOK + opp*6] |
                 piece_bbs[QUEEN + opp*6] | piece_bbs[KING + opp*6])
            )
            if opp_attackers != 0:
                break
        
        gain[d] = attacker_val - gain[d-1]
        attacker_val = lva_val
        occupancy ^= lva_bb
        
        # --- Update X-Ray Attackers ---
        # ANY piece moving can reveal an attacker behind it.
        # So we unconditionally check for revealed attacks.
        
        new_b_attacks = get_bishop_attacks(to_sq, occupancy) & (piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6])
        attackers |= new_b_attacks

        new_r_attacks = get_rook_attacks(to_sq, occupancy) & (piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6])
        attackers |= new_r_attacks
        
        attackers &= ~lva_bb
        curr_stm = 1 - curr_stm
        
        if d >= 30: break

    while d > 1:
        d -= 1
        gain[d-1] = -max(-gain[d-1], gain[d])

    return gain[0]
