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
    KNIGHT_ATTACKS,
    KING_ATTACKS,
    PAWN_ATTACKS,
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.bitboard_utils import find_piece_type_on_square

@numba.njit(cache=True)
def _check_on_line(sq1, sq2, sq3):
    """
    Check if sq1, sq2, and sq3 are collinear (on the same rank, file, or diagonal).
    """
    r1, f1 = sq1 // 8, sq1 % 8
    r2, f2 = sq2 // 8, sq2 % 8
    r3, f3 = sq3 // 8, sq3 % 8

    # Check Rank
    if r1 == r2 == r3: return True
    # Check File
    if f1 == f2 == f3: return True
    # Check Diagonal (dy/dx == 1 or -1)
    if (r1 - f1) == (r2 - f2) == (r3 - f3): return True # Anti-diagonal
    if (r1 + f1) == (r2 + f2) == (r3 + f3): return True # Main diagonal

    return False

@numba.njit(cache=True)
def _get_attackers_to_square(square, piece_bbs, occupancy):
    """
    獲取所有攻擊給定方格的位元棋盤。
    """
    white_attackers = PAWN_ATTACKS[BLACK][square] & piece_bbs[PAWN]
    black_attackers = PAWN_ATTACKS[WHITE][square] & piece_bbs[PAWN + 6]

    # Knights
    knight_attack_bb = KNIGHT_ATTACKS[square]
    white_attackers |= knight_attack_bb & piece_bbs[KNIGHT]
    black_attackers |= knight_attack_bb & piece_bbs[KNIGHT + 6]

    # Bishops and Queens (Diagonals)
    bishop_queen_bb = piece_bbs[BISHOP] | piece_bbs[QUEEN]
    black_bishop_queen_bb = piece_bbs[BISHOP + 6] | piece_bbs[QUEEN + 6]
    bishop_attacks_bb = get_bishop_attacks(square, occupancy)
    white_attackers |= bishop_attacks_bb & bishop_queen_bb
    black_attackers |= bishop_attacks_bb & black_bishop_queen_bb

    # Rooks and Queens (Files/Ranks)
    rook_queen_bb = piece_bbs[ROOK] | piece_bbs[QUEEN]
    black_rook_queen_bb = piece_bbs[ROOK + 6] | piece_bbs[QUEEN + 6]
    rook_attacks_bb = get_rook_attacks(square, occupancy)
    white_attackers |= rook_attacks_bb & rook_queen_bb
    black_attackers |= rook_attacks_bb & black_rook_queen_bb

    # Kings
    king_attack_bb = KING_ATTACKS[square]
    white_attackers |= king_attack_bb & piece_bbs[KING]
    black_attackers |= king_attack_bb & piece_bbs[KING + 6]

    return white_attackers, black_attackers


@numba.njit(cache=True)
def see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold=-32000):
    """
    Static Exchange Evaluation (SEE).
    """

    # Cast inputs to known integer types to prevent type inference issues
    current_side = np.int32(side_to_move)
    from_sq_i = np.int32(from_sq)
    to_sq_i = np.int32(to_sq)

    # --- Initial Setup ---
    victim_piece_type = find_piece_type_on_square(piece_bbs, to_sq_i)
    moved_piece_type = find_piece_type_on_square(piece_bbs, from_sq_i)

    occupancy = occupancy_bbs[2]
    from_sq_bb = BB_SQUARES[from_sq_i]
    to_sq_bb = BB_SQUARES[to_sq_i]

    # --- En Passant Handling ---
    # If victim is empty and attacker is Pawn moving diagonally, it's En Passant.
    is_en_passant = False
    if victim_piece_type == -1:
        if moved_piece_type == PAWN or moved_piece_type == PAWN + 6: # Check generic pawn type (0 or 6) is handled by %6
            # Check diagonal move
            if (from_sq_i % 8) != (to_sq_i % 8):
                # En Passant confirmed (assuming pseudo-legal input)
                is_en_passant = True
                victim_piece_type = PAWN # Victim is a pawn

                # Adjust Occupancy for EP:
                # 1. Remove from_sq (handled later)
                # 2. Add to_sq (handled later)
                # 3. Remove captured pawn square
                if current_side == WHITE:
                    cap_sq = to_sq_i - 8
                else:
                    cap_sq = to_sq_i + 8

                occupancy ^= BB_SQUARES[cap_sq]

    if victim_piece_type == -1 and not is_en_passant:
        # Not a capture
        return np.int32(0)

    # Initial value calculation

    gain = np.zeros(32, dtype=np.int32)
    depth = 0

    gain[depth] = MG_MATERIAL_VALUES[victim_piece_type % 6]

    attacker_value = MG_MATERIAL_VALUES[moved_piece_type % 6]

    # Update Occupancy for the first move
    occupancy ^= from_sq_bb
    if is_en_passant:
        occupancy |= to_sq_bb # Attacker lands on empty square

    # Get all attackers
    white_attackers, black_attackers = _get_attackers_to_square(
        np.uint8(to_sq_i), piece_bbs, occupancy
    )
    all_attackers = white_attackers | black_attackers
    
    # Remove the initial attacker (the one that just moved) from available attackers
    all_attackers &= ~from_sq_bb
    
    # --- Correctly Handle X-Ray after the first move (the initial capture) ---
    # The piece moving from 'from_sq' might reveal an attack from behind.
    bishop_attacks_bb = get_bishop_attacks(np.uint8(to_sq_i), occupancy)
    new_bishops = bishop_attacks_bb & (piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6])
    
    rook_attacks_bb = get_rook_attacks(np.uint8(to_sq_i), occupancy)
    new_rooks = rook_attacks_bb & (piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6])
    
    potential_xray = (new_bishops | new_rooks) & ~all_attackers
    if potential_xray:
        all_attackers |= potential_xray
        white_attackers |= (potential_xray & (piece_bbs[BISHOP] | piece_bbs[ROOK] | piece_bbs[QUEEN]))
        black_attackers |= (potential_xray & (piece_bbs[BISHOP+6] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]))

    while True:
        depth += 1
        if depth >= 31: break
            
        current_side = 1 - current_side

        attackers_for_side = black_attackers if current_side == BLACK else white_attackers
        attackers_for_side &= all_attackers # Filter out removed ones

        if not attackers_for_side:
            break

        # --- Find least valuable attacker that is LEGAL (not pinned) ---
        aggressor_sq = -1
        aggressor_piece_type = -1
        aggressor_sq_bb = np.uint64(0)

        # Determine King Square for Pin Check
        my_king_bb = piece_bbs[KING] if current_side == WHITE else piece_bbs[KING + 6]
        king_sq = get_lsb_index(my_king_bb) if my_king_bb else -1

        # Iterate piece types P->N->B->R->Q->K
        found = False
        for piece_type in range(PAWN, KING + 1):
            pt_idx = piece_type + (6 if current_side == BLACK else 0)
            piece_attackers = piece_bbs[pt_idx] & attackers_for_side

            while piece_attackers:
                sq = get_lsb_index(piece_attackers)
                sq_bb = BB_SQUARES[sq]

                # --- Pin Detection ---
                # Check if 'sq' is pinned.
                is_pinned = False
                if king_sq != -1:
                    if _check_on_line(king_sq, sq, to_sq_i):
                        pass
                    else:
                        occ_without_agg = occupancy ^ sq_bb

                        # Explicit casting to ensure integer indexing
                        offset = 6 - 6 * current_side
                        enemy_b_q = piece_bbs[BISHOP + offset] | piece_bbs[QUEEN + offset]
                        enemy_r_q = piece_bbs[ROOK + offset] | piece_bbs[QUEEN + offset]

                        if (get_bishop_attacks(king_sq, occ_without_agg) & enemy_b_q):
                            is_pinned = True
                        elif not is_pinned and (get_rook_attacks(king_sq, occ_without_agg) & enemy_r_q):
                            is_pinned = True

                if not is_pinned:
                    aggressor_sq = sq
                    aggressor_sq_bb = sq_bb
                    aggressor_piece_type = piece_type
                    found = True
                    break # Break inner while

                piece_attackers ^= sq_bb

            if found:
                break # Break outer for
        
        if aggressor_sq == -1:
            break
            
        # Update Gain
        gain[depth] = attacker_value - gain[depth - 1]
        
        # Prepare for next iteration
        attacker_value = MG_MATERIAL_VALUES[aggressor_piece_type]
        
        # Update Occupancy
        occupancy ^= aggressor_sq_bb
        
        # X-Ray Discovery
        bishop_attacks_bb = get_bishop_attacks(np.uint8(to_sq_i), occupancy)
        new_bishops = bishop_attacks_bb & (piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6])
        
        rook_attacks_bb = get_rook_attacks(np.uint8(to_sq_i), occupancy)
        new_rooks = rook_attacks_bb & (piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6])
        
        potential_xray = (new_bishops | new_rooks) & ~all_attackers
        if potential_xray:
            all_attackers |= potential_xray
            white_attackers |= (potential_xray & (piece_bbs[BISHOP] | piece_bbs[ROOK] | piece_bbs[QUEEN]))
            black_attackers |= (potential_xray & (piece_bbs[BISHOP+6] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]))

        all_attackers &= ~aggressor_sq_bb


    # Propagate scores
    while depth > 1:
        depth -= 1
        gain[depth-1] = -max(-gain[depth-1], gain[depth])

    return gain[0]
