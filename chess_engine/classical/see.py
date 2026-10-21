
import numba
import numpy as np
from chess_engine.classical.constants import (
    BB_SQUARES,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    MG_MATERIAL_VALUES
)
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature
)
from chess_engine.classical.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side, SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS, LINE_BB
import numba.types as nbt
# Import Magic Bitboard functions
from chess_engine.classical.move_generator import (
    get_bishop_attacks, get_rook_attacks,
    PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS,
    get_pinned_pieces
)
from chess_engine.classical.bitboard_utils import get_lsb_index, count_bits


# Piece values for SEE: P/N/B/R/Q directly use MG_MATERIAL_VALUES;
# King is set to 20000 so SEE never treats K as a free sacrificial swap piece.
SEE_PIECE_VALUES = np.array([
    MG_MATERIAL_VALUES[PAWN],
    MG_MATERIAL_VALUES[KNIGHT],
    MG_MATERIAL_VALUES[BISHOP],
    MG_MATERIAL_VALUES[ROOK],
    MG_MATERIAL_VALUES[QUEEN],
    20000,
], dtype=np.int32)

# Basic Attack Tables (Initialize once if possible, or use logic)
# Numba caches function compilation, so we can embed logic.

DIAGONAL_DIRECTIONS = np.array([-9, -7, 7, 9], dtype=np.int32)
ORTHOGONAL_DIRECTIONS = np.array([-8, -1, 1, 8], dtype=np.int32)

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def get_sliding_attacks(square, occupied, is_diagonal):
    """
    Get sliding attacks using Magic Bitboards (O(1) lookup).
    Replaces the previous slow ray-casting loop.
    """
    if is_diagonal:
        return get_bishop_attacks(square, occupied)
    else:
        return get_rook_attacks(square, occupied)


@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def get_all_attackers_for_see(square, occupied, piece_bbs, sliders_diag, sliders_orth):
    """
    Get bitboard of all pieces (both White and Black) attacking 'square'.
    Combines both sides in a single pass to eliminate duplicate Magic Bitboard
    sliding attack calculations (bishop and rook attacks on the same square and occupied).
    """
    attackers = (
        (PAWN_ATTACKS[WHITE, square] & piece_bbs[PAWN])
        | (PAWN_ATTACKS[BLACK, square] & piece_bbs[6 + PAWN])
        | (KNIGHT_ATTACKS[square] & (piece_bbs[KNIGHT] | piece_bbs[6 + KNIGHT]))
        | (KING_ATTACKS[square] & (piece_bbs[KING] | piece_bbs[6 + KING]))
    )
    if sliders_diag & BISHOP_RAYS[square]:
        attackers |= (get_bishop_attacks(square, occupied) & sliders_diag)
    if sliders_orth & ROOK_RAYS[square]:
        attackers |= (get_rook_attacks(square, occupied) & sliders_orth)
    return attackers

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_attackers_for_see(square, occupied, piece_bbs, side_mask):
    """
    Get bitboard of all pieces of 'side_mask' (0=White, 1=Black) attacking 'square'.
    Optimized for SEE loop (uses passed 'occupied' which has holes).
    """
    attackers = np.uint64(0)
    offset = 0 if side_mask == WHITE else 6

    # 1. Pawns - Use precomputed attack tables.
    # Note: PAWN_ATTACKS[side, sq] returns squares where a pawn of 'side' 
    # must be to attack 'sq'.
    attackers |= (PAWN_ATTACKS[side_mask, square] & piece_bbs[offset + PAWN])

    # 2. Knights - Use precomputed attack tables
    knights = piece_bbs[KNIGHT + offset]
    if knights:
        attackers |= (KNIGHT_ATTACKS[square] & knights)

    # 3. Sliders
    # We use the current 'occupied' which has holes where pieces were captured!
    # This allows X-Ray attacks to be found "through" the captured square.
 
    # Performance optimization: Use ray pre-filters to avoid expensive Magic Bitboard lookups
    # when no sliding pieces of the given side are on the relevant rays.

    # Bishops + Queens
    sliders_diag = piece_bbs[BISHOP + offset] | piece_bbs[QUEEN + offset]
    if sliders_diag & BISHOP_RAYS[square]:
        attackers |= (get_sliding_attacks(square, occupied, True) & sliders_diag)

    # Rooks + Queens
    sliders_orth = piece_bbs[ROOK + offset] | piece_bbs[QUEEN + offset]
    if sliders_orth & ROOK_RAYS[square]:
        attackers |= (get_sliding_attacks(square, occupied, False) & sliders_orth)

    # 4. King - Use precomputed attack tables
    kings = piece_bbs[KING + offset]
    if kings:
        attackers |= (KING_ATTACKS[square] & kings)

    return attackers

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def get_lva_and_remove(attackers, piece_bbs, side_mask):
    """
    Finds LVA, returns (value, bitboard, type_idx).
    """
    offset = 0 if side_mask == WHITE else 6
    for i in range(6): # P, N, B, R, Q, K
        subset = attackers & piece_bbs[i + offset]
        if subset:
            sq_bb = subset & -subset
            return SEE_PIECE_VALUES[i], np.uint64(sq_bb), i
    return 0, np.uint64(0), -1


@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def filter_pinned_attackers(
    stm_attackers, pinned_mask, king_bb, to_sq,
    occupied, initial_occ, occupancy_bbs, current_side,
    sliders_diag, sliders_orth,
):
    """
    Filters out pinned pieces from `stm_attackers` that cannot legally capture on `to_sq`.
    - Pinned pieces moving along their pin ray (`LINE_BB[king_sq, to_sq]`) remain legal.
    - If a square on a pinned piece's pin ray (`LINE_BB[king_sq, p_sq]`) was vacated during
      the SEE exchange (either by the initial move `from_sq` or an earlier capture into `to_sq`),
      dynamically verifies whether an enemy slider still pins `p_sq` in `occupied`.
    """
    if not (stm_attackers & pinned_mask):
        return stm_attackers
    if not king_bb:
        return stm_attackers & ~pinned_mask
    king_sq = get_lsb_index(king_bb)
    off_ray = stm_attackers & pinned_mask & ~LINE_BB[king_sq, to_sq]
    if not off_ray:
        return stm_attackers

    removed_occ = initial_occ ^ occupied
    opp_occ = occupancy_bbs[1 - current_side] & occupied
    cand = off_ray
    while cand:
        p_sq = get_lsb_index(cand)
        bb_p = BB_SQUARES[p_sq]
        pin_line = LINE_BB[king_sq, p_sq]
        if (removed_occ & pin_line) == 0:
            # Nothing on this pin ray was vacated; piece remains pinned
            stm_attackers ^= bb_p
        else:
            is_diag = (BISHOP_RAYS[king_sq] & bb_p) != 0
            enemy_sliders = (sliders_diag if is_diag else sliders_orth) & opp_occ & pin_line
            if enemy_sliders and (get_sliding_attacks(p_sq, occupied, is_diag) & enemy_sliders):
                stm_attackers ^= bb_p
        cand &= cand - np.uint64(1)
    return stm_attackers


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _see_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, pinned_white, pinned_black, attacker_type=-1, victim_type=-1, promo_type=QUEEN):
    """
    Static Exchange Evaluation (SEE).
    Mimics Stockfish's logic:
    - Stack based Swap algorithm
    - Dynamic pinned piece filtering (unpins when pinner leaves ray)
    - Self-contained En Passant handling
    - Full promotion and underpromotion support
    """
    # 1. Identify Initial Attacker and Victim
    if attacker_type == -1:
        attacker_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
    
    if attacker_type == -1:
        return 0  # Should not happen for legal moves
    attacker_type %= 6

    initial_occ = occupancy_bbs[2]
    is_promo_rank = (to_sq >= 56 or to_sq < 8)
    is_ep = False
    if (initial_occ & BB_SQUARES[to_sq]) == 0:
        # Target square is empty: En Passant if diagonal pawn move, else quiet move
        if attacker_type == PAWN and not is_promo_rank and abs((from_sq % 8) - (to_sq % 8)) == 1:
            value = SEE_PIECE_VALUES[PAWN]
            is_ep = True
        else:
            value = 0
    elif victim_type == -2:
        value = 0
    else:
        if victim_type == -1:
            victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - side_to_move)
        if victim_type >= 0:
            value = SEE_PIECE_VALUES[victim_type % 6]
        else:
            value = 0

    attacker_value = SEE_PIECE_VALUES[attacker_type]
    # Pawn Promotion handling (supports underpromotions via promo_type)
    if attacker_type == PAWN and is_promo_rank:
        p_type = promo_type if (KNIGHT <= promo_type <= QUEEN) else QUEEN
        promo_val = SEE_PIECE_VALUES[p_type]
        value += promo_val - SEE_PIECE_VALUES[PAWN]
        attacker_value = promo_val

    # Scores stack
    scores = np.zeros(32, dtype=np.int32)
    scores[0] = value

    # Setup Occupied Bitboard (XOR is faster than AND-NOT when bit is guaranteed set)
    occupied = initial_occ ^ BB_SQUARES[from_sq]

    # If it was an En Passant capture, also remove the captured pawn
    if is_ep:
        ep_sq = (from_sq // 8) * 8 + (to_sq % 8)
        occupied ^= BB_SQUARES[ep_sq]

    # Pre-compute slider masks outside loop (avoid recomputation each iteration)
    sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
    sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]

    # Find all attackers to `to_sq` in a unified pass (eliminates duplicate Magic Bitboard calls)
    attackers = get_all_attackers_for_see(to_sq, occupied, piece_bbs, sliders_diag, sliders_orth)

    current_side = numba.int32(side_to_move)
    current_attacker_val = attacker_value

    depth = 1

    while True:
        # Swap side
        current_side = numba.int32(1 - current_side)

        # Filter attackers that are still on board (occupied)
        attackers &= occupied

        # Calculate gain for this depth
        scores[depth] = current_attacker_val - scores[depth - 1]

        # Filter attackers belonging to current_side and dynamically filter pinned pieces
        stm_attackers = attackers & (occupancy_bbs[0] if current_side == WHITE else occupancy_bbs[1])
        king_bb = piece_bbs[KING] if current_side == WHITE else piece_bbs[6 + KING]
        pinned_mask = np.uint64(pinned_white if current_side == WHITE else pinned_black)
        stm_attackers = filter_pinned_attackers(
            stm_attackers, pinned_mask, king_bb, to_sq,
            occupied, initial_occ, occupancy_bbs, current_side,
            sliders_diag, sliders_orth,
        )

        if stm_attackers == 0:
            break

        # Find LVA
        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # Pawn Promotion handling for subsequent captures
        if type_idx == PAWN and is_promo_rank:
            scores[depth] += SEE_PIECE_VALUES[QUEEN] - SEE_PIECE_VALUES[PAWN]
            val = SEE_PIECE_VALUES[QUEEN]

        # King check (Stockfish-style: use occupancy_bbs for opponent mask)
        if type_idx == KING:
             opp_side = numba.int32(1 - current_side)
             if attackers & occupancy_bbs[opp_side]:
                 break  # King cannot capture into check
             val = 0  # Safe King capture

        # Remove this attacker from occupied (XOR is faster)
        occupied ^= bb

        # X-Ray attack discovery (using pre-computed slider masks)
        if type_idx == PAWN or type_idx == BISHOP or type_idx == QUEEN:
             if sliders_diag & BISHOP_RAYS[to_sq]:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx == ROOK or type_idx == QUEEN:
             if sliders_orth & ROOK_RAYS[to_sq]:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

        current_attacker_val = val
        depth += 1
        if depth >= 32: break

    # Propagate Minimax
    while depth > 1:
        depth -= 1
        scores[depth-1] = -max(-scores[depth-1], scores[depth])

    return scores[0]


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black, attacker_type=-1, victim_type=-1, promo_type=QUEEN):
    """
    SEE >= Threshold (Stockfish-style).
    Uses res ^= 1 toggle + swap variable for efficient early exit.
    Supports dynamic unpinning, self-contained EP detection, underpromotions,
    and subsequent back-rank pawn promotions inside the swap loop.
    """
    # 1. Identify attacker and victim types
    if attacker_type == -1:
        attacker_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
    
    if attacker_type == -1:
        return False
    attacker_type %= 6

    initial_occ = occupancy_bbs[2]
    is_promo_rank = (to_sq >= 56 or to_sq < 8)
    # 2. Handle en passant and compute initial value directly from occupancy
    is_ep = False
    if (initial_occ & BB_SQUARES[to_sq]) == 0:
        if attacker_type == PAWN and not is_promo_rank and abs((from_sq % 8) - (to_sq % 8)) == 1:
            value = SEE_PIECE_VALUES[PAWN]
            is_ep = True
        else:
            value = 0
    elif victim_type == -2:
        value = 0
    else:
        if victim_type == -1:
            victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - side_to_move)
        if victim_type >= 0:
            value = SEE_PIECE_VALUES[victim_type % 6]
        else:
            value = 0

    attacker_val = SEE_PIECE_VALUES[attacker_type]
    # Pawn Promotion handling (supports underpromotions via promo_type)
    promo_gain = SEE_PIECE_VALUES[QUEEN] - SEE_PIECE_VALUES[PAWN]
    if attacker_type == PAWN and is_promo_rank:
        p_type = promo_type if (KNIGHT <= promo_type <= QUEEN) else QUEEN
        promo_val = SEE_PIECE_VALUES[p_type]
        value += promo_val - SEE_PIECE_VALUES[PAWN]
        attacker_val = promo_val

    # 3. Stockfish-style early exit with swap variable
    swap = value - threshold
    if swap < 0:
        return False

    swap = attacker_val - swap
    if swap <= 0:
        if not is_promo_rank or (swap + promo_gain <= 0):
            return True

    # 4. Setup occupied (XOR style)
    occupied = initial_occ ^ BB_SQUARES[from_sq]
    if is_ep:
        ep_sq = (from_sq // 8) * 8 + (to_sq % 8)
        occupied ^= BB_SQUARES[ep_sq]

    # Pre-compute slider masks outside loop
    sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
    sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]

    # 5. Get all attackers to target square in a unified pass
    attackers = get_all_attackers_for_see(to_sq, occupied, piece_bbs, sliders_diag, sliders_orth)

    current_side = side_to_move
    res = 1

    while True:
        current_side = numba.int32(1 - current_side)
        attackers &= occupied

        # Filter attackers belonging to current_side and dynamically filter pinned pieces
        stm_attackers = attackers & (occupancy_bbs[0] if current_side == WHITE else occupancy_bbs[1])
        king_bb = piece_bbs[KING] if current_side == WHITE else piece_bbs[6 + KING]
        pinned_mask = np.uint64(pinned_white if current_side == WHITE else pinned_black)
        stm_attackers = filter_pinned_attackers(
            stm_attackers, pinned_mask, king_bb, to_sq,
            occupied, initial_occ, occupancy_bbs, current_side,
            sliders_diag, sliders_orth,
        )

        if stm_attackers == 0:
            break

        # Toggle result bit (Stockfish-style null-window alpha-beta)
        res ^= 1

        # Find LVA (Least Valuable Attacker)
        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # Subsequent pawn promotion on back rank
        if is_promo_rank:
            if type_idx == PAWN:
                swap += promo_gain
                val = SEE_PIECE_VALUES[QUEEN]
            if swap < (res ^ 1):
                return bool(res ^ 1)

        # King check (Stockfish-style: direct return)
        if type_idx == KING:
             opp_side = numba.int32(1 - current_side)
             # If opponent still has attackers, king can't capture
             if attackers & occupancy_bbs[opp_side]:
                 return bool(res ^ 1)  # Undo toggle, opponent wins
             return bool(res)  # Safe king capture, we win

        # Update swap and check for early cutoff
        swap = val - swap
        if swap < res:
            if not is_promo_rank or (swap + promo_gain < res):
                break

        # Remove attacker from occupied (XOR)
        occupied ^= bb

        # X-Ray attack discovery (using pre-computed slider masks)
        if type_idx == PAWN or type_idx == BISHOP or type_idx == QUEEN:
             if sliders_diag & BISHOP_RAYS[to_sq]:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx == ROOK or type_idx == QUEEN:
             if sliders_orth & ROOK_RAYS[to_sq]:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

    return bool(res)


def see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, pinned_white, pinned_black, attacker_type=-1, victim_type=-1, promo_type=QUEEN):
    return _see_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, pinned_white, pinned_black, attacker_type, victim_type, promo_type)


def see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black, attacker_type=-1, victim_type=-1, promo_type=QUEEN):
    return _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black, attacker_type, victim_type, promo_type)
