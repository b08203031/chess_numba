
import numba
import numpy as np
from chess_engine.classical.constants import (
    BB_SQUARES,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    MG_MATERIAL_VALUES
)
from chess_engine.classical.engine_types import piece_bbs_signature, occupancy_bbs_signature
from chess_engine.classical.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side, SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS
import numba.types as nbt
# Import Magic Bitboard functions
from chess_engine.classical.move_generator import (
    get_bishop_attacks, get_rook_attacks,
    PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS,
    get_pinned_pieces
)
from chess_engine.classical.bitboard_utils import get_lsb_index, count_bits


# Piece Values for SEE (based on Stockfish's internal values for SEE)
# P=100, N=320, B=330, R=500, Q=900, K=20000
# Note: King is given a very high value to prevent it from being "captured" in simulation logic.
# We use local definition to ensure it matches exactly the SEE assumption (King value high)

SEE_PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)

# Basic Attack Tables (Initialize once if possible, or use logic)
# Numba caches function compilation, so we can embed logic.

DIAGONAL_DIRECTIONS = np.array([-9, -7, 7, 9], dtype=np.int32)
ORTHOGONAL_DIRECTIONS = np.array([-8, -1, 1, 8], dtype=np.int32)

@numba.njit(cache=True)
def get_sliding_attacks(square, occupied, is_diagonal):
    """
    Get sliding attacks using Magic Bitboards (O(1) lookup).
    Replaces the previous slow ray-casting loop.
    """
    if is_diagonal:
        return get_bishop_attacks(square, occupied)
    else:
        return get_rook_attacks(square, occupied)


@numba.njit(cache=True)
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

@numba.njit(cache=True)
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

@numba.njit(cache=True)
def see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, pinned_white, pinned_black, attacker_type=-1, victim_type=-1):
    """
    Static Exchange Evaluation (SEE).
    Mimics Stockfish's logic:
    - Stack based Swap algorithm
    - Pinned pieces DO NOT attack/capture (Strict definition)
    - En Passant handling
    
    Updated: Accepts pre-calculated pinned_white and pinned_black bitboards.
    """

    # 1. Identify Initial Victim and Attacker

    # Optimization: Check piece type by looking at specific bitboards based on side
    # But for 'from_sq', we know side_to_move.

    # Initial Attacker Type
    if attacker_type == -1:
        attacker_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
    
    if attacker_type == -1: return 0 # Should not happen for legal moves
    attacker_type %= 6

    # Initial Victim Type
    if victim_type == -1:
        victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - side_to_move)
    
    if victim_type != -1:
        victim_type %= 6

    # Value of initial capture
    is_ep = False
    if victim_type == -1:
        # En Passant?
        if attacker_type == PAWN and abs((from_sq % 8) - (to_sq % 8)) == 1:
             # Attacker is pawn, diagonal move, empty target -> EP
             value = SEE_PIECE_VALUES[PAWN]
             is_ep = True
        else:
             value = 0 # Empty square, not EP (e.g. Quiet move)
    else:
        value = SEE_PIECE_VALUES[victim_type]

    # Scores stack
    # scores[0] = Capture gain
    # scores[1] = Opponent recapture gain
    scores = np.zeros(32, dtype=np.int32)
    scores[0] = value

    attacker_value = SEE_PIECE_VALUES[attacker_type]
    # Pawn Promotion handling
    if attacker_type == PAWN and (to_sq >= 56 or to_sq < 8):
        attacker_value = SEE_PIECE_VALUES[QUEEN]

    # Setup Occupied Bitboard (XOR is faster than AND-NOT when bit is guaranteed set)
    occupied = occupancy_bbs[2] ^ BB_SQUARES[from_sq]

    # If it was an En Passant capture, also remove the captured pawn
    if is_ep:
        ep_sq = (from_sq // 8) * 8 + (to_sq % 8)
        occupied ^= BB_SQUARES[ep_sq]

    # Pinned pieces are now passed as arguments!
    # pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    # pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # Find all attackers to `to_sq`
    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move
    current_attacker_val = attacker_value

    # Pre-compute slider masks outside loop (avoid recomputation each iteration)
    sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
    sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]

    depth = 1

    while True:
        # Swap side
        current_side = 1 - current_side

        # Filter attackers that are still on board (occupied)
        # This removes the piece that just moved/captured from the attackers list
        attackers &= occupied

        # Calculate gain for this depth
        scores[depth] = current_attacker_val - scores[depth - 1]

        # Filter attackers
        # 1. Must belong to current_side
        # 2. Must not be pinned (Stockfish simplification)

        # Use pre-stored occupancy bitboards instead of 6-OR
        stm_attackers = attackers & (occupancy_bbs[0] if current_side == WHITE else occupancy_bbs[1])

        # Remove pinned pieces
        pinned_mask = pinned_white if current_side == WHITE else pinned_black
        stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            break

        # Find LVA
        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # Pawn Promotion handling for subsequent captures
        if type_idx == PAWN and (to_sq >= 56 or to_sq < 8):
            val = SEE_PIECE_VALUES[QUEEN]

        # King check (Stockfish-style: use occupancy_bbs for opponent mask)
        if type_idx == KING:
             opp_side = 1 - current_side
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


@numba.njit(cache=True)
def see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black, attacker_type=-1, victim_type=-1):
    """
    SEE >= Threshold (Stockfish-style).
    Uses res ^= 1 toggle + swap variable for efficient early exit.
    Updated: Accepts pre-calculated pinned_white and pinned_black bitboards.
    """
    # 1. Identify attacker and victim types
    if attacker_type == -1:
        attacker_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
    
    if attacker_type == -1: return False
    attacker_type %= 6

    if victim_type == -1:
        victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - side_to_move)
    
    if victim_type != -1:
        victim_type %= 6

    # 2. Handle en passant and compute initial value
    is_ep = False
    if victim_type == -1:
        if attacker_type == PAWN and abs((from_sq % 8) - (to_sq % 8)) == 1:
             value = SEE_PIECE_VALUES[PAWN]
             is_ep = True
        else:
             value = 0
    else:
        value = SEE_PIECE_VALUES[victim_type]

    # 3. Stockfish-style early exit with swap variable
    swap = value - threshold
    if swap < 0:
        return False

    attacker_val = SEE_PIECE_VALUES[attacker_type]
    # Pawn Promotion handling
    if attacker_type == PAWN and (to_sq >= 56 or to_sq < 8):
        attacker_val = SEE_PIECE_VALUES[QUEEN]
        
    swap = attacker_val - swap
    if swap <= 0:
        return True

    # 4. Setup occupied (XOR style)
    occupied = occupancy_bbs[2] ^ BB_SQUARES[from_sq]
    if is_ep:
        ep_sq = (from_sq // 8) * 8 + (to_sq % 8)
        occupied ^= BB_SQUARES[ep_sq]

    # 5. Get all attackers to target square
    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move
    res = 1

    # Pre-compute slider masks outside loop
    sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
    sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]

    while True:
        current_side = numba.int32(1 - current_side)
        attackers &= occupied

        # Use pre-stored occupancy bitboards instead of 6-OR
        stm_attackers = attackers & (occupancy_bbs[0] if current_side == WHITE else occupancy_bbs[1])

        # Remove pinned pieces
        pinned_mask = pinned_white if current_side == WHITE else pinned_black
        stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            break

        # Toggle result bit (Stockfish-style null-window alpha-beta)
        res ^= 1

        # Find LVA (Least Valuable Attacker)
        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # Pawn Promotion handling for subsequent captures
        if type_idx == PAWN and (to_sq >= 56 or to_sq < 8):
            val = SEE_PIECE_VALUES[QUEEN]

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
