
import numba
import numpy as np
from chess_engine.constants import (
    BB_SQUARES,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    MG_MATERIAL_VALUES
)
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature
from chess_engine.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side, SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS
import numba.types as nbt
# Import Magic Bitboard functions
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks,
    PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS,
    get_pinned_pieces
)
from chess_engine.bitboard_utils import get_lsb_index, count_bits


# Piece Values for SEE (based on Stockfish's internal values for SEE)
# P=100, N=320, B=330, R=500, Q=900, K=20000
# Note: King is given a very high value to prevent it from being "captured" in simulation logic.
# We use local definition to ensure it matches exactly the SEE assumption (King value high)

PIECE_VALUES = MG_MATERIAL_VALUES

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

    # Bishops + Queens
    sliders_diag = piece_bbs[BISHOP + offset] | piece_bbs[QUEEN + offset]
    if sliders_diag:
        attackers |= (get_sliding_attacks(square, occupied, True) & sliders_diag)

    # Rooks + Queens
    sliders_orth = piece_bbs[ROOK + offset] | piece_bbs[QUEEN + offset]
    if sliders_orth:
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
            return PIECE_VALUES[i], sq_bb, i
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
    if victim_type == -1:
        # En Passant?
        if attacker_type == PAWN and abs((from_sq % 8) - (to_sq % 8)) == 1:
             # Attacker is pawn, diagonal move, empty target -> EP
             value = PIECE_VALUES[PAWN]
        else:
             value = 0 # Empty square, not EP (e.g. Quiet move)
    else:
        value = PIECE_VALUES[victim_type]

    # Scores stack
    # scores[0] = Capture gain
    # scores[1] = Opponent recapture gain
    scores = np.zeros(32, dtype=np.int32)
    scores[0] = value

    attacker_value = PIECE_VALUES[attacker_type]

    # Setup Occupied Bitboard
    occupied = occupancy_bbs[2]

    # Remove the initial attacker from occupied
    occupied &= ~BB_SQUARES[from_sq]

    # Pinned pieces are now passed as arguments!
    # pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    # pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # Find all attackers to `to_sq`
    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move
    current_attacker_val = attacker_value

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

        stm_attackers = attackers & (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]) \
                        if current_side == WHITE else \
                        attackers & (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11])

        # Remove pinned pieces
        pinned_mask = pinned_white if current_side == WHITE else pinned_black
        stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            break

        # Find LVA
        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # If King is attacker, check if it steps into check?
        # Stockfish: `if (type == KING) { if (attackers & ~stm) break; }`
        if (type_idx == KING or type_idx == KING + 6):
             # Check if opponent still has attackers
             # Opponent is `1 - current_side`
             # Opponent attackers in `attackers`
             opp_mask = (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11]) \
                        if current_side == WHITE else \
                        (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5])

             if (attackers & opp_mask):
                 break # King cannot capture into check

             # If King captures safely, its risk value is 0 (it won't be captured)
             val = 0

        # Remove this attacker from occupied
        # This "reveals" pieces behind it (X-Ray)
        occupied &= ~bb

        # Add X-Ray attackers
        # Stockfish adds: `attackers |= attacks_bb<BISHOP/ROOK>(to, occupied) & pieces(Sliders)`
        # We simulate this.
        # Check diagonals
        if type_idx == PAWN or type_idx == BISHOP or type_idx == QUEEN or \
           type_idx == PAWN+6 or type_idx == BISHOP+6 or type_idx == QUEEN+6:
             sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
             if sliders_diag:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        # Check orthogonals
        if type_idx == ROOK or type_idx == QUEEN or \
           type_idx == ROOK+6 or type_idx == QUEEN+6:
             sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
             if sliders_orth:
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
    SEE >= Threshold.
    Optimized to exit early.
    Updated: Accepts pre-calculated pinned_white and pinned_black bitboards.
    """
    # Identical setup to see()

    if attacker_type == -1:
        attacker_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
    
    if attacker_type == -1: return False # Should not happen
    attacker_type %= 6

    if victim_type == -1:
        victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - side_to_move)
    
    if victim_type != -1:
        victim_type %= 6

    if victim_type == -1:
        if attacker_type == PAWN and abs((from_sq % 8) - (to_sq % 8)) == 1:
             value = PIECE_VALUES[PAWN]
        else:
             value = 0
    else:
        value = PIECE_VALUES[victim_type]

    # Swap Algorithm with Balance
    # score = victim - threshold
    balance = value - threshold

    # If initial capture already fails (e.g. capturing nothing with big threshold?), return False
    # But usually threshold is negative.
    if balance < 0:
        return False

    # Next: Opponent captures us.
    # balance = attacker_val - balance
    attacker_val = PIECE_VALUES[attacker_type]
    balance = attacker_val - balance

    if balance <= 0:
        return True

    occupied = occupancy_bbs[2]
    occupied &= ~BB_SQUARES[from_sq]

    # Pinned pieces are now passed as arguments!
    # pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    # pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move

    while True:
        current_side = 1 - current_side

        # New: Filter attackers
        attackers &= occupied

        stm_attackers = attackers & (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]) \
                        if current_side == WHITE else \
                        attackers & (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11])

        pinned_mask = pinned_white if current_side == WHITE else pinned_black
        stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            return (current_side != side_to_move)

        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # King Check
        if (type_idx == KING or type_idx == KING + 6):
             opp_mask = (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11]) \
                        if current_side == WHITE else \
                        (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5])
             if (attackers & opp_mask):
                 # King cannot capture. Treat as no moves.
                 return (current_side != side_to_move)

             # Safe King capture.
             val = 0

        occupied &= ~bb

        # X-Ray
        if type_idx == PAWN or type_idx == BISHOP or type_idx == QUEEN or \
           type_idx == PAWN+6 or type_idx == BISHOP+6 or type_idx == QUEEN+6:
             sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
             if sliders_diag:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx == ROOK or type_idx == QUEEN or \
           type_idx == ROOK+6 or type_idx == QUEEN+6:
             sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
             if sliders_orth:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

        # Update balance
        balance = val - balance

        # Check cutoff
        if balance <= 0:
            return (current_side == side_to_move)

    return True # Fallback
