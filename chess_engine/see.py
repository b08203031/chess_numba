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
    Get all attackers to a square.
    """
    # Note: Pawn attacks lookup table is structured as [COLOR][SQUARE]
    # where the SQUARE is the square being attacked, and the value is bitboard of pawns of that COLOR attacking it.

    # Wait, let's verify PAWN_ATTACKS definition in move_generator.py
    # PAWN_ATTACKS[WHITE, sq] = attacks FROM white pawn AT sq? OR attacks TO sq?
    # move_generator.py:
    # PAWN_ATTACKS[WHITE, sq] = ((bb_sq & NOT_A_FILE) >> 9) | ((bb_sq & NOT_H_FILE) >> 7)
    # This generates where a White Pawn at `sq` attacks TO.

    # In SEE, we want to know if any White Pawn attacks `square`.
    # A White Pawn at `p_sq` attacks `square` IF `square` is one of the targets of `p_sq`.
    # Conversely, `p_sq` must be one of the sources that can attack `square`.
    # The squares that can attack `square` with a White Pawn capture are the squares where a BLACK pawn at `square` would attack.
    # So we need reverse lookup.

    # If we want White Pawns attacking `square`:
    # The pawns must be at `square - 9` or `square - 7` (approx).
    # This is equivalent to where a Black Pawn at `square` would attack.
    # So `PAWN_ATTACKS[BLACK][square]` gives the squares where White Pawns must be to attack `square`.

    # Let's verify:
    # PAWN_ATTACKS[BLACK, sq] = ((bb_sq & NOT_H_FILE) << 9) | ((bb_sq & NOT_A_FILE) << 7)
    # This shifts UP (increasing index), which is valid for Black Pawns moving "down" the board (decreasing rank)?
    # No, ranks: Rank 1 is index 0-7. Rank 8 is 56-63.
    # White pawns move +8. Capture +7/+9.
    # Black pawns move -8. Capture -7/-9.

    # PAWN_ATTACKS[WHITE, sq] (White Pawn at sq attacks where?):
    # >> 9 (sq - 9) and >> 7 (sq - 7).
    # Decreasing index. This means White Pawns attack "downwards" in index?
    # Standard: A1 is 0. H1 is 7. A8 is 56. H8 is 63.
    # White Pawn at e4 (28). Attacks d5 (35) and f5 (37).
    # 28 -> 35 (+7). 28 -> 37 (+9).
    # So White attacks are +7 and +9.
    # move_generator.py uses >> 9 and >> 7. This corresponds to -9 and -7.
    # So PAWN_ATTACKS[WHITE, sq] seems to generate attacks for a white pawn at sq, assuming sq is the source?
    # If it uses >>, it goes to LOWER indices.
    # But White Pawns move to HIGHER indices (Rank 2 -> Rank 3).
    # So `move_generator.py` seems to assume `sq` is the TARGET?
    # Or `move_generator.py` has flipped orientation?
    # Let's check `is_square_attacked` logic in `move_generator.py` (via grep I saw earlier):
    # if PAWN_ATTACKS[WHITE, sq] & wp_bb: return True
    # This checks if `sq` is attacked by White Pawns (`wp_bb`).
    # So `PAWN_ATTACKS[WHITE, sq]` must return the bitboard of SQUARES WHERE WHITE PAWNS CAN BE to attack `sq`.
    # So it IS the reverse lookup!

    # If `PAWN_ATTACKS[WHITE, sq]` is "squares that white pawns attack `sq` from",
    # then `white_attackers = PAWN_ATTACKS[WHITE][square] & piece_bbs[PAWN]` is correct.

    # My `see.py` had:
    # white_attackers = PAWN_ATTACKS[BLACK][square] & piece_bbs[PAWN]
    # This was using BLACK attacks for WHITE pawns.
    # This logic assumed PAWN_ATTACKS was "attacks FROM sq".

    # If `PAWN_ATTACKS` is "Attacks TO sq", then:
    # white_attackers = PAWN_ATTACKS[WHITE][square] & piece_bbs[PAWN]
    # black_attackers = PAWN_ATTACKS[BLACK][square] & piece_bbs[PAWN + 6]

    # Let's re-read the grep output from `move_generator.py`:
    # PAWN_ATTACKS[WHITE, sq] = ((bb_sq & NOT_A_FILE) >> 9) | ((bb_sq & NOT_H_FILE) >> 7)
    # This shifts bits DOWN (lower index).
    # e4 (28) -> d3 (19), f3 (21).
    # If `sq` is e4. The bits set are d3 and f3.
    # White Pawns at d3/f3 capture on e4.
    # Correct. A White Pawn at d3 captures e4 (+9). A White Pawn at f3 captures e4 (+7).
    # So PAWN_ATTACKS[WHITE, sq] gives the squares where a White Pawn must be to attack `sq`.

    # So I should use:
    # white_attackers = PAWN_ATTACKS[WHITE][square] & piece_bbs[PAWN]
    # black_attackers = PAWN_ATTACKS[BLACK][square] & piece_bbs[PAWN + 6]

    white_attackers = PAWN_ATTACKS[WHITE][square] & piece_bbs[PAWN]
    black_attackers = PAWN_ATTACKS[BLACK][square] & piece_bbs[PAWN + 6]

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


@numba.njit(
    numba.boolean(
        piece_bbs_signature,
        occupancy_bbs_signature,
        numba.uint8,
        numba.uint8,
        numba.uint8,
        numba.int32,
    ),
    cache=True,
)
def see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold):
    """
    Static Exchange Evaluation >= Threshold Check.
    Port of Stockfish's see_ge function.
    """
    victim_piece_type = find_piece_type_on_square(piece_bbs, to_sq)
    value_to = MG_MATERIAL_VALUES[victim_piece_type % 6] if victim_piece_type != -1 else 0

    # First swap: Capture
    swap = value_to - threshold
    if swap < 0:
        return False

    moved_piece_type = find_piece_type_on_square(piece_bbs, from_sq)
    value_from = MG_MATERIAL_VALUES[moved_piece_type % 6]
    
    # Second swap: Recapture by opponent
    swap = value_from - swap
    if swap <= 0:
        return True
    
    # Setup for iteration
    occupancy = occupancy_bbs[2] ^ BB_SQUARES[from_sq] ^ BB_SQUARES[to_sq]
    stm = side_to_move
    
    white_attackers, black_attackers = _get_attackers_to_square(to_sq, piece_bbs, occupancy)
    
    from_bb = BB_SQUARES[from_sq]
    white_attackers &= ~from_bb
    black_attackers &= ~from_bb

    all_attackers = white_attackers | black_attackers

    res = 1
    
    while True:
        stm = 1 - stm # Switch side
        attackers_bb = (black_attackers if stm == BLACK else white_attackers) & occupancy

        if not attackers_bb:
            break

        res ^= 1

        aggressor_bb = np.uint64(0)
        piece_value = 0

        # We manually unroll for speed

        # PAWN
        pawn_bb = piece_bbs[PAWN + (6 if stm == BLACK else 0)]
        if (attackers_bb & pawn_bb):
            aggressor_bb = attackers_bb & pawn_bb
            piece_value = MG_MATERIAL_VALUES[PAWN]

        # KNIGHT
        elif (attackers_bb & piece_bbs[KNIGHT + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[KNIGHT + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[KNIGHT]

        # BISHOP
        elif (attackers_bb & piece_bbs[BISHOP + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[BISHOP + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[BISHOP]

        # ROOK
        elif (attackers_bb & piece_bbs[ROOK + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[ROOK + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[ROOK]

        # QUEEN
        elif (attackers_bb & piece_bbs[QUEEN + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[QUEEN + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[QUEEN]

        # KING
        elif (attackers_bb & piece_bbs[KING + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[KING + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[KING]
            
            opponent_attackers = (white_attackers if stm == BLACK else black_attackers) & occupancy
            if opponent_attackers:
                 return True if (res ^ 1) else False
            else:
                 return True if res else False

        lsb_sq = get_lsb_index(aggressor_bb)
        lsb_bb = BB_SQUARES[lsb_sq]
        
        swap = piece_value - swap
        if swap < res:
            break
            
        occupancy ^= lsb_bb

        if (piece_value == MG_MATERIAL_VALUES[PAWN] or
            piece_value == MG_MATERIAL_VALUES[BISHOP] or
            piece_value == MG_MATERIAL_VALUES[QUEEN]):

            b_attacks = get_bishop_attacks(to_sq, occupancy)
            diag_sliders = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
            new_attackers = b_attacks & diag_sliders & occupancy

            if new_attackers:
                white_attackers |= (new_attackers & (piece_bbs[BISHOP] | piece_bbs[QUEEN]))
                black_attackers |= (new_attackers & (piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]))

        if (piece_value == MG_MATERIAL_VALUES[ROOK] or
            piece_value == MG_MATERIAL_VALUES[QUEEN]):

            r_attacks = get_rook_attacks(to_sq, occupancy)
            orth_sliders = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
            new_attackers = r_attacks & orth_sliders & occupancy

            if new_attackers:
                white_attackers |= (new_attackers & (piece_bbs[ROOK] | piece_bbs[QUEEN]))
                black_attackers |= (new_attackers & (piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]))

    return True if res else False

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
    Exact Static Exchange Evaluation.
    """
    victim_piece_type = find_piece_type_on_square(piece_bbs, to_sq)
    value_to = MG_MATERIAL_VALUES[victim_piece_type % 6] if victim_piece_type != -1 else 0

    moved_piece_type = find_piece_type_on_square(piece_bbs, from_sq)
    value_from = MG_MATERIAL_VALUES[moved_piece_type % 6]

    gain = np.zeros(32, dtype=np.int32)
    depth = 0

    gain[depth] = value_to

    # Setup for iteration
    occupancy = occupancy_bbs[2] ^ BB_SQUARES[from_sq]

    white_attackers, black_attackers = _get_attackers_to_square(to_sq, piece_bbs, occupancy)

    stm = 1 - side_to_move

    # Add initial X-rays revealed by the first move
    if (moved_piece_type == PAWN or moved_piece_type == BISHOP or moved_piece_type == QUEEN or moved_piece_type == PAWN+6 or moved_piece_type == BISHOP+6 or moved_piece_type == QUEEN+6):
        b_attacks = get_bishop_attacks(to_sq, occupancy)
        diag_sliders = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
        new_attackers = b_attacks & diag_sliders & occupancy
        if new_attackers:
            white_attackers |= (new_attackers & (piece_bbs[BISHOP] | piece_bbs[QUEEN]))
            black_attackers |= (new_attackers & (piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]))

    if (moved_piece_type == ROOK or moved_piece_type == QUEEN or moved_piece_type == ROOK+6 or moved_piece_type == QUEEN+6):
        r_attacks = get_rook_attacks(to_sq, occupancy)
        orth_sliders = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
        new_attackers = r_attacks & orth_sliders & occupancy
        if new_attackers:
            white_attackers |= (new_attackers & (piece_bbs[ROOK] | piece_bbs[QUEEN]))
            black_attackers |= (new_attackers & (piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]))

    current_piece_value = value_from

    while True:
        depth += 1
        if depth >= 32: break
        
        attackers_bb = (black_attackers if stm == BLACK else white_attackers) & occupancy
        if not attackers_bb: break
        
        aggressor_bb = np.uint64(0)
        piece_value = 0

        # Find least valuable attacker
        # PAWN
        pawn_bb = piece_bbs[PAWN + (6 if stm == BLACK else 0)]
        if (attackers_bb & pawn_bb):
            aggressor_bb = attackers_bb & pawn_bb
            piece_value = MG_MATERIAL_VALUES[PAWN]
        elif (attackers_bb & piece_bbs[KNIGHT + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[KNIGHT + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[KNIGHT]
        elif (attackers_bb & piece_bbs[BISHOP + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[BISHOP + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[BISHOP]
        elif (attackers_bb & piece_bbs[ROOK + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[ROOK + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[ROOK]
        elif (attackers_bb & piece_bbs[QUEEN + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[QUEEN + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[QUEEN]
        elif (attackers_bb & piece_bbs[KING + (6 if stm == BLACK else 0)]):
            aggressor_bb = attackers_bb & piece_bbs[KING + (6 if stm == BLACK else 0)]
            piece_value = MG_MATERIAL_VALUES[KING]

        lsb_sq = get_lsb_index(aggressor_bb)
        lsb_bb = BB_SQUARES[lsb_sq]
        
        gain[depth] = current_piece_value - gain[depth - 1]
        
        current_piece_value = piece_value
        occupancy ^= lsb_bb
        stm = 1 - stm
        
        # Add X-Rays
        if (piece_value == MG_MATERIAL_VALUES[PAWN] or
            piece_value == MG_MATERIAL_VALUES[BISHOP] or
            piece_value == MG_MATERIAL_VALUES[QUEEN]):
            b_attacks = get_bishop_attacks(to_sq, occupancy)
            diag_sliders = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
            new_attackers = b_attacks & diag_sliders & occupancy
            if new_attackers:
                white_attackers |= (new_attackers & (piece_bbs[BISHOP] | piece_bbs[QUEEN]))
                black_attackers |= (new_attackers & (piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]))

        if (piece_value == MG_MATERIAL_VALUES[ROOK] or
            piece_value == MG_MATERIAL_VALUES[QUEEN]):
            r_attacks = get_rook_attacks(to_sq, occupancy)
            orth_sliders = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
            new_attackers = r_attacks & orth_sliders & occupancy
            if new_attackers:
                white_attackers |= (new_attackers & (piece_bbs[ROOK] | piece_bbs[QUEEN]))
                black_attackers |= (new_attackers & (piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]))

    # Minimax back-propagation
    while depth > 1:
        depth -= 1
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])

    return gain[0]
