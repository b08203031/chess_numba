#過去測試發現依舊打不贏象馬殺王，主引擎沒有使用
# chess_engine/classical/endgame.py
import numba
import numpy as np
from chess_engine.classical_old.constants import (
    SCALE_FACTOR_NORMAL, SCALE_FACTOR_DRAW, SCALE_FACTOR_OCB_ONE_PAWN,
    SCALE_FACTOR_OCB_TWO_PAWNS, SCALE_FACTOR_OCB_MULTIPLE_PAWNS,
    SCALE_FACTOR_KXK, SCALE_FACTOR_KBNK, SCALE_FACTOR_KBPSK_FORTRESS,
    SCALE_FACTOR_KRPKR_FORTRESS, SCALE_FACTOR_KQKR_FORTRESS, SCALE_FACTOR_KQKRPs_FORTRESS,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
)
from chess_engine.classical_old.bitboard_utils import get_lsb_index, count_bits, FILE_MASKS
from chess_engine.classical_old.move_generator import get_bishop_attacks

# --- Helper Distance Tables ---
def _create_manhattan_distance_table():
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = abs(rank1 - rank2) + abs(file1 - file2)
    return table

MANHATTAN_DISTANCE = _create_manhattan_distance_table()

def _create_chebyshev_distance_table():
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = max(abs(rank1 - rank2), abs(file1 - file2))
    return table

CHEBYSHEV_DISTANCE = _create_chebyshev_distance_table()

# PushToEdges: 8x8 table pushing enemy king to edges
PUSH_TO_EDGES = np.array([
    100, 90, 80, 70, 70, 80, 90, 100,
     90, 70, 60, 50, 50, 60, 70,  90,
     80, 60, 40, 30, 30, 40, 60,  80,
     70, 50, 30, 20, 20, 30, 50,  70,
     70, 50, 30, 20, 20, 30, 50,  70,
     80, 60, 40, 30, 30, 40, 60,  80,
     90, 70, 60, 50, 50, 60, 70,  90,
    100, 90, 80, 70, 70, 80, 90, 100
], dtype=np.int32)

PUSH_CLOSE = np.array([0, 0, 100, 80, 60, 40, 20, 10], dtype=np.int32)
PUSH_AWAY = np.array([0, 5, 20, 40, 60, 80, 90, 100], dtype=np.int32)

PUSH_TO_CORNERS = np.array([
     6400, 6080, 5760, 5440, 5120, 4800, 4480, 4160,
     6080, 5760, 5440, 5120, 4800, 4480, 4160, 4480,
     5760, 5440, 4960, 4480, 4480, 4000, 4480, 4800,
     5440, 5120, 4480, 3840, 3520, 4480, 4800, 5120,
     5120, 4800, 4480, 3520, 3840, 4480, 5120, 5440,
     4800, 4480, 4000, 4480, 4480, 4960, 5440, 5760,
     4480, 4160, 4480, 4800, 5120, 5440, 5760, 6080,
     4160, 4480, 4800, 5120, 5440, 5760, 6080, 6400
], dtype=np.int32)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def mate_kxk(strong_king_sq, weak_king_sq):
    dist = CHEBYSHEV_DISTANCE[strong_king_sq, weak_king_sq]
    edge_bonus = PUSH_TO_EDGES[weak_king_sq]
    return np.int32(PUSH_CLOSE[dist] + edge_bonus)

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def opposite_colors(sq1, sq2):
    return (((sq1 % 8) + (sq1 // 8)) & 1) != (((sq2 % 8) + (sq2 // 8)) & 1)

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def normalize_square(sq, strong_side, p_sq):
    if strong_side == 1:
        sq = sq ^ 56
    p_file = p_sq % 8
    if p_file >= 4:
        sq = sq ^ 7
    return sq

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def mate_kbnk(strong_king_sq, weak_king_sq, bishop_sq):
    is_opposite = opposite_colors(bishop_sq, 0)  # 0 is SQ_A1
    idx = (weak_king_sq ^ 56) if is_opposite else weak_king_sq
    
    dist_kings = CHEBYSHEV_DISTANCE[strong_king_sq, weak_king_sq]
    
    result = np.int32(10000) + PUSH_CLOSE[dist_kings] + PUSH_TO_CORNERS[idx]
    return result

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_special_endgame(piece_bbs, occupancy_bbs, game_state, phase):
    wp = count_bits(piece_bbs[0])
    wn = count_bits(piece_bbs[1])
    wb = count_bits(piece_bbs[2])
    wr = count_bits(piece_bbs[3])
    wq = count_bits(piece_bbs[4])
    
    bp = count_bits(piece_bbs[6])
    bn = count_bits(piece_bbs[7])
    bb = count_bits(piece_bbs[8])
    br = count_bits(piece_bbs[9])
    bq = count_bits(piece_bbs[10])
    
    w_minor_major = wn + wb + wr + wq
    b_minor_major = bn + bb + br + bq
    total_pawns = wp + bp
    
    wk_sq = get_lsb_index(piece_bbs[5])
    bk_sq = get_lsb_index(piece_bbs[11])
    
    if total_pawns == 0:
        if w_minor_major == 2 and wb == 1 and wn == 1 and b_minor_major == 0:
            bishop_sq = get_lsb_index(piece_bbs[2])
            return True, np.int32(mate_kbnk(wk_sq, bk_sq, bishop_sq))
        if b_minor_major == 2 and bb == 1 and bn == 1 and w_minor_major == 0:
            bishop_sq = get_lsb_index(piece_bbs[8])
            return True, np.int32(-mate_kbnk(bk_sq, wk_sq, bishop_sq))
            
    return False, np.int32(0)

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def is_pawn_passed_on_the_fly(sq, side, enemy_pawns_bb):
    f = sq % 8
    r = sq // 8
    mask = np.uint64(0)
    for file_idx in (f - 1, f, f + 1):
        if 0 <= file_idx <= 7:
            if side == 0:  # White
                for rank_idx in range(r + 1, 8):
                    mask |= np.uint64(1) << np.uint64(rank_idx * 8 + file_idx)
            else:  # Black
                for rank_idx in range(0, r):
                    mask |= np.uint64(1) << np.uint64(rank_idx * 8 + file_idx)
    return (mask & enemy_pawns_bb) == np.uint64(0)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_endgame_scale_factor(piece_bbs, occupancy_bbs, game_state, phase, eg_score):
    wp = count_bits(piece_bbs[0])
    wn = count_bits(piece_bbs[1])
    wb = count_bits(piece_bbs[2])
    wr = count_bits(piece_bbs[3])
    wq = count_bits(piece_bbs[4])
    
    bp = count_bits(piece_bbs[6])
    bn = count_bits(piece_bbs[7])
    bb = count_bits(piece_bbs[8])
    br = count_bits(piece_bbs[9])
    bq = count_bits(piece_bbs[10])
    
    w_minor_major = wn + wb + wr + wq
    b_minor_major = bn + bb + br + bq
    
    wk_sq = get_lsb_index(piece_bbs[5])
    bk_sq = get_lsb_index(piece_bbs[11])
    
    # Pre-determine strong side for later generic scaling
    # White is strong side (0) if eg_score >= 0, otherwise Black is strong side (1)
    strong_side = 0 if eg_score >= 0 else 1
    pawn_count_strong = wp if strong_side == 0 else bp

    # 1. KBPsK (King + Bishop + Pawns vs King)
    # SF11 endgame.cpp lines 334-394
    if w_minor_major == 1 and wb == 1 and b_minor_major == 0 and wp > 0:
        wb_sq = get_lsb_index(piece_bbs[2])
        all_on_a = (piece_bbs[0] & ~FILE_MASKS[0]) == np.uint64(0)
        all_on_h = (piece_bbs[0] & ~FILE_MASKS[7]) == np.uint64(0)
        wb_color = ((wb_sq % 8) + (wb_sq // 8)) & 1
        
        # A/H rook pawn + wrong-colored bishop: defending king near queening square
        if all_on_a:
            if wb_color == 0:  # Dark bishop vs White a8 (light)
                if CHEBYSHEV_DISTANCE[bk_sq, 56] <= 1:
                    return SCALE_FACTOR_DRAW
        elif all_on_h:
            if wb_color == 1:  # Light bishop vs White h8 (dark)
                if CHEBYSHEV_DISTANCE[bk_sq, 63] <= 1:
                    return SCALE_FACTOR_DRAW

        # B/G file case: if weak side has a pawn blocking on 7th rank with a weak king nearby
        # SF11 endgame.cpp lines 359-392
        all_on_b = (piece_bbs[0] & ~FILE_MASKS[1]) == np.uint64(0)
        all_on_g = (piece_bbs[0] & ~FILE_MASKS[6]) == np.uint64(0)
        if (all_on_b or all_on_g) and count_bits(piece_bbs[6]) >= 1:
            # Find black's frontmost pawn (highest rank = most advanced from white's view)
            pb_temp = piece_bbs[6]
            front_rank = -1
            front_sq = 0
            while pb_temp != np.uint64(0):
                sq = get_lsb_index(pb_temp)
                if sq // 8 > front_rank:
                    front_rank = sq // 8
                    front_sq = sq
                pb_temp &= np.uint64(pb_temp - 1)
            # weak pawn on 7th rank (rank index 6), blocked by white pawn directly below it
            if front_rank == 6:
                block_sq = front_sq - 8
                has_block = (piece_bbs[0] & (np.uint64(1) << np.uint64(block_sq))) != np.uint64(0)
                if has_block:
                    bishop_wrong = opposite_colors(wb_sq, front_sq)
                    if bishop_wrong or count_bits(piece_bbs[0]) == 1:
                        strong_king_dist = CHEBYSHEV_DISTANCE[front_sq, wk_sq]
                        weak_king_dist = CHEBYSHEV_DISTANCE[front_sq, bk_sq]
                        # Weak king on 7th/8th rank, close to blockade, not further than strong king
                        if bk_sq // 8 >= 6 and weak_king_dist <= 2 and weak_king_dist <= strong_king_dist:
                            return SCALE_FACTOR_DRAW

    if b_minor_major == 1 and bb == 1 and w_minor_major == 0 and bp > 0:
        bb_sq = get_lsb_index(piece_bbs[8])
        all_on_a = (piece_bbs[6] & ~FILE_MASKS[0]) == np.uint64(0)
        all_on_h = (piece_bbs[6] & ~FILE_MASKS[7]) == np.uint64(0)
        bb_color = ((bb_sq % 8) + (bb_sq // 8)) & 1
        
        # A/H rook pawn + wrong-colored bishop
        if all_on_a:
            if bb_color == 1:  # Light bishop vs Black a1 (dark)
                if CHEBYSHEV_DISTANCE[wk_sq, 0] <= 1:
                    return SCALE_FACTOR_DRAW
        elif all_on_h:
            if bb_color == 0:  # Dark bishop vs Black h1 (light)
                if CHEBYSHEV_DISTANCE[wk_sq, 7] <= 1:
                    return SCALE_FACTOR_DRAW

        # B/G file case (mirror of white's case)
        all_on_b = (piece_bbs[6] & ~FILE_MASKS[1]) == np.uint64(0)
        all_on_g = (piece_bbs[6] & ~FILE_MASKS[6]) == np.uint64(0)
        if (all_on_b or all_on_g) and count_bits(piece_bbs[0]) >= 1:
            # Find white's frontmost pawn (lowest rank = most advanced from black's view)
            pw_temp = piece_bbs[0]
            front_rank = 8
            front_sq = 0
            while pw_temp != np.uint64(0):
                sq = get_lsb_index(pw_temp)
                if sq // 8 < front_rank:
                    front_rank = sq // 8
                    front_sq = sq
                pw_temp &= np.uint64(pw_temp - 1)
            # Weak (white) pawn on 2nd rank (rank index 1), blocked by black pawn directly above
            if front_rank == 1:
                block_sq = front_sq + 8
                has_block = (piece_bbs[6] & (np.uint64(1) << np.uint64(block_sq))) != np.uint64(0)
                if has_block:
                    bishop_wrong = opposite_colors(bb_sq, front_sq)
                    if bishop_wrong or count_bits(piece_bbs[6]) == 1:
                        strong_king_dist = CHEBYSHEV_DISTANCE[front_sq, bk_sq]
                        weak_king_dist = CHEBYSHEV_DISTANCE[front_sq, wk_sq]
                        if wk_sq // 8 <= 1 and weak_king_dist <= 2 and weak_king_dist <= strong_king_dist:
                            return SCALE_FACTOR_DRAW
                    
    # 2. KBPKB (King + Bishop + Pawn vs King + Bishop)
    # SF11 endgame.cpp lines 619-645
    # 2. KBPKB (King + Bishop + Pawn vs King + Bishop)
    # SF11 endgame.cpp lines 619-645
    if w_minor_major == 1 and wb == 1 and wp == 1 and b_minor_major == 1 and bb == 1 and bp == 0:
        wb_sq = get_lsb_index(piece_bbs[2])
        bb_sq = get_lsb_index(piece_bbs[8])
        wb_color = ((wb_sq % 8) + (wb_sq // 8)) & 1
        bb_color = ((bb_sq % 8) + (bb_sq // 8)) & 1
        p_sq = get_lsb_index(piece_bbs[0])
        # Case 1: Defending king blocks the pawn and cannot be driven away
        if (bk_sq % 8 == p_sq % 8
                and p_sq // 8 < bk_sq // 8
                and (opposite_colors(bk_sq, wb_sq) or bk_sq // 8 <= 5)):
            return SCALE_FACTOR_DRAW
        # Case 2: Opposite colored bishops
        if wb_color != bb_color:
            return SCALE_FACTOR_DRAW

    if b_minor_major == 1 and bb == 1 and bp == 1 and w_minor_major == 1 and wb == 1 and wp == 0:
        wb_sq = get_lsb_index(piece_bbs[2])
        bb_sq = get_lsb_index(piece_bbs[8])
        wb_color = ((wb_sq % 8) + (wb_sq // 8)) & 1
        bb_color = ((bb_sq % 8) + (bb_sq // 8)) & 1
        p_sq = get_lsb_index(piece_bbs[6])
        # Case 1: Defending king blocks the pawn (black moves down, relative rank)
        if (wk_sq % 8 == p_sq % 8
                and p_sq // 8 > wk_sq // 8
                and (opposite_colors(wk_sq, bb_sq) or wk_sq // 8 >= 2)):
            return SCALE_FACTOR_DRAW
        # Case 2: Opposite colored bishops
        if wb_color != bb_color:
            return SCALE_FACTOR_DRAW

    # KPsK (King + 2+ Pawns vs King on same rook file)
    # SF11 endgame.cpp lines 596-616
    if w_minor_major == 0 and wp >= 2 and b_minor_major == 0 and bp == 0:
        all_on_a = (piece_bbs[0] & ~FILE_MASKS[0]) == np.uint64(0)
        all_on_h = (piece_bbs[0] & ~FILE_MASKS[7]) == np.uint64(0)
        if all_on_a or all_on_h:
            p_file = 0 if all_on_a else 7
            if abs(bk_sq % 8 - p_file) <= 1:
                pw_temp = piece_bbs[0]
                ok = True
                while pw_temp:
                    sq = get_lsb_index(pw_temp)
                    if sq // 8 >= bk_sq // 8:
                        ok = False
                        break
                    pw_temp &= pw_temp - np.uint64(1)
                if ok:
                    return SCALE_FACTOR_DRAW

    if b_minor_major == 0 and bp >= 2 and w_minor_major == 0 and wp == 0:
        all_on_a = (piece_bbs[6] & ~FILE_MASKS[0]) == np.uint64(0)
        all_on_h = (piece_bbs[6] & ~FILE_MASKS[7]) == np.uint64(0)
        if all_on_a or all_on_h:
            p_file = 0 if all_on_a else 7
            if abs(wk_sq % 8 - p_file) <= 1:
                pb_temp = piece_bbs[6]
                ok = True
                while pb_temp:
                    sq = get_lsb_index(pb_temp)
                    if sq // 8 <= wk_sq // 8:
                        ok = False
                        break
                    pb_temp &= pb_temp - np.uint64(1)
                if ok:
                    return SCALE_FACTOR_DRAW

    # KNPKB (Knight + Pawn vs Bishop)
    # SF11 endgame.cpp lines 758-776
    if w_minor_major == 1 and wn == 1 and wp == 1 and b_minor_major == 1 and bb == 1 and bp == 0:
        p_sq = get_lsb_index(piece_bbs[0])
        b_sq = get_lsb_index(piece_bbs[8])
        b_attacks = get_bishop_attacks(b_sq, occupancy_bbs[2])
        p_file = p_sq % 8
        p_rank = p_sq // 8
        forward_mask = FILE_MASKS[p_file] & ~((np.uint64(1) << np.uint64((p_rank + 1) * 8)) - np.uint64(1))
        if b_attacks & forward_mask:
            return np.int32(CHEBYSHEV_DISTANCE[bk_sq, p_sq])

    if b_minor_major == 1 and bn == 1 and bp == 1 and w_minor_major == 1 and wb == 1 and wp == 0:
        p_sq = get_lsb_index(piece_bbs[6])
        b_sq = get_lsb_index(piece_bbs[2])
        b_attacks = get_bishop_attacks(b_sq, occupancy_bbs[2])
        p_file = p_sq % 8
        p_rank = p_sq // 8
        forward_mask = FILE_MASKS[p_file] & ((np.uint64(1) << np.uint64(p_rank * 8)) - np.uint64(1))
        if b_attacks & forward_mask:
            return np.int32(CHEBYSHEV_DISTANCE[wk_sq, p_sq])

    # KBPPKB (Bishop + 2 Pawns vs Bishop, opposite-colored)
    # SF11 endgame.cpp lines 649-713
    if w_minor_major == 1 and wb == 1 and wp == 2 and b_minor_major == 1 and bb == 1 and bp == 0:
        wbsq = get_lsb_index(piece_bbs[2])
        bbsq = get_lsb_index(piece_bbs[8])
        if opposite_colors(wbsq, bbsq):
            pw_temp = piece_bbs[0]
            psq1 = get_lsb_index(pw_temp)
            pw_temp &= pw_temp - np.uint64(1)
            psq2 = get_lsb_index(pw_temp)
            if psq1 // 8 > psq2 // 8:
                blockSq1 = psq1 + 8
                blockSq2 = (psq2 % 8) + (psq1 // 8) * 8
            else:
                blockSq1 = psq2 + 8
                blockSq2 = (psq1 % 8) + (psq2 // 8) * 8
            file_dist = abs(psq1 % 8 - psq2 % 8)
            if file_dist == 0:
                if bk_sq % 8 == blockSq1 % 8 and bk_sq // 8 >= blockSq1 // 8 and opposite_colors(bk_sq, wbsq):
                    return SCALE_FACTOR_DRAW
            elif file_dist == 1:
                b_attacks = get_bishop_attacks(bbsq, occupancy_bbs[2])
                rank_dist = abs(psq1 // 8 - psq2 // 8)
                if bk_sq == blockSq1 and opposite_colors(bk_sq, wbsq):
                    if bbsq == blockSq2 or (b_attacks & (np.uint64(1) << np.uint64(blockSq2))) or rank_dist >= 2:
                        return SCALE_FACTOR_DRAW
                elif bk_sq == blockSq2 and opposite_colors(bk_sq, wbsq):
                    if bbsq == blockSq1 or (b_attacks & (np.uint64(1) << np.uint64(blockSq1))):
                        return SCALE_FACTOR_DRAW

    if b_minor_major == 1 and bb == 1 and bp == 2 and w_minor_major == 1 and wb == 1 and wp == 0:
        wbsq = get_lsb_index(piece_bbs[2])
        bbsq = get_lsb_index(piece_bbs[8])
        if opposite_colors(wbsq, bbsq):
            pb_temp = piece_bbs[6]
            psq1 = get_lsb_index(pb_temp)
            pb_temp &= pb_temp - np.uint64(1)
            psq2 = get_lsb_index(pb_temp)
            if psq1 // 8 < psq2 // 8:
                blockSq1 = psq1 - 8
                blockSq2 = (psq2 % 8) + (psq1 // 8) * 8
            else:
                blockSq1 = psq2 - 8
                blockSq2 = (psq1 % 8) + (psq2 // 8) * 8
            file_dist = abs(psq1 % 8 - psq2 % 8)
            if file_dist == 0:
                if wk_sq % 8 == blockSq1 % 8 and wk_sq // 8 <= blockSq1 // 8 and opposite_colors(wk_sq, bbsq):
                    return SCALE_FACTOR_DRAW
            elif file_dist == 1:
                w_attacks = get_bishop_attacks(wbsq, occupancy_bbs[2])
                rank_dist = abs(psq1 // 8 - psq2 // 8)
                if wk_sq == blockSq1 and opposite_colors(wk_sq, bbsq):
                    if wbsq == blockSq2 or (w_attacks & (np.uint64(1) << np.uint64(blockSq2))) or rank_dist >= 2:
                        return SCALE_FACTOR_DRAW
                elif wk_sq == blockSq2 and opposite_colors(wk_sq, bbsq):
                    if wbsq == blockSq1 or (w_attacks & (np.uint64(1) << np.uint64(blockSq1))):
                        return SCALE_FACTOR_DRAW

    # KRPPKRP (Rook + 2 Pawns vs Rook + Pawn)
    # SF11 endgame.cpp lines 567-593
    if w_minor_major == 1 and wr == 1 and wp == 2 and b_minor_major == 1 and br == 1 and bp == 1:
        pw_temp = piece_bbs[0]
        wpsq1 = get_lsb_index(pw_temp)
        pw_temp &= pw_temp - np.uint64(1)
        wpsq2 = get_lsb_index(pw_temp)
        bp_bb = piece_bbs[6]
        if not (is_pawn_passed_on_the_fly(wpsq1, 0, bp_bb) or is_pawn_passed_on_the_fly(wpsq2, 0, bp_bb)):
            r = max(wpsq1 // 8, wpsq2 // 8)
            if (abs(bk_sq % 8 - wpsq1 % 8) <= 1
                    and abs(bk_sq % 8 - wpsq2 % 8) <= 1
                    and bk_sq // 8 > r):
                if 1 < r < 6:
                    return np.int32([0, 9, 10, 14, 21, 44, 0, 0][r])

    if b_minor_major == 1 and br == 1 and bp == 2 and w_minor_major == 1 and wr == 1 and wp == 1:
        pb_temp = piece_bbs[6]
        wpsq1 = get_lsb_index(pb_temp)
        pb_temp &= pb_temp - np.uint64(1)
        wpsq2 = get_lsb_index(pb_temp)
        wp_bb = piece_bbs[0]
        if not (is_pawn_passed_on_the_fly(wpsq1, 1, wp_bb) or is_pawn_passed_on_the_fly(wpsq2, 1, wp_bb)):
            r = max(7 - wpsq1 // 8, 7 - wpsq2 // 8)
            if (abs(wk_sq % 8 - wpsq1 % 8) <= 1
                    and abs(wk_sq % 8 - wpsq2 % 8) <= 1
                    and (7 - wk_sq // 8) > r):
                if 1 < r < 6:
                    return np.int32([0, 9, 10, 14, 21, 44, 0, 0][r])

    # 3. KBPKN (King + Bishop + Pawn vs King + Knight)
    # SF11 endgame.cpp lines 716-736
    # Draw if defending king blocks pawn's file while in front, AND
    # (king is on opposite color to bishop, OR king is not yet on 7th rank)
    if w_minor_major == 1 and wb == 1 and wp == 1 and b_minor_major == 1 and bn == 1 and bp == 0:
        wb_sq = get_lsb_index(piece_bbs[2])
        p_sq = get_lsb_index(piece_bbs[0])
        if (bk_sq % 8 == p_sq % 8
                and p_sq // 8 < bk_sq // 8  # king in front of pawn
                and (opposite_colors(bk_sq, wb_sq) or bk_sq // 8 <= 5)):  # rel_rank <= RANK_6
            return SCALE_FACTOR_DRAW

    if b_minor_major == 1 and bb == 1 and bp == 1 and w_minor_major == 1 and wn == 1 and wp == 0:
        bb_sq = get_lsb_index(piece_bbs[8])
        p_sq = get_lsb_index(piece_bbs[6])
        if (wk_sq % 8 == p_sq % 8
                and p_sq // 8 > wk_sq // 8  # king in front of pawn (black pawn moves down)
                and (opposite_colors(wk_sq, bb_sq) or wk_sq // 8 >= 2)):  # rel_rank <= RANK_6
            return SCALE_FACTOR_DRAW

    # 4. KNPK (King + Knight + Pawn vs King)
    # SF11 endgame.cpp lines 739-755
    # Draw only when: normalized pawn is on a7 AND defending king is within 1 of a8
    if w_minor_major == 1 and wn == 1 and wp == 1 and b_minor_major == 0 and bp == 0:
        wp_sq = get_lsb_index(piece_bbs[0])
        wpsq_norm = normalize_square(wp_sq, 0, wp_sq)
        bksq_norm = normalize_square(bk_sq, 0, wp_sq)
        if wpsq_norm == 48 and CHEBYSHEV_DISTANCE[56, bksq_norm] <= 1:
            return SCALE_FACTOR_DRAW

    if b_minor_major == 1 and bn == 1 and bp == 1 and w_minor_major == 0 and wp == 0:
        bp_sq = get_lsb_index(piece_bbs[6])
        bpsq_norm = normalize_square(bp_sq, 1, bp_sq)
        wksq_norm = normalize_square(wk_sq, 1, bp_sq)
        if bpsq_norm == 48 and CHEBYSHEV_DISTANCE[56, wksq_norm] <= 1:
            return SCALE_FACTOR_DRAW

    # 3. KRPKB (Rook + Pawn vs Bishop)
    if w_minor_major == 1 and wr == 1 and wp == 1 and b_minor_major == 1 and bb == 1 and bp == 0:
        bk_sq_orig = bk_sq
        bb_sq_orig = get_lsb_index(piece_bbs[8])
        wp_sq_orig = get_lsb_index(piece_bbs[0])
        if wp_sq_orig % 8 == 0 or wp_sq_orig % 8 == 7:
            ksq = normalize_square(bk_sq_orig, 0, wp_sq_orig)
            bsq = normalize_square(bb_sq_orig, 0, wp_sq_orig)
            psq = normalize_square(wp_sq_orig, 0, wp_sq_orig)
            rk = psq // 8
            if rk == 4 and not opposite_colors(bsq, psq):
                d = CHEBYSHEV_DISTANCE[psq + 24, ksq]
                if d <= 2:
                    return 24
                else:
                    return 48
            elif rk == 5:
                prom_sq = psq + 16
                if CHEBYSHEV_DISTANCE[prom_sq, ksq] <= 1:
                    if abs(bsq // 8 - (psq + 8) // 8) == abs(bsq % 8 - (psq + 8) % 8):
                        if abs(bsq % 8 - psq % 8) >= 2:
                            return 8

    if b_minor_major == 1 and br == 1 and bp == 1 and w_minor_major == 1 and wb == 1 and wp == 0:
        wk_sq_orig = wk_sq
        wb_sq_orig = get_lsb_index(piece_bbs[2])
        bp_sq_orig = get_lsb_index(piece_bbs[6])
        if bp_sq_orig % 8 == 0 or bp_sq_orig % 8 == 7:
            ksq = normalize_square(wk_sq_orig, 1, bp_sq_orig)
            bsq = normalize_square(wb_sq_orig, 1, bp_sq_orig)
            psq = normalize_square(bp_sq_orig, 1, bp_sq_orig)
            rk = psq // 8
            if rk == 4 and not opposite_colors(bsq, psq):
                d = CHEBYSHEV_DISTANCE[psq + 24, ksq]
                if d <= 2:
                    return 24
                else:
                    return 48
            elif rk == 5:
                prom_sq = psq + 16
                if CHEBYSHEV_DISTANCE[prom_sq, ksq] <= 1:
                    if abs(bsq // 8 - (psq + 8) // 8) == abs(bsq % 8 - (psq + 8) % 8):
                        if abs(bsq % 8 - psq % 8) >= 2:
                            return 8

    # 4. KRPKN (Rook + Pawn vs Knight)
    if w_minor_major == 1 and wr == 1 and wp == 1 and b_minor_major == 1 and bn == 1 and bp == 0:
        p_sq = get_lsb_index(piece_bbs[0])
        p_file = p_sq % 8
        p_rank = p_sq // 8
        bk_file = bk_sq % 8
        bk_rank = bk_sq // 8
        bn_sq = get_lsb_index(piece_bbs[7])
        if bk_file == p_file and bk_rank > p_rank:
            if CHEBYSHEV_DISTANCE[bk_sq, p_sq + 8] <= 2 and CHEBYSHEV_DISTANCE[bn_sq, p_sq + 8] <= 2:
                return SCALE_FACTOR_DRAW

    if b_minor_major == 1 and br == 1 and bp == 1 and w_minor_major == 1 and wn == 1 and wp == 0:
        p_sq = get_lsb_index(piece_bbs[6])
        p_file = p_sq % 8
        p_rank = p_sq // 8
        wk_file = wk_sq % 8
        wk_rank = wk_sq // 8
        wn_sq = get_lsb_index(piece_bbs[1])
        if wk_file == p_file and wk_rank < p_rank:
            if CHEBYSHEV_DISTANCE[wk_sq, p_sq - 8] <= 2 and CHEBYSHEV_DISTANCE[wn_sq, p_sq - 8] <= 2:
                return SCALE_FACTOR_DRAW

    # 5. KRKP (Rook vs Pawn)
    if w_minor_major == 1 and wr == 1 and wp == 0 and b_minor_major == 0 and bp == 1:
        p_sq = get_lsb_index(piece_bbs[6])
        p_rank = p_sq // 8
        p_file = p_sq % 8
        if p_rank == 1:  # Black pawn on 2nd rank
            promo_sq = p_file
            if CHEBYSHEV_DISTANCE[wk_sq, promo_sq] > 3 and CHEBYSHEV_DISTANCE[bk_sq, promo_sq] <= 1:
                return SCALE_FACTOR_KRPKR_FORTRESS
                
    if b_minor_major == 1 and br == 1 and bp == 0 and w_minor_major == 0 and wp == 1:
        p_sq = get_lsb_index(piece_bbs[0])
        p_rank = p_sq // 8
        p_file = p_sq % 8
        if p_rank == 6:  # White pawn on 7th rank
            promo_sq = p_file + 56
            if CHEBYSHEV_DISTANCE[bk_sq, promo_sq] > 3 and CHEBYSHEV_DISTANCE[wk_sq, promo_sq] <= 1:
                return SCALE_FACTOR_KRPKR_FORTRESS

    # 6. KRPKR (Rook + Pawn vs Rook) - PHILIDOR & GLAURUNG RULES
    is_krpkr = False
    strong_side_rp = 0
    if w_minor_major == 1 and wr == 1 and wp == 1 and b_minor_major == 1 and br == 1 and bp == 0:
        is_krpkr = True
        strong_side_rp = 0
        wksq_orig, bksq_orig = wk_sq, bk_sq
        wrsq_orig, brsq_orig = get_lsb_index(piece_bbs[3]), get_lsb_index(piece_bbs[9])
        wpsq_orig = get_lsb_index(piece_bbs[0])
    elif b_minor_major == 1 and br == 1 and bp == 1 and w_minor_major == 1 and wr == 1 and wp == 0:
        is_krpkr = True
        strong_side_rp = 1
        wksq_orig, bksq_orig = bk_sq, wk_sq
        wrsq_orig, brsq_orig = get_lsb_index(piece_bbs[9]), get_lsb_index(piece_bbs[3])
        wpsq_orig = get_lsb_index(piece_bbs[6])

    if is_krpkr:
        wksq = normalize_square(wksq_orig, strong_side_rp, wpsq_orig)
        bksq = normalize_square(bksq_orig, strong_side_rp, wpsq_orig)
        wrsq = normalize_square(wrsq_orig, strong_side_rp, wpsq_orig)
        wpsq = normalize_square(wpsq_orig, strong_side_rp, wpsq_orig)
        brsq = normalize_square(brsq_orig, strong_side_rp, wpsq_orig)

        f = wpsq % 8
        r = wpsq // 8
        queening_sq = f + 56
        side_to_move = game_state[0]
        tempo = 1 if side_to_move == strong_side_rp else 0

        # Third-rank defense
        if (r <= 4 
            and CHEBYSHEV_DISTANCE[bksq, queening_sq] <= 1 
            and wksq <= 39 
            and (brsq // 8 == 5 or (r <= 2 and wrsq // 8 != 5))):
            return SCALE_FACTOR_DRAW

        # Check from behind
        if (r == 5 
            and CHEBYSHEV_DISTANCE[bksq, queening_sq] <= 1 
            and (wksq // 8) + tempo <= 5 
            and (brsq // 8 == 0 or (tempo == 0 and abs(brsq % 8 - f) >= 3))):
            return SCALE_FACTOR_DRAW

        # Queening square + rook behind
        if (r >= 5 
            and bksq == queening_sq 
            and brsq // 8 == 0 
            and (tempo == 0 or CHEBYSHEV_DISTANCE[wksq, wpsq] >= 2)):
            return SCALE_FACTOR_DRAW

        # A7 + A8 corner rule
        if (wpsq == 48 
            and wrsq == 56 
            and (bksq == 55 or bksq == 54) 
            and brsq % 8 == 0 
            and (brsq // 8 <= 2 or wksq % 8 >= 3 or wksq // 8 <= 4)):
            return SCALE_FACTOR_DRAW

        # King blocks pawn
        if (r <= 4 
            and bksq == wpsq + 8 
            and CHEBYSHEV_DISTANCE[wksq, wpsq] - tempo >= 2 
            and CHEBYSHEV_DISTANCE[wksq, brsq] - tempo >= 2):
            return SCALE_FACTOR_DRAW

        # Pawn on 7th rank progressive scaling (SCALE_FACTOR_MAX = 128)
        if (r == 6 
            and f != 0 
            and wrsq % 8 == f 
            and wrsq != queening_sq 
            and CHEBYSHEV_DISTANCE[wksq, queening_sq] < CHEBYSHEV_DISTANCE[bksq, queening_sq] - 2 + tempo 
            and CHEBYSHEV_DISTANCE[wksq, queening_sq] < CHEBYSHEV_DISTANCE[bksq, wrsq] + tempo):
            return max(0, 128 - 2 * CHEBYSHEV_DISTANCE[wksq, queening_sq])

        # Pawn further back progressive scaling (SCALE_FACTOR_MAX = 128)
        if (f != 0 
            and wrsq % 8 == f 
            and wrsq < wpsq 
            and CHEBYSHEV_DISTANCE[wksq, queening_sq] < CHEBYSHEV_DISTANCE[bksq, queening_sq] - 2 + tempo 
            and CHEBYSHEV_DISTANCE[wksq, wpsq + 8] < CHEBYSHEV_DISTANCE[bksq, wpsq + 8] - 2 + tempo 
            and (CHEBYSHEV_DISTANCE[bksq, wrsq] + tempo >= 3 
                 or (CHEBYSHEV_DISTANCE[wksq, queening_sq] < CHEBYSHEV_DISTANCE[bksq, wrsq] + tempo 
                     and CHEBYSHEV_DISTANCE[wksq, wpsq + 8] < CHEBYSHEV_DISTANCE[bksq, wrsq] + tempo))):
            val = 128 - 8 * CHEBYSHEV_DISTANCE[wpsq, queening_sq] - 2 * CHEBYSHEV_DISTANCE[wksq, queening_sq]
            return max(0, val)

        # Pawn not far advanced progressive scaling
        if r <= 3 and bksq > wpsq:
            if bksq % 8 == f:
                return 10
            if abs(bksq % 8 - f) == 1 and CHEBYSHEV_DISTANCE[wksq, bksq] > 2:
                val = 24 - 2 * CHEBYSHEV_DISTANCE[wksq, bksq]
                return max(0, val)

    # 7. KQKRPs (Queen vs Rook + Pawn)
    if w_minor_major == 1 and wq == 1 and wp == 0 and b_minor_major == 1 and br == 1 and bp == 1:
        p_sq = get_lsb_index(piece_bbs[6])
        br_sq = get_lsb_index(piece_bbs[9])
        if 7 - (bk_sq // 8) <= 1 and 7 - (wk_sq // 8) >= 3 and 7 - (br_sq // 8) == 2:
            if CHEBYSHEV_DISTANCE[bk_sq, p_sq] <= 1:
                if p_sq // 8 == br_sq // 8 + 1 and abs(p_sq % 8 - br_sq % 8) == 1:
                    return SCALE_FACTOR_DRAW
                    
    if b_minor_major == 1 and bq == 1 and bp == 0 and w_minor_major == 1 and wr == 1 and wp == 1:
        p_sq = get_lsb_index(piece_bbs[0])
        wr_sq = get_lsb_index(piece_bbs[3])
        if wk_sq // 8 <= 1 and bk_sq // 8 >= 3 and wr_sq // 8 == 2:
            if CHEBYSHEV_DISTANCE[wk_sq, p_sq] <= 1:
                if p_sq // 8 == wr_sq // 8 - 1 and abs(p_sq % 8 - wr_sq % 8) == 1:
                    return SCALE_FACTOR_DRAW

    # 8. material.cpp GENERAL NO-PAWN SCALING
    # SF11 material.cpp: when stronger side has no pawns and advantage <= bishop value
    # Covers: KRKB/KRKN (sf=4), KRKR/symmetric (sf=14)
    # BishopValueMg threshold ≈ 340 cp, RookValueMg threshold ≈ 530 cp
    if pawn_count_strong == 0:
        npm_w = wn * 310 + wb * 340 + wr * 530 + wq * 950
        npm_b = bn * 310 + bb * 340 + br * 530 + bq * 950
        npm_strong = npm_w if strong_side == 0 else npm_b
        npm_weak   = npm_b if strong_side == 0 else npm_w
        if npm_strong - npm_weak <= 340:  # advantage <= BishopValueMg
            if npm_strong < 530:  # weaker than a rook: pure minor advantage, nearly draw
                return SCALE_FACTOR_DRAW
            elif npm_weak <= 340:  # weak side has only a minor piece: sf=4
                return 4
            else:  # symmetric or near-symmetric (KRKR, etc.): sf=14
                return 14

    # 9. GENERAL OCB & PAWN COUNT SCALING (Fall-through if no specialized scale factor was triggered)
    # SF11 evaluate.cpp scale_factor() lines 732-751
    is_ocb = False
    if wb == 1 and bb == 1:
        wb_sq = get_lsb_index(piece_bbs[2])
        bb_sq = get_lsb_index(piece_bbs[8])
        if opposite_colors(wb_sq, bb_sq):
            is_ocb = True

    sf = SCALE_FACTOR_NORMAL  # 64

    if is_ocb and wn == 0 and bn == 0 and wr == 0 and br == 0 and wq == 0 and bq == 0:
        # Pure OCB: no other pieces except kings, opposite-colored bishops and pawns
        sf = 22
    else:
        # General pawn count scaling: multiplier is 2 if opposite-colored bishops exist, 7 otherwise
        mult = 2 if is_ocb else 7
        sf = min(sf, 36 + mult * pawn_count_strong)

    # 50-move rule decay (halfmove clock is game_state[3])
    rule50 = int(game_state[3])
    if rule50 > 12:
        sf = max(0, sf - (rule50 - 12) // 4)

    return sf
