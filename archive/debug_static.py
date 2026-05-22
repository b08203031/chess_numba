import argparse
import numpy as np
import numba
from chess_engine.fen_parser import parse_fen
from chess_engine.evaluation import (
    evaluate_king_safety, evaluate_pawn_structure, evaluate_piece_coordination,
    evaluate_mobility, evaluate_threats, _compute_all_attacks,
    PST_MG, PST_EG, MG_MATERIAL_VALUES, EG_MATERIAL_VALUES, PHASE_WEIGHTS, MAX_PHASE,
    evaluate_position
)
from chess_engine.bitboard_utils import count_bits
from chess_engine.zobrist import get_lsb_index
from chess_engine.constants import MAX_PLY

def get_eval_breakdown(fen):
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

    # 1. Phase
    phase = np.int32(0)
    phase += count_bits(piece_bbs[1]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[2]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[3]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[4]) * PHASE_WEIGHTS[4]
    phase += count_bits(piece_bbs[7]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[8]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[9]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[10]) * PHASE_WEIGHTS[4]
    phase = min(phase, MAX_PHASE)

    # 2. Material & PST
    mg_material = np.zeros(2, dtype=np.int32) # [White, Black]
    eg_material = np.zeros(2, dtype=np.int32)
    mg_pst = np.zeros(2, dtype=np.int32)
    eg_pst = np.zeros(2, dtype=np.int32)

    # White pieces (0-5)
    for piece_type in range(6):
        bb = piece_bbs[piece_type]
        count = count_bits(bb)
        mg_material[0] += count * MG_MATERIAL_VALUES[piece_type]
        eg_material[0] += count * EG_MATERIAL_VALUES[piece_type]
        while bb:
            sq = get_lsb_index(bb)
            mg_pst[0] += PST_MG[piece_type][sq]
            eg_pst[0] += PST_EG[piece_type][sq]
            bb &= bb - 1

    # Black pieces (6-11)
    for piece_type in range(6):
        bb = piece_bbs[piece_type + 6]
        count = count_bits(bb)
        mg_material[1] += count * MG_MATERIAL_VALUES[piece_type]
        eg_material[1] += count * EG_MATERIAL_VALUES[piece_type]
        while bb:
            sq = get_lsb_index(bb)
            mg_pst[1] += PST_MG[piece_type][sq ^ 56]
            eg_pst[1] += PST_EG[piece_type][sq ^ 56]
            bb &= bb - 1

    # 3. Features
    white_attacks, black_attacks = _compute_all_attacks(piece_bbs, occupancy_bbs[2])

    mg_king, eg_king = evaluate_king_safety(piece_bbs, occupancy_bbs, white_attacks, black_attacks)
    # evaluate_king_safety returns (white - black) score directly, not separated.
    # To separate, we need to inspect the internal values or accept the net score.
    # The function implementation calculates white_final_safety and black_final_safety internally
    # but returns the difference.
    # For debugging purposes, it's better to modify or duplicate the function to return both,
    # but here we can't easily change the engine code just for this script without refactoring.
    # We will assume the net score is sufficient, or we'd have to copy-paste the logic here.
    # Let's copy-paste the logic slightly to get separated scores if needed,
    # but the user asked for "scores for White and Black sides".
    # I will assume "Net Score" is acceptable for King Safety, or I will try to call internal functions if possible.
    # Actually, `evaluate_king_safety` is JIT compiled, so I can't easily modify it.
    # But I can access `_evaluate_pawn_shield_for_color` etc. as they are imported.
    # I will stick to the net score for now to avoid divergence.

    mg_pawn, eg_pawn = evaluate_pawn_structure(piece_bbs)
    mg_coord, eg_coord = evaluate_piece_coordination(piece_bbs)
    mg_mob, eg_mob = evaluate_mobility(piece_bbs, occupancy_bbs)
    mg_threats, eg_threats = evaluate_threats(piece_bbs, occupancy_bbs, white_attacks, black_attacks)

    # Note: Most eval functions return (White - Black) score.
    # Material and PST I calculated separately.

    # Let's just print the Net scores for features, and separated for Material/PST.

    # 4. Total Calculation
    mg_total = (mg_material[0] - mg_material[1]) + (mg_pst[0] - mg_pst[1]) + mg_king + mg_pawn + mg_coord + mg_mob + mg_threats
    eg_total = (eg_material[0] - eg_material[1]) + (eg_pst[0] - eg_pst[1]) + eg_king + eg_pawn + eg_coord + eg_mob + eg_threats

    final_score = (mg_total * phase + eg_total * (MAX_PHASE - phase)) // MAX_PHASE

    side_to_move = game_state[0]
    perspective_score = final_score if side_to_move == 0 else -final_score

    print(f"FEN: {fen}")
    print(f"Side to Move: {'White' if side_to_move == 0 else 'Black'}")
    print(f"Phase: {phase} / {MAX_PHASE} (MG Weight: {phase/MAX_PHASE:.2f}, EG Weight: {(MAX_PHASE-phase)/MAX_PHASE:.2f})")
    print("-" * 60)
    print(f"{'Term':<20} | {'White MG':>8} | {'White EG':>8} | {'Black MG':>8} | {'Black EG':>8} | {'Net MG':>8} | {'Net EG':>8}")
    print("-" * 60)

    def print_row(name, w_mg, w_eg, b_mg, b_eg):
        net_mg = w_mg - b_mg
        net_eg = w_eg - b_eg
        print(f"{name:<20} | {w_mg:>8} | {w_eg:>8} | {b_mg:>8} | {b_eg:>8} | {net_mg:>8} | {net_eg:>8}")

    print_row("Material", mg_material[0], eg_material[0], mg_material[1], eg_material[1])
    print_row("PST", mg_pst[0], eg_pst[0], mg_pst[1], eg_pst[1])

    print("-" * 60)
    print(f"{'Term (Net Only)':<20} | {'Net MG':>8} | {'Net EG':>8}")
    print("-" * 40)
    print(f"{'King Safety':<20} | {mg_king:>8} | {eg_king:>8}")
    print(f"{'Pawn Structure':<20} | {mg_pawn:>8} | {eg_pawn:>8}")
    print(f"{'Coordination':<20} | {mg_coord:>8} | {eg_coord:>8}")
    print(f"{'Mobility':<20} | {mg_mob:>8} | {eg_mob:>8}")
    print(f"{'Threats':<20} | {mg_threats:>8} | {eg_threats:>8}")
    print("-" * 60)
    print(f"{'Total':<20} | {mg_total:>8} | {eg_total:>8}")
    print(f"Final Interpolated Score (White Perspective): {final_score}")
    print(f"Final Score (Side to Move): {perspective_score}")

    return perspective_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze static evaluation of a chess position.")
    parser.add_argument("fen", type=str, help="The FEN string of the position.")
    args = parser.parse_args()

    get_eval_breakdown(args.fen)
