import time
import numpy as np
import numba
from chess_engine.fen_parser import parse_fen
from chess_engine.board_operations import make_move
from chess_engine.move_generator import generate_legal_moves
from chess_engine.move import move_to_uci, get_special_move_flag, get_to_square, get_from_square, SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT
from chess_engine.constants import *
from chess_engine.evaluation import (
    evaluate_pawn_structure, evaluate_piece_coordination, evaluate_king_safety, evaluate_mobility,
    PST_MG, PST_EG, MG_MATERIAL_VALUES, EG_MATERIAL_VALUES, PHASE_WEIGHTS, MAX_PHASE
)
from chess_engine.bitboard_utils import count_bits
from chess_engine.zobrist import get_lsb_index
from chess_engine.search import iterative_deepening_search
from chess_engine.engine_types import SearchContext
from chess_engine.transposition_table import create_transposition_table
from chess_engine.see import see

# Define the failed puzzles
failed_puzzles = [
    {
        "name": "Lichess Puzzle 000mr",
        "fen": "5r1k/5rp1/p7/1b2B2p/1P1P1Pq1/2R3Q1/P3p1P1/2R3K1 b - - 1 41",
        "solution": "f7f4",
        "engine_move": "g4g3"
    },
    # {
    #     "name": "Lichess Puzzle 004d8",
    #     "fen": "8/4kr2/R2p4/1p1Pp3/5pp1/3K1P2/PPP5/8 w - - 0 40",
    #     "solution": "a6a7",
    #     "engine_move": "f3g4"
    # },
    {
        "name": "Lichess Puzzle 005qG",
        "fen": "8/8/1p1k1p1p/3np3/2B2p2/PP1K1PP1/7P/8 w - - 0 37",
        "solution": "c4d5",
        "engine_move": "d3e4"
    },
    # {
    #     "name": "Lichess Puzzle 0068D",
    #     "fen": "7r/pppk4/2pN1r2/8/3P2p1/2P5/PP2RPP1/4R1K1 b - - 0 26",
    #     "solution": "f6h6",
    #     "engine_move": "d7d6"
    # },
    {
        "name": "Lichess Puzzle 006eO",
        "fen": "8/8/2p5/1p1p1k2/3P4/1PP1pK2/8/8 w - - 4 65",
        "solution": "b3b4",
        "engine_move": "f3e3"
    },
    {
        "name": "Lichess Puzzle 006of",
        "fen": "r2qr2k/1pp2Qp1/1b4np/pP2P3/P4n2/B1N2N1P/5PP1/R3R1K1 b - - 0 20",
        "solution": "d8d3",
        "engine_move": "f4d3"
    },
    {       "name": "Lichess Puzzle 002rd",
            "fen": "r6k/q1p2p1p/1b2bPr1/p1ppP2Q/3P2p1/4B3/PP2NRPP/3R2K1 w - - 2 26",
            "solution": "e2f4",
            "engine_move": "d4c5"
    },
    {
        "name": "Lichess Puzzle 0078T",
        "fen": "rk5r/1b3R2/pp2p2q/4P2p/B6B/4p2P/PP4P1/5Q1K w - - 0 28",
        "solution": "f7b7",
        "engine_move": "h1h2"
    },
    # {
    #     "name": "Lichess Puzzle 006NL",
    #     "fen": "1r6/k2qn1b1/p1N1p1p1/2PpPpN1/2n2P1P/p4B2/1PP2Q2/1K1R3R b - - 0 32",
    #     "solution": "e7c6",
    #     "engine_move": "d7c6"
    # },
    # {
    #     "name": "Lichess Puzzle 007gO",
    #     "fen": "2r3rk/5p2/4p2p/4q3/1Q6/8/1P3PPP/2R2RK1 b - - 1 31",
    #     "solution": "e5g5",
    #     "engine_move": "e5g7"
    # },
]

def find_move_by_uci(piece_bbs, occupancy_bbs, game_state, uci_move_str):
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    for move in moves:
        if move_to_uci(move) == uci_move_str:
            return move
    return None

def is_capture(piece_bbs, occupancy_bbs, game_state, move):
    to_sq = get_to_square(move)
    side_to_move = game_state[0]
    opponent_occupancy = occupancy_bbs[1] if side_to_move == 0 else occupancy_bbs[0]
    
    # Standard capture
    if (1 << to_sq) & opponent_occupancy:
        return True
    
    # En Passant
    if get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT:
        return True
        
    return False

def is_promotion(move):
    return get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION

def get_eval_breakdown(piece_bbs, occupancy_bbs, game_state):
    """
    Deconstructs the evaluation function to return components.
    """
    
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
    mg_material = 0
    eg_material = 0
    mg_pst = 0
    eg_pst = 0

    for piece_type in range(6):
        # White Material
        count = count_bits(piece_bbs[piece_type])
        mg_material += count * MG_MATERIAL_VALUES[piece_type]
        eg_material += count * EG_MATERIAL_VALUES[piece_type]
        
        # Black Material
        count = count_bits(piece_bbs[piece_type + 6])
        mg_material -= count * MG_MATERIAL_VALUES[piece_type]
        eg_material -= count * EG_MATERIAL_VALUES[piece_type]

    for piece_type in range(6):
        bb = piece_bbs[piece_type]
        while bb:
            sq = get_lsb_index(bb)
            mg_pst += PST_MG[piece_type][sq]
            eg_pst += PST_EG[piece_type][sq]
            bb &= bb - 1

    for piece_type in range(6):
        bb = piece_bbs[piece_type + 6]
        while bb:
            sq = get_lsb_index(bb)
            mg_pst -= PST_MG[piece_type][sq ^ 56]
            eg_pst -= PST_EG[piece_type][sq ^ 56]
            bb &= bb - 1

    # 3. Features
    mg_king, eg_king = evaluate_king_safety(piece_bbs, occupancy_bbs)
    mg_pawn, eg_pawn = evaluate_pawn_structure(piece_bbs)
    mg_coord, eg_coord = evaluate_piece_coordination(piece_bbs)
    mg_mob, eg_mob = evaluate_mobility(piece_bbs, occupancy_bbs)

    # 4. Total
    mg_total = mg_material + mg_pst + mg_king + mg_pawn + mg_coord + mg_mob
    eg_total = eg_material + eg_pst + eg_king + eg_pawn + eg_coord + eg_mob
    
    final_score = (mg_total * phase + eg_total * (MAX_PHASE - phase)) // MAX_PHASE
    
    side_to_move = game_state[0]
    perspective_score = final_score if side_to_move == 0 else -final_score

    return {
        "Phase": phase,
        "MG_Material": mg_material, "EG_Material": eg_material,
        "MG_PST": mg_pst, "EG_PST": eg_pst,
        "MG_KingSafety": mg_king, "EG_KingSafety": eg_king,
        "MG_PawnStructure": mg_pawn, "EG_PawnStructure": eg_pawn,
        "MG_Coordination": mg_coord, "EG_Coordination": eg_coord,
        "MG_Mobility": mg_mob, "EG_Mobility": eg_mob,
        "Final_Raw": final_score,
        "Final_Perspective": perspective_score
    }

def print_diff(label, val_correct, val_engine):
    diff = val_correct - val_engine
    print(f"{label:<20}: Correct={val_correct:>6}, Engine={val_engine:>6}, Diff={diff:>6}")

def analyze():
    print("Starting Analysis of Failed Puzzles...")
    
    # Setup Search Context
    transposition_table = create_transposition_table(16) # 16MB TT
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)

    for puzzle in failed_puzzles:
        print("\n" + "="*50)
        print(f"Puzzle: {puzzle['name']}")
        print(f"FEN: {puzzle['fen']}")
        
        piece_bbs, occupancy_bbs, game_state = parse_fen(puzzle['fen'])
        
        correct_move_uci = puzzle['solution']
        engine_move_uci = puzzle['engine_move']
        
        print(f"Correct Move: {correct_move_uci}")
        print(f"Engine Move : {engine_move_uci}")
        
        correct_move = find_move_by_uci(piece_bbs, occupancy_bbs, game_state, correct_move_uci)
        engine_move = find_move_by_uci(piece_bbs, occupancy_bbs, game_state, engine_move_uci)
        
        if correct_move is None:
            print(f"ERROR: Could not find legal move for correct solution {correct_move_uci}")
            continue
        if engine_move is None:
            print(f"ERROR: Could not find legal move for engine move {engine_move_uci}")
            # Even if engine move is illegal (weird), we proceed with correct move analysis
        
        # --- 1. SEE Analysis (from Root) ---
        # see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq)
        see_correct = see(piece_bbs, occupancy_bbs, game_state[0], get_from_square(correct_move), get_to_square(correct_move))
        see_engine = 0
        if engine_move is not None:
             see_engine = see(piece_bbs, occupancy_bbs, game_state[0], get_from_square(engine_move), get_to_square(engine_move))
             
        print(f"\n--- Move Properties ---")
        print(f"Correct Move SEE: {see_correct}")
        print(f"Engine Move SEE : {see_engine}")
        print(f"Correct Capture?: {is_capture(piece_bbs, occupancy_bbs, game_state, correct_move)}")
        print(f"Correct Promotion?: {is_promotion(correct_move)}")

        # --- 2. Static Eval Breakdown (After Move) ---
        print(f"\n--- Static Evaluation Breakdown (After Move) ---")
        
        # Apply Correct Move
        c_p_bbs, c_o_bbs, c_gs = piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy()
        make_move(c_p_bbs, c_o_bbs, c_gs, correct_move)
        eval_correct = get_eval_breakdown(c_p_bbs, c_o_bbs, c_gs)
        
        # Apply Engine Move
        if engine_move is not None:
            e_p_bbs, e_o_bbs, e_gs = piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy()
            make_move(e_p_bbs, e_o_bbs, e_gs, engine_move)
            eval_engine = get_eval_breakdown(e_p_bbs, e_o_bbs, e_gs)
            
            # Note: The perspective of the evaluation is now the OTHER side.
            # To compare "goodness" for the puzzle solver, we must invert the score if we want "Higher is Better for Solver".
            # The solver (us) just moved. It is now Opponent's turn. 
            # get_eval_breakdown returns score from Opponent's perspective.
            # So a NEGATIVE score is good for us.
            
            print(f"Note: Scores are from the perspective of the side to move AFTER the move.")
            print(f"(Lower is better for the puzzle solver)")
            
            for key in eval_correct:
                print_diff(key, eval_correct[key], eval_engine[key])
        
        # --- 3. Search Comparison ---
        print(f"\n--- Search Comparison (Depth 8) ---")
        time_config = {'optimum_time': 1000, 'maximum_time': 1000}
        
        # Search after Correct Move
        print(f"Searching after Correct Move ({correct_move_uci})...")
        (best_move_c, score_c, nodes_c, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = iterative_deepening_search(
            c_p_bbs, c_o_bbs, c_gs, 8, time_config, search_context
        )
        # Search after Engine Move
        if engine_move is not None:
            print(f"Searching after Engine Move ({engine_move_uci})...")
            (best_move_e, score_e, nodes_e, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = iterative_deepening_search(
                e_p_bbs, e_o_bbs, e_gs, 8, time_config, search_context
            )
            
            # Again, score is from opponent's perspective. Lower is better for us.
            print(f"Score after Correct Move: {score_c} (Best Reply: {move_to_uci(best_move_c)})")
            print(f"Score after Engine Move : {score_e} (Best Reply: {move_to_uci(best_move_e)})")
            print(f"Diff (Correct - Engine): {score_c - score_e}")
            if score_c < score_e:
                print(">> Search correctly evaluates Correct Move as better (lower score for opponent).")
            else:
                print(">> Search INCORRECTLY evaluates Engine Move as better (or equal).")

if __name__ == "__main__":
    analyze()
