import re

EVAL_PATH = "chess_engine/classical/evaluation.py"
TUNABLE_PATH = "tuner/tunable_eval.py"
INDICES_PATH = "tuner/codegen/new_indices.txt"

with open(EVAL_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Grab all indices
with open(INDICES_PATH, 'r', encoding='utf-8') as f:
    indices_text = f.read()

# Replace V2 relative imports with our new absolute ones
text = text.replace("from chess_engine.classical.constants import *", "from chess_engine.classical.constants import *")
text = text.replace("from chess_engine.bitboard_utils import", "from chess_engine.classical.bitboard_utils import")
text = text.replace("from chess_engine.engine_types import", "from chess_engine.classical.engine_types import")
text = text.replace("from chess_engine.move_generator import", "from chess_engine.classical.move_generator import")
text = text.replace("from chess_engine.evaluation import", "from chess_engine.classical.evaluation import")

# Insert tuner getters after the imports
insert_pos = text.find("NOT_A_FILE =")
if insert_pos == -1:
    insert_pos = text.find("def _create_manhattan_distance_table")

getters = """from tuner.constants_to_be_tuned import (
    PHASE_WEIGHTS, MAX_PHASE, INITIATIVE_PHASE_THRESHOLD,
    KING_TROPISM_MAX_DISTANCE, EG_SAFETY_SCALE
)

""" + indices_text + """

@numba.njit(numba.int32(numba.float64[:], numba.int32), cache=True, inline='always')
def get_int(theta, idx):
    return numba.int32(theta[idx])

@numba.njit(numba.int32(numba.float64[:], numba.int32, numba.int32), cache=True, inline='always')
def get_array_val(theta, base_idx, offset):
    return numba.int32(theta[base_idx + offset])

@numba.njit(numba.int32(numba.float64[:], numba.int32, numba.int32, numba.int32, numba.int32), cache=True, inline='always')
def get_2d_val(theta, base_idx, row, col_size, col):
    return numba.int32(theta[base_idx + row * col_size + col])

"""

text = text[:insert_pos] + getters + text[insert_pos:]


# Constants replacements
single_vars = [
    "CONNECTED_SUPPORT_WEIGHT", "KING_DANGER_WEAK_SQ", "KING_DANGER_UNSAFE_CHECK", "KING_DANGER_ATTACK_ON_KING_SQ", 
    "KING_DANGER_NO_QUEEN", "KING_DANGER_PINNED", "KING_DANGER_DIVISOR", "SAFE_CHECK_KNIGHT", "SAFE_CHECK_BISHOP", 
    "SAFE_CHECK_ROOK", "SAFE_CHECK_QUEEN", "PAWN_SHIELD_MISSING_PENALTY", "PAWN_SHIELD_INTACT_BONUS", 
    "PAWN_SHIELD_ADVANCED_BONUS", "PAWN_SHIELD_PUSHED_PENALTY", "KING_OPEN_FILE_PENALTY", "KING_SEMI_OPEN_FILE_PENALTY", 
    "KING_SAFETY_WEAK_SQUARE_PENALTY", "INITIATIVE_BONUS"
]

arr1d = [
    "MG_MATERIAL_VALUES", "EG_MATERIAL_VALUES", "BISHOP_PAIR_BONUS", "ROOK_ON_SEMI_OPEN_FILE_BONUS",
    "ROOK_ON_OPEN_FILE_BONUS", "ROOK_ON_SEVENTH_BONUS", "ISOLATED_PAWN_PENALTY", "DOUBLED_PAWN_PENALTY",
    "BACKWARD_PAWN_PENALTY", "CONNECTED_BONUS", "OUTPOST_HOLE_BONUS", "KING_SAFETY_ATTACK_UNITS",
    "KING_TROPISM_WEIGHTS", "PAWN_STORM_PENALTY_BY_RANK", "SCALING_WEIGHTS", "THREAT_SAFE_PAWN",
    "THREAT_BY_KING", "THREAT_HANGING", "THREAT_RESTRICTED_PIECE", "THREAT_PAWN_PUSH", "KING_PROTECTOR",
    "BISHOP_PAWNS_PENALTY", "TRAPPED_ROOK"
]

arr2d_64 = [ "PST_MG", "PST_EG" ]
arr2d_2 = [
    "KNIGHT_MOBILITY_BONUS", "BISHOP_MOBILITY_BONUS", "ROOK_MOBILITY_BONUS", "QUEEN_MOBILITY_BONUS",
    "PASSED_PAWN_BONUS", "CANDIDATE_PASSED_PAWN_BONUS", "OUTPOST_BONUS_KNIGHT", "OUTPOST_BONUS_BISHOP",
    "THREAT_BY_MINOR", "THREAT_BY_ROOK"
]

for v in single_vars:
    text = re.sub(r'\b' + v + r'\b', f'get_int(theta, IDX_{v})', text)

for v in arr1d:
    text = re.sub(r'\b' + v + r'\[([^\]]+)\]', rf'get_array_val(theta, IDX_{v}, \1)', text)

for v in arr2d_64:
    text = re.sub(r'\b' + v + r'\[([^,]+),\s*([^\]]+)\]', rf'get_2d_val(theta, IDX_{v}, \1, 64, \2)', text)

for v in arr2d_2:
    text = re.sub(r'\b' + v + r'\[([^,]+),\s*([^\]]+)\]', rf'get_2d_val(theta, IDX_{v}, \1, 2, \2)', text)


# Function modifications
def add_theta_to_def(text, func_name):
    # Match standard signatures
    text = re.sub(
        r'def ' + func_name + r'\((.*?)\):',
        r'def ' + func_name + r'(\1, theta):',
        text
    )
    return text

funcs = [
    "evaluate_pawn_structure", 
    "evaluate_king_safety",
    "evaluate_piece_coordination",
    "evaluate_attacks_mobility_threats",
    "_evaluate_king_pawn_endgame",
    "_process_piece_score_and_count",
    "_evaluate_king_attackers",
    "_evaluate_pawn_shield_for_color"
]
for func in funcs:
    text = add_theta_to_def(text, func)
    text = re.sub(r'\b' + func + r'\((.*?)\)(?![:,])', r'' + func + r'(\1, theta)', text)

# Handle evaluate_position to evaluate_position_tunable
# Careful with the lazy=False parameter
text = text.replace("def evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy: bool = False):", 
                    "def evaluate_position_tunable(piece_bbs, occupancy_bbs, game_state, theta, lazy=False):")


text = text.replace("@numba.njit(numba.int32(numba.int32, numba.uint64, numba.uint64, numba.int32), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.int32(numba.int32, numba.uint64, numba.uint64, numba.int32, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

text = text.replace("@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, occupancy_bbs_signature, numba.uint64, numba.uint64, numba.uint64), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, occupancy_bbs_signature, numba.uint64, numba.uint64, numba.uint64, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

# Numba signature fixes (Inject numba.float64[:] before closing bracket)
# This is tricky since the signatures are long.
text = text.replace("@numba.njit(numba.types.UniTuple(numba.int32, 6)(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.types.UniTuple(numba.int32, 6)(piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

text = text.replace("@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, piece_counts_signature), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, piece_counts_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

text = text.replace("@numba.njit(numba.types.Tuple((numba.int32, numba.int32, numba.int32, numba.int32, numba.int32, numba.int32))(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.types.Tuple((numba.int32, numba.int32, numba.int32, numba.int32, numba.int32, numba.int32))(piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

text = text.replace("@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature, numba.uint64, numba.uint64, numba.int32, numba.int32, numba.int32, numba.int32, piece_counts_signature, numba.uint64, numba.uint64, numba.uint64, numba.uint64), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature, numba.uint64, numba.uint64, numba.int32, numba.int32, numba.int32, numba.int32, piece_counts_signature, numba.uint64, numba.uint64, numba.uint64, numba.uint64, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

text = text.replace("@numba.njit(numba.types.Tuple((numba.int32, numba.int32))(piece_bbs_signature, numba.types.UniTuple(numba.int32, 12)), cache=True, boundscheck=False, fastmath=True, inline='always')",
                    "@numba.njit(numba.types.Tuple((numba.int32, numba.int32))(piece_bbs_signature, numba.types.UniTuple(numba.int32, 12), numba.float64[:]), cache=True, boundscheck=False, fastmath=True, inline='always')")

text = re.sub(r'(@numba\.njit\(numba\.types\.Tuple\(.*?piece_bbs_signature, occupancy_bbs_signature)\)',
              r'\1, numba.float64[:])', text)

text = text.replace("@numba.njit(numba.int32(piece_bbs_signature, numba.uint64), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.int32(piece_bbs_signature, numba.uint64, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)")

text = text.replace("@numba.njit(numba.types.Tuple((numba.int32, numba.int32, numba.int32))(numba.int32, numba.uint64, numba.boolean), cache=True, boundscheck=False, fastmath=True, inline='always')",
                    "@numba.njit(numba.types.Tuple((numba.int32, numba.int32, numba.int32))(numba.int32, numba.uint64, numba.boolean, numba.float64[:]), cache=True, boundscheck=False, fastmath=True, inline='always')")

text = text.replace("@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.boolean), cache=True, boundscheck=False, fastmath=True)",
                    "@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.float64[:], numba.boolean), cache=True, boundscheck=False, fastmath=True)")

with open(TUNABLE_PATH, 'w', encoding='utf-8') as f:
    f.write(text)

print("Process completed successfully!")
