"""
Maps a flat SPSA theta vector <-> classical HCE constants.

Only parameters that are (1) used by evaluation/pawns/material and
(2) intended for tuning are registered. Structural bitboards, search
constants, scale-factor draw tables, and known-win thresholds are excluded.
"""
from __future__ import annotations

import numpy as np

import tuner.constants_snapshot as constants

# ---------------------------------------------------------------------------
# Registration order defines theta layout. Keep in sync with codegen.
# Each entry: (name, value_from_snapshot)
# ---------------------------------------------------------------------------

def _collect_param_specs():
    c = constants
    return [
        # Material
        ("MG_MATERIAL_VALUES", c.MG_MATERIAL_VALUES),
        ("EG_MATERIAL_VALUES", c.EG_MATERIAL_VALUES),
        # PSTs
        ("PST_MG", c.PST_MG),
        ("PST_EG", c.PST_EG),
        # Mobility
        ("KNIGHT_MOBILITY_BONUS", c.KNIGHT_MOBILITY_BONUS),
        ("BISHOP_MOBILITY_BONUS", c.BISHOP_MOBILITY_BONUS),
        ("ROOK_MOBILITY_BONUS", c.ROOK_MOBILITY_BONUS),
        ("QUEEN_MOBILITY_BONUS", c.QUEEN_MOBILITY_BONUS),
        # Coordination
        ("BISHOP_PAIR_BONUS", c.BISHOP_PAIR_BONUS),
        ("BISHOP_PAIR_PAWN_SCALE", c.BISHOP_PAIR_PAWN_SCALE),
        ("ROOK_ON_SEMI_OPEN_FILE_BONUS", c.ROOK_ON_SEMI_OPEN_FILE_BONUS),
        ("ROOK_ON_OPEN_FILE_BONUS", c.ROOK_ON_OPEN_FILE_BONUS),
        # Pawn structure
        ("PASSED_PAWN_BONUS", c.PASSED_PAWN_BONUS),
        ("PASSED_FILE_BONUS", c.PASSED_FILE_BONUS),
        ("PASSED_CONNECTED_BONUS", c.PASSED_CONNECTED_BONUS),
        ("PASSED_PHALANX_BONUS", c.PASSED_PHALANX_BONUS),
        ("CANDIDATE_PASSER_DIVISOR", c.CANDIDATE_PASSER_DIVISOR),
        ("PASSED_PATH_SAFE_NONE_ATTACK", c.PASSED_PATH_SAFE_NONE_ATTACK),
        ("PASSED_PATH_SAFE_EDGE_ATTACK", c.PASSED_PATH_SAFE_EDGE_ATTACK),
        ("PASSED_PATH_SAFE_BLOCK_ONLY", c.PASSED_PATH_SAFE_BLOCK_ONLY),
        ("PASSED_PATH_SUPPORT_BONUS", c.PASSED_PATH_SUPPORT_BONUS),
        ("PASSED_DYNAMICS_MULT", c.PASSED_DYNAMICS_MULT),
        ("PASSED_DYNAMICS_OFFSET", c.PASSED_DYNAMICS_OFFSET),
        ("ISOLATED_PAWN_PENALTY", c.ISOLATED_PAWN_PENALTY),
        ("DOUBLED_PAWN_PENALTY", c.DOUBLED_PAWN_PENALTY),
        ("BACKWARD_PAWN_PENALTY", c.BACKWARD_PAWN_PENALTY),
        ("WEAK_LEVER_PENALTY", c.WEAK_LEVER_PENALTY),
        ("WEAK_UNOPPOSED_PENALTY", c.WEAK_UNOPPOSED_PENALTY),
        ("CONNECTED_BONUS", c.CONNECTED_BONUS),
        ("CONNECTED_BONUS_EG", c.CONNECTED_BONUS_EG),
        ("CONNECTED_SUPPORT_WEIGHT", c.CONNECTED_SUPPORT_WEIGHT),
        ("CONNECTED_SUPPORT_WEIGHT_EG", c.CONNECTED_SUPPORT_WEIGHT_EG),
        # Outposts / minors
        ("OUTPOST_BONUS_KNIGHT", c.OUTPOST_BONUS_KNIGHT),
        ("OUTPOST_BONUS_BISHOP", c.OUTPOST_BONUS_BISHOP),
        ("REACHABLE_OUTPOST_BONUS", c.REACHABLE_OUTPOST_BONUS),
        ("MINOR_BEHIND_PAWN", c.MINOR_BEHIND_PAWN),
        ("BISHOP_PAWNS_PENALTY", c.BISHOP_PAWNS_PENALTY),
        ("BISHOP_PAWNS_CENTER_BLOCKED_FACTOR", c.BISHOP_PAWNS_CENTER_BLOCKED_FACTOR),
        ("LONG_DIAGONAL_BISHOP", c.LONG_DIAGONAL_BISHOP),
        ("TRAPPED_ROOK", c.TRAPPED_ROOK),
        ("ROOK_ON_QUEEN_FILE", c.ROOK_ON_QUEEN_FILE),
        ("WEAK_QUEEN", c.WEAK_QUEEN),
        ("KING_PROTECTOR", c.KING_PROTECTOR),
        # King danger / checks
        ("KING_ATTACK_WEIGHTS", c.KING_ATTACK_WEIGHTS),
        ("SAFE_CHECK_QUEEN", c.SAFE_CHECK_QUEEN),
        ("SAFE_CHECK_ROOK", c.SAFE_CHECK_ROOK),
        ("SAFE_CHECK_BISHOP", c.SAFE_CHECK_BISHOP),
        ("SAFE_CHECK_KNIGHT", c.SAFE_CHECK_KNIGHT),
        ("KING_DANGER_WEAK_SQ", c.KING_DANGER_WEAK_SQ),
        ("KING_DANGER_UNSAFE_CHECK", c.KING_DANGER_UNSAFE_CHECK),
        ("KING_DANGER_BLOCKERS", c.KING_DANGER_BLOCKERS),
        ("KING_DANGER_ATTACK_ON_KING_SQ", c.KING_DANGER_ATTACK_ON_KING_SQ),
        ("KING_DANGER_NO_QUEEN", c.KING_DANGER_NO_QUEEN),
        ("KING_DANGER_NO_QUEEN_ROOKLESS", c.KING_DANGER_NO_QUEEN_ROOKLESS),
        ("KING_DANGER_KNIGHT_DEF", c.KING_DANGER_KNIGHT_DEF),
        ("KING_DANGER_OFFSET", c.KING_DANGER_OFFSET),
        ("KING_DANGER_THRESHOLD", c.KING_DANGER_THRESHOLD),
        # Shelter / storm
        ("SHELTER_STRENGTH", c.SHELTER_STRENGTH),
        ("UNBLOCKED_STORM", c.UNBLOCKED_STORM),
        ("BLOCKED_STORM", c.BLOCKED_STORM),
        ("BLOCKED_STORM_EG", c.BLOCKED_STORM_EG),
        ("SHELTER_BASE_MG", c.SHELTER_BASE_MG),
        ("SHELTER_BASE_EG", c.SHELTER_BASE_EG),
        ("KING_PAWN_DIST_PENALTY_EG", c.KING_PAWN_DIST_PENALTY_EG),
        ("PAWNLESS_FLANK", c.PAWNLESS_FLANK),
        ("FLANK_ATTACKS", c.FLANK_ATTACKS),
        # Threats
        ("THREAT_SAFE_PAWN", c.THREAT_SAFE_PAWN),
        ("THREAT_KNIGHT_ON_QUEEN", c.THREAT_KNIGHT_ON_QUEEN),
        ("THREAT_SLIDER_ON_QUEEN", c.THREAT_SLIDER_ON_QUEEN),
        ("THREAT_BY_MINOR", c.THREAT_BY_MINOR),
        ("THREAT_BY_ROOK", c.THREAT_BY_ROOK),
        ("THREAT_BY_KING", c.THREAT_BY_KING),
        ("THREAT_HANGING", c.THREAT_HANGING),
        ("THREAT_RESTRICTED_PIECE", c.THREAT_RESTRICTED_PIECE),
        ("THREAT_PAWN_PUSH", c.THREAT_PAWN_PUSH),
        # Space / tempo / initiative (small DOF)
        ("SPACE_THRESHOLD", c.SPACE_THRESHOLD),
        ("SPACE_BONUS_DIVISOR", c.SPACE_BONUS_DIVISOR),
        ("SPACE_SCALE_DIVISOR", c.SPACE_SCALE_DIVISOR),
        ("TEMPO_BONUS", c.TEMPO_BONUS),
        ("INITIATIVE_PASSED_WEIGHT_MG", c.INITIATIVE_PASSED_WEIGHT_MG),
        ("INITIATIVE_PAWN_WEIGHT_MG", c.INITIATIVE_PAWN_WEIGHT_MG),
        ("INITIATIVE_OUTFLANKING_WEIGHT_MG", c.INITIATIVE_OUTFLANKING_WEIGHT_MG),
        ("INITIATIVE_INFILTRATION_WEIGHT_MG", c.INITIATIVE_INFILTRATION_WEIGHT_MG),
        ("INITIATIVE_BOTH_FLANKS_WEIGHT_MG", c.INITIATIVE_BOTH_FLANKS_WEIGHT_MG),
        ("INITIATIVE_PAWN_ENDGAME_WEIGHT_MG", c.INITIATIVE_PAWN_ENDGAME_WEIGHT_MG),
        ("INITIATIVE_ALMOST_UNWIN_WEIGHT_MG", c.INITIATIVE_ALMOST_UNWIN_WEIGHT_MG),
        ("INITIATIVE_OFFSET_MG", c.INITIATIVE_OFFSET_MG),
        ("INITIATIVE_MG_OFFSET", c.INITIATIVE_MG_OFFSET),
        ("INITIATIVE_PASSED_WEIGHT_EG", c.INITIATIVE_PASSED_WEIGHT_EG),
        ("INITIATIVE_PAWN_WEIGHT_EG", c.INITIATIVE_PAWN_WEIGHT_EG),
        ("INITIATIVE_OUTFLANKING_WEIGHT_EG", c.INITIATIVE_OUTFLANKING_WEIGHT_EG),
        ("INITIATIVE_INFILTRATION_WEIGHT_EG", c.INITIATIVE_INFILTRATION_WEIGHT_EG),
        ("INITIATIVE_BOTH_FLANKS_WEIGHT_EG", c.INITIATIVE_BOTH_FLANKS_WEIGHT_EG),
        ("INITIATIVE_PAWN_ENDGAME_WEIGHT_EG", c.INITIATIVE_PAWN_ENDGAME_WEIGHT_EG),
        ("INITIATIVE_ALMOST_UNWIN_WEIGHT_EG", c.INITIATIVE_ALMOST_UNWIN_WEIGHT_EG),
        ("INITIATIVE_OFFSET_EG", c.INITIATIVE_OFFSET_EG),
        # Imbalance (optional group; easy to freeze via --exclude)
        ("IMBALANCE_QUADRATIC_OURS", c.IMBALANCE_QUADRATIC_OURS),
        ("IMBALANCE_QUADRATIC_THEIRS", c.IMBALANCE_QUADRATIC_THEIRS),
        ("IMBALANCE_DIVISOR", c.IMBALANCE_DIVISOR),
        ("IMBALANCE_SCALE_MG", c.IMBALANCE_SCALE_MG),
        ("IMBALANCE_SCALE_EG", c.IMBALANCE_SCALE_EG),
        # Passed king proximity helpers
        ("KING_PROXIMITY_MAX_DIST", c.KING_PROXIMITY_MAX_DIST),
        ("KING_PROX_ENEMY_MULT", c.KING_PROX_ENEMY_MULT),
        ("KING_PROX_ENEMY_DIV", c.KING_PROX_ENEMY_DIV),
        ("KING_PROX_FRIENDLY_MULT", c.KING_PROX_FRIENDLY_MULT),
        ("KING_PROX_MIN_BONUS", c.KING_PROX_MIN_BONUS),
        ("KING_PROX_MAX_BONUS", c.KING_PROX_MAX_BONUS),
        ("UNSTOPPABLE_PAWN_BONUS", c.UNSTOPPABLE_PAWN_BONUS),
        ("KING_PAWN_PROXIMITY_FACTOR", c.KING_PAWN_PROXIMITY_FACTOR),
    ]


class ParameterManager:
    """Maps flat theta <-> structured HCE constants."""

    def __init__(self):
        self.param_map = []
        self.theta = []
        self._initialize_mapping()

    def _add_param(self, name, value):
        start_idx = len(self.theta)
        if isinstance(value, (int, float, np.integer, np.floating)):
            self.theta.append(float(value))
            self.param_map.append(
                {"name": name, "start": start_idx, "count": 1, "shape": (1,)}
            )
        elif isinstance(value, (list, np.ndarray)):
            arr = np.array(value, dtype=np.float64)
            flat = arr.flatten()
            self.theta.extend(flat.tolist())
            self.param_map.append(
                {
                    "name": name,
                    "start": start_idx,
                    "count": len(flat),
                    "shape": arr.shape,
                }
            )
        else:
            raise ValueError(f"Unsupported type for parameter {name}: {type(value)}")

    def _initialize_mapping(self):
        for name, value in _collect_param_specs():
            self._add_param(name, value)

    def get_initial_theta(self):
        return np.array(self.theta, dtype=np.float64)

    def get_param_indices(self, name):
        for p in self.param_map:
            if p["name"] == name:
                return p["start"], p["count"]
        raise ValueError(f"Parameter {name} not found.")

    def list_param_names(self):
        return [p["name"] for p in self.param_map]

    def update_constants_string(self, new_theta):
        lines = [
            "# Updated constants generated by tuner.py",
            "# Apply into chess_engine/classical/constants.py after review.",
            "import numpy as np",
            "",
        ]
        for p in self.param_map:
            name = p["name"]
            start = p["start"]
            count = p["count"]
            shape = p["shape"]
            values = new_theta[start : start + count]
            int_values = np.round(values).astype(int)

            if count == 1:
                lines.append(f"{name} = {int_values[0]}")
            elif len(shape) == 1:
                arr_str = ", ".join(map(str, int_values))
                lines.append(f"{name} = np.array([{arr_str}], dtype=np.int32)")
            elif len(shape) == 2:
                reshaped = int_values.reshape(shape)
                lines.append(f"{name} = np.array([")
                for row in reshaped:
                    row_str = ", ".join(map(str, row))
                    lines.append(f"    [{row_str}],")
                lines.append("], dtype=np.int32)")
            else:
                reshaped = int_values.reshape(shape)
                lines.append(f"{name} = np.array({reshaped.tolist()}, dtype=np.int32)")
            lines.append("")
        return "\n".join(lines)
