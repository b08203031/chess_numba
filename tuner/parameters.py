import numpy as np
import tuner.constants_to_be_tuned as constants

class ParameterManager:
    """
    Manages the mapping between the flat parameter array (theta) and the structured constants.
    """
    def __init__(self):
        self.param_map = []
        self.theta = []
        self._initialize_mapping()

    def _add_param(self, name, value):
        start_idx = len(self.theta)
        
        if isinstance(value, (int, float, np.integer, np.floating)):
            self.theta.append(float(value))
            self.param_map.append({'name': name, 'start': start_idx, 'count': 1, 'shape': (1,)})
        elif isinstance(value, (list, np.ndarray)):
            arr = np.array(value, dtype=np.float64)
            flat_arr = arr.flatten()
            self.theta.extend(flat_arr.tolist())
            self.param_map.append({'name': name, 'start': start_idx, 'count': len(flat_arr), 'shape': arr.shape})
        else:
            raise ValueError(f"Unsupported type for parameter {name}: {type(value)}")

    def _initialize_mapping(self):
        """
        Reads values from constants.py and populates the mapping.
        """
        # --- Material ---
        self._add_param("MG_MATERIAL_VALUES", constants.MG_MATERIAL_VALUES)
        self._add_param("EG_MATERIAL_VALUES", constants.EG_MATERIAL_VALUES)
        
        # --- PSTs ---
        self._add_param("PST_MG", constants.PST_MG)
        self._add_param("PST_EG", constants.PST_EG)

        # --- Mobility ---
        self._add_param("KNIGHT_MOBILITY_BONUS", constants.KNIGHT_MOBILITY_BONUS)
        self._add_param("BISHOP_MOBILITY_BONUS", constants.BISHOP_MOBILITY_BONUS)
        self._add_param("ROOK_MOBILITY_BONUS", constants.ROOK_MOBILITY_BONUS)
        self._add_param("QUEEN_MOBILITY_BONUS", constants.QUEEN_MOBILITY_BONUS)

        # --- Coordination ---
        self._add_param("BISHOP_PAIR_BONUS", constants.BISHOP_PAIR_BONUS)
        self._add_param("ROOK_ON_SEMI_OPEN_FILE_BONUS", constants.ROOK_ON_SEMI_OPEN_FILE_BONUS)
        self._add_param("ROOK_ON_OPEN_FILE_BONUS", constants.ROOK_ON_OPEN_FILE_BONUS)
        # ROOK_ON_SEVENTH_BONUS removed from classical (not in SF11)

        # --- Pawn Structure ---
        self._add_param("PASSED_PAWN_BONUS", constants.PASSED_PAWN_BONUS)
        self._add_param("CANDIDATE_PASSED_PAWN_BONUS", constants.CANDIDATE_PASSED_PAWN_BONUS)
        self._add_param("ISOLATED_PAWN_PENALTY", constants.ISOLATED_PAWN_PENALTY)
        self._add_param("DOUBLED_PAWN_PENALTY", constants.DOUBLED_PAWN_PENALTY)
        self._add_param("BACKWARD_PAWN_PENALTY", constants.BACKWARD_PAWN_PENALTY)
        self._add_param("CONNECTED_BONUS", constants.CONNECTED_BONUS)
        self._add_param("CONNECTED_SUPPORT_WEIGHT", constants.CONNECTED_SUPPORT_WEIGHT)

        # --- Outposts ---
        self._add_param("OUTPOST_BONUS_KNIGHT", constants.OUTPOST_BONUS_KNIGHT)
        self._add_param("OUTPOST_BONUS_BISHOP", constants.OUTPOST_BONUS_BISHOP)
        # OUTPOST_HOLE_BONUS removed from classical (SF11 has no separate hole term)

        # --- King Safety ---
        self._add_param("KING_SAFETY_ATTACK_UNITS", constants.KING_SAFETY_ATTACK_UNITS)
        self._add_param("KING_DANGER_WEAK_SQ", constants.KING_DANGER_WEAK_SQ)
        self._add_param("KING_DANGER_UNSAFE_CHECK", constants.KING_DANGER_UNSAFE_CHECK)
        self._add_param("KING_DANGER_ATTACK_ON_KING_SQ", constants.KING_DANGER_ATTACK_ON_KING_SQ)
        self._add_param("KING_DANGER_NO_QUEEN", constants.KING_DANGER_NO_QUEEN)
        self._add_param("KING_DANGER_PINNED", constants.KING_DANGER_PINNED)
        self._add_param("KING_DANGER_DIVISOR", constants.KING_DANGER_DIVISOR)
        self._add_param("SAFE_CHECK_KNIGHT", constants.SAFE_CHECK_KNIGHT)
        self._add_param("SAFE_CHECK_BISHOP", constants.SAFE_CHECK_BISHOP)
        self._add_param("SAFE_CHECK_ROOK", constants.SAFE_CHECK_ROOK)
        self._add_param("SAFE_CHECK_QUEEN", constants.SAFE_CHECK_QUEEN)
        self._add_param("KING_TROPISM_WEIGHTS", constants.KING_TROPISM_WEIGHTS)
        self._add_param("PAWN_STORM_PENALTY_BY_RANK", constants.PAWN_STORM_PENALTY_BY_RANK)
        self._add_param("SCALING_WEIGHTS", constants.SCALING_WEIGHTS)
        
        self._add_param("PAWN_SHIELD_MISSING_PENALTY", constants.PAWN_SHIELD_MISSING_PENALTY)
        self._add_param("PAWN_SHIELD_INTACT_BONUS", constants.PAWN_SHIELD_INTACT_BONUS)
        self._add_param("PAWN_SHIELD_ADVANCED_BONUS", constants.PAWN_SHIELD_ADVANCED_BONUS)
        self._add_param("PAWN_SHIELD_PUSHED_PENALTY", constants.PAWN_SHIELD_PUSHED_PENALTY)
        self._add_param("KING_OPEN_FILE_PENALTY", constants.KING_OPEN_FILE_PENALTY)
        self._add_param("KING_SEMI_OPEN_FILE_PENALTY", constants.KING_SEMI_OPEN_FILE_PENALTY)
        # NOTE: KING_SAFETY_WEAK_SQUARE_PENALTY intentionally skipped — not used in evaluation.py

        # --- Threats ---
        self._add_param("THREAT_SAFE_PAWN", constants.THREAT_SAFE_PAWN)
        self._add_param("THREAT_BY_MINOR", constants.THREAT_BY_MINOR)
        self._add_param("THREAT_BY_ROOK", constants.THREAT_BY_ROOK)
        self._add_param("THREAT_BY_KING", constants.THREAT_BY_KING)
        self._add_param("THREAT_HANGING", constants.THREAT_HANGING)
        # NOTE: THREAT_RESTRICTED_PIECE intentionally skipped — not used in evaluation.py
        self._add_param("THREAT_PAWN_PUSH", constants.THREAT_PAWN_PUSH)

        # --- Per-Piece ---
        self._add_param("KING_PROTECTOR", constants.KING_PROTECTOR)
        self._add_param("BISHOP_PAWNS_PENALTY", constants.BISHOP_PAWNS_PENALTY)
        self._add_param("TRAPPED_ROOK", constants.TRAPPED_ROOK)
        
        # --- Other ---
        self._add_param("MAX_KING_ATTACKERS", constants.MAX_KING_ATTACKERS)
        self._add_param("PROXIMITY_ENEMY_WEIGHT", constants.PROXIMITY_ENEMY_WEIGHT)
        self._add_param("PROXIMITY_FRIENDLY_WEIGHT", constants.PROXIMITY_FRIENDLY_WEIGHT)
        self._add_param("MAX_PROXIMITY_BONUS", constants.MAX_PROXIMITY_BONUS)
        self._add_param("KING_DANGER_SINGLE_ATTACKER_DIVISOR", constants.KING_DANGER_SINGLE_ATTACKER_DIVISOR)
        self._add_param("UNSTOPPABLE_PAWN_BONUS", constants.UNSTOPPABLE_PAWN_BONUS)
        self._add_param("EG_KING_PAWN_PROXIMITY_WEIGHT", constants.EG_KING_PAWN_PROXIMITY_WEIGHT)
        self._add_param("BLOCKED_PASSER_DIVISOR", constants.BLOCKED_PASSER_DIVISOR)

    def get_initial_theta(self):
        return np.array(self.theta, dtype=np.float64)

    def get_param_indices(self, name):
        for p in self.param_map:
            if p['name'] == name:
                return p['start'], p['count']
        raise ValueError(f"Parameter {name} not found.")

    def update_constants_string(self, new_theta):
        lines = []
        lines.append("# Updated constants generated by tuner.py")
        lines.append("import numpy as np")
        lines.append("")

        for p in self.param_map:
            name = p['name']
            start = p['start']
            count = p['count']
            shape = p['shape']
            
            values = new_theta[start : start + count]
            int_values = np.round(values).astype(int)
            
            if count == 1:
                lines.append(f"{name} = {int_values[0]}")
            else:
                if len(shape) == 1:
                    arr_str = ", ".join(map(str, int_values))
                    lines.append(f"{name} = np.array([{arr_str}], dtype=np.int32)")
                elif len(shape) == 2:
                    reshaped = int_values.reshape(shape)
                    lines.append(f"{name} = np.array([")
                    for row in reshaped:
                        row_str = ", ".join(map(str, row))
                        lines.append(f"    [{row_str}],")
                    lines.append(f"], dtype=np.int32)")
                elif len(shape) == 3: 
                     reshaped = int_values.reshape(shape)
                     lines.append(f"{name} = np.array([")
                     for row in reshaped:
                        lines.append(f"    # {name} slice")
                        lines.append(f"    {list(row)},") 
                     lines.append(f"], dtype=np.int32)")
            
            lines.append("")
            
        return "\n".join(lines)
