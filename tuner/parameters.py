
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
        """
        Adds a parameter or array of parameters to the mapping.
        """
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
        更動要調整的項目時要到tuner.py更改mask!!!!!!!
        以下都不要註解掉
        """
        # --- Material ---
        self._add_param("MG_MATERIAL_VALUES", constants.MG_MATERIAL_VALUES)
        self._add_param("EG_MATERIAL_VALUES", constants.EG_MATERIAL_VALUES)
        
        # --- PSTs ---
        self._add_param("PST_MG", constants.PST_MG)
        self._add_param("PST_EG", constants.PST_EG)

        # --- Mobility ---
        self._add_param("KNIGHT_MOBILITY_WEIGHT", constants.KNIGHT_MOBILITY_WEIGHT)
        self._add_param("BISHOP_MOBILITY_WEIGHT", constants.BISHOP_MOBILITY_WEIGHT)
        self._add_param("ROOK_MOBILITY_WEIGHT", constants.ROOK_MOBILITY_WEIGHT)
        self._add_param("QUEEN_MOBILITY_WEIGHT", constants.QUEEN_MOBILITY_WEIGHT)

        # --- Coordination ---
        self._add_param("BISHOP_PAIR_BONUS", constants.BISHOP_PAIR_BONUS)
        self._add_param("ROOK_ON_SEMI_OPEN_FILE_BONUS", constants.ROOK_ON_SEMI_OPEN_FILE_BONUS)
        self._add_param("ROOK_ON_OPEN_FILE_BONUS", constants.ROOK_ON_OPEN_FILE_BONUS)
        self._add_param("ROOK_ON_SEVENTH_BONUS", constants.ROOK_ON_SEVENTH_BONUS)

        # --- Pawn Structure ---
        self._add_param("PASSED_PAWN_BONUS", constants.PASSED_PAWN_BONUS)
        self._add_param("ISOLATED_PAWN_PENALTY", constants.ISOLATED_PAWN_PENALTY)
        self._add_param("DOUBLED_PAWN_PENALTY", constants.DOUBLED_PAWN_PENALTY)
        self._add_param("CONNECTED_PASSED_PAWN_BONUS", constants.CONNECTED_PASSED_PAWN_BONUS)
        self._add_param("BACKWARD_PAWN_PENALTY", constants.BACKWARD_PAWN_PENALTY)

        # --- Outposts ---
        self._add_param("OUTPOST_BONUS_KNIGHT", constants.OUTPOST_BONUS_KNIGHT)
        self._add_param("OUTPOST_BONUS_BISHOP", constants.OUTPOST_BONUS_BISHOP)
        self._add_param("OUTPOST_HOLE_BONUS", constants.OUTPOST_HOLE_BONUS)

        # --- King Safety ---
        self._add_param("KING_SAFETY_WEAK_UNITS", constants.KING_SAFETY_WEAK_UNITS)
        self._add_param("KING_SAFETY_ATTACK_UNITS", constants.KING_SAFETY_ATTACK_UNITS)
        self._add_param("KING_SAFETY_TABLE", constants.KING_SAFETY_TABLE)
        self._add_param("KING_TROPISM_WEIGHTS", constants.KING_TROPISM_WEIGHTS)
        self._add_param("PAWN_STORM_PENALTY_BY_RANK", constants.PAWN_STORM_PENALTY_BY_RANK)
        
        # King Safety Scalars
        self._add_param("PAWN_SHIELD_MISSING_PENALTY", constants.PAWN_SHIELD_MISSING_PENALTY)
        self._add_param("PAWN_SHIELD_INTACT_BONUS", constants.PAWN_SHIELD_INTACT_BONUS)
        self._add_param("PAWN_SHIELD_ADVANCED_BONUS", constants.PAWN_SHIELD_ADVANCED_BONUS)
        self._add_param("PAWN_SHIELD_PUSHED_PENALTY", constants.PAWN_SHIELD_PUSHED_PENALTY)
        self._add_param("KING_OPEN_FILE_PENALTY", constants.KING_OPEN_FILE_PENALTY)
        self._add_param("KING_SEMI_OPEN_FILE_PENALTY", constants.KING_SEMI_OPEN_FILE_PENALTY)

        # --- Threats ---
        self._add_param("THREAT_SAFE_PAWN", constants.THREAT_SAFE_PAWN)
        self._add_param("THREAT_MINOR_ON_MAJOR", constants.THREAT_MINOR_ON_MAJOR)
        self._add_param("THREAT_ROOK_ON_QUEEN", constants.THREAT_ROOK_ON_QUEEN)
        self._add_param("THREAT_HANGING", constants.THREAT_HANGING)
        
        # --- Other ---
        self._add_param("INITIATIVE_BONUS", constants.INITIATIVE_BONUS)

    def get_initial_theta(self):
        return np.array(self.theta, dtype=np.float64)

    def get_param_indices(self, name):
        """Returns the start index and count for a given parameter name."""
        for p in self.param_map:
            if p['name'] == name:
                return p['start'], p['count']
        raise ValueError(f"Parameter {name} not found.")

    def update_constants_string(self, new_theta):
        """
        Generates a string containing the updated constants code.
        """
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
            
            # Convert back to int for constants (most are ints)
            # Use rounding to nearest integer
            int_values = np.round(values).astype(int)
            
            if count == 1:
                lines.append(f"{name} = {int_values[0]}")
            else:
                # Format as numpy array
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
