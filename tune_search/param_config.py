
"""
Configuration for search parameter tuning.
Defines which parameters to tune, their ranges, and step sizes.
"""

# Define the parameters to tune
# Format: "PARAM_NAME": (default_value, min_value, max_value, step_size)
SEARCH_PARAMS = {
    # --- Razoring ---
    "RAZORING_MARGIN": (700, 300, 1200, 50),

    # --- Futility Pruning ---
    "FP_MARGIN_D1": (400, 100, 800, 50),
    "FP_MARGIN_D2": (700, 300, 1200, 50),
    "FP_BASE": (200, 100, 500, 50),
    "FP_MULTIPLIER": (200, 100, 500, 50),

    # --- Reverse Futility Pruning (Static Null Move Pruning) ---
    "RFP_MARGIN_D1": (250, 50, 600, 25),

    # --- Null Move Pruning ---
    "NMP_STATIC_MARGIN": (600, 0, 1200, 50),
    "NULL_MOVE_REDUCTION": (2, 1, 4, 1), # Typically integer

    # --- Late Move Reductions ---
    "LMR_MIN_DEPTH": (4, 2, 6, 1),
    "LMR_MIN_QUIET_MOVE_INDEX": (4, 2, 8, 1),
    
    # --- SEE Pruning ---
    "SEE_THRESHOLD": (-100, -300, 0, 25),
    "PRUNING_CAPTURE_SEE_MARGIN": (-200, -500, 0, 50),
    "PRUNING_QUIET_SEE_MARGIN": (-100, -400, 0, 25),

    # --- History Pruning ---
    "PRUNING_HISTORY_THRESHOLD": (-1500, -3000, -500, 100),
    
    # --- Delta Pruning ---
    "DELTA_PRUNING_MARGIN": (1200, 500, 2000, 100),
    
    # --- ProbCut ---
    "PROBCUT_MARGIN": (150, 50, 400, 25),
    
    # --- Singular Extensions ---
    "SINGULAR_EXTENSION_MARGIN": (150, 50, 400, 25),

    # --- Other Parameters ---
    "MAX_QUIESCENCE_DEPTH": (5, 4, 8, 1),
}

def get_param_names():
    return list(SEARCH_PARAMS.keys())

def get_default_params():
    return {k: v[0] for k, v in SEARCH_PARAMS.items()}

def get_bounds(param_name):
    if param_name in SEARCH_PARAMS:
        return SEARCH_PARAMS[param_name][1], SEARCH_PARAMS[param_name][2]
    return None, None

def get_step(param_name):
    if param_name in SEARCH_PARAMS:
        return SEARCH_PARAMS[param_name][3]
    return 1
