
import numpy as np
import numba
import time
import os
import argparse
from tuner.parameters import ParameterManager
from tuner.tunable_eval import evaluate_position_tunable
from tuner.constraint_manager import ConstraintManager

# --- Cost Function (Numba Optimized) ---

@numba.njit(parallel=True, fastmath=True)
def compute_mse(piece_bbs, occupancy_bbs, game_states, results, theta, K_factor=1.0):
    n = len(results)
    total_error = 0.0
    
    # 400 * ln(10) to convert 10^(-s/400) to exp
    # 1 / (1 + 10^(-s/400)) = 1 / (1 + exp(-s * ln(10) / 400))
    # Let K' = ln(10)/400 ~= 0.005756
    scale = 0.00575646273 * K_factor

    for i in numba.prange(n):
        # Evaluate returns score relative to the side to move
        score = evaluate_position_tunable(
            piece_bbs[i],
            occupancy_bbs[i],
            game_states[i],
            theta,
            False
        )
        
        # Convert back to White's perspective to match the labels
        if game_states[i][0] == 1: # Black to move
            score = -score
        
        # Sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-score * scale))
        
        diff = results[i] - sigmoid
        total_error += diff * diff
        
    return total_error / n

# --- SPSA Optimizer ---

class SPSAOptimizer:
    def __init__(self, data_path, param_manager):
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        self.piece_bbs = data['piece_bbs']
        self.occupancy_bbs = data['occupancy_bbs']
        self.game_states = data['game_states']
        self.results = data['results']
        self.param_manager = param_manager
        
        # Initialize ConstraintManager
        self.constraint_manager = ConstraintManager(param_manager)
        self._setup_constraints()
        
        self.n_samples = len(self.results)
        print(f"Loaded {self.n_samples} positions.")
        
        # Display label distribution statistics
        r = self.results
        print(f"Label stats: min={r.min():.4f}, max={r.max():.4f}, mean={r.mean():.4f}, std={r.std():.4f}")
        # Check if labels are continuous or discrete
        unique_count = len(np.unique(np.round(r, 3)))
        if unique_count <= 5:
            print(f"  Warning: Only {unique_count} unique label values detected (likely discrete labels).")
            print(f"  Consider regenerating data with continuous Sigmoid labels for better results.")
        else:
            print(f"  Continuous labels detected ({unique_count} unique values). Good.")
        
        # Initial best
        self.best_theta = self.param_manager.get_initial_theta()
        print("Calculating initial error...")
        self.best_loss = self._cost(self.best_theta)
        print(f"Initial MSE: {self.best_loss:.6f}")

    def _setup_constraints(self):
        """
        Configures the constraints for the optimization.
        You can add more constraints here using self.constraint_manager.
        """
        cm = self.constraint_manager
        
        # --- Positive Values (Stored as positive) ---
        cm.add_range("MG_MATERIAL_VALUES", min_val=0)
        cm.add_range("EG_MATERIAL_VALUES", min_val=0)
        cm.add_range("BISHOP_PAIR_BONUS", min_val=0)
        cm.add_range("ROOK_ON_SEMI_OPEN_FILE_BONUS", min_val=0)
        cm.add_range("ROOK_ON_OPEN_FILE_BONUS", min_val=0)
        cm.add_range("PASSED_PAWN_BONUS", min_val=0)
        cm.add_range("CANDIDATE_PASSED_PAWN_BONUS", min_val=0)
        
        # Outposts
        cm.add_range("OUTPOST_BONUS_KNIGHT", min_val=0)
        cm.add_range("OUTPOST_BONUS_BISHOP", min_val=0)

        # King Safety / King Danger 
        cm.add_range("KING_SAFETY_ATTACK_UNITS", min_val=0)
        cm.add_range("KING_DANGER_WEAK_SQ", min_val=1)
        cm.add_range("KING_DANGER_UNSAFE_CHECK", min_val=1)
        cm.add_range("KING_DANGER_ATTACK_ON_KING_SQ", min_val=1)
        cm.add_range("KING_DANGER_NO_QUEEN", min_val=1)
        cm.add_range("KING_DANGER_PINNED", min_val=1)
        cm.add_range("KING_DANGER_DIVISOR", min_val=1)
        cm.add_range("SAFE_CHECK_KNIGHT", min_val=0)
        cm.add_range("SAFE_CHECK_BISHOP", min_val=0)
        cm.add_range("SAFE_CHECK_ROOK", min_val=0)
        cm.add_range("SAFE_CHECK_QUEEN", min_val=0)
        
        cm.add_range("KING_TROPISM_WEIGHTS", min_val=0)
        cm.add_range("PAWN_STORM_PENALTY_BY_RANK", min_val=0)
        cm.add_range("SCALING_WEIGHTS", min_val=0)
        
        cm.add_range("PAWN_SHIELD_MISSING_PENALTY", min_val=0)
        cm.add_range("PAWN_SHIELD_INTACT_BONUS", min_val=0)
        cm.add_range("PAWN_SHIELD_ADVANCED_BONUS", min_val=0)
        cm.add_range("PAWN_SHIELD_PUSHED_PENALTY", min_val=0)
        cm.add_range("KING_OPEN_FILE_PENALTY", min_val=0)
        cm.add_range("KING_SEMI_OPEN_FILE_PENALTY", min_val=0)
        # NOTE: KING_SAFETY_WEAK_SQUARE_PENALTY removed — not used in evaluation.py
        # Threats
        cm.add_range("THREAT_SAFE_PAWN", min_val=0)
        cm.add_range("THREAT_BY_MINOR", min_val=0)
        cm.add_range("THREAT_BY_ROOK", min_val=0)
        cm.add_range("THREAT_BY_KING", min_val=0)
        cm.add_range("THREAT_HANGING", min_val=0)
        # NOTE: THREAT_RESTRICTED_PIECE removed — not used in evaluation.py
        cm.add_range("THREAT_PAWN_PUSH", min_val=0)
        
        cm.add_range("TRAPPED_ROOK", min_val=0)
        cm.add_range("BISHOP_PAWNS_PENALTY", min_val=0)
        cm.add_range("KING_PROTECTOR", min_val=0)
        cm.add_range("CONNECTED_BONUS", min_val=0)
        cm.add_range("CONNECTED_SUPPORT_WEIGHT", min_val=0)
        cm.add_range("MAX_KING_ATTACKERS", min_val=0)
        cm.add_range("PROXIMITY_ENEMY_WEIGHT", min_val=0)
        cm.add_range("PROXIMITY_FRIENDLY_WEIGHT", min_val=0)
        cm.add_range("MAX_PROXIMITY_BONUS", min_val=0)
        cm.add_range("KING_DANGER_SINGLE_ATTACKER_DIVISOR", min_val=1)
        cm.add_range("UNSTOPPABLE_PAWN_BONUS", min_val=0)
        cm.add_range("EG_KING_PAWN_PROXIMITY_WEIGHT", min_val=0)
        cm.add_range("BLOCKED_PASSER_DIVISOR", min_val=1)
        
        # --- Negative Values (Stored as negative) ---
        cm.add_range("ISOLATED_PAWN_PENALTY", max_val=0)
        cm.add_range("DOUBLED_PAWN_PENALTY", max_val=0)
        cm.add_range("BACKWARD_PAWN_PENALTY", max_val=0)
        

        # --- Monotonic Constraints ---
        
        # CRITICAL: Lock Pawn values to serving as the absolute reference unit (100 MG / 120 EG).
        # This prevents the entire evaluation scale from drifting/shrinking (e.g. Pawn becoming 36).
        mg_start, _ = cm.pm.get_param_indices("MG_MATERIAL_VALUES")
        eg_start, _ = cm.pm.get_param_indices("EG_MATERIAL_VALUES")
        cm.add_range_on_absolute_index(mg_start + 0, min_val=100.0, max_val=100.0) # Pawn MG = 100
        cm.add_range_on_absolute_index(eg_start + 0, min_val=120.0, max_val=120.0) # Pawn EG = 120

        # Add floors for pieces to prevent them from becoming too cheap relative to pawns
        # [P, N, B, R, Q, K]
        # P is pinned above. N, B, R, Q should have minimums.
        for i in range(1, 4): # N, B, R
             cm.add_range_on_absolute_index(mg_start + i, min_val=250.0)
             cm.add_range_on_absolute_index(eg_start + i, min_val=250.0)
        cm.add_range_on_absolute_index(mg_start + 4, min_val=700.0) # Queen MG min
        cm.add_range_on_absolute_index(eg_start + 4, min_val=700.0) # Queen EG min

        # CRITICAL: MG/EG_MATERIAL_VALUES has 6 elements [P, N, B, R, Q, K].
        # King (index 5) must be EXCLUDED from the increasing constraint; otherwise
        # the optimizer forces King >= Queen value, which is catastrophic.
        # end_idx=5 means only indices 0..4 (P, N, B, R, Q) are sorted.
        cm.add_monotonic("MG_MATERIAL_VALUES", axis=0, direction='increasing', end_idx=5)
        cm.add_monotonic("EG_MATERIAL_VALUES", axis=0, direction='increasing', end_idx=5)
        # Pin King material (index 5) to exactly 0 so it never drifts.
        cm.add_range_on_absolute_index(mg_start + 5, min_val=0, max_val=0)  # King MG = 0
        cm.add_range_on_absolute_index(eg_start + 5, min_val=0, max_val=0)  # King EG = 0
        
        # Lock SCALING_WEIGHTS to original values [0, 4, 4, 6, 10]
        # This prevents it from being zeroed out and breaking the scaling formula.
        sc_start, _ = cm.pm.get_param_indices("SCALING_WEIGHTS")
        cm.add_range_on_absolute_index(sc_start + 0, min_val=0.0, max_val=0.0)
        cm.add_range_on_absolute_index(sc_start + 1, min_val=4.0, max_val=4.0)
        cm.add_range_on_absolute_index(sc_start + 2, min_val=4.0, max_val=4.0)
        cm.add_range_on_absolute_index(sc_start + 3, min_val=6.0, max_val=6.0)
        cm.add_range_on_absolute_index(sc_start + 4, min_val=10.0, max_val=10.0)
        
        cm.add_monotonic("PASSED_PAWN_BONUS", axis=0, direction='increasing', end_idx=7)
        cm.add_monotonic("CANDIDATE_PASSED_PAWN_BONUS", axis=0, direction='increasing', end_idx=7)
        cm.add_monotonic("CONNECTED_BONUS", axis=0, direction='increasing', end_idx=7)
        
        cm.add_monotonic("KING_TROPISM_WEIGHTS", axis=0, direction='increasing')
        cm.add_monotonic("KING_SAFETY_ATTACK_UNITS", axis=0, direction='increasing')
        
        # Mobility tables monotonicity
        cm.add_monotonic("KNIGHT_MOBILITY_BONUS", axis=0, direction='increasing')
        cm.add_monotonic("BISHOP_MOBILITY_BONUS", axis=0, direction='increasing')
        cm.add_monotonic("ROOK_MOBILITY_BONUS", axis=0, direction='increasing')
        cm.add_monotonic("QUEEN_MOBILITY_BONUS", axis=0, direction='increasing')

        # --- Invariant Pins (Logical Zeros) ---
        
        # Passed Pawns: Rank 1, 2, and 8 (indices 0, 1, 7). 
        # Rank 8 (index 7) is 0, so we used end_idx=7 above to stop monotonicity at index 6.
        pp_start, _ = cm.pm.get_param_indices("PASSED_PAWN_BONUS")
        for r in [0, 1, 7]:
            cm.add_range_on_absolute_index(pp_start + 2*r, min_val=0, max_val=0) # MG
            cm.add_range_on_absolute_index(pp_start + 2*r + 1, min_val=0, max_val=0) # EG
            
        cp_start, _ = cm.pm.get_param_indices("CANDIDATE_PASSED_PAWN_BONUS")
        for r in [0, 1, 7]:
            cm.add_range_on_absolute_index(cp_start + 2*r, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(cp_start + 2*r + 1, min_val=0, max_val=0)
            
        con_start, _ = cm.pm.get_param_indices("CONNECTED_BONUS")
        cm.add_range_on_absolute_index(con_start + 0, min_val=0, max_val=0) # Rank 1
        cm.add_range_on_absolute_index(con_start + 7, min_val=0, max_val=0) # Rank 8
        
        # Outposts: Ranks 1, 2, and 8 should be 0 (Indices 0, 1, 7)
        ok_start, _ = cm.pm.get_param_indices("OUTPOST_BONUS_KNIGHT")
        ob_start, _ = cm.pm.get_param_indices("OUTPOST_BONUS_BISHOP")
        for r in [0, 1, 7]:
            cm.add_range_on_absolute_index(ok_start + 2*r, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(ok_start + 2*r + 1, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(ob_start + 2*r, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(ob_start + 2*r + 1, min_val=0, max_val=0)
            
        # Also need monotonic for Outposts up to Index 6
        cm.add_monotonic("OUTPOST_BONUS_KNIGHT", axis=0, direction='increasing', end_idx=7)
        cm.add_monotonic("OUTPOST_BONUS_BISHOP", axis=0, direction='increasing', end_idx=7)

        # Tropism: Pawn (index 0) usually has no tropism bonus
        tr_start, _ = cm.pm.get_param_indices("KING_TROPISM_WEIGHTS")
        cm.add_range_on_absolute_index(tr_start + 0, min_val=0, max_val=0)
        
        # Pawn Storm: Rank 1 (Index 7) is logically 0.
        # Rank 7 (Index 1) is MOST dangerous, Rank 2 (Index 6) is LEAST.
        ps_start, _ = cm.pm.get_param_indices("PAWN_STORM_PENALTY_BY_RANK")
        cm.add_range_on_absolute_index(ps_start + 7, min_val=0, max_val=0) # Rank 1
        
        # Re-configure monotonicity for PAWN_STORM to cover indices 1-6 (Decreasing)
        # 1 > 2 > 3 > 4 > 5 > 6
        # To use add_monotonic with axis=0, we need to handle the start index too.
        # Since our CM doesn't support start_idx, we just use range constraints for now.
        for i in range(1, 7):
            cm.add_range_on_absolute_index(ps_start + i, min_val=0.0)
        
        # Threats vs King (Index 5) in Minor/Rook tables should be 0
        tm_start, _ = cm.pm.get_param_indices("THREAT_BY_MINOR")
        cm.add_range_on_absolute_index(tm_start + 10, min_val=0, max_val=0) # vs King MG
        cm.add_range_on_absolute_index(tm_start + 11, min_val=0, max_val=0) # vs King EG
        tr_start_threat, _ = cm.pm.get_param_indices("THREAT_BY_ROOK")
        cm.add_range_on_absolute_index(tr_start_threat + 10, min_val=0, max_val=0)
        cm.add_range_on_absolute_index(tr_start_threat + 11, min_val=0, max_val=0)

        print(f"Constraints configured: {len(cm.constraints)} constraints active.")

    def _cost(self, theta, indices=None):
        # Support batching via indices slicing
        if indices is None:
            # Full dataset
            return compute_mse(
                self.piece_bbs, 
                self.occupancy_bbs, 
                self.game_states, 
                self.results, 
                theta
            )
        else:
            # Subset
            # Numpy slicing with integer array returns a copy. This is acceptable for mini-batches.
            return compute_mse(
                self.piece_bbs[indices], 
                self.occupancy_bbs[indices], 
                self.game_states[indices], 
                self.results[indices], 
                theta
            )
    
    def _create_mask(self, include_list=None, exclude_list=None):
        """
        Creates a mask vector where 1 indicates the parameter should be tuned,
        and 0 indicates it should be frozen.
        """
        mask = np.zeros_like(self.best_theta)
        
        # Map parameter names to indices
        name_to_indices = {}
        for p in self.param_manager.param_map:
            name = p['name']
            start = p['start']
            count = p['count']
            name_to_indices[name] = (start, count)

        if include_list:
            print(f"Freezing all parameters except: {include_list}")
            for name in include_list:
                if name in name_to_indices:
                    start, count = name_to_indices[name]
                    mask[start : start + count] = 1.0
                else:
                    print(f"Warning: Unknown parameter '{name}' in include list.")
        else:
            mask[:] = 1.0

        if exclude_list:
            print(f"Freezing parameters: {exclude_list}")
            for name in exclude_list:
                if name in name_to_indices:
                    start, count = name_to_indices[name]
                    mask[start : start + count] = 0.0
                else:
                    print(f"Warning: Unknown parameter '{name}' in exclude list.")
        
        active_params = np.sum(mask)
        total_params = len(mask)
        print(f"Active parameters: {int(active_params)} / {total_params}")
        
        return mask

    def optimize(self, iterations=1000, alpha=0.1, gamma=0.101, c=2.0, A=100, 
                 save_path="tuner/tuned_constants.py", include_list=None, exclude_list=None,
                 batch_size=None):
        """
        Standard SPSA implementation with Constraint Enforcement, Parameter Selection, and Mini-batching.
        """
        theta = self.best_theta.copy()
        mask = self._create_mask(include_list, exclude_list)
        
        print(f"Starting SPSA optimization for {iterations} iterations...")
        if batch_size:
            print(f"Using mini-batch size: {batch_size}")
        
        start_time = time.time()
        
        for k in range(1, iterations + 1):
            # Select batch if enabled
            batch_indices = None
            if batch_size is not None and batch_size < self.n_samples:
                batch_indices = np.random.choice(self.n_samples, batch_size, replace=False)

            # Decay steps
            ak = alpha / ((A + k) ** 0.602)
            ck = c / (k ** gamma)
            
            # Perturbation
            raw_delta = np.random.randint(0, 2, size=theta.shape) * 2 - 1
            delta = raw_delta * mask

            # Two measurements (using the same batch for both)
            theta_plus = theta + ck * delta
            self.constraint_manager.apply(theta_plus)
            
            theta_minus = theta - ck * delta
            self.constraint_manager.apply(theta_minus)
            
            loss_plus = self._cost(theta_plus, indices=batch_indices)
            loss_minus = self._cost(theta_minus, indices=batch_indices)
            
            # Gradient estimate
            ghat = (loss_plus - loss_minus) / (2 * ck) * delta
            
            # Update theta
            theta = theta - ak * ghat
            
            # Apply Constraints
            self.constraint_manager.apply(theta)
            
            # Check for new best
            if k % 10 == 0:
                # Calculate full loss for accurate reporting and "Best" tracking
                current_loss = self._cost(theta, indices=None) # Full dataset check
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_theta = theta.copy()
                    print(f"Iter {k}: New Best MSE = {self.best_loss:.7f}")
                    self.save_results(save_path)
                else:
                    print(f"Iter {k}: MSE = {current_loss:.7f} (Best: {self.best_loss:.7f})")
            
        total_time = time.time() - start_time
        print(f"Optimization finished in {total_time:.2f}s.")
        print(f"Final Best MSE: {self.best_loss:.7f}")
        self.save_results(save_path)

    def save_results(self, path):
        content = self.param_manager.update_constants_string(self.best_theta)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=20000, help="Number of SPSA iterations")
    parser.add_argument("--alpha", type=float, default=30000.0, help="Learning rate scaling (a)")
    parser.add_argument("--c", type=float, default=5.0, help="Perturbation scaling (c)")
    
    # New arguments
    parser.add_argument("--tune", nargs='+', help="List of parameter names to tune (others will be frozen)")
    parser.add_argument("--exclude", nargs='+', help="List of parameter names to exclude/freeze")
    parser.add_argument("--batch-size", type=int, default=32768, help="Mini-batch size for gradient estimation (e.g. 16384)")
    parser.add_argument("--dataset", type=str, default="tuner/dataset.npz", help="Path to the preprocessed dataset .npz file")
    # default=[
    #     "KING_SAFETY_ATTACK_UNITS", "KING_DANGER_WEAK_SQ", "KING_DANGER_UNSAFE_CHECK", 
    #     "KING_DANGER_ATTACK_ON_KING_SQ", "KING_DANGER_NO_QUEEN", "KING_DANGER_PINNED", 
    #     "KING_DANGER_DIVISOR", "SAFE_CHECK_KNIGHT", "SAFE_CHECK_BISHOP", "SAFE_CHECK_ROOK", 
    #     "SAFE_CHECK_QUEEN", "KING_TROPISM_WEIGHTS", "KING_PROTECTOR",
    #     "PAWN_SHIELD_MISSING_PENALTY", "PAWN_SHIELD_INTACT_BONUS", "PAWN_SHIELD_ADVANCED_BONUS",
    #     "PAWN_SHIELD_PUSHED_PENALTY", "KING_OPEN_FILE_PENALTY", "KING_SEMI_OPEN_FILE_PENALTY"
    # ]
    args = parser.parse_args()

    pm = ParameterManager()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset not found at '{args.dataset}'. Please run preprocess_data.py first.")
    else:
        optimizer = SPSAOptimizer(args.dataset, pm)
        optimizer.optimize(
            iterations=args.iter, 
            alpha=args.alpha, 
            c=args.c,
            include_list=args.tune,
            exclude_list=args.exclude,
            batch_size=args.batch_size
        )
