
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
        score = evaluate_position_tunable(
            piece_bbs[i],
            occupancy_bbs[i],
            game_states[i],
            theta
        )
        
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
        
        # --- 1. Positive Values (Bonuses & Penalties stored as positive) ---
        # Material
        cm.add_range("MG_MATERIAL_VALUES", min_val=0)
        cm.add_range("EG_MATERIAL_VALUES", min_val=0)
        
        # Mobility Weights
        cm.add_range("KNIGHT_MOBILITY_WEIGHT", min_val=0)
        cm.add_range("BISHOP_MOBILITY_WEIGHT", min_val=0)
        cm.add_range("ROOK_MOBILITY_WEIGHT", min_val=0)
        cm.add_range("QUEEN_MOBILITY_WEIGHT", min_val=0)
        
        # Coordination Bonuses
        cm.add_range("BISHOP_PAIR_BONUS", min_val=0)
        cm.add_range("ROOK_ON_SEMI_OPEN_FILE_BONUS", min_val=0)
        cm.add_range("ROOK_ON_OPEN_FILE_BONUS", min_val=0)
        cm.add_range("ROOK_ON_SEVENTH_BONUS", min_val=0)
        
        # Pawn Bonuses
        cm.add_range("PASSED_PAWN_BONUS", min_val=0)
        cm.add_range("CONNECTED_PASSED_PAWN_BONUS", min_val=0)
        
        # Penalties (Subtracted in eval, so must be positive in theta)
        cm.add_range("BACKWARD_PAWN_PENALTY", min_val=0)
        
        # Outposts
        cm.add_range("OUTPOST_BONUS_KNIGHT", min_val=0)
        cm.add_range("OUTPOST_BONUS_BISHOP", min_val=0)
        cm.add_range("OUTPOST_HOLE_BONUS", min_val=0)

        # King Safety (Most are positive parameters used as penalties or bonuses)
        cm.add_range("KING_SAFETY_WEAK_UNITS", min_val=0)
        cm.add_range("KING_SAFETY_ATTACK_UNITS", min_val=0)
        cm.add_range("KING_SAFETY_TABLE", min_val=0)
        cm.add_range("KING_TROPISM_WEIGHTS", min_val=0)
        cm.add_range("PAWN_STORM_PENALTY_BY_RANK", min_val=0)
        cm.add_range("SCALING_WEIGHTS", min_val=0)
        cm.add_range("PAWN_SHIELD_MISSING_PENALTY", min_val=0)
        cm.add_range("PAWN_SHIELD_INTACT_BONUS", min_val=0)
        cm.add_range("PAWN_SHIELD_ADVANCED_BONUS", min_val=0)
        cm.add_range("PAWN_SHIELD_PUSHED_PENALTY", min_val=0)
        cm.add_range("KING_OPEN_FILE_PENALTY", min_val=0)
        cm.add_range("KING_SEMI_OPEN_FILE_PENALTY", min_val=0)
        
        # Threats
        cm.add_range("THREAT_SAFE_PAWN", min_val=0)
        cm.add_range("THREAT_MINOR_ON_MAJOR", min_val=0)
        cm.add_range("THREAT_ROOK_ON_QUEEN", min_val=0)
        cm.add_range("THREAT_HANGING", min_val=0)
        
        # Initiative
        cm.add_range("INITIATIVE_BONUS", min_val=0)

        # --- 2. Negative Values (Penalties stored as negative) ---
        cm.add_range("ISOLATED_PAWN_PENALTY", max_val=0)
        cm.add_range("DOUBLED_PAWN_PENALTY", max_val=0)

        # --- 3. Monotonic Constraints ---

        # 子力分數: P < N < B < R < Q (Axis 0)
        cm.add_monotonic("MG_MATERIAL_VALUES", axis=0, direction='increasing')
        cm.add_monotonic("EG_MATERIAL_VALUES", axis=0, direction='increasing')
        # Passed Pawn Bonus: Should increase with Rank (Axis 0)
        cm.add_monotonic("PASSED_PAWN_BONUS", axis=0, direction='increasing')
        
        # King Safety Table: More attack units -> Higher penalty (Axis 0)
        cm.add_monotonic("KING_SAFETY_TABLE", axis=0, direction='increasing')
        
        # King Tropism Weights: P < N < B < R < Q (Axis 0)
        cm.add_monotonic("KING_TROPISM_WEIGHTS", axis=0, direction='increasing')
        
        # King Safety Attack Units: P < N < B < R < Q (Axis 0)
        cm.add_monotonic("KING_SAFETY_ATTACK_UNITS", axis=0, direction='increasing')

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
    parser.add_argument("--iter", type=int, default=1000, help="Number of SPSA iterations")
    parser.add_argument("--alpha", type=float, default=5000.0, help="Learning rate scaling (a)")
    parser.add_argument("--c", type=float, default=5.0, help="Perturbation scaling (c)")
    
    # New arguments
    parser.add_argument("--tune", nargs='+', help="List of parameter names to tune (others will be frozen)")
    parser.add_argument("--exclude", nargs='+', help="List of parameter names to exclude/freeze")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size for gradient estimation (e.g. 16384)")
    
    args = parser.parse_args()

    pm = ParameterManager()
    
    if not os.path.exists("tuner/dataset.npz"):
        print("Dataset not found. Please run preprocess_data.py first.")
    else:
        optimizer = SPSAOptimizer("tuner/dataset.npz", pm)
        optimizer.optimize(
            iterations=args.iter, 
            alpha=args.alpha, 
            c=args.c,
            include_list=args.tune,
            exclude_list=args.exclude,
            batch_size=args.batch_size
        )
