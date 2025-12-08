
import numpy as np
import numba
import time
import os
import argparse
from tuner.parameters import ParameterManager
from tuner.tunable_eval import evaluate_position_tunable

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
        
        self.n_samples = len(self.results)
        print(f"Loaded {self.n_samples} positions.")
        
        # Initial best
        self.best_theta = self.param_manager.get_initial_theta()
        print("Calculating initial error...")
        self.best_loss = self._cost(self.best_theta)
        print(f"Initial MSE: {self.best_loss:.6f}")

    def _cost(self, theta):
        return compute_mse(
            self.piece_bbs, 
            self.occupancy_bbs, 
            self.game_states, 
            self.results, 
            theta
        )

    def optimize(self, iterations=1000, alpha=0.1, gamma=0.101, c=2.0, A=100, save_path="tuner/tuned_constants.py"):
        """
        Standard SPSA implementation.
        """
        theta = self.best_theta.copy()

        # --- 新增：定義要調整的範圍 (Mask) ---
        # 預設全部鎖定 (全部為 0)
        mask = np.zeros_like(theta)

        # 開放你想調整的參數索引
        # 例如：只調整 MG_MATERIAL (索引 0~5)
        # 你可以去 parameters.py 或直接 print 出來確認索引範圍
        # 參考：
        # IDX_MG_MATERIAL = 0 (長度 6)
        # IDX_PST_MG = 12 (長度 384)

        mask[0:] = 1    # 開放 MG Material
        # mask[12:396] = 1 # 開放 PST MG (12 + 384 = 396)

        # Determine fixed/variable indices (Optional: if we wanted to freeze some params)
        # For now, tune all float values in theta.
        
        print(f"Starting SPSA optimization for {iterations} iterations...")
        
        start_time = time.time()
        
        for k in range(1, iterations + 1):
            # Decay steps
            ak = alpha / ((A + k) ** 0.602)
            ck = c / (k ** gamma)
            
            # Perturbation vector (Bernoulli +/- 1)
            raw_delta = np.random.randint(0, 2, size=theta.shape) * 2 - 1
            # 乘上 Mask：不想調的地方變成 0，想調的地方保持 -1 或 1
            delta = raw_delta * mask

            # Two measurements
            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta
            
            loss_plus = self._cost(theta_plus)
            loss_minus = self._cost(theta_minus)
            
            # Gradient estimate
            # g = (y+ - y-) / (2*ck) * delta^-1
            # Since delta is +/- 1, delta^-1 is equal to delta
            ghat = (loss_plus - loss_minus) / (2 * ck) * delta
            
            # Update theta
            theta = theta - ak * ghat
            
            # Check for new best
            # (We only do this periodically to save time, or use the current estimate)
            # SPSA is noisy, so the current theta might not be strictly better.
            # But we track the "stable" loss periodically.
            
            if k % 10 == 0:
                current_loss = self._cost(theta)
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_theta = theta.copy()
                    print(f"Iter {k}: New Best MSE = {self.best_loss:.7f}")
                    # Save intermediate
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
        # print(f"Saved tuned constants to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=5000, help="Number of SPSA iterations")
    parser.add_argument("--alpha", type=float, default=20000.0, help="Learning rate scaling (a)")
    parser.add_argument("--c", type=float, default=5.0, help="Perturbation scaling (c)")
    args = parser.parse_args()

    pm = ParameterManager()
    
    if not os.path.exists("tuner/dataset.npz"):
        print("Dataset not found. Please run preprocess_data.py first.")
    else:
        optimizer = SPSAOptimizer("tuner/dataset.npz", pm)
        # Alpha needs to be relatively high because parameter values are large (e.g. 100-500)
        # so gradients need to move them by integer amounts. 
        # But MSE is small (0.25 max). Gradient is small. 
        # ghat ~ (diff 0.01) / (2 * 2) ~ 0.0025. 
        # update = ak * 0.0025. To move by 1 unit, ak needs to be ~400.
        # So alpha=100 is a reasonable start.
        optimizer.optimize(iterations=args.iter, alpha=args.alpha, c=args.c)
