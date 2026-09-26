
import numpy as np
import numba
import time
import os
import argparse
from tuner.parameters import ParameterManager
from tuner.tunable_eval import evaluate_position_tunable
from tuner.constraint_manager import ConstraintManager
from tuner.constants_snapshot import (
    PHASE_MIDGAME,
    PHASE_ENDGAME_RATIO_NUM,
    PHASE_ENDGAME_RATIO_DEN,
)

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


@numba.njit(parallel=True, fastmath=True)
def compute_squared_errors(
    piece_bbs, occupancy_bbs, game_states, results, theta, K_factor=1.0
):
    """Per-position squared errors for fixed-bucket validation reporting."""
    n = len(results)
    errors = np.empty(n, dtype=np.float64)
    scale = 0.00575646273 * K_factor

    for i in numba.prange(n):
        score = evaluate_position_tunable(
            piece_bbs[i],
            occupancy_bbs[i],
            game_states[i],
            theta,
            False,
        )
        if game_states[i][0] == 1:
            score = -score
        sigmoid = 1.0 / (1.0 + np.exp(-score * scale))
        diff = results[i] - sigmoid
        errors[i] = diff * diff

    return errors

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
            print(f"  Discrete WDL labels detected ({unique_count} values): valid pure Texel targets.")
        else:
            print(f"  Continuous/mixed labels detected ({unique_count} unique values).")
        
        # Initial best (full/eval MSE is computed in optimize() on the chosen eval set)
        self.best_theta = self.param_manager.get_initial_theta()
        self.best_loss = float("inf")

    def _setup_constraints(self):
        """Chess-logic constraints for the SF11-aligned HCE parameter set."""
        cm = self.constraint_manager
        names = {p["name"] for p in self.param_manager.param_map}

        def maybe_range(name, **kw):
            if name in names:
                cm.add_range(name, **kw)

        # --- Material / bonuses (non-negative) ---
        maybe_range("MG_MATERIAL_VALUES", min_val=0)
        maybe_range("EG_MATERIAL_VALUES", min_val=0)
        maybe_range("ROOK_ON_SEMI_OPEN_FILE_BONUS", min_val=0)
        maybe_range("ROOK_ON_OPEN_FILE_BONUS", min_val=0)
        maybe_range("PASSED_PAWN_BONUS", min_val=0)
        maybe_range("PASSED_CONNECTED_BONUS", min_val=0)
        maybe_range("PASSED_PHALANX_BONUS", min_val=0)
        maybe_range("BISHOP_PAIR_BONUS", min_val=0)
        maybe_range("BISHOP_PAIR_PAWN_SCALE", min_val=0)
        maybe_range("OUTPOST_BONUS_KNIGHT", min_val=0)
        maybe_range("OUTPOST_BONUS_BISHOP", min_val=0)
        maybe_range("REACHABLE_OUTPOST_BONUS", min_val=0)
        maybe_range("CONNECTED_BONUS", min_val=0)
        maybe_range("CONNECTED_BONUS_EG", min_val=0)
        # Support weights: multiplicative in connected formula — pin (do not SPSA).
        # (Unconstrained runs previously collapsed these to 0 / floor.)
        maybe_range("MINOR_BEHIND_PAWN", min_val=0)  # SF11 S(18,3) always non-negative
        maybe_range("REACHABLE_OUTPOST_BONUS", min_val=0)
        maybe_range("PASSED_FILE_BONUS", min_val=0)
        maybe_range("PASSED_PATH_SAFE_NONE_ATTACK", min_val=0)
        maybe_range("PASSED_PATH_SAFE_EDGE_ATTACK", min_val=0)
        maybe_range("PASSED_PATH_SAFE_BLOCK_ONLY", min_val=0)
        maybe_range("PASSED_PATH_SUPPORT_BONUS", min_val=0)
        # King safety: non-negative danger units (SF kingDanger coeffs)
        maybe_range("SAFE_CHECK_KNIGHT", min_val=200, max_val=900)   # ~617
        maybe_range("SAFE_CHECK_BISHOP", min_val=150, max_val=800)   # ~496
        maybe_range("SAFE_CHECK_ROOK", min_val=400, max_val=1200)    # ~843 highest in SF
        maybe_range("SAFE_CHECK_QUEEN", min_val=200, max_val=1000)   # ~609
        maybe_range("KING_DANGER_WEAK_SQ", min_val=40, max_val=280)          # ~144
        maybe_range("KING_DANGER_UNSAFE_CHECK", min_val=30, max_val=240)     # ~115
        maybe_range("KING_DANGER_BLOCKERS", min_val=20, max_val=180)         # ~76
        maybe_range("KING_DANGER_ATTACK_ON_KING_SQ", min_val=10, max_val=140)  # ~53
        maybe_range("KING_DANGER_NO_QUEEN", min_val=300, max_val=950)        # ~682; must stay large
        maybe_range("KING_DANGER_NO_QUEEN_ROOKLESS", min_val=300, max_val=950)
        maybe_range("KING_DANGER_KNIGHT_DEF", min_val=20, max_val=180)       # ~78
        maybe_range("KING_DANGER_OFFSET", min_val=0, max_val=80)             # ~28
        maybe_range("KING_ATTACK_WEIGHTS", min_val=0)
        # Shelter / storm (SF tables can be negative)
        maybe_range("SHELTER_BASE_MG", min_val=0, max_val=20)
        maybe_range("SHELTER_BASE_EG", min_val=0, max_val=15)
        maybe_range("BLOCKED_STORM", max_val=0, min_val=-150)       # SF storm penalty (negative)
        maybe_range("BLOCKED_STORM_EG", max_val=0, min_val=-120)
        maybe_range("KING_PAWN_DIST_PENALTY_EG", max_val=0, min_val=-40)
        maybe_range("PAWNLESS_FLANK", max_val=0)   # penalty S-like
        # FLANK_ATTACKS: SF S(8,0) → our table often MG-negative; floor/ceiling set below via soft band
        maybe_range("THREAT_SAFE_PAWN", min_val=0)
        maybe_range("THREAT_BY_MINOR", min_val=0)
        maybe_range("THREAT_BY_ROOK", min_val=0)
        maybe_range("THREAT_BY_KING", min_val=0)
        maybe_range("THREAT_HANGING", min_val=0)
        maybe_range("THREAT_RESTRICTED_PIECE", min_val=0)
        maybe_range("THREAT_PAWN_PUSH", min_val=0)
        maybe_range("THREAT_KNIGHT_ON_QUEEN", min_val=0)
        maybe_range("THREAT_SLIDER_ON_QUEEN", min_val=0)
        maybe_range("TRAPPED_ROOK", min_val=0)
        maybe_range("BISHOP_PAWNS_PENALTY", min_val=0)
        maybe_range("LONG_DIAGONAL_BISHOP", min_val=0)
        maybe_range("ROOK_ON_QUEEN_FILE", min_val=0)
        maybe_range("WEAK_QUEEN", min_val=0)
        maybe_range("KING_PROTECTOR", min_val=0)
        maybe_range("UNSTOPPABLE_PAWN_BONUS", min_val=0)
        # TEMPO / SPACE_THRESHOLD: soft bands applied later (not hard-pin)
        maybe_range("TEMPO_BONUS", min_val=12, max_val=36)  # SF 28 scaled ≈22; allow modest search-coupled move
        maybe_range("SPACE_THRESHOLD", min_val=3500, max_val=6500)  # SF 12222 scaled ≈5.5k; keep mid-open gate
        maybe_range("IMBALANCE_DIVISOR", min_val=1)
        maybe_range("CANDIDATE_PASSER_DIVISOR", min_val=1)
        maybe_range("KING_PROX_ENEMY_DIV", min_val=1)
        maybe_range("SPACE_BONUS_DIVISOR", min_val=1)
        maybe_range("SPACE_SCALE_DIVISOR", min_val=1)

        # Penalties stored as negative
        maybe_range("ISOLATED_PAWN_PENALTY", max_val=0)
        maybe_range("DOUBLED_PAWN_PENALTY", max_val=0)
        maybe_range("BACKWARD_PAWN_PENALTY", max_val=0)
        maybe_range("WEAK_LEVER_PENALTY", max_val=0)
        maybe_range("WEAK_UNOPPOSED_PENALTY", max_val=0)

        # Pin scale: MG Pawn = 100 (display/search anchor). EG Pawn soft ±10% around θ₀.
        # King material = 0 always.
        mg_start, _ = cm.pm.get_param_indices("MG_MATERIAL_VALUES")
        eg_start, _ = cm.pm.get_param_indices("EG_MATERIAL_VALUES")
        cm.add_range_on_absolute_index(mg_start + 0, min_val=100.0, max_val=100.0)
        eg_pawn0 = float(cm.pm.theta[eg_start + 0])  # typically 120
        eg_pawn_half = abs(eg_pawn0) * 0.10
        cm.add_range_on_absolute_index(
            eg_start + 0,
            min_val=max(1.0, eg_pawn0 - eg_pawn_half),
            max_val=eg_pawn0 + eg_pawn_half,
        )
        for i in range(1, 4):
            cm.add_range_on_absolute_index(mg_start + i, min_val=250.0)
            cm.add_range_on_absolute_index(eg_start + i, min_val=250.0)
        cm.add_range_on_absolute_index(mg_start + 4, min_val=700.0)
        cm.add_range_on_absolute_index(eg_start + 4, min_val=700.0)
        cm.add_monotonic("MG_MATERIAL_VALUES", axis=0, direction="increasing", end_idx=5)
        cm.add_monotonic("EG_MATERIAL_VALUES", axis=0, direction="increasing", end_idx=5)
        cm.add_range_on_absolute_index(mg_start + 5, min_val=0, max_val=0)
        cm.add_range_on_absolute_index(eg_start + 5, min_val=0, max_val=0)

        # Rank tables: pin unused ranks only. Do NOT force mono on PASSED/OUTPOST —
        # SF11 tables peak mid-board and are not strictly increasing.
        if "PASSED_PAWN_BONUS" in names:
            pp_start, _ = cm.pm.get_param_indices("PASSED_PAWN_BONUS")
            for r in (0, 7):  # rank-1 and rank-8 unused
                cm.add_range_on_absolute_index(pp_start + 2 * r, min_val=0, max_val=0)
                cm.add_range_on_absolute_index(pp_start + 2 * r + 1, min_val=0, max_val=0)

        if "CONNECTED_BONUS" in names:
            cm.add_monotonic("CONNECTED_BONUS", axis=0, direction="increasing", end_idx=7)
            con_start, _ = cm.pm.get_param_indices("CONNECTED_BONUS")
            cm.add_range_on_absolute_index(con_start + 0, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(con_start + 7, min_val=0, max_val=0)
        if "CONNECTED_BONUS_EG" in names:
            cm.add_monotonic("CONNECTED_BONUS_EG", axis=0, direction="increasing", end_idx=7)
            con_eg, _ = cm.pm.get_param_indices("CONNECTED_BONUS_EG")
            cm.add_range_on_absolute_index(con_eg + 0, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(con_eg + 7, min_val=0, max_val=0)

        if "PASSED_CONNECTED_BONUS" in names:
            cm.add_monotonic("PASSED_CONNECTED_BONUS", axis=0, direction="increasing", end_idx=7)
            pc_start, _ = cm.pm.get_param_indices("PASSED_CONNECTED_BONUS")
            cm.add_range_on_absolute_index(pc_start + 0, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(pc_start + 7, min_val=0, max_val=0)

        if "PASSED_PHALANX_BONUS" in names:
            cm.add_monotonic("PASSED_PHALANX_BONUS", axis=0, direction="increasing", end_idx=7)
            pp_start, _ = cm.pm.get_param_indices("PASSED_PHALANX_BONUS")
            cm.add_range_on_absolute_index(pp_start + 0, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(pp_start + 7, min_val=0, max_val=0)

        # SF11: candidate passers use bonus / 2 only — pin divisor.
        if "CANDIDATE_PASSER_DIVISOR" in names:
            cp_i, _ = cm.pm.get_param_indices("CANDIDATE_PASSER_DIVISOR")
            cm.add_range_on_absolute_index(cp_i, min_val=2.0, max_val=2.0)

        def pin_to_theta0(name: str) -> None:
            """Hard-pin every element of a registered param to snapshot θ₀."""
            if name not in names:
                return
            start, count = cm.pm.get_param_indices(name)
            th0 = cm.pm.theta
            for i in range(count):
                v = float(th0[start + i])
                cm.add_range_on_absolute_index(start + i, min_val=v, max_val=v)

        def soft_band_theta0(
            name: str,
            frac: float = 0.5,
            min_half: float = 3.0,
            abs_min=None,
            abs_max=None,
        ) -> None:
            """
            Per-element band around θ₀: [v ± max(|v|*frac, min_half)],
            optionally clipped to [abs_min, abs_max].
            Replaces hard pin for features that SF treats as free Score knobs
            but SPSA used to collapse without a floor/ceiling.
            """
            if name not in names:
                return
            start, count = cm.pm.get_param_indices(name)
            th0 = cm.pm.theta
            for i in range(count):
                v = float(th0[start + i])
                half = max(abs(v) * frac, min_half)
                lo, hi = v - half, v + half
                if abs_min is not None:
                    lo = max(lo, abs_min)
                if abs_max is not None:
                    hi = min(hi, abs_max)
                if lo > hi:
                    lo, hi = hi, lo
                cm.add_range_on_absolute_index(start + i, min_val=lo, max_val=hi)

        # Multiplicative / structural factors still hard-pinned (not passer dynamics / king-prox).
        for name, lo, hi in (
            # King–pawn endgame: (b_dist - w_dist) * factor
            ("KING_PAWN_PROXIMITY_FACTOR", 5.0, 5.0),
            # Bad bishop: penalty * (1 + factor * center_blocked)
            ("BISHOP_PAWNS_CENTER_BLOCKED_FACTOR", 1.0, 1.0),
            # Large known-win constant (not a free HCE knob)
            ("UNSTOPPABLE_PAWN_BONUS", 800.0, 800.0),
        ):
            if name in names:
                idx, _ = cm.pm.get_param_indices(name)
                cm.add_range_on_absolute_index(idx, min_val=lo, max_val=hi)

        def abs_pm1(name: str, lo_floor: float | None = 1.0) -> None:
            """Integer-ish scalar: θ₀ ± 1. Optional lo_floor (None = no floor, allows negatives)."""
            if name not in names:
                return
            idx, _ = cm.pm.get_param_indices(name)
            v = float(cm.pm.theta[idx])
            lo, hi = v - 1.0, v + 1.0
            if lo_floor is not None:
                lo = max(lo_floor, lo)
            cm.add_range_on_absolute_index(idx, min_val=lo, max_val=hi)

        def pct_band(
            name: str, frac: float = 0.05, min_half: float = 0.0
        ) -> None:
            """Scalar ±frac around θ₀, with an optional minimum half-width.

            Integer-valued evaluator knobs need at least one centipawn/integer of
            room; otherwise a percentage band such as 4 ± 10% quantizes back to
            the starting value on both sides and is not actually tunable.
            """
            if name not in names:
                return
            idx, _ = cm.pm.get_param_indices(name)
            v = float(cm.pm.theta[idx])
            half = max(abs(v) * frac, min_half)
            cm.add_range_on_absolute_index(idx, min_val=v - half, max_val=v + half)

        def soft_band_keep_zeros(
            name: str, frac: float = 0.05, min_half: float = 0.0
        ) -> None:
            """Per-element ±frac; exact zeros at θ₀ stay pinned at 0 (SF structural holes)."""
            if name not in names:
                return
            start, count = cm.pm.get_param_indices(name)
            th0 = cm.pm.theta
            for i in range(count):
                v = float(th0[start + i])
                if v == 0.0:
                    cm.add_range_on_absolute_index(start + i, min_val=0.0, max_val=0.0)
                else:
                    half = max(abs(v) * frac, min_half)
                    cm.add_range_on_absolute_index(
                        start + i, min_val=v - half, max_val=v + half
                    )

        # Connected support weights: multiplicative but small DOF — θ₀ ± 1
        abs_pm1("CONNECTED_SUPPORT_WEIGHT", lo_floor=1.0)
        abs_pm1("CONNECTED_SUPPORT_WEIGHT_EG", lo_floor=1.0)
        # Divisors: θ₀ ± 1 (IMBALANCE/SPACE_BONUS ~16; SPACE_SCALE ~4 → [3,5])
        abs_pm1("IMBALANCE_DIVISOR", lo_floor=1.0)
        abs_pm1("SPACE_BONUS_DIVISOR", lo_floor=1.0)
        abs_pm1("SPACE_SCALE_DIVISOR", lo_floor=1.0)
        # Path-safety k (SF 35/20/9/+5 scaled): θ₀ ± 1, never drop to 0
        for path_name in (
            "PASSED_PATH_SAFE_NONE_ATTACK",
            "PASSED_PATH_SAFE_EDGE_ATTACK",
            "PASSED_PATH_SAFE_BLOCK_ONLY",
            "PASSED_PATH_SUPPORT_BONUS",
        ):
            abs_pm1(path_name, lo_floor=1.0)
        # Imbalance display scale ±10%
        pct_band("IMBALANCE_SCALE_MG", 0.1)
        pct_band("IMBALANCE_SCALE_EG", 0.1)
        # Quadratic imbalance tables: ±10% on non-zeros, but at least ±1 so
        # small integer cells remain reachable; structural zeros stay pinned.
        soft_band_keep_zeros("IMBALANCE_QUADRATIC_OURS", 0.1, min_half=1.0)
        soft_band_keep_zeros("IMBALANCE_QUADRATIC_THEIRS", 0.1, min_half=1.0)

        # Passed dynamics w = MULT*rank + OFFSET (SF 5*r-13): θ₀ ± 1
        abs_pm1("PASSED_DYNAMICS_MULT", lo_floor=1.0)       # 5 → [4, 6]
        abs_pm1("PASSED_DYNAMICS_OFFSET", lo_floor=None)    # -13 → [-14, -12]
        # King proximity for passers (EG): θ₀ ± 1 on distance / enemy ratio coeffs
        abs_pm1("KING_PROXIMITY_MAX_DIST", lo_floor=1.0)    # 5 → [4, 6]
        abs_pm1("KING_PROX_ENEMY_MULT", lo_floor=1.0)       # 9 → [8, 10]
        abs_pm1("KING_PROX_ENEMY_DIV", lo_floor=1.0)        # 4 → [3, 5]
        # Friendly mult: discrete {1, 2} (SF-scale small integer)
        if "KING_PROX_FRIENDLY_MULT" in names:
            fm_i, _ = cm.pm.get_param_indices("KING_PROX_FRIENDLY_MULT")
            cm.add_range_on_absolute_index(fm_i, min_val=1.0, max_val=2.0)
        # Proximity clamp ±10%
        pct_band("KING_PROX_MIN_BONUS", 0.10)               # -70 → [-77, -63]
        pct_band("KING_PROX_MAX_BONUS", 0.10)               #  70 → [63, 77]

        # Tempo: SF 28 → ours ≈22 (×100/128). Allow modest move (search-coupled).
        if "TEMPO_BONUS" in names:
            t_i, _ = cm.pm.get_param_indices("TEMPO_BONUS")
            # Prefer band around θ₀ but keep global 12–36 from maybe_range as outer box.
            v = float(cm.pm.theta[t_i])
            cm.add_range_on_absolute_index(t_i, min_val=max(12.0, v - 6.0), max_val=min(36.0, v + 6.0))

        # Space gate: SF 12222 Value → our npm scale ~5.5k proportional; θ₀=4700 co-tuned.
        if "SPACE_THRESHOLD" in names:
            s_i, _ = cm.pm.get_param_indices("SPACE_THRESHOLD")
            v = float(cm.pm.theta[s_i])
            cm.add_range_on_absolute_index(
                s_i, min_val=max(3500.0, v - 800.0), max_val=min(6500.0, v + 800.0)
            )

        # Initiative complexity weights/offsets — soft ±10% (was hard-pin),
        # with at least ±1 because the evaluator consumes integer constants.
        for name in (
            "INITIATIVE_PASSED_WEIGHT_MG",
            "INITIATIVE_PAWN_WEIGHT_MG",
            "INITIATIVE_OUTFLANKING_WEIGHT_MG",
            "INITIATIVE_INFILTRATION_WEIGHT_MG",
            "INITIATIVE_BOTH_FLANKS_WEIGHT_MG",
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_MG",
            "INITIATIVE_ALMOST_UNWIN_WEIGHT_MG",
            "INITIATIVE_OFFSET_MG",
            "INITIATIVE_MG_OFFSET",
            "INITIATIVE_PASSED_WEIGHT_EG",
            "INITIATIVE_PAWN_WEIGHT_EG",
            "INITIATIVE_OUTFLANKING_WEIGHT_EG",
            "INITIATIVE_INFILTRATION_WEIGHT_EG",
            "INITIATIVE_BOTH_FLANKS_WEIGHT_EG",
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_EG",
            "INITIATIVE_ALMOST_UNWIN_WEIGHT_EG",
            "INITIATIVE_OFFSET_EG",
        ):
            pct_band(name, 0.10, min_half=1.0)

        # --- Formerly hard-pinned Score knobs (SF free constants; SPSA used to collapse) ---
        # Soft-band around θ₀ instead of pin. Outer sign floors still from maybe_range.
        # TRAPPED_ROOK SF S(52,10) → ours ~[41,6]; allow ±50% (min half 4)
        soft_band_theta0("TRAPPED_ROOK", frac=0.5, min_half=4.0, abs_min=0.0, abs_max=120.0)
        # RestrictedPiece SF S(7,7) → ours ~[4,0]; allow room for small EG
        soft_band_theta0("THREAT_RESTRICTED_PIECE", frac=0.75, min_half=3.0, abs_min=0.0, abs_max=40.0)
        # FlankAttacks SF S(8,0) → ours MG often negative [-6,0]; keep EG ≥ 0, MG ≤ 0
        # Element-wise: MG-like slots can go more negative; EG non-negative.
        if "FLANK_ATTACKS" in names:
            fa_start, fa_count = cm.pm.get_param_indices("FLANK_ATTACKS")
            th0 = cm.pm.theta
            for i in range(fa_count):
                v = float(th0[fa_start + i])
                half = max(abs(v) * 0.75, 3.0)
                lo, hi = v - half, v + half
                # shape (2,) = [MG, EG]: MG typically ≤0 penalty-style, EG ≥0
                if i == 0:
                    lo, hi = max(lo, -20.0), min(hi, 0.0)
                else:
                    lo, hi = max(lo, 0.0), min(hi, 20.0)
                if lo > hi:
                    lo, hi = hi, lo
                cm.add_range_on_absolute_index(fa_start + i, min_val=lo, max_val=hi)

        for out_name in ("OUTPOST_BONUS_KNIGHT", "OUTPOST_BONUS_BISHOP"):
            if out_name not in names:
                continue
            o_start, _ = cm.pm.get_param_indices(out_name)
            for r in (0, 1, 7):  # already zero in classical tables
                cm.add_range_on_absolute_index(o_start + 2 * r, min_val=0, max_val=0)
                cm.add_range_on_absolute_index(o_start + 2 * r + 1, min_val=0, max_val=0)

        for mob in (
            "KNIGHT_MOBILITY_BONUS",
            "BISHOP_MOBILITY_BONUS",
            "ROOK_MOBILITY_BONUS",
            "QUEEN_MOBILITY_BONUS",
        ):
            if mob in names:
                # A constant shift of every mobility bin is indistinguishable
                # from material. Anchor the zero-mobility MG/EG pair at theta0.
                mob_start, mob_count = cm.pm.get_param_indices(mob)
                if mob_count >= 2:
                    cm.add_range_on_absolute_index(
                        mob_start,
                        min_val=float(cm.pm.theta[mob_start]),
                        max_val=float(cm.pm.theta[mob_start]),
                    )
                    cm.add_range_on_absolute_index(
                        mob_start + 1,
                        min_val=float(cm.pm.theta[mob_start + 1]),
                        max_val=float(cm.pm.theta[mob_start + 1]),
                    )
                cm.add_monotonic(mob, axis=0, direction="increasing")

        # Threat tables: no "vs King" term (piece index 5 → flat offset 10,11 for 6x2 layout)
        for threat in ("THREAT_BY_MINOR", "THREAT_BY_ROOK"):
            if threat not in names:
                continue
            t_start, count = cm.pm.get_param_indices(threat)
            # shape (6,2) → indices 10,11 are vs King
            if count >= 12:
                cm.add_range_on_absolute_index(t_start + 10, min_val=0, max_val=0)
                cm.add_range_on_absolute_index(t_start + 11, min_val=0, max_val=0)

        # --- King safety structural pins (SF11-aligned) ---
        # KING_ATTACK_WEIGHTS: [NO_PIECE, PAWN, N, B, R, Q] — first two are 0 in SF
        if "KING_ATTACK_WEIGHTS" in names:
            kaw, _ = cm.pm.get_param_indices("KING_ATTACK_WEIGHTS")
            cm.add_range_on_absolute_index(kaw + 0, min_val=0, max_val=0)
            cm.add_range_on_absolute_index(kaw + 1, min_val=0, max_val=0)
            # N/B/R/Q bands around current scaled SF values (63,40,34,7)
            cm.add_range_on_absolute_index(kaw + 2, min_val=20, max_val=120)  # N
            cm.add_range_on_absolute_index(kaw + 3, min_val=10, max_val=100)  # B
            cm.add_range_on_absolute_index(kaw + 4, min_val=10, max_val=90)   # R
            cm.add_range_on_absolute_index(kaw + 5, min_val=1, max_val=40)    # Q weight small in SF

        # Danger activation threshold: ±5% around θ₀ (gate of whole kingDanger quadratic)
        if "KING_DANGER_THRESHOLD" in names:
            thr_i, _ = cm.pm.get_param_indices("KING_DANGER_THRESHOLD")
            v = float(cm.pm.theta[thr_i])
            half = abs(v) * 0.05
            cm.add_range_on_absolute_index(thr_i, min_val=v - half, max_val=v + half)

        # Shelter / storm tables: last rank column unused (index 7) → 0 like SF padding
        for table in ("SHELTER_STRENGTH", "UNBLOCKED_STORM"):
            if table not in names:
                continue
            t0, count = cm.pm.get_param_indices(table)
            # shape (4, 8) row-major: cols 0..7, pin col 7 for each row
            if count >= 32:
                for row in range(4):
                    cm.add_range_on_absolute_index(t0 + row * 8 + 7, min_val=0, max_val=0)

        # --- PST (piece-square tables) ---
        # Layout: shape (6, 64) piece-major, sq A1=0 .. H8=63 (see constants.py).
        # Pieces: 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K.
        # Free DOF is large (768); constrain before any PST SPSA.
        _PIECE_PAWN, _PIECE_KING = 0, 5
        for table_name, half_width in (("PST_MG", 35.0), ("PST_EG", 40.0)):
            if table_name not in names:
                continue
            base, count = cm.pm.get_param_indices(table_name)
            th0 = cm.pm.theta
            # Hard safety box (same ~cp scale as material P=100)
            maybe_range(table_name, min_val=-100.0, max_val=100.0)
            # Soft band around current θ₀ so SPSA only fine-tunes locally
            for i in range(count):
                v = float(th0[base + i])
                cm.add_range_on_absolute_index(
                    base + i, min_val=v - half_width, max_val=v + half_width
                )
            # Pawn: rank-1 and rank-8 unused (no pawn stays there) — pin 0
            for sq in list(range(0, 8)) + list(range(56, 64)):
                cm.add_range_on_absolute_index(
                    base + _PIECE_PAWN * 64 + sq, min_val=0.0, max_val=0.0
                )
            # King PST: narrow band (not full pin). Applied after the piece-wide soft
            # band so the tighter interval wins. Rationale vs N/B/R/Q (±35/±40):
            #   - MG couples castling / shelter; keep e1≺g1/c1 ordering mostly intact
            #   - EG centralization is cleaner; slightly wider OK
            # Defaults: MG ±10, EG ±15 (table span ~80–90; ~10–15% local step).
            king_half = 10.0 if table_name == "PST_MG" else 15.0
            for sq in range(64):
                idx = base + _PIECE_KING * 64 + sq
                v = float(th0[idx])
                cm.add_range_on_absolute_index(
                    idx, min_val=v - king_half, max_val=v + king_half
                )

            # Material and a constant offset applied to an entire PST piece
            # row encode the same feature. Preserve each theta0 row mean so
            # SPSA learns square-to-square shape while material owns the base.
            for piece in range(6):
                squares = range(8, 56) if piece == _PIECE_PAWN else range(64)
                relative = [piece * 64 + sq for sq in squares]
                target_mean = float(np.mean([th0[base + i] for i in relative]))
                cm.add_fixed_mean(
                    table_name, relative, target_mean, quantized=True
                )

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

    def _build_validation_buckets(self, indices, reference_theta):
        """Build fixed theta0 bucket membership for comparable MSE reports."""
        sample_bbs = np.ascontiguousarray(self.piece_bbs[indices])
        n = len(sample_bbs)
        if n == 0:
            return []

        # Vectorized uint64 popcount via a byte lookup table. This is run once
        # for the bounded metrics sample, outside the SPSA/JIT hot path.
        popcount_lut = np.array(
            [bin(i).count("1") for i in range(256)], dtype=np.uint8
        )
        byte_view = sample_bbs.view(np.uint8).reshape(n, 12, 8)
        piece_counts = np.sum(
            popcount_lut[byte_view], axis=2, dtype=np.int16
        )
        total_pieces = np.sum(piece_counts, axis=1, dtype=np.int16)

        mg_start, _ = self.param_manager.get_param_indices("MG_MATERIAL_VALUES")
        mat_n = int(reference_theta[mg_start + 1])
        mat_b = int(reference_theta[mg_start + 2])
        mat_r = int(reference_theta[mg_start + 3])
        mat_q = int(reference_theta[mg_start + 4])
        npm = (
            (piece_counts[:, 1] + piece_counts[:, 7]) * mat_n
            + (piece_counts[:, 2] + piece_counts[:, 8]) * mat_b
            + (piece_counts[:, 3] + piece_counts[:, 9]) * mat_r
            + (piece_counts[:, 4] + piece_counts[:, 10]) * mat_q
        ).astype(np.int64)
        midgame_limit = 2 * (2 * mat_n + 2 * mat_b + 2 * mat_r + mat_q)
        endgame_limit = (
            midgame_limit * int(PHASE_ENDGAME_RATIO_NUM)
            + int(PHASE_ENDGAME_RATIO_DEN) // 2
        ) // int(PHASE_ENDGAME_RATIO_DEN)
        phase_denominator = midgame_limit - endgame_limit
        if phase_denominator <= 0:
            phase = np.full(n, int(PHASE_MIDGAME), dtype=np.int16)
        else:
            npm_clamped = np.clip(npm, endgame_limit, midgame_limit)
            phase = (
                (npm_clamped - endgame_limit) * int(PHASE_MIDGAME)
                // phase_denominator
            ).astype(np.int16)

        minor_count = (
            piece_counts[:, 1] + piece_counts[:, 2]
            + piece_counts[:, 7] + piece_counts[:, 8]
        )
        rook_count = piece_counts[:, 3] + piece_counts[:, 9]
        queen_count = piece_counts[:, 4] + piece_counts[:, 10]
        non_pawn_count = minor_count + rook_count + queen_count
        return [
            ("phase=0", phase == 0),
            ("phase=1..31", (phase >= 1) & (phase <= 31)),
            ("phase=32..63", (phase >= 32) & (phase <= 63)),
            ("phase=64..95", (phase >= 64) & (phase <= 95)),
            ("phase=96..128", (phase >= 96) & (phase <= 128)),
            ("broad-EG phase<=64", phase <= 64),
            ("pieces=8..12", (total_pieces >= 8) & (total_pieces <= 12)),
            ("pieces=13..20", (total_pieces >= 13) & (total_pieces <= 20)),
            ("queenless", queen_count == 0),
            ("pure king+pawns", non_pawn_count == 0),
            (
                "minor-only EG",
                (phase <= 64) & (minor_count > 0)
                & (rook_count == 0) & (queen_count == 0),
            ),
            (
                "rook EG queenless",
                (phase <= 64) & (rook_count > 0) & (queen_count == 0),
            ),
        ]

    def _report_validation_buckets(self, theta, indices, buckets):
        if indices is None or not buckets:
            return
        errors = compute_squared_errors(
            self.piece_bbs[indices],
            self.occupancy_bbs[indices],
            self.game_states[indices],
            self.results[indices],
            theta,
        )
        print("  Fixed-theta0 validation buckets (MSE, n):")
        for name, mask in buckets:
            count = int(np.count_nonzero(mask))
            mse = float(np.mean(errors[mask])) if count else float("nan")
            mse_text = f"{mse:.7f}" if count else "n/a"
            print(f"    {name:20s} {mse_text:>10s}  n={count:,}")

    def _report_quantized_changes(self, theta, reference_theta, active_mask):
        """Report engine-effective integer deltas, not hidden float movement."""
        current = np.rint(theta).astype(np.int64)
        reference = np.rint(reference_theta).astype(np.int64)
        delta = current - reference
        print("  Effective integer parameter changes from run baseline:")
        any_change = False
        for param in self.param_manager.param_map:
            start = param["start"]
            end = start + param["count"]
            if not np.any(active_mask[start:end] != 0):
                continue
            local = delta[start:end]
            changed = int(np.count_nonzero(local))
            if changed == 0:
                continue
            any_change = True
            if param["count"] == 1:
                print(
                    f"    {param['name']}: {reference[start]} -> {current[start]}"
                )
            else:
                print(
                    f"    {param['name']}: changed={changed}/{param['count']} "
                    f"sum_delta={int(np.sum(local)):+d} "
                    f"L1={int(np.sum(np.abs(local)))}"
                )
        if not any_change:
            print("    (none; all active values still round to baseline integers)")
    
    @staticmethod
    def _parse_pst_pieces(spec) -> list:
        """Parse piece filter: 'NBRQ', 'P', '1,2,3,4', or list of letters/indices."""
        if spec is None:
            return []
        if isinstance(spec, (list, tuple)):
            raw = "".join(str(x) for x in spec)
        else:
            raw = str(spec)
        raw = raw.replace(",", " ").replace(" ", "").upper()
        letter_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
        pieces = []
        if raw.isdigit():
            pieces = [int(ch) for ch in raw]
        else:
            for ch in raw:
                if ch in letter_map:
                    pieces.append(letter_map[ch])
                elif ch.isdigit():
                    pieces.append(int(ch))
        # unique preserve order
        seen = set()
        out = []
        for p in pieces:
            if 0 <= p <= 5 and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _create_mask(self, include_list=None, exclude_list=None,
                     king_pst_only=False, non_king_pst_only=False, pst_pieces=None):
        """
        Creates a mask vector where 1 indicates the parameter should be tuned,
        and 0 indicates it should be frozen.

        king_pst_only: only PST king rows (piece 5).
        non_king_pst_only: only PST pieces P/N/B/R/Q.
        pst_pieces: explicit piece indices/letters (e.g. NBRQ or P); overrides
        the two flags when non-empty. If material arrays are also included,
        only matching material entries are enabled for coupled tuning.
        """
        if king_pst_only and non_king_pst_only:
            raise ValueError("Cannot set both king_pst_only and non_king_pst_only")

        piece_filter = self._parse_pst_pieces(pst_pieces)
        if king_pst_only and not piece_filter:
            piece_filter = [5]
        elif non_king_pst_only and not piece_filter:
            piece_filter = [0, 1, 2, 3, 4]

        mask = np.zeros_like(self.best_theta)
        
        # Map parameter names to indices
        name_to_indices = {}
        for p in self.param_manager.param_map:
            name = p['name']
            start = p['start']
            count = p['count']
            name_to_indices[name] = (start, count)

        def _enable_non_pst_from_include() -> None:
            if not include_list:
                return
            for name in include_list:
                if name in ("PST_MG", "PST_EG"):
                    continue
                if name in name_to_indices:
                    start, count = name_to_indices[name]
                    if name in ("MG_MATERIAL_VALUES", "EG_MATERIAL_VALUES"):
                        # Couple only the matching material entries rather than
                        # opening the whole material array in PST-piece mode.
                        for piece in piece_filter:
                            if piece < count:
                                mask[start + piece] = 1.0
                    else:
                        mask[start : start + count] = 1.0
                else:
                    print(f"Warning: Unknown parameter '{name}' in include list.")

        def _enable_pst_rows(pieces: list) -> None:
            names = ["P", "N", "B", "R", "Q", "K"]
            label = ",".join(names[i] for i in pieces)
            print(f"PST piece filter: [{label}] rows active on PST_MG/EG")
            for table_name in ("PST_MG", "PST_EG"):
                if table_name not in name_to_indices:
                    print(f"Warning: {table_name} not registered.")
                    continue
                start, count = name_to_indices[table_name]
                for pi in pieces:
                    if count < (pi + 1) * 64:
                        print(f"Warning: {table_name} too small for piece {pi}")
                        continue
                    mask[start + pi * 64 : start + (pi + 1) * 64] = 1.0

        if piece_filter:
            extra = [n for n in (include_list or []) if n not in ("PST_MG", "PST_EG")]
            print(f"PST-piece mode active"
                  + (f"; also: {extra}" if extra else ""))
            _enable_pst_rows(piece_filter)
            _enable_non_pst_from_include()
        elif include_list:
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

        # Imbalance matrices are lower-triangular. Upper-triangle entries are
        # never read, THEIRS diagonal terms cancel in the white-minus-black
        # polynomial, and the two remaining zero OURS cells are deliberately
        # kept as the SF/stability gauge. Exclude every theta0 zero from SPSA
        # perturbations as well as pinning it in _setup_constraints(), so the
        # reported active count reflects real degrees of freedom.
        for table_name in (
            "IMBALANCE_QUADRATIC_OURS",
            "IMBALANCE_QUADRATIC_THEIRS",
        ):
            if table_name not in name_to_indices:
                continue
            start, count = name_to_indices[table_name]
            local = slice(start, start + count)
            if not np.any(mask[local] > 0):
                continue
            zero_cells = np.asarray(self.best_theta[local]) == 0.0
            mask[local][zero_cells] = 0.0
            print(
                f"{table_name}: froze {int(np.count_nonzero(zero_cells))} "
                "theta0-zero structural/stability cells"
            )

        # No-piece and pawn king-attack weights are structural zeros in the
        # SF king-danger model. They are range-pinned above; remove them from
        # SPSA perturbations too so only N/B/R/Q count as active parameters.
        if "KING_ATTACK_WEIGHTS" in name_to_indices:
            start, count = name_to_indices["KING_ATTACK_WEIGHTS"]
            if count >= 2 and np.any(mask[start:start + count] > 0):
                mask[start + 0] = 0.0
                mask[start + 1] = 0.0
                print(
                    "KING_ATTACK_WEIGHTS: keep NO_PIECE/PAWN frozen "
                    "(indices 0,1)"
                )

        # Material scale: MG Pawn is the 100cp anchor; both Kings are zero.
        # EG Pawn remains tunable inside its theta0 ±10% constraint.
        if "MG_MATERIAL_VALUES" in name_to_indices:
            start, count = name_to_indices["MG_MATERIAL_VALUES"]
            if count >= 6 and np.any(mask[start : start + count] > 0):
                mask[start + 0] = 0.0
                mask[start + 5] = 0.0
                print("MG_MATERIAL_VALUES: keep Pawn/King frozen (indices 0,5)")
        if "EG_MATERIAL_VALUES" in name_to_indices:
            start, count = name_to_indices["EG_MATERIAL_VALUES"]
            if count >= 6 and np.any(mask[start : start + count] > 0):
                mask[start + 5] = 0.0
                print("EG_MATERIAL_VALUES: keep King frozen (index 5)")
        
        active_params = np.sum(mask)
        total_params = len(mask)
        print(f"Active parameters: {int(active_params)} / {total_params}")
        
        return mask

    def optimize(self, iterations=1000, alpha=0.1, gamma=0.101, c=2.0, A=100,
                 save_path="tuner/tuned_constants.py", include_list=None, exclude_list=None,
                 batch_size=None, eval_sample=1_000_000, report_every=50, patience=30,
                 metrics_sample=500_000, seed=0, king_pst_only=False,
                 non_king_pst_only=False, pst_pieces=None):
        """
        Standard SPSA with constraints, parameter mask, mini-batch gradients.

        eval_sample: size of a **held-out validation set** used only for MSE
        reporting / best tracking (default 1M). Those indices are never used
        in gradient mini-batches. Set <= 0 to disable hold-out (train+eval = full data).

        patience: stop after this many consecutive *report* steps without a new best
        (default 30). Set to 0 to disable early stopping.

        metrics_sample: fixed subset of the hold-out used for phase/material bucket
        MSE reporting (default 500k). Membership is computed once from theta0 so
        different candidates are directly comparable. Set <= 0 to disable.

        seed: SPSA perturbation / training mini-batch seed. Validation and bucket
        membership stay fixed across seeds for directly comparable runs.

        king_pst_only / non_king_pst_only / pst_pieces: restrict which PST rows move.
        """
        theta = self.best_theta.copy()
        run_baseline_theta = theta.copy()
        mask = self._create_mask(
            include_list, exclude_list,
            king_pst_only=king_pst_only,
            non_king_pst_only=non_king_pst_only,
            pst_pieces=pst_pieces,
        )

        split_rng = np.random.default_rng(0)
        train_rng = np.random.default_rng(seed)
        n = self.n_samples

        # Held-out validation split (never used for gradient steps)
        if eval_sample is None or eval_sample <= 0 or eval_sample >= n:
            eval_indices = None
            train_indices = np.arange(n, dtype=np.int64)
            print(f"No hold-out: train+eval on full dataset ({n:,})")
        else:
            n_val = int(eval_sample)
            eval_indices = np.sort(split_rng.choice(n, size=n_val, replace=False))
            is_val = np.zeros(n, dtype=bool)
            is_val[eval_indices] = True
            train_indices = np.flatnonzero(~is_val).astype(np.int64)
            print(
                f"Hold-out val: {len(eval_indices):,}  |  train pool: {len(train_indices):,}  "
                f"(val never enters mini-batches)"
            )

        n_train = len(train_indices)

        metric_indices = None
        validation_buckets = []
        if metrics_sample is not None and metrics_sample > 0:
            metric_pool = eval_indices if eval_indices is not None else train_indices
            metric_count = min(int(metrics_sample), len(metric_pool))
            metrics_rng = np.random.default_rng(1)
            if metric_count < len(metric_pool):
                metric_indices = np.sort(
                    metrics_rng.choice(metric_pool, size=metric_count, replace=False)
                )
            else:
                metric_indices = np.asarray(metric_pool, dtype=np.int64).copy()
            validation_buckets = self._build_validation_buckets(
                metric_indices, self.best_theta
            )
            print(
                f"Bucket metrics: {len(metric_indices):,} fixed validation positions "
                "(membership from initial theta0)"
            )

        # Initial best on validation (or full set if no hold-out)
        self.best_loss = self._cost(self.best_theta, indices=eval_indices)
        print(f"Initial val MSE: {self.best_loss:.6f}")
        self._report_validation_buckets(
            self.best_theta, metric_indices, validation_buckets
        )

        print(f"Starting SPSA optimization for up to {iterations} iterations...")
        if batch_size:
            print(f"Using mini-batch size: {batch_size} (from train pool only)")
        if patience and patience > 0:
            print(
                f"Early stop: patience={patience} reports without improvement "
                f"(report every {report_every} iters)"
            )
        else:
            print("Early stop: disabled (patience=0)")

        start_time = time.time()
        stalled_reports = 0
        stopped_early = False

        for k in range(1, iterations + 1):
            # Mini-batch strictly from train_indices
            if batch_size is not None and batch_size < n_train:
                pick = train_rng.choice(n_train, size=int(batch_size), replace=False)
                batch_indices = train_indices[pick]
            else:
                batch_indices = train_indices

            ak = alpha / ((A + k) ** 0.602)
            ck = c / (k ** gamma)

            raw_delta = train_rng.integers(0, 2, size=theta.shape) * 2 - 1
            delta = raw_delta * mask

            theta_plus = theta + ck * delta
            self.constraint_manager.apply(theta_plus, active_mask=mask)

            theta_minus = theta - ck * delta
            self.constraint_manager.apply(theta_minus, active_mask=mask)

            loss_plus = self._cost(theta_plus, indices=batch_indices)
            loss_minus = self._cost(theta_minus, indices=batch_indices)

            ghat = (loss_plus - loss_minus) / (2 * ck) * delta
            theta = theta - ak * ghat
            self.constraint_manager.apply(theta, active_mask=mask)

            if k % report_every == 0:
                current_loss = self._cost(theta, indices=eval_indices)
                self._report_validation_buckets(
                    theta, metric_indices, validation_buckets
                )
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_theta = theta.copy()
                    stalled_reports = 0
                    print(f"Iter {k}: New Best val MSE = {self.best_loss:.7f}")
                    self._report_quantized_changes(
                        self.best_theta, run_baseline_theta, mask
                    )
                    self.save_results(save_path)
                else:
                    stalled_reports += 1
                    print(
                        f"Iter {k}: val MSE = {current_loss:.7f} "
                        f"(Best: {self.best_loss:.7f}, stalled {stalled_reports}/{patience or '∞'})"
                    )
                    if patience and patience > 0 and stalled_reports >= patience:
                        print(
                            f"Early stopping at iter {k}: "
                            f"no improvement for {patience} consecutive reports."
                        )
                        stopped_early = True
                        break

        total_time = time.time() - start_time
        print(f"Optimization finished in {total_time:.2f}s"
              f"{' (early stop)' if stopped_early else ''}.")
        print(f"Final Best val MSE: {self.best_loss:.7f}")
        print("Final best bucket report:")
        self._report_validation_buckets(
            self.best_theta, metric_indices, validation_buckets
        )
        self._report_quantized_changes(
            self.best_theta, run_baseline_theta, mask
        )
        self.save_results(save_path)

    def save_results(self, path):
        content = self.param_manager.update_constants_string(self.best_theta)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=20000, help="Number of SPSA iterations")
    parser.add_argument("--alpha", type=float, default=30000.0, help="Learning rate scaling (a)")
    parser.add_argument(
        "--c", type=float, default=3.0,
        help="Perturbation scaling (c); integer HCE parameters normally need c >= 1",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="SPSA perturbation/mini-batch seed (validation split remains fixed)",
    )
    
    # New arguments
    parser.add_argument("--tune", nargs='+', help="List of parameter names to tune (others will be frozen)")
    parser.add_argument("--exclude", nargs='+', help="List of parameter names to exclude/freeze")
    parser.add_argument("--batch-size", type=int, default=65536, help="Mini-batch size for gradient estimation")
    parser.add_argument(
        "--eval-sample",
        type=int,
        default=1_000_000,
        help="Held-out validation size for MSE/best (never used in train batches; 0 = no hold-out)",
    )
    parser.add_argument("--report-every", type=int, default=50, help="Log MSE every N iterations")
    parser.add_argument(
        "--metrics-sample",
        type=int,
        default=500_000,
        help="Fixed validation subset for phase/endgame bucket MSE (0 = disable)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stop after N consecutive reports without a new best (0 = never stop early)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tuner/data/hce/dataset.npz",
        help="Path to the preprocessed dataset .npz file",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Print registered parameter names and exit",
    )
    parser.add_argument(
        "--king-pst-only",
        action="store_true",
        help="Only tune PST_MG/PST_EG king rows (128 params; narrow-band constraints apply)",
    )
    parser.add_argument(
        "--non-king-pst-only",
        action="store_true",
        help="Only tune PST_MG/PST_EG for P/N/B/R/Q (640 params); king row frozen",
    )
    parser.add_argument(
        "--pst-pieces",
        type=str,
        default=None,
        help="Restrict PST rows to these pieces, e.g. NBRQ or P (letters P,N,B,R,Q,K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuner/tuned_constants.py",
        help="Path for best/final tuned constants dump",
    )
    args = parser.parse_args()

    pm = ParameterManager()
    if args.list_params:
        for name in pm.list_param_names():
            start, count = pm.get_param_indices(name)
            print(f"{name:40s} start={start:5d} count={count}")
        raise SystemExit(0)

    if not os.path.exists(args.dataset):
        print(
            f"Dataset not found at '{args.dataset}'. "
            "Generate labels with tuner/data_pipeline then preprocess_data.py."
        )
    else:
        optimizer = SPSAOptimizer(args.dataset, pm)
        optimizer.optimize(
            iterations=args.iter,
            alpha=args.alpha,
            c=args.c,
            include_list=args.tune,
            exclude_list=args.exclude,
            batch_size=args.batch_size,
            eval_sample=args.eval_sample,
            metrics_sample=args.metrics_sample,
            seed=args.seed,
            report_every=args.report_every,
            patience=args.patience,
            king_pst_only=args.king_pst_only,
            non_king_pst_only=args.non_king_pst_only,
            pst_pieces=args.pst_pieces,
            save_path=args.output,
        )
