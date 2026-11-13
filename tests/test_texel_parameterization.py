"""Lightweight invariants for the HCE Texel/SPSA parameterization."""
from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chess_engine.classical.constants import (
    BISHOP,
    ENDGAME_LIMIT,
    KNIGHT,
    MG_MATERIAL_VALUES,
    MIDGAME_LIMIT,
    PHASE_ENDGAME_RATIO_DEN,
    PHASE_ENDGAME_RATIO_NUM,
    QUEEN,
    ROOK,
)
from tuner.apply_tuned import _eval_const_module, _extract_assignments, apply_tuned_source
from tuner.constraint_manager import ConstraintManager
from tuner.data_pipeline.annotate_sf_labels import (
    UNANNOTATED_CP,
    arrays_to_fen,
    parse_sf_search_output,
)
from tuner.data_pipeline.preprocess_data import parse_fen_to_arrays
from tuner.notebook_session import (
    CONSTANTS_PATH,
    TexelTuningSession,
    _parse_proc_meminfo_gb,
    build_blended_labels,
)
from tuner.parameters import ParameterManager
from tuner.tuner import SPSAOptimizer, split_dataset_indices


class _DummyParameterManager:
    def __init__(self):
        self.param_map = [
            {"name": "TABLE", "start": 0, "count": 4, "shape": (4,)}
        ]


class TestPhaseMaterialCoupling(unittest.TestCase):
    def test_limits_follow_current_mg_material(self):
        expected_midgame = 2 * (
            2 * int(MG_MATERIAL_VALUES[KNIGHT])
            + 2 * int(MG_MATERIAL_VALUES[BISHOP])
            + 2 * int(MG_MATERIAL_VALUES[ROOK])
            + int(MG_MATERIAL_VALUES[QUEEN])
        )
        expected_endgame = (
            expected_midgame * int(PHASE_ENDGAME_RATIO_NUM)
            + int(PHASE_ENDGAME_RATIO_DEN) // 2
        ) // int(PHASE_ENDGAME_RATIO_DEN)

        self.assertEqual(int(MIDGAME_LIMIT), expected_midgame)
        self.assertEqual(int(ENDGAME_LIMIT), expected_endgame)
        self.assertGreater(int(MIDGAME_LIMIT), int(ENDGAME_LIMIT))


class TestFixedMeanConstraint(unittest.TestCase):
    def test_projection_preserves_mean_and_range(self):
        manager = ConstraintManager(_DummyParameterManager())
        manager.add_range("TABLE", min_val=0.0, max_val=30.0)
        manager.add_fixed_mean("TABLE", range(4), target_mean=15.0)
        theta = np.array([50.0, -10.0, 25.0, 30.0], dtype=np.float64)

        manager.apply(theta)

        self.assertAlmostEqual(float(np.mean(theta)), 15.0, places=9)
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 30.0))

    def test_projection_does_not_move_frozen_entries(self):
        manager = ConstraintManager(_DummyParameterManager())
        manager.add_range("TABLE", min_val=0.0, max_val=30.0)
        manager.add_fixed_mean("TABLE", range(4), target_mean=15.0)
        theta = np.array([0.0, 20.0, 20.0, 30.0], dtype=np.float64)
        active_mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

        manager.apply(theta, active_mask=active_mask)

        self.assertAlmostEqual(float(np.mean(theta)), 15.0, places=9)
        self.assertEqual(theta[0], 0.0)
        self.assertEqual(theta[3], 30.0)

    def test_quantized_projection_preserves_float_and_integer_sum(self):
        manager = ConstraintManager(_DummyParameterManager())
        manager.add_fixed_mean(
            "TABLE", range(4), target_mean=0.0, quantized=True
        )
        # Continuous sum is already zero, but banker's rounding sums to -1.
        theta = np.array([0.49, 0.49, 0.49, -1.47], dtype=np.float64)

        manager.apply(theta)

        self.assertAlmostEqual(float(np.sum(theta)), 0.0, places=9)
        self.assertEqual(int(np.sum(np.rint(theta))), 0)
        self.assertTrue(np.any(np.abs(theta - np.rint(theta)) > 1e-6))


class TestIntegerAwareTuningBands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = ParameterManager()
        optimizer = SPSAOptimizer.__new__(SPSAOptimizer)
        optimizer.param_manager = cls.pm
        optimizer.constraint_manager = ConstraintManager(cls.pm)
        optimizer._setup_constraints()
        optimizer.best_theta = cls.pm.get_initial_theta()
        cls.optimizer = optimizer
        cls.cm = optimizer.constraint_manager

    def test_small_initiative_weight_can_reach_adjacent_integers(self):
        name = "INITIATIVE_OUTFLANKING_WEIGHT_EG"
        start, _ = self.pm.get_param_indices(name)
        value = float(self.pm.theta[start])
        lower, upper = self.cm._absolute_bounds(name, np.array([start]))

        self.assertLessEqual(lower[0], value - 1.0)
        self.assertGreaterEqual(upper[0], value + 1.0)

    def test_imbalance_structural_zeros_stay_pinned(self):
        name = "IMBALANCE_QUADRATIC_OURS"
        start, count = self.pm.get_param_indices(name)
        values = np.asarray(self.pm.theta[start:start + count], dtype=np.float64)
        zero_rel = int(np.flatnonzero(values == 0.0)[0])
        nonzero_rel = int(np.flatnonzero((values != 0.0) & (np.abs(values) < 10.0))[0])
        indices = np.array([start + zero_rel, start + nonzero_rel])
        lower, upper = self.cm._absolute_bounds(name, indices)

        self.assertEqual((lower[0], upper[0]), (0.0, 0.0))
        self.assertLessEqual(lower[1], values[nonzero_rel] - 1.0)
        self.assertGreaterEqual(upper[1], values[nonzero_rel] + 1.0)

    def test_imbalance_zero_cells_are_excluded_from_spsa_mask(self):
        names = ["IMBALANCE_QUADRATIC_OURS", "IMBALANCE_QUADRATIC_THEIRS"]
        mask = self.optimizer._create_mask(include_list=names)

        for name in names:
            start, count = self.pm.get_param_indices(name)
            values = np.asarray(
                self.pm.theta[start:start + count], dtype=np.float64
            )
            local_mask = mask[start:start + count]
            self.assertTrue(np.all(local_mask[values == 0.0] == 0.0))
            self.assertTrue(np.all(local_mask[values != 0.0] == 1.0))

    def test_king_attack_structural_weights_are_excluded_from_mask(self):
        name = "KING_ATTACK_WEIGHTS"
        mask = self.optimizer._create_mask(include_list=[name])
        start, count = self.pm.get_param_indices(name)

        self.assertEqual(count, 6)
        np.testing.assert_array_equal(
            mask[start:start + count],
            np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        )

    def test_dead_initiative_pawn_endgame_weights_are_pinned_and_frozen(self):
        dead = [
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_MG",
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_EG",
        ]
        mask = self.optimizer._create_mask(include_list=dead)
        for name in dead:
            start, _ = self.pm.get_param_indices(name)
            v0 = float(self.pm.theta[start])
            lower, upper = self.cm._absolute_bounds(name, np.array([start]))
            self.assertEqual((lower[0], upper[0]), (v0, v0))
            self.assertEqual(mask[start], 0.0)

    def test_param_scales_respect_unit_classes_and_tight_boxes(self):
        theta0 = self.pm.get_initial_theta()
        mask = self.optimizer._create_mask(
            include_list=[
                "MG_MATERIAL_VALUES",
                "PST_MG",
                "SAFE_CHECK_ROOK",
                "PASSED_PAWN_BONUS",
                "PASSED_PATH_SAFE_NONE_ATTACK",
                "IMBALANCE_QUADRATIC_OURS",
            ]
        )
        scales = self.optimizer._build_param_scales(theta0, mask, c=2.5)

        mg_s, mg_c = self.pm.get_param_indices("MG_MATERIAL_VALUES")
        pst_s, pst_c = self.pm.get_param_indices("PST_MG")
        np.testing.assert_allclose(scales[mg_s : mg_s + mg_c], 1.0)
        np.testing.assert_allclose(scales[pst_s : pst_s + pst_c], 1.0)

        scr_s, _ = self.pm.get_param_indices("SAFE_CHECK_ROOK")
        self.assertGreaterEqual(scales[scr_s], 6.0)

        pp_s, _ = self.pm.get_param_indices("PASSED_PAWN_BONUS")
        self.assertEqual(scales[pp_s + 2], 1.0)      # small low-rank entry
        self.assertGreater(scales[pp_s + 12], 3.0)   # rank-6 132cp entry

        path_s, _ = self.pm.get_param_indices("PASSED_PATH_SAFE_NONE_ATTACK")
        self.assertEqual(scales[path_s], 1.0)        # ±1 box stays 1.0

    def test_king_safety_output_and_flank_coeffs_are_registered_and_scaled(self):
        new_ks_names = [
            "KING_DANGER_OUT_MG_NUM",
            "KING_DANGER_OUT_EG_NUM",
            "EG_SAFETY_SCALE",
            "KING_FLANK_ATTACK_NUM",
            "KING_FLANK_DEFENSE_MULT",
            "KING_SHELTER_FEEDBACK_NUM",
        ]
        registered = set(self.pm.list_param_names())
        for name in new_ks_names:
            self.assertIn(name, registered)
            start, count = self.pm.get_param_indices(name)
            self.assertEqual(count, 1)
            lower, upper = self.cm._absolute_bounds(name, np.array([start]))
            self.assertGreater(upper[0], lower[0], msg=f"{name} should not be pinned")

        theta0 = self.pm.get_initial_theta()
        mask = self.optimizer._create_mask(include_list=new_ks_names)
        scales = self.optimizer._build_param_scales(theta0, mask, c=2.0)

        eg_out_s, _ = self.pm.get_param_indices("KING_DANGER_OUT_EG_NUM")
        mg_out_s, _ = self.pm.get_param_indices("KING_DANGER_OUT_MG_NUM")
        eg_safe_s, _ = self.pm.get_param_indices("EG_SAFETY_SCALE")
        self.assertGreaterEqual(scales[eg_out_s], 100.0)
        self.assertGreater(scales[mg_out_s], 1.5)
        self.assertGreater(scales[eg_safe_s], 1.5)

    def test_discrete_factor_and_divisor_knobs_are_unlocked(self):
        for name in (
            "KING_PAWN_PROXIMITY_FACTOR",
            "BISHOP_PAWNS_CENTER_BLOCKED_FACTOR",
            "IMBALANCE_DIVISOR",
        ):
            start, count = self.pm.get_param_indices(name)
            self.assertEqual(count, 1)
            lower, upper = self.cm._absolute_bounds(name, np.array([start]))
            self.assertGreaterEqual(
                upper[0] - lower[0],
                2.0,
                msg=f"{name} should allow >= 3 discrete integers",
            )

    def test_notebook_covers_all_active_registered_parameters(self):
        nb_path = Path(__file__).resolve().parents[1] / "tuner" / "texel_tuner_runner.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        for idx, src in enumerate(code_cells):
            cleaned = "\n".join(
                line for line in src.splitlines()
                if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(cleaned, filename=f"{nb_path.name}#code_cell_{idx}")
        code_text = "\n".join(code_cells)
        tokens = set(re.findall(r'["\']([A-Z][A-Z0-9_]*)(?:=[^"\']*)?["\']', code_text))
        intentionally_pinned_or_dead = {
            "CANDIDATE_PASSER_DIVISOR",
            "UNSTOPPABLE_PAWN_BONUS",
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_MG",
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_EG",
        }
        missing = sorted(
            set(self.pm.list_param_names()) - tokens - intentionally_pinned_or_dead
        )
        self.assertEqual(missing, [], msg=f"Active params missing from notebook: {missing}")


class TestSplitDatasetIndices(unittest.TestCase):
    def test_three_way_split_is_disjoint_and_exhaustive(self):
        train_idx, eval_idx, hold_idx = split_dataset_indices(
            20_000, eval_sample=1_000, holdout_sample=2_000
        )
        self.assertIsNotNone(eval_idx)
        self.assertIsNotNone(hold_idx)
        self.assertEqual(len(train_idx), 17_000)
        self.assertEqual(len(eval_idx), 1_000)
        self.assertEqual(len(hold_idx), 2_000)

        s_train = set(train_idx.tolist())
        s_eval = set(eval_idx.tolist())
        s_hold = set(hold_idx.tolist())
        self.assertEqual(len(s_train & s_eval), 0)
        self.assertEqual(len(s_train & s_hold), 0)
        self.assertEqual(len(s_eval & s_hold), 0)
        self.assertEqual(len(s_train | s_eval | s_hold), 20_000)

    def test_small_smoke_dataset_disables_holdout_gracefully(self):
        train_idx, eval_idx, hold_idx = split_dataset_indices(
            500, eval_sample=1_000, holdout_sample=2_000
        )
        self.assertIsNone(eval_idx)
        self.assertIsNone(hold_idx)
        self.assertEqual(len(train_idx), 500)


class TestApplyTunedPreservesComments(unittest.TestCase):
    def test_ast_replacement_preserves_inline_and_following_comments(self):
        target_src = (
            "import numpy as np\n\n"
            "INITIATIVE_MG_OFFSET = np.int32(39)  # keep inline comment\n\n"
            "KING_DANGER_NO_QUEEN = 560\n"
            "# Important explanation block line 1\n"
            "# Important explanation block line 2\n"
            "KING_DANGER_NO_QUEEN_ROOKLESS = 810\n"
            "PASSED_FILE_BONUS = np.array([16, 4], dtype=np.int32)\n"
        )
        tuned_src = (
            "import numpy as np\n"
            "INITIATIVE_MG_OFFSET = 42\n"
            "KING_DANGER_NO_QUEEN = 575\n"
            "KING_DANGER_NO_QUEEN_ROOKLESS = 810\n"
            "PASSED_FILE_BONUS = np.array([14, 5], dtype=np.int32)\n"
        )
        target_ns = {"np": np}
        exec(target_src, target_ns)
        tuned_ns = {"np": np}
        exec(tuned_src, tuned_ns)
        tuned_assigns = _extract_assignments(tuned_src)

        new_src, changed, skipped = apply_tuned_source(
            target_src=target_src,
            tuned_ns=tuned_ns,
            target_ns=target_ns,
            tuned_assigns=tuned_assigns,
            names=list(tuned_assigns.keys()),
            verbose=False,
        )
        self.assertEqual(
            changed,
            ["INITIATIVE_MG_OFFSET", "KING_DANGER_NO_QUEEN", "PASSED_FILE_BONUS"],
        )
        self.assertEqual(skipped, 1)
        self.assertIn(
            "INITIATIVE_MG_OFFSET = np.int32(42)  # keep inline comment\n\n",
            new_src,
        )
        self.assertIn(
            "KING_DANGER_NO_QUEEN = 575\n"
            "# Important explanation block line 1\n"
            "# Important explanation block line 2\n"
            "KING_DANGER_NO_QUEEN_ROOKLESS = 810\n",
            new_src,
        )
        self.assertIn(
            "PASSED_FILE_BONUS = np.array([14, 5], dtype=np.int32)\n",
            new_src,
        )


class TestPreprocessCastlingAndEp(unittest.TestCase):
    def test_castling_rights_and_legal_ep_encoding(self):
        # Pseudo-EP (no adjacent black pawn on d4/f4) -> ep stored as 64
        _, _, gs_pseudo = parse_fen_to_arrays(
            "r3k2r/pppppppp/8/8/4P3/8/PPPP1PPP/R3K2R b KQkq e3 0 7"
        )
        self.assertEqual(int(gs_pseudo[0]), 1)
        self.assertEqual(int(gs_pseudo[1]), 15)  # K=1 | Q=2 | k=4 | q=8
        self.assertEqual(int(gs_pseudo[2]), 64)
        self.assertEqual(int(gs_pseudo[3]), 0)

        # Legal EP (black pawn on d4 can capture on e3 = sq 20)
        _, _, gs_legal = parse_fen_to_arrays(
            "r3k2r/ppp1pppp/8/8/3pP3/8/PPPP1PPP/R3K2R b KQk e3 0 7"
        )
        self.assertEqual(int(gs_legal[1]), 7)  # K=1 | Q=2 | k=4
        self.assertEqual(int(gs_legal[2]), 20)


class TestCustomConstantsAndSessionModes(unittest.TestCase):
    def test_parameter_manager_reads_custom_constants_path_without_side_effects(self):
        default_pm = ParameterManager()
        idx, _ = default_pm.get_param_indices("KING_DANGER_NO_QUEEN")
        orig_val = int(default_pm.theta[idx])

        with tempfile.TemporaryDirectory() as tmp:
            custom_path = Path(tmp) / "custom_constants.py"
            src = CONSTANTS_PATH.read_text(encoding="utf-8")
            src = src.replace(
                f"KING_DANGER_NO_QUEEN = {orig_val}",
                f"KING_DANGER_NO_QUEEN = {orig_val + 17}",
                1,
            )
            custom_path.write_text(src, encoding="utf-8", newline="\n")

            custom_pm = ParameterManager(constants_source=custom_path)
            self.assertEqual(int(custom_pm.theta[idx]), orig_val + 17)
            self.assertEqual(int(ParameterManager().theta[idx]), orig_val)

    def test_session_pgn_toggle_target_engine_reconfigure_and_accept_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_npz = tmp_dir / "train_mini.npz"
            pgn_npz = tmp_dir / "val_pgn_mini.npz"
            classical_copy = tmp_dir / "classical_constants.py"
            old_copy = tmp_dir / "classical_old_constants.py"
            shutil.copyfile(CONSTANTS_PATH, classical_copy)
            shutil.copyfile(CONSTANTS_PATH, old_copy)

            pb0, ob0, gs0 = parse_fen_to_arrays(
                "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8"
            )
            pb1, ob1, gs1 = parse_fen_to_arrays(
                "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 40"
            )
            n_train = 60
            piece_bbs = np.tile(np.stack([pb0, pb1]), (n_train // 2, 1))
            occ_bbs = np.tile(np.stack([ob0, ob1]), (n_train // 2, 1))
            game_states = np.tile(np.stack([gs0, gs1]), (n_train // 2, 1))
            results_train = np.full(n_train, 0.80, dtype=np.float32)
            np.savez(
                train_npz,
                piece_bbs=piece_bbs,
                occupancy_bbs=occ_bbs,
                game_states=game_states,
                results=results_train,
            )

            # Give val_pgn opposite WDL (0.10) so a score increase improves holdout
            # (PASS under use_pgn_val=False) but worsens val_pgn (REJECT under True).
            n_pgn = 40
            np.savez(
                pgn_npz,
                piece_bbs=piece_bbs[:n_pgn],
                occupancy_bbs=occ_bbs[:n_pgn],
                game_states=game_states[:n_pgn],
                results=np.full(n_pgn, 0.10, dtype=np.float32),
            )

            init_idx, _ = ParameterManager().get_param_indices("INITIATIVE_MG_OFFSET")

            def _fake_eval(pb, ob, gs, theta, lazy=False):
                return np.int32(int(np.rint(theta[init_idx])))

            def _fake_scores(pbs, obs, gss, theta):
                jitter = (np.arange(len(pbs), dtype=np.int32) % 7) - 3
                return np.int32(int(np.rint(theta[init_idx]))) + jitter

            with (
                patch("tuner.notebook_session.evaluate_position_tunable", side_effect=_fake_eval),
                patch("tuner.notebook_session.compute_white_scores", side_effect=_fake_scores),
                patch("tuner.validate_tuned.compute_white_scores", side_effect=_fake_scores),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                with self.assertRaises(ValueError):
                    TexelTuningSession(
                        train_npz=train_npz,
                        pgn_val_npz=pgn_npz,
                        eval_sample=10,
                        holdout_sample=20,
                        use_pgn_val=False,
                        target_engine="invalid_engine",
                        target_constants_path=old_copy,
                        runs_dir=tmp_dir / "runs_err",
                    )

                # Mode A: holdout-only targeting classical_old copy
                session = TexelTuningSession(
                    train_npz=train_npz,
                    pgn_val_npz=pgn_npz,
                    eval_sample=10,
                    holdout_sample=20,
                    use_pgn_val=False,
                    target_engine="classical_old",
                    target_constants_path=old_copy,
                    runs_dir=tmp_dir / "runs_old",
                )
                self.assertFalse(session.use_pgn_val)
                self.assertIsNone(session.pgn_data)
                self.assertEqual(session.target_engine, "classical_old")

                base_val = int(session.step_baseline_theta[init_idx])
                cand_theta = session.step_baseline_theta.copy()
                cand_theta[init_idx] = base_val + 40
                cand_file = session.runs_dir / "block_test.py"
                cand_file.write_text(
                    session.pm.update_constants_string(cand_theta),
                    encoding="utf-8",
                    newline="\n",
                )

                rec_holdout = session.validate_candidate_theta(
                    "block_test", cand_theta, out_path=cand_file
                )
                self.assertEqual(rec_holdout["verdict"], "PASS")
                self.assertIsNone(rec_holdout["pgn_val"])

                changed = session.accept_if_pass("block_test")
                self.assertEqual(changed, ["INITIATIVE_MG_OFFSET"])
                self.assertEqual(
                    int(_eval_const_module(old_copy)["INITIATIVE_MG_OFFSET"]),
                    base_val + 40,
                )
                # classical copy untouched
                self.assertEqual(
                    int(_eval_const_module(classical_copy)["INITIATIVE_MG_OFFSET"]),
                    base_val,
                )

                # Rollback restores classical_old copy cleanly
                session.rollback("block_test")
                self.assertEqual(
                    int(_eval_const_module(old_copy)["INITIATIVE_MG_OFFSET"]),
                    base_val,
                )

                # Reconfigure in-place to use_pgn_val=True & classical copy without reloading train_data
                prev_train_data_id = id(session.train_data)
                session._custom_runs_dir = tmp_dir / "runs_classical"
                session.reconfigure(
                    use_pgn_val=True,
                    target_engine="classical",
                    pgn_val_npz=pgn_npz,
                    target_constants_path=classical_copy,
                )
                self.assertEqual(id(session.train_data), prev_train_data_id)
                self.assertTrue(session.use_pgn_val)
                self.assertIsNotNone(session.pgn_data)
                self.assertEqual(session.target_engine, "classical")

                cand_file2 = session.runs_dir / "block_test.py"
                cand_file2.write_text(
                    session.pm.update_constants_string(cand_theta),
                    encoding="utf-8",
                    newline="\n",
                )
                rec_dual = session.validate_candidate_theta(
                    "block_test", cand_theta, out_path=cand_file2
                )
                self.assertEqual(rec_dual["verdict"], "REJECT")
                self.assertIsNotNone(rec_dual["pgn_val"])
                self.assertEqual(session.accept_if_pass("block_test"), [])

                # Verify dynamic random_seed toggle
                self.assertFalse(session.random_seed)
                session.reconfigure(random_seed=True)
                self.assertTrue(session.random_seed)

    def test_pgn_gate_judges_calibrated_kstar_not_raw_k1(self):
        sess = TexelTuningSession.__new__(TexelTuningSession)
        sess.use_pgn_val = True
        sess.target_engine = "classical"
        sess.train_path = Path("epd.npz")

        changes = [{"name": "INITIATIVE_PAWN_WEIGHT_EG"}]
        hold_res = {
            "buckets": {
                "all": {"n": 1000, "delta": -4.0e-5, "z": -13.0},
                "broad-EG phase<=64": {"n": 400, "delta": -5.0e-5, "z": -8.0},
            },
            "fitted_k": {"n": 1000, "delta": -4.2e-5, "z": -13.5},
        }
        # Raw K=1 shows large positive z due solely to scale calibration offset, while K* is neutral
        pgn_res_ok = {
            "buckets": {
                "all": {"n": 1000, "delta": +3.2e-5, "z": +41.0},
                "castling!=0": {"n": 300, "delta": -1.5e-5, "z": -5.6},
            },
            "fitted_k": {"n": 1000, "delta": +1.0e-6, "z": +0.46},
        }
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(sess._gate_summary(changes, hold_res, pgn_res_ok), "PASS")

            pgn_res_bad = {
                "buckets": {
                    "all": {"n": 1000, "delta": -1.0e-5, "z": -2.0},
                    "castling!=0": {"n": 300, "delta": -1.0e-6, "z": -0.2},
                },
                "fitted_k": {"n": 1000, "delta": +3.0e-5, "z": +2.6},
            }
            self.assertEqual(sess._gate_summary(changes, hold_res, pgn_res_bad), "REJECT")

    def test_parse_proc_meminfo_gb_handles_linux_meminfo_and_fallback(self):
        meminfo = (
            "MemTotal:       13289612 kB\n"
            "MemFree:         2097152 kB\n"
            "MemAvailable:   10485760 kB\n"
        )
        avail, total = _parse_proc_meminfo_gb(meminfo)
        self.assertAlmostEqual(avail, 10.0, places=6)
        self.assertAlmostEqual(total, 13289612 / (1024 ** 2), places=6)

        meminfo_fallback = "MemTotal: 4194304 kB\nMemFree: 2097152 kB\n"
        avail2, total2 = _parse_proc_meminfo_gb(meminfo_fallback)
        self.assertAlmostEqual(avail2, 2.0, places=6)
        self.assertAlmostEqual(total2, 4.0, places=6)

        self.assertEqual(_parse_proc_meminfo_gb("malformed"), (None, None))

    def test_drive_backup_dir_mirrors_constants_runs_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_npz = tmp_dir / "train_drive.npz"
            const_copy = tmp_dir / "classical_constants.py"
            drive_dir = tmp_dir / "drive_backup"
            shutil.copyfile(CONSTANTS_PATH, const_copy)

            pb0, ob0, gs0 = parse_fen_to_arrays(
                "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8"
            )
            n_train = 60
            np.savez(
                train_npz,
                piece_bbs=np.tile(pb0, (n_train, 1)),
                occupancy_bbs=np.tile(ob0, (n_train, 1)),
                game_states=np.tile(gs0, (n_train, 1)),
                results=np.full(n_train, 0.80, dtype=np.float32),
            )

            init_idx, _ = ParameterManager().get_param_indices("INITIATIVE_MG_OFFSET")

            def _fake_eval(pb, ob, gs, theta, lazy=False):
                return np.int32(int(np.rint(theta[init_idx])))

            def _fake_scores(pbs, obs, gss, theta):
                jitter = (np.arange(len(pbs), dtype=np.int32) % 7) - 3
                return np.int32(int(np.rint(theta[init_idx]))) + jitter

            with (
                patch("tuner.notebook_session.evaluate_position_tunable", side_effect=_fake_eval),
                patch("tuner.notebook_session.compute_white_scores", side_effect=_fake_scores),
                patch("tuner.validate_tuned.compute_white_scores", side_effect=_fake_scores),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                session = TexelTuningSession(
                    train_npz=train_npz,
                    eval_sample=10,
                    holdout_sample=20,
                    use_pgn_val=False,
                    target_engine="classical",
                    target_constants_path=const_copy,
                    runs_dir=tmp_dir / "runs_drive",
                    drive_backup_dir=drive_dir,
                )
                base_val = int(session.step_baseline_theta[init_idx])
                cand_theta = session.step_baseline_theta.copy()
                cand_theta[init_idx] = base_val + 40
                cand_file = session.runs_dir / "block_drive.py"
                cand_file.write_text(
                    session.pm.update_constants_string(cand_theta),
                    encoding="utf-8",
                    newline="\n",
                )

                rec = session.validate_candidate_theta(
                    "block_drive", cand_theta, out_path=cand_file
                )
                self.assertEqual(rec["verdict"], "PASS")
                changed = session.accept_if_pass("block_drive")
                self.assertEqual(changed, ["INITIATIVE_MG_OFFSET"])

                mirrored_top = drive_dir / "classical_constants.py"
                mirrored_run_const = drive_dir / "runs_drive" / "classical_constants.py"
                mirrored_cand = drive_dir / "runs_drive" / "block_drive.py"
                mirrored_backups = list((drive_dir / "runs_drive" / "backups").glob("*.py"))

                self.assertTrue(mirrored_top.is_file())
                self.assertTrue(mirrored_run_const.is_file())
                self.assertTrue(mirrored_cand.is_file())
                self.assertTrue(mirrored_backups)
                self.assertEqual(
                    int(_eval_const_module(mirrored_top)["INITIATIVE_MG_OFFSET"]),
                    base_val + 40,
                )

                session.rollback("block_drive")
                self.assertEqual(
                    int(_eval_const_module(mirrored_top)["INITIATIVE_MG_OFFSET"]),
                    base_val,
                )


class TestSfMixedLabelsAndKStarGate(unittest.TestCase):
    def test_arrays_to_fen_roundtrip_preserves_castling_ep_stm_and_rule50(self):
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/ppp1pppp/8/8/3pP3/8/PPPP1PPP/R3K2R b KQk e3 12 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 37 1",
        ]
        for fen in fens:
            pb, ob, gs = parse_fen_to_arrays(fen)
            rebuilt = arrays_to_fen(pb, gs)
            self.assertEqual(rebuilt, fen)
            pb2, ob2, gs2 = parse_fen_to_arrays(rebuilt)
            np.testing.assert_array_equal(pb, pb2)
            np.testing.assert_array_equal(ob, ob2)
            np.testing.assert_array_equal(gs, gs2)

    def test_parse_sf_search_output_ignores_aspiration_bounds_and_handles_black_stm(self):
        lines = [
            "info depth 1 seldepth 4 multipv 1 score cp 22 nodes 35 pv e2e4",
            "info depth 8 seldepth 11 multipv 1 score cp 48 nodes 3100 pv e2e4 e7e5",
            "info depth 9 seldepth 14 multipv 1 score cp -140 upperbound nodes 4005 pv e2e4",
        ]
        # White to move: keeps exact depth 8 (+48) and depth 1 (+22), ignoring depth 9 upperbound
        w_cp, w_d1 = parse_sf_search_output(lines, stm_is_black=False)
        self.assertEqual((w_cp, w_d1), (48, 22))

        # Black to move: negates both into White-POV
        b_cp, b_d1 = parse_sf_search_output(lines, stm_is_black=True)
        self.assertEqual((b_cp, b_d1), (-48, -22))

    def test_build_blended_labels_calibrates_ksf_and_filters_tactical_spikes(self):
        wdl = np.array([1.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float32)
        sf_cp = np.array([150, 0, -150, 20, -320], dtype=np.int16)
        # Row 4 has |sf_cp - sf_d1_cp| = |-320 - 30| = 350 > 180 -> tactical spike!
        sf_d1 = np.array([130, 10, -140, 15, 30], dtype=np.int16)

        blended, stats = build_blended_labels(
            wdl,
            sf_cp,
            0.25,
            sf_d1_cp=sf_d1,
            max_tactical_gap=180,
            fit_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        )
        self.assertEqual(stats["n_tactical_filtered"], 1)
        self.assertFalse(bool(stats["quiet_annotated_mask"][4]))
        # Row 4 falls back to pure WDL (0.0), while quiet rows 0 and 2 move smoothly inward
        self.assertEqual(float(blended[4]), 0.0)
        self.assertGreater(float(blended[2]), 0.0)
        self.assertLess(float(blended[0]), 1.0)
        self.assertAlmostEqual(float(blended[1]), 0.5, places=5)

    def test_session_sf_labels_keep_holdout_pure_wdl_and_reject_kstar_scale_inflation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_npz = tmp_dir / "train_sf.npz"
            sf_sidecar = tmp_dir / "train_sf.sf19.npz"
            const_copy = tmp_dir / "constants_copy.py"
            shutil.copyfile(CONSTANTS_PATH, const_copy)

            pb0, ob0, gs0 = parse_fen_to_arrays(
                "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8"
            )
            n_rows = 80
            piece_bbs = np.tile(pb0, (n_rows, 1))
            occ_bbs = np.tile(ob0, (n_rows, 1))
            game_states = np.tile(gs0, (n_rows, 1))
            rng = np.random.default_rng(7)
            true_cp = rng.integers(-220, 221, size=n_rows, dtype=np.int16)
            probs = 1.0 / (1.0 + np.exp(-true_cp.astype(np.float64) * 0.0066))
            wdl = np.where(probs > 0.62, 1.0, np.where(probs < 0.38, 0.0, 0.5)).astype(np.float32)

            np.savez(
                train_npz,
                piece_bbs=piece_bbs,
                occupancy_bbs=occ_bbs,
                game_states=game_states,
                results=wdl,
            )
            # Leave one row unannotated and one row with a tactical jump > 180cp
            sf_cp = true_cp.copy()
            sf_d1 = true_cp.copy()
            sf_cp[0] = UNANNOTATED_CP
            sf_d1[0] = UNANNOTATED_CP
            sf_d1[1] = np.int16(int(sf_cp[1]) + 300)
            np.savez(sf_sidecar, sf_cp=sf_cp, sf_d1_cp=sf_d1)

            init_idx, _ = ParameterManager().get_param_indices("INITIATIVE_MG_OFFSET")
            base_init = int(ParameterManager().theta[init_idx])

            def _fake_eval(pb, ob, gs, theta, lazy=False):
                return np.int32(int(np.rint(theta[init_idx])))

            def _fake_scores(pbs, obs, gss, theta):
                delta = int(np.rint(theta[init_idx])) - base_init
                # Base matches true_cp directionally on arbitrary slice length; large +delta adds pure bias
                ramp = ((np.arange(len(pbs), dtype=np.int32) * 37) % 200) - 100
                return ramp + delta * 15

            with (
                patch("tuner.notebook_session.evaluate_position_tunable", side_effect=_fake_eval),
                patch("tuner.notebook_session.compute_white_scores", side_effect=_fake_scores),
                patch("tuner.validate_tuned.compute_white_scores", side_effect=_fake_scores),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                session = TexelTuningSession(
                    train_npz=train_npz,
                    use_pgn_val=False,
                    eval_sample=15,
                    holdout_sample=20,
                    target_engine="classical",
                    target_constants_path=const_copy,
                    runs_dir=tmp_dir / "runs_sf",
                    use_sf_labels=True,
                    sf_lambda=0.25,
                    sf_labels_path=sf_sidecar,
                    sf_max_tactical_gap=180,
                )
                self.assertTrue(session.use_sf_labels)
                self.assertIsNotNone(session.sf_blend_stats)
                self.assertNotIn(0, set(session.train_idx.tolist()))
                self.assertNotIn(1, set(session.train_idx.tolist()))

                # Unseen holdout strictly preserves pure WDL {0.0, 0.5, 1.0}
                np.testing.assert_array_equal(
                    session.train_data["results"][session.holdout_idx],
                    wdl[session.holdout_idx],
                )
                # Active train rows have fractional blended labels
                self.assertTrue(
                    np.any(
                        ~np.isin(
                            session.train_data["results"][session.train_idx],
                            [0.0, 0.5, 1.0],
                        )
                    )
                )

                # Reconfigure sf_lambda dynamically in memory without reloading train_data
                prev_id = id(session.train_data)
                session.reconfigure(sf_lambda=0.30)
                self.assertEqual(id(session.train_data), prev_id)
                self.assertAlmostEqual(session.sf_lambda, 0.30)

                # Switch back to pure WDL cleanly
                session.reconfigure(use_sf_labels=False)
                self.assertFalse(session.use_sf_labels)
                np.testing.assert_array_equal(session.train_data["results"], wdl)


if __name__ == "__main__":
    unittest.main()
