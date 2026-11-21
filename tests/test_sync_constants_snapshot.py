import tempfile
import unittest
from pathlib import Path

from tuner.sync_constants_snapshot import (
    affected_classical_modules,
    clear_classical_numba_cache,
    diff_constant_names,
)


class TestSyncConstantsSnapshot(unittest.TestCase):
    def test_clear_only_rebuildable_numba_cache_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for name in ("eval.x.nbi", "eval.x.1.nbc", "constants.cpython.pyc", "keep.txt"):
                (cache_dir / name).write_bytes(b"cache")

            self.assertEqual(clear_classical_numba_cache(cache_dir), 2)
            self.assertFalse((cache_dir / "eval.x.nbi").exists())
            self.assertFalse((cache_dir / "eval.x.1.nbc").exists())
            self.assertTrue((cache_dir / "constants.cpython.pyc").exists())
            self.assertTrue((cache_dir / "keep.txt").exists())

    def test_selective_clear_keeps_unrelated_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for name in (
                "evaluation.evaluate-1.py310.nbi",
                "evaluation.evaluate-1.py310.1.nbc",
                "search._search-10.py310.nbi",
                "move_generator.get_rook_attacks-5.py310.1.nbc",
                "engine_types._ctx-2.py310.nbi",
            ):
                (cache_dir / name).write_bytes(b"cache")

            removed = clear_classical_numba_cache(cache_dir, modules={"evaluation", "search"})
            self.assertEqual(removed, 3)
            self.assertFalse((cache_dir / "evaluation.evaluate-1.py310.nbi").exists())
            self.assertFalse((cache_dir / "evaluation.evaluate-1.py310.1.nbc").exists())
            self.assertFalse((cache_dir / "search._search-10.py310.nbi").exists())
            self.assertTrue((cache_dir / "move_generator.get_rook_attacks-5.py310.1.nbc").exists())
            self.assertTrue((cache_dir / "engine_types._ctx-2.py310.nbi").exists())
            self.assertEqual(clear_classical_numba_cache(cache_dir, modules=set()), 0)

    def test_diff_constant_names_ignores_comments_and_detects_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old_f = d / "old.py"
            new_f = d / "new.py"
            old_f.write_text(
                "import numpy as np\nA = 1\nARR = np.array([1, 2], dtype=np.int32)\n",
                encoding="utf-8",
            )
            new_f.write_text(
                "# comment\nimport numpy as np\nA = 1\nARR = np.array([1, 3], dtype=np.int32)\n",
                encoding="utf-8",
            )
            self.assertEqual(diff_constant_names(old_f, new_f), {"ARR"})
            new_f.write_text(
                "# comment only\nimport numpy as np\nA = 1\nARR = np.array([1, 2], dtype=np.int32)\n",
                encoding="utf-8",
            )
            self.assertEqual(diff_constant_names(old_f, new_f), set())

    def test_affected_classical_modules_tracks_direct_and_transitive_jit_deps(self):
        self.assertEqual(affected_classical_modules([]), set())
        # Runtime-tuned search scalar read only in SearchContext.__init__ (pure Python)
        self.assertEqual(affected_classical_modules(["LMR_BASE_OFFSET"]), set())
        # Eval scalar propagates to evaluation and its @njit caller search
        self.assertEqual(affected_classical_modules(["EG_SAFETY_SCALE"]), {"evaluation", "search"})
        # MG_MATERIAL_VALUES affects see (via SEE_PIECE_VALUES), endgame, search_heuristics, evaluation, search
        mat_mods = affected_classical_modules(["MG_MATERIAL_VALUES"])
        self.assertTrue(
            {"see", "endgame", "search_heuristics", "evaluation", "search"}.issubset(mat_mods)
        )
        self.assertTrue(
            {"engine_types", "move_generator", "bitboard_utils", "board_operations", "zobrist"}.isdisjoint(mat_mods)
        )


if __name__ == "__main__":
    unittest.main()
