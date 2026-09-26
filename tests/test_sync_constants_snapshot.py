import tempfile
import unittest
from pathlib import Path

from tuner.sync_constants_snapshot import clear_classical_numba_cache


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


if __name__ == "__main__":
    unittest.main()
