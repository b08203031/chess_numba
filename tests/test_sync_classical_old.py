import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools.sync_classical_old import do_diff_code, do_sync


class TestSyncClassicalOld(unittest.TestCase):
    def test_sync_code_preserves_eval_and_updates_search_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new_dir = root / 'classical'
            old_dir = root / 'classical_old'
            new_dir.mkdir()
            old_dir.mkdir()

            (new_dir / 'constants.py').write_text(
                'EVAL_WEIGHT = 2\n'
                '# --- Search Constants / 搜尋常量 ---\n'
                'SEARCH_MARGIN = 30\n',
                encoding='utf-8',
            )
            (old_dir / 'constants.py').write_text(
                'EVAL_WEIGHT = 1\n'
                '# --- Search Constants / 搜尋常量 ---\n'
                'SEARCH_MARGIN = 10\n',
                encoding='utf-8',
            )
            (new_dir / 'evaluation.py').write_text(
                'from chess_engine.classical.constants import EVAL_WEIGHT\n',
                encoding='utf-8',
            )
            (old_dir / 'evaluation.py').write_text(
                'from chess_engine.classical_old.constants import EVAL_WEIGHT\nold = True\n',
                encoding='utf-8',
            )

            with contextlib.redirect_stdout(io.StringIO()):
                do_sync(str(new_dir), str(old_dir), preserve_constants=True)

            old_constants = (old_dir / 'constants.py').read_text(encoding='utf-8')
            self.assertIn('EVAL_WEIGHT = 1', old_constants)
            self.assertIn('SEARCH_MARGIN = 30', old_constants)
            self.assertNotIn('EVAL_WEIGHT = 2', old_constants)
            self.assertEqual(
                (old_dir / 'evaluation.py').read_text(encoding='utf-8'),
                'from chess_engine.classical_old.constants import EVAL_WEIGHT\n',
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(do_diff_code(str(new_dir), str(old_dir)))


if __name__ == '__main__':
    unittest.main()
