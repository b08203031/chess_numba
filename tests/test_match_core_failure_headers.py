import unittest

from tools.match_core import SearchLimit, play_game


class _FailingEngine:
    name = 'Failing'

    def new_game(self, _start_fen):
        return None

    def get_move(self, _limit):
        raise RuntimeError('boom\nwith detail')


class _IdleEngine:
    name = 'Idle'

    def new_game(self, _start_fen):
        return None


class TestMatchCoreFailureHeaders(unittest.TestCase):
    def test_search_failure_records_safe_diagnostic_headers(self):
        result = play_game(
            _FailingEngine(),
            _IdleEngine(),
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=1,
        )
        self.assertEqual(result.pgn.headers['Termination'], 'engine failure')
        self.assertEqual(result.pgn.headers['Result'], '0-1')
        self.assertEqual(result.pgn.headers['FailurePly'], '1.')
        self.assertEqual(result.pgn.headers['Failure'], 'Failing: RuntimeError: boom with detail')


if __name__ == '__main__':
    unittest.main()
