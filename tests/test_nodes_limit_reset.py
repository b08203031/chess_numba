import unittest
from pathlib import Path


class TestNodeLimitReset(unittest.TestCase):
    def test_search_resets_timer_counter_at_root_and_each_iteration(self):
        for filename in ('chess_engine/classical/search.py', 'chess_engine/nnue/search.py'):
            source = Path(filename).read_text(encoding='utf-8')
            reset = 'nodes_searched_array[0] = np.uint64(0)'
            self.assertGreaterEqual(source.count(reset), 2, filename)
        classical = Path('chess_engine/classical/search.py').read_text(encoding='utf-8')
        nnue = Path('chess_engine/nnue/search.py').read_text(encoding='utf-8')
        self.assertIn('Fixed-node stopping is', classical)
        self.assertIn('checked directly in JIT through nodes_searched_array[1].', classical)
        self.assertIn('current root/depth', nnue)
        self.assertIn('iteration.', nnue)

    def test_classical_return_keeps_search_and_qnodes_separate(self):
        source = Path('chess_engine/classical/search.py').read_text(encoding='utf-8')
        self.assertIn('total_search_nodes += res[2]', source)
        self.assertIn(
            'return (final_best_move, last_score, total_search_nodes, total_q_nodes,',
            source,
        )
        self.assertIn('Preserve completed child telemetry even when the timer fired', source)
        self.assertIn('Count the child before propagating a stop.', source)

    def test_classical_stop_check_cadence_is_bounded_for_100k_searches(self):
        constants = Path('chess_engine/classical/constants.py').read_text(encoding='utf-8')
        source = Path('chess_engine/classical/search.py').read_text(encoding='utf-8')
        self.assertIn('STOP_CHECK_MASK = 4095', constants)
        self.assertEqual(source.count('& STOP_CHECK_MASK'), 2)
        self.assertGreaterEqual(source.count('nodes_searched_array[1]'), 6)
        engine_types = Path('chess_engine/classical/engine_types.py').read_text(encoding='utf-8')
        self.assertIn('nodes_searched_array = np.zeros(2, dtype=np.uint64)', engine_types)


if __name__ == '__main__':
    unittest.main()
