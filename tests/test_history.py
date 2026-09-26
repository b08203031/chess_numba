"""Light tests for the production Phase-1a history tables."""
import unittest

import numpy as np

from chess_engine.classical.constants import (
    CORRECTION_HISTORY_SIZE,
    HISTORY_MAX_CAPTURE,
    HISTORY_MAX_CONTINUATION,
    HISTORY_MAX_MAIN,
    HISTORY_MAX_PAWN,
    HISTORY_PRUNE_CONT1_FLOOR,
    MAX_PLY,
    PAWN_HISTORY_SIZE,
)
from chess_engine.classical.engine_types import SearchContext
from chess_engine.classical.move import encode_move
from chess_engine.classical.search_heuristics import (
    capture_victim_index,
    get_lmr_stat_score,
    get_quiet_stat_components,
    get_quiet_stat_score,
    update_capture_history,
    update_continuation_history,
    update_history,
    update_pawn_history,
)
from chess_engine.classical.transposition_table import create_transposition_table


def make_context():
    return SearchContext(
        create_transposition_table(1),
        np.zeros(MAX_PLY * 2, dtype=np.uint16),
        np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16),
        np.zeros((12, 64), dtype=np.int32),
        np.zeros((64, 64), dtype=np.int32),
        np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
        np.zeros((12, 64, 6), dtype=np.int32),
        np.full((PAWN_HISTORY_SIZE, 12, 64), -1238, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
    )


class TestHistoryTables(unittest.TestCase):
    def test_shapes_match_phase_1a(self):
        ctx = make_context()
        self.assertEqual(ctx.history_table.shape, (12, 64))
        self.assertEqual(ctx.butterfly_history.shape, (64, 64))
        self.assertEqual(ctx.capture_history.shape, (12, 64, 6))
        self.assertEqual(ctx.continuation_history.shape, (5, 12, 64, 12, 64))
        self.assertEqual(ctx.pawn_history.shape, (PAWN_HISTORY_SIZE, 12, 64))
        self.assertEqual(ctx.counter_moves.shape, (12, 64))
        self.assertEqual(ctx.low_ply_history.shape, (5, 65536))

    def test_gravity_updates_stay_bounded(self):
        ctx = make_context()
        for _ in range(100):
            update_history(ctx.history_table, 1, 18, 4000)
            update_capture_history(ctx.capture_history, 1, 18, 0, 4000)
            update_pawn_history(ctx.pawn_history, 7, 1, 18, 4000)

        self.assertLessEqual(abs(int(ctx.history_table[1, 18])), HISTORY_MAX_MAIN)
        self.assertLessEqual(abs(int(ctx.capture_history[1, 18, 0])), HISTORY_MAX_CAPTURE)
        self.assertLessEqual(abs(int(ctx.pawn_history[7, 1, 18])), HISTORY_MAX_PAWN)

    def test_continuation_offsets_and_lmr_read_use_same_keys(self):
        ctx = make_context()
        previous = encode_move(8, 16, 0, 0)
        current = encode_move(1, 18, 0, 0)
        ctx.move_stack[0] = previous
        ctx.piece_stack[0] = 0

        update_continuation_history(ctx, 0, previous, 0, current, 1, 600)
        value = int(ctx.continuation_history[0, 0, 16, 1, 18])
        self.assertNotEqual(value, 0)
        self.assertLessEqual(abs(value), HISTORY_MAX_CONTINUATION)
        ctx.history_table[1, 18] = 100
        stat_score = get_lmr_stat_score(ctx, 1, 1, 18, 1)
        self.assertEqual(int(stat_score), 2 * 100 + value)

    def test_capture_victim_index_is_color_relative(self):
        self.assertEqual(int(capture_victim_index(0)), 0)
        self.assertEqual(int(capture_victim_index(5)), 5)
        self.assertEqual(int(capture_victim_index(6)), 0)
        self.assertEqual(int(capture_victim_index(11)), 5)
        self.assertEqual(int(capture_victim_index(-1)), -1)

    def test_quiet_stat_attribution_reconstructs_production_score(self):
        ctx = make_context()
        older = encode_move(48, 40, 0, 0)
        newer = encode_move(8, 16, 0, 0)
        ctx.move_stack[0] = older
        ctx.piece_stack[0] = 6
        ctx.move_stack[1] = newer
        ctx.piece_stack[1] = 0
        ctx.history_table[1, 18] = 100
        ctx.butterfly_history[1, 18] = -50
        ctx.pawn_history[7, 1, 18] = 30
        ctx.continuation_history[0, 0, 16, 1, 18] = 40
        ctx.continuation_history[1, 6, 40, 1, 18] = -20

        components = tuple(
            int(v) for v in get_quiet_stat_components(ctx, 2, 1, 18, 1, 7)
        )
        self.assertEqual(components, (300, -50, 60, 200, -40, 0, 0, 0))
        self.assertEqual(
            sum(components), int(get_quiet_stat_score(ctx, 2, 1, 18, 1, 7))
        )

    def test_history_prune_cont1_floor_does_not_change_default_score(self):
        ctx = make_context()
        previous = encode_move(8, 16, 0, 0)
        ctx.move_stack[0] = previous
        ctx.piece_stack[0] = 0
        ctx.pawn_history[7, 1, 18] = 0
        ctx.continuation_history[0, 0, 16, 1, 18] = -2000

        raw_score = int(get_quiet_stat_score(ctx, 1, 1, 18, 1, 7))
        prune_score = int(get_quiet_stat_score(
            ctx, 1, 1, 18, 1, 7, HISTORY_PRUNE_CONT1_FLOOR,
        ))
        self.assertEqual(raw_score, -10000)
        self.assertEqual(prune_score, HISTORY_PRUNE_CONT1_FLOOR)

if __name__ == "__main__":
    unittest.main()
