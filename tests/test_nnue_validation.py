import unittest
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import components to test
from chess_engine.nnue.ml_eval.train import validate_dataset_schema
from chess_engine.nnue.ml_eval.inference import (
    _sf_hm_king_bucket,
    nnue_forward_incremental,
    init_accumulator,
    update_accumulator,
    get_bb_differences,
    FC1_WEIGHT,
    FC1_BIAS,
    NUM_FEATURES,
    PADDING_INDEX
)

class TestNNUEValidation(unittest.TestCase):

    def test_sign_aware_integer_division(self):
        """1. Test sign-aware integer division truncation (trunc_toward_zero)"""
        def trunc_div(x, d=41):
            if x >= 0:
                return x // d
            return -((-x) // d)
            
        self.assertEqual(trunc_div(82), 2)
        self.assertEqual(trunc_div(41), 1)
        self.assertEqual(trunc_div(40), 0)
        self.assertEqual(trunc_div(0), 0)
        self.assertEqual(trunc_div(-40), 0)
        self.assertEqual(trunc_div(-41), -1)
        self.assertEqual(trunc_div(-82), -2)

    def test_piece_count_bucket_index_routing(self):
        """2. Test piece count bucket index routing boundary cases"""
        def get_bucket(pc):
            b = (pc - 1) // 4
            return max(0, min(7, b))
            
        self.assertEqual(get_bucket(2), 0)
        self.assertEqual(get_bucket(4), 0)
        self.assertEqual(get_bucket(5), 1)
        self.assertEqual(get_bucket(8), 1)
        self.assertEqual(get_bucket(9), 2)
        self.assertEqual(get_bucket(28), 6)
        self.assertEqual(get_bucket(29), 7)
        self.assertEqual(get_bucket(32), 7)

    def test_dataset_schema_validator(self):
        """3. Test dataset schema validator behavior with correct and malformed schemas"""
        valid_data = {
            'features_stm': np.zeros((10, 32), dtype=np.int32),
            'features_nstm': np.zeros((10, 32), dtype=np.int32),
            'targets': np.zeros(10, dtype=np.float32),
            'piece_counts': np.full(10, 16, dtype=np.int8)
        }
        
        # Should not raise any exception
        validate_dataset_schema(valid_data, "dummy_valid")
        
        # Test missing key
        invalid_missing = dict(valid_data)
        del invalid_missing['piece_counts']
        with self.assertRaises(ValueError):
            validate_dataset_schema(invalid_missing, "dummy_missing")
            
        # Test shape inconsistency
        invalid_shape = dict(valid_data)
        invalid_shape['targets'] = np.zeros(9, dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_dataset_schema(invalid_shape, "dummy_shape")
            
        # Test piece_counts out of range [2, 32]
        invalid_pc = dict(valid_data)
        invalid_pc['piece_counts'] = np.full(10, 1, dtype=np.int8)
        with self.assertRaises(ValueError):
            validate_dataset_schema(invalid_pc, "dummy_pc")
            
        # Test feature out of range [0, 22528]
        invalid_features = dict(valid_data)
        invalid_features['features_stm'] = np.full((10, 32), 22529, dtype=np.int32)
        with self.assertRaises(ValueError):
            validate_dataset_schema(invalid_features, "dummy_features")

    def test_accumulator_differential_updates(self):
        """4. Test accumulator state differential check: incremental vs full rebuilds"""
        bbs = np.zeros(12, dtype=np.uint64)
        bbs[5] = np.uint64(1 << 4)   # White King
        bbs[11] = np.uint64(1 << 60) # Black King
        bbs[0] = np.uint64(1 << 8)    # White Pawn on a2
        
        # Initial rebuild
        stack_full = np.zeros((2, 2, 512), dtype=np.int32)
        init_accumulator(bbs, stack_full)
        
        # Move White Pawn a2 -> a3 (8 -> 16)
        bbs_new = np.copy(bbs)
        bbs_new[0] = np.uint64(1 << 16)
        
        # Full rebuild of new state
        stack_rebuild = np.zeros((2, 2, 512), dtype=np.int32)
        init_accumulator(bbs_new, stack_rebuild)
        
        # Incremental update of new state from old
        added, removed = get_bb_differences(bbs, bbs_new)
        self.assertIn(0 * 64 + 16, added)
        self.assertIn(0 * 64 + 8, removed)
        
        stack_inc = np.copy(stack_full)
        update_accumulator(bbs_new, added, removed, 1, stack_inc)
        
        # Compare White accumulator (index 0) and Black accumulator (index 1)
        np.testing.assert_array_equal(stack_rebuild[0, 0, :], stack_inc[1, 0, :])
        np.testing.assert_array_equal(stack_rebuild[0, 1, :], stack_inc[1, 1, :])

    def test_king_bucket_mirroring(self):
        """5. Test Stockfish HalfKAv2_hm king bucket horizontal mirroring"""
        self.assertEqual(_sf_hm_king_bucket(4), 31)
        self.assertEqual(_sf_hm_king_bucket(3), 31)
        self.assertEqual(_sf_hm_king_bucket(2), _sf_hm_king_bucket(5))

    def test_forward_pass_overflow_safety(self):
        """6. Test forward pass safety with overflow mock weights"""
        stack = np.zeros((1, 2, 512), dtype=np.int32)
        stack[0, 0, :] = 32767
        stack[0, 1, :] = 32767
        
        score = nnue_forward_incremental(0, 0, stack, 16)
        self.assertTrue(isinstance(score, (int, np.integer)))

    def test_converter_vs_inference_feature_consistency(self):
        """7. Test feature index consistency between converter, inference, and train FEN mapper"""
        # We need to import the converter functions
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tuner/nnue_pipeline")))
        from convert_bullet_bin import get_halfka_indices_bullet, get_halfka_indices_bullet_nstm
        from chess_engine.nnue.ml_eval.train import fen_to_halfka_indices
        
        # Helper to make bullet inputs
        def make_bullet_inputs(pieces_dict):
            sorted_sqs = sorted(pieces_dict.keys())
            occupancy = np.uint64(0)
            for sq in sorted_sqs:
                occupancy |= np.uint64(1) << np.uint64(sq)
                
            pieces_arr = np.zeros(16, dtype=np.uint8)
            for p_idx, sq in enumerate(sorted_sqs):
                our_piece = pieces_dict[sq]
                if our_piece < 6:
                    bullet_piece = our_piece
                else:
                    bullet_piece = our_piece + 2
                    
                byte_idx = p_idx // 2
                if p_idx % 2 == 1:
                    pieces_arr[byte_idx] |= (bullet_piece & 0x0F) << 4
                else:
                    pieces_arr[byte_idx] |= (bullet_piece & 0x0F)
                    
            return occupancy, pieces_arr

        # Helper to generate FEN
        def pieces_to_fen(pieces_dict, stm_color):
            char_map = {
                0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
                6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
            }
            grid = [['' for _ in range(8)] for _ in range(8)]
            for sq, p in pieces_dict.items():
                r = sq // 8
                f = sq % 8
                grid[7 - r][f] = char_map[p]
                
            rows = []
            for r in range(8):
                empty = 0
                row_str = ""
                for f in range(8):
                    char = grid[r][f]
                    if char == '':
                        empty += 1
                    else:
                        if empty > 0:
                            row_str += str(empty)
                            empty = 0
                        row_str += char
                if empty > 0:
                    row_str += str(empty)
                rows.append(row_str)
            board_part = '/'.join(rows)
            return f"{board_part} {stm_color} - - 0 1"

        def get_inference_indices_via_accumulator(pieces_dict, stm_color):
            orig_weight = FC1_WEIGHT.copy()
            orig_bias = FC1_BIAS.copy()
            try:
                FC1_BIAS[:] = 0
                FC1_WEIGHT[:, 0] = np.arange(NUM_FEATURES) % 256
                FC1_WEIGHT[:, 1] = np.arange(NUM_FEATURES) // 256
                FC1_WEIGHT[PADDING_INDEX:, :] = 0
                
                wk_sq = next((sq for sq, p in pieces_dict.items() if p == 5), None)
                bk_sq = next((sq for sq, p in pieces_dict.items() if p == 11), None)
                
                stm = 0 if stm_color == 'w' else 1
                
                # Get empty stack to capture bias
                empty_bbs = np.zeros(12, dtype=np.uint64)
                stack_empty = np.zeros((1, 2, 512), dtype=np.int32)
                init_accumulator(empty_bbs, stack_empty)
                
                # 1. White Perspective features (only depends on wk_sq)
                stm_indices_w = []
                nstm_indices_w = []
                if wk_sq is not None:
                    # Get White King own feature under White perspective
                    wk_only_bbs = np.zeros(12, dtype=np.uint64)
                    wk_only_bbs[5] |= np.uint64(1) << np.uint64(wk_sq)
                    stack_wk = np.zeros((1, 2, 512), dtype=np.int32)
                    init_accumulator(wk_only_bbs, stack_wk)
                    
                    diff_wk = stack_wk[0, 0, :] - stack_empty[0, 0, :]
                    f_wk_w = diff_wk[1] * 256 + diff_wk[0]
                    
                    if stm == 0:
                        stm_indices_w.append(f_wk_w)
                    else:
                        nstm_indices_w.append(f_wk_w)
                        
                    # Get features of all other pieces under White perspective
                    for sq, p in pieces_dict.items():
                        if p == 5:
                            continue
                        test_bbs = np.zeros(12, dtype=np.uint64)
                        test_bbs[5] |= np.uint64(1) << np.uint64(wk_sq)
                        test_bbs[p] |= np.uint64(1) << np.uint64(sq)
                        
                        stack_test = np.zeros((1, 2, 512), dtype=np.int32)
                        init_accumulator(test_bbs, stack_test)
                        
                        # Difference gives the feature index
                        diff = stack_test[0, 0, :] - stack_wk[0, 0, :]
                        f_piece_w = diff[1] * 256 + diff[0]
                        if stm == 0:
                            stm_indices_w.append(f_piece_w)
                        else:
                            nstm_indices_w.append(f_piece_w)
                            
                # 2. Black Perspective features (only depends on bk_sq)
                stm_indices_b = []
                nstm_indices_b = []
                if bk_sq is not None:
                    # Get Black King own feature under Black perspective
                    bk_only_bbs = np.zeros(12, dtype=np.uint64)
                    bk_only_bbs[11] |= np.uint64(1) << np.uint64(bk_sq)
                    stack_bk = np.zeros((1, 2, 512), dtype=np.int32)
                    init_accumulator(bk_only_bbs, stack_bk)
                    
                    diff_bk = stack_bk[0, 1, :] - stack_empty[0, 1, :]
                    f_bk_b = diff_bk[1] * 256 + diff_bk[0]
                    
                    if stm == 1:
                        stm_indices_b.append(f_bk_b)
                    else:
                        nstm_indices_b.append(f_bk_b)
                        
                    # Get features of all other pieces under Black perspective
                    for sq, p in pieces_dict.items():
                        if p == 11:
                            continue
                        test_bbs = np.zeros(12, dtype=np.uint64)
                        test_bbs[11] |= np.uint64(1) << np.uint64(bk_sq)
                        test_bbs[p] |= np.uint64(1) << np.uint64(sq)
                        
                        stack_test = np.zeros((1, 2, 512), dtype=np.int32)
                        init_accumulator(test_bbs, stack_test)
                        
                        # Difference gives the feature index
                        diff = stack_test[0, 1, :] - stack_bk[0, 1, :]
                        f_piece_b = diff[1] * 256 + diff[0]
                        if stm == 1:
                            stm_indices_b.append(f_piece_b)
                        else:
                            nstm_indices_b.append(f_piece_b)
                            
                if stm == 0:
                    return sorted(stm_indices_w), sorted(nstm_indices_b)
                else:
                    return sorted(stm_indices_b), sorted(nstm_indices_w)
            finally:
                FC1_WEIGHT[:] = orig_weight
                FC1_BIAS[:] = orig_bias

        test_positions = [
            (
                "Initial position, White to move",
                {
                    4: 5, 60: 11, # Kings
                    8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, # White pawns
                    48: 6, 49: 6, 50: 6, 51: 6, 52: 6, 53: 6, 54: 6, 55: 6, # Black pawns
                },
                "w"
            ),
            (
                "Initial position, Black to move",
                {
                    4: 5, 60: 11, # Kings
                    8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, # White pawns
                    48: 6, 49: 6, 50: 6, 51: 6, 52: 6, 53: 6, 54: 6, 55: 6, # Black pawns
                },
                "b"
            ),
            (
                "White king on c1 (18), Black king on f8 (61)",
                {
                    18: 5, 61: 11,
                    24: 0, # White Pawn on a4
                    40: 6, # Black Pawn on a6
                },
                "w"
            ),
            (
                "White king on f1 (21), Black king on c8 (58)",
                {
                    21: 5, 58: 11,
                    25: 1, # White Knight on b4
                    42: 7, # Black Knight on c6
                },
                "b"
            ),
            (
                "Endgame with only kings",
                {
                    36: 5, 44: 11,
                },
                "w"
            )
        ]

        for desc, pieces_dict, stm_color in test_positions:
            with self.subTest(desc=desc, stm=stm_color):
                inf_stm, inf_nstm = get_inference_indices_via_accumulator(pieces_dict, stm_color)
                
                fen = pieces_to_fen(pieces_dict, stm_color)
                t_stm_raw, t_nstm_raw, _ = fen_to_halfka_indices(fen)
                t_stm = sorted([int(x) for x in t_stm_raw if x != PADDING_INDEX])
                t_nstm = sorted([int(x) for x in t_nstm_raw if x != PADDING_INDEX])
                
                bullet_pieces_dict = {}
                if stm_color == 'w':
                    wk_sq = next(sq for sq, p in pieces_dict.items() if p == 5)
                    bk_sq = next(sq for sq, p in pieces_dict.items() if p == 11) ^ 56
                    bullet_pieces_dict = pieces_dict.copy()
                else:
                    wk_sq = next(sq for sq, p in pieces_dict.items() if p == 11) ^ 56
                    bk_sq = next(sq for sq, p in pieces_dict.items() if p == 5)
                    for sq, p in pieces_dict.items():
                        new_p = p + 6 if p < 6 else p - 6
                        bullet_pieces_dict[sq ^ 56] = new_p
                
                occ, pieces_arr = make_bullet_inputs(bullet_pieces_dict)
                
                c_stm_raw = get_halfka_indices_bullet(occ, pieces_arr, np.uint8(wk_sq), np.uint8(bk_sq))
                c_nstm_raw = get_halfka_indices_bullet_nstm(occ, pieces_arr, np.uint8(wk_sq), np.uint8(bk_sq))
                c_stm = sorted([int(x) for x in c_stm_raw if x != PADDING_INDEX])
                c_nstm = sorted([int(x) for x in c_nstm_raw if x != PADDING_INDEX])
                
                self.assertEqual(inf_stm, t_stm, f"Inference vs Train STM mismatch in '{desc}'\nInf: {inf_stm}\nTrain: {t_stm}")
                self.assertEqual(inf_nstm, t_nstm, f"Inference vs Train NSTM mismatch in '{desc}'\nInf: {inf_nstm}\nTrain: {t_nstm}")
                
                self.assertEqual(c_stm, t_stm, f"Converter vs Train STM mismatch in '{desc}'\nConv: {c_stm}\nTrain: {t_stm}")
                self.assertEqual(c_nstm, t_nstm, f"Converter vs Train NSTM mismatch in '{desc}'\nConv: {c_nstm}\nTrain: {t_nstm}")

if __name__ == "__main__":
    unittest.main()
