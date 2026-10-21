import unittest
import numpy as np

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.see import see, see_ge
from chess_engine.classical.move_generator import get_pinned_pieces


class TestSEE(unittest.TestCase):
    def test_quiet_promotion_undefended(self):
        # White pawn on e7 (52), promoting to e8 (60), no defenders
        pb, ob, gs = parse_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        stm = np.int32(gs[0])
        pw = get_pinned_pieces(pb, ob, 0)
        pbk = get_pinned_pieces(pb, ob, 1)

        val = see(pb, ob, stm, 52, 60, pw, pbk)
        self.assertEqual(val, 1070)
        self.assertTrue(see_ge(pb, ob, stm, 52, 60, 1070, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 52, 60, 0, pw, pbk))
        self.assertFalse(see_ge(pb, ob, stm, 52, 60, 1071, pw, pbk))

    def test_quiet_promotion_supported(self):
        # White pawn on e7 (52), Black rook on a8 (56), White rook on e1 (4)
        # 1. e8=Q Rxe8 2. Rxe8 -> net +472 (wins Rook 572 for Pawn 100)
        pb, ob, gs = parse_fen("r6k/4P3/8/8/8/8/8/4R2K w - - 0 1")
        stm = np.int32(gs[0])
        pw = get_pinned_pieces(pb, ob, 0)
        pbk = get_pinned_pieces(pb, ob, 1)

        val = see(pb, ob, stm, 52, 60, pw, pbk)
        self.assertEqual(val, 472)
        self.assertTrue(see_ge(pb, ob, stm, 52, 60, 472, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 52, 60, 0, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 52, 60, -91, pw, pbk))
        self.assertFalse(see_ge(pb, ob, stm, 52, 60, 473, pw, pbk))

    def test_quiet_promotion_unsupported_loses_pawn(self):
        # White pawn on e7 (52), Black rook on a8 (56), no White defender
        # 1. e8=Q Rxe8 -> net -100 (loses Pawn)
        pb, ob, gs = parse_fen("r6k/4P3/8/8/8/8/8/7K w - - 0 1")
        stm = np.int32(gs[0])
        pw = get_pinned_pieces(pb, ob, 0)
        pbk = get_pinned_pieces(pb, ob, 1)

        val = see(pb, ob, stm, 52, 60, pw, pbk)
        self.assertEqual(val, -100)
        self.assertFalse(see_ge(pb, ob, stm, 52, 60, 0, pw, pbk))
        self.assertFalse(see_ge(pb, ob, stm, 52, 60, -80, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 52, 60, -100, pw, pbk))

    def test_capture_promotion_against_defended_rook(self):
        # White pawn on e7 (52), Black rook on d8 (59) defended by King on c8 (58)
        # 1. exd8=Q+ Kxd8 -> net +472 (traded Pawn 100 for Rook 572)
        pb, ob, gs = parse_fen("2kr4/4P3/8/8/8/8/8/7K w - - 0 1")
        stm = np.int32(gs[0])
        pw = get_pinned_pieces(pb, ob, 0)
        pbk = get_pinned_pieces(pb, ob, 1)

        val = see(pb, ob, stm, 52, 59, pw, pbk)
        self.assertEqual(val, 472)
        self.assertTrue(see_ge(pb, ob, stm, 52, 59, 0, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 52, 59, -91, pw, pbk))

    def test_normal_exchanges_and_en_passant(self):
        # KiwiPete position
        kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        pb, ob, gs = parse_fen(kiwipete)
        stm = np.int32(gs[0])
        pw = get_pinned_pieces(pb, ob, 0)
        pbk = get_pinned_pieces(pb, ob, 1)

        # d5xe6 (35 to 44)
        val = see(pb, ob, stm, 35, 44, pw, pbk)
        self.assertEqual(val, 0)
        self.assertTrue(see_ge(pb, ob, stm, 35, 44, 0, pw, pbk))
        self.assertFalse(see_ge(pb, ob, stm, 35, 44, 1, pw, pbk))

        # En passant
        ep_fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        pb_ep, ob_ep, gs_ep = parse_fen(ep_fen)
        stm_ep = np.int32(gs_ep[0])
        pw_ep = get_pinned_pieces(pb_ep, ob_ep, 0)
        pbk_ep = get_pinned_pieces(pb_ep, ob_ep, 1)

        # e5xf6 (36 to 45) - White trades pawns: 1. exf6 gxf6 -> net 0
        val_ep = see(pb_ep, ob_ep, stm_ep, 36, 45, pw_ep, pbk_ep)
        self.assertEqual(val_ep, 0)
        self.assertTrue(see_ge(pb_ep, ob_ep, stm_ep, 36, 45, 0, pw_ep, pbk_ep))
        self.assertFalse(see_ge(pb_ep, ob_ep, stm_ep, 36, 45, 1, pw_ep, pbk_ep))

    def test_sentinel_minus_two_empty_square(self):
        # Empty square quiet move with victim_type = -2 must evaluate as 0, not -2 % 6 == 4 (QUEEN = 900)
        fen = "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"
        pb, ob, gs = parse_fen(fen)
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        # Quiet move e2 to e4 (12 to 28), victim_type = -2
        val = see(pb, ob, stm, 12, 28, pw, pbk, attacker_type=3, victim_type=-2)
        self.assertEqual(val, 0)

    def test_pinned_piece_along_pin_ray(self):
        # White King e1 (4), White Rook e2 (12). Black Rook e8 (60), Black King h8 (63).
        # Black Rook on e8 pins White Rook on e2 to King on e1.
        # Direct capture of pinner along pin ray (Rxe8+): legal and wins Rook (+500).
        fen = "4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1"
        pb, ob, gs = parse_fen(fen)
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        self.assertTrue(bool(pw & (np.uint64(1) << 12)))
        val = see(pb, ob, stm, 12, 60, pw, pbk)
        self.assertEqual(val, 572)
        self.assertTrue(see_ge(pb, ob, stm, 12, 60, 572, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 12, 60, 0, pw, pbk))

    def test_pinned_piece_blocked_off_pin_ray(self):
        # White: King e1 (4), Rook e2 (12), Bishop c3 (18)
        # Black: King h8 (63), Rook e8 (60), Knight d4 (27 defended by Black Queen d8 (59))
        # White Rook on e2 is pinned on file E by Black Rook on e8.
        # 1. Bxd4 (18 to 27) takes Knight (+347)
        # 2. Qxd4 (59 to 27) takes Bishop (-364)
        # White Rook on e2 is pinned along file E and CANNOT recapture on d4 (off pin ray).
        # Net exchange on d4: 347 - 364 = -17.
        fen = "3q3k/8/8/8/3n4/2B5/4R3/4K3 w - - 0 1"
        pb, ob, gs = parse_fen(fen)
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        val = see(pb, ob, stm, 18, 27, pw, pbk)
        self.assertEqual(val, -17)
        self.assertFalse(see_ge(pb, ob, stm, 18, 27, 0, pw, pbk))
        self.assertTrue(see_ge(pb, ob, stm, 18, 27, -17, pw, pbk))

    def test_recapture_promotion_in_loop(self):
        # 測試 while 迴圈內 (depth >= 1) 的後續反吃升變：
        # 白車 d1 (3)、白兵 e7 (52)；黑車 a8 (56)、黑車 d8 (59)、黑王 c7 (50)
        # （黑王在 c7 讓出第 8 橫排，使 a8 黑車得以直線反吃 d8）
        # 1. Rd1xd8 (+572) 2. Ra8xd8 (-572, 累計 0) 3. e7xd8=Q+ (+572+1070) 4. Kc7xd8 (-1170)
        # 白方淨賺 +472
        pb, ob, gs = parse_fen("r2r4/2k1P3/8/8/8/8/8/3R3K w - - 0 1")
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        val = see(pb, ob, stm, 3, 59, pw, pbk)
        self.assertEqual(val, 472)
        self.assertTrue(see_ge(pb, ob, stm, 3, 59, 472, pw, pbk))
        self.assertFalse(see_ge(pb, ob, stm, 3, 59, 473, pw, pbk))

    def test_en_passant_xray_through_captured_pawn(self):
        # 驗證過路兵被吃掉的格子 (f5=37) 有確實從 occupied 移除：
        # 白兵 e5 (36)、白車 f1 (5)；黑兵 f5 (37)、黑馬 h6 (47，攻擊 f6=45)
        # 1. exf6 e.p. 後，f5 被清空，f1 白車得以直線穿透保護 f6，黑馬不敢反吃 -> 白方淨賺 +100
        pb, ob, gs = parse_fen("6k1/8/7n/4Pp2/8/8/8/5R1K w - f6 0 1")
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        # 同時測試預設 (-1) 與 search.py 傳入的 (-2) 哨兵值
        self.assertEqual(see(pb, ob, stm, 36, 45, pw, pbk), 100)
        self.assertEqual(see(pb, ob, stm, 36, 45, pw, pbk, 0, -2), 100)
        self.assertTrue(see_ge(pb, ob, stm, 36, 45, 100, pw, pbk, 0, -2))
        self.assertFalse(see_ge(pb, ob, stm, 36, 45, 101, pw, pbk, 0, -2))

    def test_quiet_move_sentinel_minus_two(self):
        # 驗證 search.py / search_heuristics.py 傳入 victim_type = -2 時，
        # _see_jit 與 _see_ge_jit 不會因為 -2 % 6 == 4 而把空目標格當成皇后 (900)
        # 黑兵在 d6 (43) 斜下攻擊空白格 e5 (36)，白馬 f3 (21) 走到受保護的 e5 -> 淨損 -320
        pb, ob, gs = parse_fen("6k1/8/3p4/8/8/5N2/8/6K1 w - - 0 1")
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        self.assertEqual(see(pb, ob, stm, 21, 36, pw, pbk, 1, -2), -347)
        self.assertFalse(see_ge(pb, ob, stm, 21, 36, 0, pw, pbk, 1, -2))
        self.assertTrue(see_ge(pb, ob, stm, 21, 36, -347, pw, pbk, 1, -2))

    def test_black_promotion_and_pinned_defender(self):
        # 黑兵 e2 (12) 吃白車 d1 (3) 升變，白馬 c3 (18) 原本防守 d1 但被 a5 (32) 黑主教牽制在 e1 (4) 白王前
        # 同時黑車 d8 (59) 保護 d1，防止白王 e1 反吃 d1
        # 因 c3 白馬被牽制無法反吃 d1 且白王不可走入將軍，黑方淨賺 +572(車) + 1070(升變) = +1642
        pb, ob, gs = parse_fen("3r2k1/8/8/b7/8/2N5/4p3/3RK3 b - - 0 1")
        stm = np.int32(gs[0])
        pw = np.uint64(get_pinned_pieces(pb, ob, 0))
        pbk = np.uint64(get_pinned_pieces(pb, ob, 1))

        val = see(pb, ob, stm, 12, 3, pw, pbk)
        self.assertEqual(val, 1642)
        self.assertTrue(see_ge(pb, ob, stm, 12, 3, 1642, pw, pbk))
        self.assertFalse(see_ge(pb, ob, stm, 12, 3, 1643, pw, pbk))


if __name__ == '__main__':
    unittest.main()
