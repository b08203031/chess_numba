import unittest
import numpy as np
from pathlib import Path

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.constants import (
    THREAT_BY_KING, MINOR_BEHIND_PAWN, WEAK_QUEEN,
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
)
from chess_engine.classical.evaluation import (
    evaluate_position, _evaluate_position_jit,
    evaluate_attacks_mobility_threats,
)


def _vflip(bb: np.uint64) -> np.uint64:
    res = np.uint64(0)
    for r in range(8):
        byte = (bb >> np.uint64(r * 8)) & np.uint64(0xFF)
        res |= byte << np.uint64((7 - r) * 8)
    return res


def _flip_position(piece_bbs, occupancy_bbs, game_state):
    new_pb = np.zeros(12, dtype=np.uint64)
    for pt in range(6):
        new_pb[pt] = _vflip(piece_bbs[pt + 6])
        new_pb[pt + 6] = _vflip(piece_bbs[pt])
    new_occ = np.zeros(3, dtype=np.uint64)
    for p in range(6):
        new_occ[0] |= new_pb[p]
        new_occ[1] |= new_pb[p + 6]
    new_occ[2] = new_occ[0] | new_occ[1]

    new_gs = game_state.copy()
    new_gs[0] = 1 - game_state[0]
    cr = int(game_state[1])
    new_cr = ((cr & 3) << 2) | ((cr >> 2) & 3)
    new_gs[1] = np.uint8(new_cr)
    ep = int(game_state[2])
    if ep != 64:
        new_gs[2] = np.int8(ep ^ 56)
    return new_pb, new_occ, new_gs


class TestPhase4SF11Alignment(unittest.TestCase):
    """
    Phase 4: 黑白鏡像對稱化（_trunc_div）與 Stockfish 11 特徵對齊單元測試
    覆蓋：
      - BUG-3: 500-FEN 黑白鏡像翻轉對稱性測試（Bit-Exact 0 誤差）
      - BUG-5: ThreatByKing 確保為布林單次觸發（不乘 popcount）
      - BUG-6: MinorBehindPawn 支援敵方兵掩體／封鎖哨站
      - BUG-7: WeakQueen 排除長程狙擊子疊線假阻擋
      - BUG-8: kingAttacksCount 嚴格限定於 king_zone 內累加
    """

    def test_bug3_color_flip_mirror_symmetry_500_fen(self):
        epd_path = Path(__file__).resolve().parent.parent / "data" / "hist_diag_openings_500.epd"
        self.assertTrue(epd_path.exists(), f"EPD test file missing: {epd_path}")

        mismatches = []
        with open(epd_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                fen = line.strip().split(";")[0].strip()
                if not fen:
                    continue
                pb, occ, gs = parse_fen(fen)
                s1, _, _ = evaluate_position(pb, occ, gs)
                pb_f, occ_f, gs_f = _flip_position(pb, occ, gs)
                s2, _, _ = evaluate_position(pb_f, occ_f, gs_f)
                if s1 != s2:
                    mismatches.append((idx + 1, fen, s1, s2, s1 - s2))

        self.assertEqual(len(mismatches), 0, f"Mirror symmetry mismatches found: {mismatches[:5]}")

    def test_bug5_threat_by_king_boolean_not_popcount(self):
        # Position 1: White King attacks ONE weak piece (Nc6)
        # Hanging adds 12/21, ThreatByKing adds 18/38 -> Total 30 / 59
        fen1 = "8/8/2n5/3K4/8/8/8/4k3 w - - 0 1"
        pb1, occ1, gs1 = parse_fen(fen1)
        res1 = evaluate_attacks_mobility_threats(
            pb1, occ1, 35, 4, np.uint64(0), np.uint64(0), np.uint8(0))
        mg_thr1, eg_thr1 = res1[10], res1[11]

        # Position 2: White King attacks TWO weak pieces (Nc6 and Be6)
        # In SF11, ThreatByKing is a boolean check (b = KING_ATTACKS & weak; if b: ...),
        # so ThreatByKing triggers at most ONCE (not multiplied by popcount).
        # Hanging pieces = 2 (+24 mg, +42 eg), ThreatByKing = 1 (+18 mg, +38 eg) -> Total 42 / 80.
        # (Prior to fix, ThreatByKing added 18*2 = 36 mg -> Total would be 60 mg)
        fen2 = "8/8/2n1b3/3K4/8/8/8/4k3 w - - 0 1"
        pb2, occ2, gs2 = parse_fen(fen2)
        res2 = evaluate_attacks_mobility_threats(
            pb2, occ2, 35, 4, np.uint64(0), np.uint64(0), np.uint8(0))
        mg_thr2, eg_thr2 = res2[10], res2[11]

        self.assertEqual(mg_thr1, 30)
        self.assertEqual(eg_thr1, 59)
        self.assertEqual(mg_thr2, 42)
        self.assertEqual(eg_thr2, 80)
        # Crucial check: difference between 2 weak pieces and 1 weak piece is ONLY hanging bonus (12 mg), NOT doubling ThreatByKing
        self.assertEqual(mg_thr2 - mg_thr1, 12)
        self.assertEqual(eg_thr2 - eg_thr1, 21)

    def test_bug6_minor_behind_enemy_pawn(self):
        # White Knight on d4 behind an ENEMY pawn on d5
        # 8/8/8/3p4/3N4/8/8/k3K3 w - - 0 1
        fen_behind_enemy = "8/8/8/3p4/3N4/8/8/k3K3 w - - 0 1"
        pb1, occ1, gs1 = parse_fen(fen_behind_enemy)
        res1 = evaluate_attacks_mobility_threats(
            pb1, occ1, 4, 56, np.uint64(0), np.uint64(0), np.uint8(0))
        mg_mob1, eg_mob1 = res1[8], res1[9]

        # Contrast with White Knight on d4 without pawn on d5
        # 8/8/8/8/3N4/8/8/k3K3 w - - 0 1
        fen_no_pawn = "8/8/8/8/3N4/8/8/k3K3 w - - 0 1"
        pb2, occ2, gs2 = parse_fen(fen_no_pawn)
        res2 = evaluate_attacks_mobility_threats(
            pb2, occ2, 4, 56, np.uint64(0), np.uint64(0), np.uint8(0))
        mg_mob2, eg_mob2 = res2[8], res2[9]

        # Difference in mobility must reflect MINOR_BEHIND_PAWN
        self.assertEqual(mg_mob1 - mg_mob2, MINOR_BEHIND_PAWN[0])
        self.assertEqual(eg_mob1 - eg_mob2, MINOR_BEHIND_PAWN[1])

    def test_bug7_weak_queen_multiple_snipers_file(self):
        # White Queen on e1, Black Rooks on e5 and e8, Intervening blocker on e3
        # 4r3/8/8/4r3/8/4N3/8/4Q2k w - - 0 1
        fen = "4r3/8/8/4r3/8/4N3/8/4Q2k w - - 0 1"
        pb, occ, gs = parse_fen(fen)
        res = evaluate_attacks_mobility_threats(
            pb, occ, 55, 60, np.uint64(0), np.uint64(0), np.uint8(0))
        mg_mob, eg_mob = res[8], res[9]

        # If weak queen wasn't detected, mobility would not include -WEAK_QUEEN
        # Verify that weak queen penalty is triggered
        # Contrast with blocker removed (queen directly under attack, not relative pin through 1 blocker)
        fen_clear = "4r3/8/8/4r3/8/8/8/4Q2k w - - 0 1"
        pb_clear, occ_clear, gs_clear = parse_fen(fen_clear)
        res_clear = evaluate_attacks_mobility_threats(
            pb_clear, occ_clear, 55, 60, np.uint64(0), np.uint64(0), np.uint8(0))
        # With blocker present, WeakQueen penalty is applied
        # Difference in mobility includes WEAK_QUEEN deduction
        self.assertLess(mg_mob, res_clear[8])

    def test_bug8_king_attacks_count_in_zone(self):
        # White Knight on f6 attacking Black King on h8 (king zone and adjacent squares)
        # 7k/8/5N2/8/8/8/8/4K3 w - - 0 1
        fen_att = "7k/8/5N2/8/8/8/8/4K3 w - - 0 1"
        pb1, occ1, gs1 = parse_fen(fen_att)
        res1 = evaluate_attacks_mobility_threats(
            pb1, occ1, 4, 63, np.uint64(0), np.uint64(0), np.uint8(0))
        # b_attacker_count is index 26, b_king_attacks_count is index 27
        b_ac1, b_kac1 = res1[26], res1[27]
        self.assertEqual(b_ac1, 1)
        self.assertGreater(b_kac1, 0)

        # White Knight on a1 far away from Black King on h8
        fen_far = "7k/8/8/8/8/8/8/N3K3 w - - 0 1"
        pb2, occ2, gs2 = parse_fen(fen_far)
        res2 = evaluate_attacks_mobility_threats(
            pb2, occ2, 4, 63, np.uint64(0), np.uint64(0), np.uint8(0))
        b_ac2, b_kac2 = res2[26], res2[27]
        self.assertEqual(b_ac2, 0)
        self.assertEqual(b_kac2, 0)


if __name__ == "__main__":
    unittest.main()
