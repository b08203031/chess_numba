import unittest
import numpy as np
import numba
from chess_engine.see import see
from chess_engine.constants import (
    BB_SQUARES
)
from chess_engine.engine_types import (
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
)
from chess_engine.fen_parser import parse_fen
from chess_engine.bitboard_utils import find_piece_type_on_square

class TestSEE(unittest.TestCase):

    def setUp(self):
        # Warmup JIT
        pass

    def test_simple_capture(self):
        # White Rook takes Black Pawn, undefended
        fen = "8/8/8/3p4/3R4/8/8/8 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        # d4 (R) takes d5 (p)
        from_sq = np.uint8(27) # d4
        to_sq = np.uint8(35)   # d5
        side = np.uint8(WHITE)

        # Value: Pawn (100) - 0 (no recapture) = 100
        val = see(piece_bbs, occupancy_bbs, side, from_sq, to_sq)
        self.assertEqual(val, 100)

    def test_exchange_winning(self):
        # White Knight takes Black Pawn, protected by Knight
        # Exchange: +100 (P) - 320 (N) = -220? No.
        # Sequence:
        # 1. W N takes P (+100)
        # 2. B N takes N (-320) -> Net for White: -220

        fen = "8/8/8/3p4/3N4/4n3/8/8 w - - 0 1" # Black N at e3 protecting d5
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        # d4 (N) takes d5 (p)
        from_sq = np.uint8(27) # d4
        to_sq = np.uint8(35)   # d5
        side = np.uint8(WHITE)

        # Attackers: W-N
        # Defenders: B-N
        # 1. Capture: Gain 100.
        # 2. Recapture: Gain 320 (Knight).
        # Total for White: 100 - 320 = -220.
        val = see(piece_bbs, occupancy_bbs, side, from_sq, to_sq)
        self.assertEqual(val, -220)

    def test_battery_xray(self):
        # White Rook on d1, White Queen on d2. Black Pawn on d5.
        # Attackers: Q (d2), R (d1).
        # Sequence: QxP, ...
        # But Q is more valuable.
        # Let's say White Rook d4, Queen d1.
        fen = "8/8/8/3p4/3R4/8/3Q4/8 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        # d4 (R) takes d5 (p)
        from_sq = np.uint8(27) # d4
        to_sq = np.uint8(35)   # d5
        side = np.uint8(WHITE)

        # 1. RxP (+100). Occupancy d4 cleared.
        # 2. Revealed: Q attacks d5.
        # If there was a defender, Q would be next attacker.
        # Here no defender.
        val = see(piece_bbs, occupancy_bbs, side, from_sq, to_sq)
        self.assertEqual(val, 100)

    def test_pinned_piece_should_not_defend(self):
        # White R h1. Black K h8.
        # Black R h5 (Pinned).
        # White Q a5 captures b5 (Pawn).
        # Black R h5 attacks b5 (horizontally)
        # But Black R is pinned vertically to King.
        # If Black R recaptures on b5, King is in check from h1.
        # So SEE should ignore Black R.

        fen = "7k/8/8/Qp5r/8/8/8/7R w - - 0 1"
        # W R h1. B K h8. B R h5.
        # W Q a5. B P b5.
        # Q(a5) x P(b5).
        # B R(h5) attacks b5.
        # But B R is pinned.

        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        from_sq = np.uint8(32) # a5 (rank 4, file 0)
        to_sq = np.uint8(33)   # b5 (rank 4, file 1)
        side = np.uint8(WHITE)

        # Antares currently IGNORES pins, so it thinks Black R can recapture.
        # Q (900) takes P (100). Net +100.
        # Black R recaptures Q. Gain 900.
        # White loses 800.
        # SEE should be -800 if Pin is ignored.
        # SEE should be +100 if Pin is respected (R cannot move).

        val = see(piece_bbs, occupancy_bbs, side, from_sq, to_sq)
        print(f"Pinned piece test val: {val}")

        # We want it to be 100.
        self.assertEqual(val, 100)

    def test_en_passant_value(self):
        # White P e5. Black P d5 (moved d7-d5).
        # White captures d6 ep.
        # Victim is P on d5.
        # NOTE: In this specific FEN, d6 is defended by Black (e.g. Q d8, P c7).
        # So SEE = 100 (capture) - 100 (recapture) = 0.

        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        # e5 (36) captures d6 (43)
        from_sq = np.uint8(36)
        to_sq = np.uint8(43)
        side = np.uint8(WHITE)

        val = see(piece_bbs, occupancy_bbs, side, from_sq, to_sq)
        print(f"EP test val: {val}")
        self.assertEqual(val, 0)

if __name__ == '__main__':
    unittest.main()
