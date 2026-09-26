import unittest
import chess
import chess.pgn

class MockVar:
    def __init__(self, value=""):
        self._value = value
    def get(self):
        return self._value
    def set(self, val):
        self._value = val

class AssistantPgnEpTest(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.game_pgn = chess.pgn.Game.from_board(self.board)
        self.game_node = self.game_pgn
        self.ep_var = MockVar("-")
        self.castling_var = MockVar("KQkq")
        self.side_var = MockVar("w")
        self.best_move_arrow = None

    def get_pgn_string(self):
        if self.game_pgn is None:
            return ""
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return self.game_pgn.accept(exporter)

    def init_game_from_board(self, board):
        self.board = chess.Board(board.fen())
        self.game_pgn = chess.pgn.Game.from_board(self.board)
        self.game_node = self.game_pgn
        ep = chess.square_name(self.board.ep_square) if self.board.has_legal_en_passant() else "-"
        self.ep_var.set(ep)
        self.castling_var.set(self.board.fen().split()[2])
        self.side_var.set("w" if self.board.turn == chess.WHITE else "b")
        self.best_move_arrow = None

    def find_matching_legal_move(self, target_board):
        if target_board is None:
            return None
        target_pieces = target_board.board_fen() if hasattr(target_board, 'board_fen') else str(target_board).split()[0]
        for move in self.board.legal_moves:
            self.board.push(move)
            matched = (self.board.board_fen() == target_pieces)
            self.board.pop()
            if matched:
                return move
        return None

    def apply_move(self, move: chess.Move):
        if move not in self.board.legal_moves:
            return False
        self.board.push(move)
        if self.game_node is not None:
            self.game_node = self.game_node.add_variation(move)
        ep = chess.square_name(self.board.ep_square) if self.board.has_legal_en_passant() else "-"
        self.ep_var.set(ep)
        self.castling_var.set(self.board.fen().split()[2])
        self.side_var.set("w" if self.board.turn == chess.WHITE else "b")
        return True

    def test_e4_en_passant_is_dash(self):
        """1. e4 should NOT have legal en passant, ep_var should be '-'."""
        move = chess.Move.from_uci("e2e4")
        self.assertTrue(self.apply_move(move))
        
        self.assertEqual(self.ep_var.get(), "-", "After 1. e4, ep_var MUST be '-' because black cannot take EP")
        self.assertEqual(self.board.fen().split()[3], "-", "FEN 4th field should be '-'")
        self.assertEqual(self.side_var.get(), "b")
        self.assertIn("1. e4", self.get_pgn_string())

    def test_legal_en_passant_detection(self):
        """1. e4 Nf6 2. e5 d5 -> White can capture exd6, so ep_var MUST be 'd6'."""
        self.apply_move(chess.Move.from_uci("e2e4"))
        self.apply_move(chess.Move.from_uci("g8f6"))
        self.apply_move(chess.Move.from_uci("e4e5"))
        self.assertEqual(self.ep_var.get(), "-")
        
        # Black pushes d7->d5 adjacent to White pawn at e5
        self.apply_move(chess.Move.from_uci("d7d5"))
        self.assertEqual(self.ep_var.get(), "d6", "When pawn pushes d5 next to e5, White has legal EP capture exd6")
        self.assertEqual(self.board.fen().split()[3], "d6")
        
        # White captures en passant
        ep_move = chess.Move.from_uci("e5d6")
        self.assertTrue(self.board.is_en_passant(ep_move))
        self.apply_move(ep_move)
        self.assertEqual(self.ep_var.get(), "-")
        self.assertIsNone(self.board.piece_at(chess.D5), "Captured pawn on d5 should be removed")

    def test_find_matching_legal_move(self):
        """Test matching legal move from board piece placement."""
        target = chess.Board()
        target.push_san("e4")
        
        matched = self.find_matching_legal_move(target)
        self.assertIsNotNone(matched)
        self.assertEqual(matched, chess.Move.from_uci("e2e4"))

    def test_new_game_reset_and_start_pgn(self):
        """Opening a new game clears previous PGN and starts from the current board."""
        # Play some moves in old game
        self.apply_move(chess.Move.from_uci("e2e4"))
        self.apply_move(chess.Move.from_uci("e7e5"))
        self.assertIn("1. e4 e5", self.get_pgn_string())
        
        # New game from current board
        current_board = self.board.copy()
        self.init_game_from_board(current_board)
        
        # Old PGN moves are cleared, PGN starts from this FEN
        pgn_str = self.get_pgn_string()
        self.assertNotIn("1. e4 e5", pgn_str)
        self.assertIn("FEN", pgn_str)
        
        # Next move starts as the first move of the new PGN
        self.apply_move(chess.Move.from_uci("g1f3"))
        self.assertIn("Nf3", self.get_pgn_string())

if __name__ == '__main__':
    unittest.main()
