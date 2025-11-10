# chess_engine/board.py
from chess_engine.fen_parser import parse_fen

class Board:
    def __init__(self, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.piece_bbs, self.occupancy_bbs, self.game_state = parse_fen(fen)

    def __str__(self):
        # (A basic string representation could be added here if needed for debugging)
        return "Board State"
