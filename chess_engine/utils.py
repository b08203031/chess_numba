import numpy as np
from .constants import PIECE_CHAR

def print_board(piece_bbs: tuple):
    """
    Prints a human-readable chess board from the piece bitboards tuple.

    Args:
        piece_bbs (tuple): A tuple of 12 numpy.uint64 bitboards,
                           representing the board state.
    """
    board = ['.'] * 64

    for piece_type, bb in enumerate(piece_bbs):
        while bb > 0:
            square_index = int(np.log2(bb & -bb))
            board[square_index] = PIECE_CHAR[piece_type]
            bb &= bb - 1

    print("\\n  +-----------------+")
    for rank in range(7, -1, -1):
        print(f"{rank + 1} |", end=" ")
        for file in range(8):
            square = rank * 8 + file
            # To correctly map board array to visual board, we need to flip the rank for display
            visual_square_index = (7-rank) * 8 + file
            print(f"{board[square]:>s}", end=" ")
        print("|")
    print("  +-----------------+")
    print("    a b c d e f g h\\n")
