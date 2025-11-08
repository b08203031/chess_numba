
import numpy as np
import sys

PIECE_CHARS = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
"""
A list of characters representing the pieces.
"""

def print_board(piece_bbs, name: str = "Board State"):
    """
    Prints a visual representation of the board showing piece characters.

    Args:
        piece_bbs: A tuple of 12 uint64 bitboards, one for each piece type.
        name: The name of the board state.
    """
    print(f"\n--- {name} ---")
    print("   a b c d e f g h")
    print("  +-----------------+")

    board_repr = ['.'] * 64
    for piece_idx, bb in enumerate(piece_bbs):
        while bb > 0:
            sq = int(np.log2(bb & -bb))
            board_repr[sq] = PIECE_CHARS[piece_idx]
            bb &= bb - np.uint64(1)

    for rank in range(7, -1, -1):
        print(f"{rank + 1} |", end="")
        for file in range(8):
            sq = rank * 8 + file
            print(f" {board_repr[sq]}", end="")
        print(" |")
    print("  +-----------------+")


def compare_move_lists(moves1, moves2, labels=("List 1", "List 2")):
    """
    Compares two lists of moves and prints the differences.

    Args:
        moves1: The first list of moves.
        moves2: The second list of moves.
        labels: A tuple of two strings representing the names of the two lists.
    """
    set1 = set(moves1)
    set2 = set(moves2)

    missing_from_2 = sorted(list(set1 - set2))
    extra_in_2 = sorted(list(set2 - set1))

    print(f"\n--- Comparing Move Lists: {labels[0]} vs {labels[1]} ---")

    if not missing_from_2 and not extra_in_2:
        print("✓ Lists match perfectly!")
        return

    if missing_from_2:
        print(f"✗ Moves in '{labels[0]}' but missing from '{labels[1]}':")
        for move in missing_from_2:
            print(f"  - {move}")

    if extra_in_2:
        print(f"✗ Moves in '{labels[1]}' but not in '{labels[0]}':")
        for move in extra_in_2:
            print(f"  - {move}")

    print("-------------------------------------------------")


def log_info(message):
    """
    Prints debug information to stderr, prefixed with "info string" for UCI compatibility.
    """
    print(f"info string {message}", file=sys.stderr, flush=True)
