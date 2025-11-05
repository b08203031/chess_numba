# main.py
from chess_engine.board import Board
import chess_engine.move_generator as move_gen
import numpy

def print_bitboard(bb: numpy.uint64):
    """Prints a bitboard in a human-readable 8x8 format."""
    print("\n  a b c d e f g h")
    print(" +-----------------+")
    for rank in range(7, -1, -1):
        print(f"{rank + 1}|", end=" ")
        for file in range(8):
            square = rank * 8 + file
            mask = numpy.uint64(1) << numpy.uint64(square)
            print("X" if bb & mask else ".", end=" ")
        print(f"|")
    print(" +-----------------+")
    print(f"Decimal value: {bb}\n")


if __name__ == "__main__":
    board = Board()
    print("Initial Board State:")
    board.print_board()

    # --- Test python-chess style sliding piece attacks ---
    print("--- Testing python-chess Style Sliding Attacks ---")

    # Create a custom board setup for testing rooks
    test_board = Board()
    test_board.P = numpy.uint64(0) # Clear pawns for simplicity
    test_board.p = numpy.uint64(0)

    # Place a white rook on d4 (square 27)
    # Place some blockers: a black pawn on d7 (sq 51) and a white pawn on f4 (sq 29)
    test_board.R = move_gen.BB_SQUARES[27]
    test_board.p = move_gen.BB_SQUARES[51]
    test_board.P = move_gen.BB_SQUARES[29]

    print("--- Rook Test ---")
    test_board.print_board()

    # Get all pieces on the test board
    occupied = test_board.all_pieces

    # Get rook attacks from d4
    rook_attacks = move_gen.get_rook_attacks(27, occupied)

    print("--- Rook Attacks from d4 ---")
    print_bitboard(rook_attacks)
