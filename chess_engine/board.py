# chess_engine/board.py

import numpy

class Board:
    """
    Represents the chessboard using bitboards.

    A bitboard is a 64-bit unsigned integer where each bit corresponds to a square
    on the board. A '1' indicates a piece is on that square, and a '0' indicates
    it's empty. We follow the standard convention where A1 is the least significant
    bit (LSB, bit 0) and H8 is the most significant bit (MSB, bit 63).
    """
    def __init__(self):
        """
        Initializes the bitboards to the standard chess starting position.
        """
        # White pieces bitboards
        self.P = numpy.uint64(0b11111111 << 8)
        self.R = numpy.uint64(0b10000001)
        self.N = numpy.uint64(0b01000010)
        self.B = numpy.uint64(0b00100100)
        self.Q = numpy.uint64(0b00001000)
        self.K = numpy.uint64(0b00010000)

        # Black pieces bitboards
        self.p = numpy.uint64(0b11111111 << 48)
        self.r = numpy.uint64(0b10000001 << 56)
        self.n = numpy.uint64(0b01000010 << 56)
        self.b = numpy.uint64(0b00100100 << 56)
        self.q = numpy.uint64(0b00001000 << 56)
        self.k = numpy.uint64(0b00010000 << 56)

    def print_board(self):
        """
        Displays the board in a human-readable format.
        The board is printed from the perspective of white, with rank 8 at the top
        and rank 1 at the bottom.
        """
        print("\n  a b c d e f g h")
        print(" +-----------------+")
        for rank in range(7, -1, -1):
            print(f"{rank + 1}|", end=" ")
            for file in range(8):
                square = rank * 8 + file
                piece = "."

                # Create a bitmask for the current square
                mask = numpy.uint64(1) << numpy.uint64(square)

                # Check which bitboard has a piece on the current square
                if self.P & mask: piece = "P"
                elif self.R & mask: piece = "R"
                elif self.N & mask: piece = "N"
                elif self.B & mask: piece = "B"
                elif self.Q & mask: piece = "Q"
                elif self.K & mask: piece = "K"
                elif self.p & mask: piece = "p"
                elif self.r & mask: piece = "r"
                elif self.n & mask: piece = "n"
                elif self.b & mask: piece = "b"
                elif self.q & mask: piece = "q"
                elif self.k & mask: piece = "k"

                print(f"{piece}", end=" ")
            print(f"|")
        print(" +-----------------+")
        print("  a b c d e f g h\n")
