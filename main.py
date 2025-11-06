# main.py
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
    # =========================================================================
    # --- Test Case 1: Rook Moves (Center) ---
    # =========================================================================
    print("--- Testing Magic Bitboard Rook Moves ---")

    # --- Setup ---
    # Position: White Rook on d4, White Pawn on f4, Black Pawn on d7.
    rook_square_1 = 27  # d4
    white_pawn_square_1 = 29 # f4
    black_pawn_square_1 = 51 # d7

    white_rooks_1 = move_gen.BB_SQUARES[rook_square_1]
    white_pawns_1 = move_gen.BB_SQUARES[white_pawn_square_1]
    black_pawns_1 = move_gen.BB_SQUARES[black_pawn_square_1]

    white_pieces_1 = white_rooks_1 | white_pawns_1
    black_pieces_1 = black_pawns_1
    all_pieces_1 = white_pieces_1 | black_pieces_1

    print("\n--- Initial Position (Rook Test 1) ---")
    print("All Occupied Squares:")
    print_bitboard(all_pieces_1)

    # --- Generate and Test Rook Moves ---
    rook_moves_1 = move_gen.get_rook_moves(rook_square_1, all_pieces_1, white_pieces_1)

    print("\n--- Legal Rook Moves from d4 ---")
    print("Should attack up to and including d7, and up to but not including f4.")
    print_bitboard(rook_moves_1)

    # =========================================================================
    # --- Test Case 2: Bishop Moves ---
    # =========================================================================
    print("\n\n--- Testing Magic Bitboard Bishop Moves ---")

    # --- Setup ---
    # Position: White Bishop on c4, White Pawn on e6, Black Pawn on a6.
    bishop_square_2 = 26 # c4
    white_pawn_square_2 = 44 # e6
    black_pawn_square_2 = 40 # a6

    white_bishops_2 = move_gen.BB_SQUARES[bishop_square_2]
    white_pawns_2 = move_gen.BB_SQUARES[white_pawn_square_2]
    black_pawns_2 = move_gen.BB_SQUARES[black_pawn_square_2]

    white_pieces_2 = white_bishops_2 | white_pawns_2
    black_pieces_2 = black_pawns_2
    all_pieces_2 = white_pieces_2 | black_pieces_2

    print("\n--- Initial Position (Bishop Test) ---")
    print("All Occupied Squares:")
    print_bitboard(all_pieces_2)

    # --- Generate and Test Bishop Moves ---
    bishop_moves_2 = move_gen.get_bishop_moves(bishop_square_2, all_pieces_2, white_pieces_2)

    print("\n--- Legal Bishop Moves from c4 ---")
    print("Should attack up to and including a6, and up to but not including e6.")
    print_bitboard(bishop_moves_2)

    # =========================================================================
    # --- Test Case 3: Rook Moves (Corner) ---
    # =========================================================================
    print("\n\n--- Testing Magic Bitboard Rook Moves (Corner Case) ---")

    # --- Setup ---
    # Position: White Rook on a1, White Pawn on a4, Black Pawn on e1.
    rook_square_3 = 0   # a1
    white_pawn_square_3 = 24  # a4
    black_pawn_square_3 = 4   # e1

    white_rooks_3 = move_gen.BB_SQUARES[rook_square_3]
    white_pawns_3 = move_gen.BB_SQUARES[white_pawn_square_3]
    black_pawns_3 = move_gen.BB_SQUARES[black_pawn_square_3]

    white_pieces_3 = white_rooks_3 | white_pawns_3
    black_pieces_3 = black_pawns_3
    all_pieces_3 = white_pieces_3 | black_pieces_3

    print("\n--- Initial Position (Rook Test 2) ---")
    print("All Occupied Squares:")
    print_bitboard(all_pieces_3)

    # --- Generate and Test Rook Moves ---
    rook_moves_3 = move_gen.get_rook_moves(rook_square_3, all_pieces_3, white_pieces_3)

    print("\n--- Legal Rook Moves from a1 ---")
    print("Should attack up to e1 (inclusive) and up to a3 (exclusive of a4).")
    print_bitboard(rook_moves_3)
