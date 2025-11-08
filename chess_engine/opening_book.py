# chess_engine/opening_book.py

import numpy as np
import os
import sys

# Polyglot book entry struct:
# key: uint64
# move: uint16
# weight: uint16
# learn: uint32
# Total: 16 bytes
POLYGLOT_DTYPE = np.dtype([
    ('key', '>u8'),      # Big-endian 64-bit unsigned integer
    ('move', '>u2'),     # Big-endian 16-bit unsigned integer
    ('weight', '>u2'),   # Big-endian 16-bit unsigned integer
    ('learn', '>u4'),    # Big-endian 32-bit unsigned integer
])

class OpeningBook:
    """
    A class to handle reading and looking up moves from a Polyglot opening book.
    """
    def __init__(self, book_path):
        """
        Initializes the OpeningBook by loading the .bin file.

        Args:
            book_path (str): The path to the polyglot.bin file.
        """
        self.book = None
        if os.path.exists(book_path):
            try:
                self.book = np.fromfile(book_path, dtype=POLYGLOT_DTYPE)
            except IOError as e:
                print(f"info string Error opening or reading book file: {e}", file=sys.stderr)

    def lookup(self, zobrist_key, board_state):
        """
        Looks up a zobrist key in the opening book and returns a list of possible moves.

        Args:
            zobrist_key (np.uint64): The zobrist key of the current position.
            board_state: The current board state tuple.

        Returns:
            A list of (move, weight) tuples, or an empty list if no match is found.
        """
        if self.book is None:
            return []

        # Find the first entry with a key >= zobrist_key
        # 'key' is the name of the field in our structured array
        first_match_idx = np.searchsorted(self.book['key'], zobrist_key)

        moves = []
        # Iterate through all entries that match the key
        for i in range(first_match_idx, len(self.book)):
            entry = self.book[i]
            if entry['key'] == zobrist_key:
                engine_move = _convert_polyglot_move_to_engine_move(entry['move'], board_state)
                moves.append((engine_move, entry['weight']))
            else:
                # Since the book is sorted, we can stop as soon as the key no longer matches
                break

        return moves

from .move import encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION, PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN
def _convert_polyglot_move_to_engine_move(poly_move: np.uint16, board_state) -> np.uint16:
    # Polyglot move format:
    # 15 14 13 12 11 10 9  8  7  6  5  4  3  2  1  0
    # promo. | to_rank | to_file | from_rank | from_file
    # (3 bits)(3 bits)  (3 bits)  (3 bits)    (3 bits)

    from_file = poly_move & 0b111
    from_rank = (poly_move >> 3) & 0b111
    to_file = (poly_move >> 6) & 0b111
    to_rank = (poly_move >> 9) & 0b111
    promotion_piece_poly = (poly_move >> 12) & 0b111

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file

    promotion_piece = 0
    special_flag = SPECIAL_MOVE_FLAG_NORMAL

    # Polyglot promotion piece encoding: 1=N, 2=B, 3=R, 4=Q
    if promotion_piece_poly > 0:
        special_flag = SPECIAL_MOVE_FLAG_PROMOTION
        if promotion_piece_poly == 1:
            promotion_piece = PROMO_KNIGHT
        elif promotion_piece_poly == 2:
            promotion_piece = PROMO_BISHOP
        elif promotion_piece_poly == 3:
            promotion_piece = PROMO_ROOK
        elif promotion_piece_poly == 4:
            promotion_piece = PROMO_QUEEN

    # Note: Polyglot does not explicitly handle castling or en passant.
    # These moves must be inferred if necessary, but for lookup, this direct conversion is sufficient.

    return encode_move(from_sq, to_sq, promotion_piece, special_flag)
