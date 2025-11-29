# chess_engine/opening_book.py

import numpy as np
import os
import sys

# Polyglot book entry struct:
# Polyglot 開局書項目結構：
# key: uint64
# move: uint16
# weight: uint16
# learn: uint32
# Total: 16 bytes
POLYGLOT_DTYPE = np.dtype([
    ('key', '>u8'),      # Big-endian 64-bit unsigned integer / 大端序 64 位無符號整數
    ('move', '>u2'),     # Big-endian 16-bit unsigned integer / 大端序 16 位無符號整數
    ('weight', '>u2'),   # Big-endian 16-bit unsigned integer / 大端序 16 位無符號整數
    ('learn', '>u4'),    # Big-endian 32-bit unsigned integer / 大端序 32 位無符號整數
])

class OpeningBook:
    """
    用於處理 Polyglot 開局書讀取和查找的類別。
    """
    def __init__(self, book_path):
        """
        通過加載 .bin 文件初始化 OpeningBook。

        Args:
            book_path (str): polyglot.bin 文件的路徑。
        """
        self.book = None
        if os.path.exists(book_path):
            try:
                self.book = np.fromfile(book_path, dtype=POLYGLOT_DTYPE)
            except IOError as e:
                print(f"info string Error opening or reading book file: {e}", file=sys.stderr)

    def lookup(self, zobrist_key, board_state):
        """
        在開局書中查找 Zobrist 鍵值，並返回可能的移動列表。

        Args:
            zobrist_key (np.uint64): 當前局面的 Zobrist 鍵值。
            board_state: 當前棋盤狀態元組。

        Returns:
            list: (move, weight) 元組的列表，如果未找到匹配項則返回空列表。
        """
        if self.book is None:
            return []

        # Find the first entry with a key >= zobrist_key
        # 'key' is the name of the field in our structured array
        # 尋找第一個鍵值 >= zobrist_key 的項目
        first_match_idx = np.searchsorted(self.book['key'], zobrist_key)

        moves = []
        # Iterate through all entries that match the key
        # 遍歷所有匹配該鍵值的項目
        for i in range(first_match_idx, len(self.book)):
            entry = self.book[i]
            if entry['key'] == zobrist_key:
                engine_move = _convert_polyglot_move_to_engine_move(entry['move'], board_state)
                moves.append((engine_move, entry['weight']))
            else:
                # Since the book is sorted, we can stop as soon as the key no longer matches
                # 由於書是排序的，一旦鍵值不匹配即可停止
                break

        return moves

from .move import encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION, PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN
def _convert_polyglot_move_to_engine_move(poly_move: np.uint16, board_state) -> np.uint16:
    """
    將 Polyglot 移動格式轉換為引擎內部的移動格式。

    Polyglot move format:
    15 14 13 12 11 10 9  8  7  6  5  4  3  2  1  0
    promo. | to_rank | to_file | from_rank | from_file
    (3 bits)(3 bits)  (3 bits)  (3 bits)    (3 bits)
    """

    to_file = poly_move & 0b111
    to_rank = (poly_move >> 3) & 0b111
    from_file = (poly_move >> 6) & 0b111
    from_rank = (poly_move >> 9) & 0b111
    promotion_piece_poly = (poly_move >> 12) & 0b111

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file

    promotion_piece = 0
    special_flag = SPECIAL_MOVE_FLAG_NORMAL

    # Polyglot promotion piece encoding: 1=N, 2=B, 3=R, 4=Q
    # Polyglot 升變棋子編碼：1=馬, 2=象, 3=車, 4=后
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
    # 注意：Polyglot 不顯式處理王車易位或吃過路兵。
    # 這些移動如果需要必須推斷，但對於查找，這種直接轉換已足夠。

    return encode_move(from_sq, to_sq, promotion_piece, special_flag)
