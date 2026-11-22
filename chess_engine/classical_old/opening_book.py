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

from .move import (
    encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION,
    SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN,
)

# Polyglot encodes castling as "king takes own rook"; map to the engine's king-destination encoding.
# Polyglot 以「王吃己方車」編碼易位；轉為引擎使用的王目標格編碼。
_POLYGLOT_CASTLING_MAP = {
    (4, 7): 6,     # e1h1 -> e1g1
    (4, 0): 2,     # e1a1 -> e1c1
    (60, 63): 62,  # e8h8 -> e8g8
    (60, 56): 58,  # e8a8 -> e8c8
}


def _piece_bit(board_state, piece_index, sq) -> bool:
    return bool((int(board_state[0][piece_index]) >> sq) & 1)


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

    # Polyglot encodes castling as king-takes-rook and does not flag en passant, so both must
    # be inferred from the current position (Bug #5). Without this, the engine would emit illegal
    # UCI such as `bestmove e1h1` straight from the book.
    # Polyglot 以王吃車編碼易位且不標記吃過路兵，必須依當前局面推斷。
    if special_flag == SPECIAL_MOVE_FLAG_NORMAL and board_state is not None:
        from_sq_i = int(from_sq)
        to_sq_i = int(to_sq)
        castle_to = _POLYGLOT_CASTLING_MAP.get((from_sq_i, to_sq_i))
        if castle_to is not None and (_piece_bit(board_state, 5, from_sq_i) or _piece_bit(board_state, 11, from_sq_i)):
            to_sq = castle_to
            special_flag = SPECIAL_MOVE_FLAG_CASTLING
        else:
            is_pawn_move = _piece_bit(board_state, 0, from_sq_i) or _piece_bit(board_state, 6, from_sq_i)
            diagonal_to_empty = (
                from_file != to_file
                and ((int(board_state[1][2]) >> to_sq_i) & 1) == 0
            )
            if is_pawn_move and diagonal_to_empty:
                special_flag = SPECIAL_MOVE_FLAG_EN_PASSANT

    return encode_move(int(from_sq), int(to_sq), promotion_piece, special_flag)
