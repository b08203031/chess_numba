
import numpy as np
import sys

PIECE_CHARS = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
"""
代表棋子的字符列表。
"""

def print_board(piece_bbs, name: str = "Board State"):
    """
    打印棋盤的可視化表示，顯示棋子字符。

    Args:
        piece_bbs (list): 12 個 uint64 位元棋盤的列表，每個對應一種棋子類型。
        name (str): 棋盤狀態的名稱。
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
    比較兩個移動列表並打印差異。

    Args:
        moves1 (list): 第一個移動列表。
        moves2 (list): 第二個移動列表。
        labels (tuple): 代表兩個列表名稱的元組。
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
    將調試信息打印到 stderr，並加上 "info string" 前綴以兼容 UCI 協議。

    Args:
        message (str): 要記錄的消息。
    """
    print(f"info string {message}", file=sys.stderr, flush=True)
