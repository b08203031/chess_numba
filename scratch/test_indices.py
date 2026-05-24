import os
import sys

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

from chess_engine.nnue.ml_eval.train import fen_to_halfka_indices

def decode_feature(f):
    if f == 22528:
        return "PADDING"
    bucket = f // 704
    rem = f % 704
    channel = rem // 64
    sq = rem % 64
    
    # channels:
    # 0: own Pawn, 1: opp Pawn
    # 2: own Knight, 3: opp Knight
    # 4: own Bishop, 5: opp Bishop
    # 6: own Rook, 7: opp Rook
    # 8: own Queen, 9: opp Queen
    # 10: King
    channel_names = {
        0: "own P", 1: "opp P",
        2: "own N", 3: "opp N",
        4: "own B", 5: "opp B",
        6: "own R", 7: "opp R",
        8: "own Q", 9: "opp Q",
        10: "King"
    }
    
    sq_name = f"{chr(ord('a') + sq % 8)}{sq // 8 + 1}"
    return f"Bucket {bucket}, {channel_names.get(channel, f'Ch{channel}')} at {sq_name} (sq={sq})"

def main():
    positions = {
        "Starting pos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "White missing b1 knight": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
        "White missing d1 queen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    }
    
    for name, fen in positions.items():
        print(f"\n=== {name} ===")
        stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
        print("Active features (STM perspective):")
        for f in stm_idx:
            if f != 22528:
                print(f"  {f:<5}: {decode_feature(f)}")

if __name__ == "__main__":
    main()
