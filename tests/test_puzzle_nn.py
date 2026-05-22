import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

import time
import sys
import json
import numpy as np
import shutil
from pathlib import Path

# --- Transposition Table Setup ---
from chess_engine.nnue.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table
transposition_table = create_transposition_table(TT_SIZE_MB)


def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories in the project.
    """
    # log_info("--- Clearing Numba Cache ---")
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))

    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            # log_info(f"Removing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
    # log_info("--- Cache Cleared ---")

from chess_engine.nnue.debug_utils import log_info

# Clear cache before importing the engine to avoid stale cache issues
# clear_numba_cache()

from chess_engine.nnue.fen_parser import parse_fen
from chess_engine.nnue.nnue.search import iterative_deepening_search
from chess_engine.nnue.move import move_to_uci
from chess_engine.nnue.core import SQUARE_TO_ALGEBRAIC

# Maximum search depth (Ply) for arrays like killer moves
from chess_engine.nnue.constants import MAX_PLY

puzzles = [
        {
            "name": "Â∞çÊ?chess.com 2900?ÇÈ??∞Á?ÔºåÂ??éÊ?Â∞ã‰??∞Ô?‰ΩÜ‰??úÂà∞Â∞±Áü•?ìÂ§ß??,
            "fen": "4rrk1/p2p1p1p/1p2p1p1/2nPq2P/2P5/4B3/PbB2PP1/1R1Q2KR w - - 2 20",
            "solution": ["b1b2", "h5g6"],
            "rating": "2400",
            "theme": "sacrifice"
        },
        {
            "name": "Lichess Puzzle 006eO",
            "fen": "8/8/2p5/1p1p1k2/3P4/1PP1pK2/8/8 w - - 4 65",
            "solution": "b3b4",
            "rating": "2186",
            "theme": "defensiveMove endgame equality oneMove pawnEndgame"
        },
        {
            "name": "Lichess Puzzle 005qG",
            "fen": "8/8/1p1k1p1p/3np3/2B2p2/PP1K1PP1/7P/8 w - - 0 37",
            "solution": "c4d5",
            "rating": "2244",
            "theme": "crushing defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle 005yO",
            "fen": "r1r2k2/ppq3bQ/4p2p/4n3/3p4/2P5/PBB2PPP/4R1K1 w - - 3 25",
            "solution": "b2a3",
            "rating": "2793",
            "theme": "advantage exposedKing middlegame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 006of",
            "fen": "r2qr2k/1pp2Qp1/1b4np/pP2P3/P4n2/B1N2N1P/5PP1/R3R1K1 b - - 0 20",
            "solution": "d8d3",
            "rating": "2500",
            "theme": "advantage kingsideAttack long middlegame"
        },
        {
            "name": "Lichess Puzzle 0078T",
            "fen": "rk5r/1b3R2/pp2p2q/4P2p/B6B/4p2P/PP4P1/5Q1K w - - 0 28",
            "solution": "f7b7",
            "rating": "2286",
            "theme": "attraction crushing defensiveMove exposedKing long middlegame queensideAttack sacrifice"
        },
        {
            "name": "Lichess Puzzle 000mr",
            "fen": "5r1k/5rp1/p7/1b2B2p/1P1P1Pq1/2R3Q1/P3p1P1/2R3K1 b - - 1 41",
            "solution": "f7f4",
            "rating": "1478",
            "theme": "crushing middlegame short"
        },
        {
            "name": "Lichess Puzzle 002rd",
            "fen": "r6k/q1p2p1p/1b2bPr1/p1ppP2Q/3P2p1/4B3/PP2NRPP/3R2K1 w - - 2 26",
            "solution": "e2f4",
            "rating": "1776",
            "theme": "crushing kingsideAttack long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 004d8",
            "fen": "8/4kr2/R2p4/1p1Pp3/5pp1/3K1P2/PPP5/8 w - - 0 40",
            "solution": "a6a7",
            "rating": "1730",
            "theme": "crushing endgame long rookEndgame"
        },
        {
            "name": "Lichess Puzzle 004sY",
            "fen": "8/2k3n1/K2p2p1/2pP2Pp/2P4P/7B/8/8 b - - 1 57",
            "solution": "c7d8",
            "rating": "2191",
            "theme": "crushing endgame short"
        },
        #------------------------
        {
            "name": "Lichess Puzzle 002LW",
            "fen": "3r1rk1/1b3pp1/3p4/p3nPPQ/4P3/3q1BN1/8/2R2RK1 w - - 2 29",
            "solution": "f5f6",
            "rating": "2489",
            "theme": "advantage middlegame short"
        },
        {
            "name": "Lichess Puzzle 005f3",
            "fen": "r5k1/2p1pp2/pp4p1/1q5r/5P2/2QP2R1/PP6/1K4R1 w - - 1 33",
            "solution": "g3g6",
            "rating": "1986",
            "theme": "crushing endgame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 005wy",
            "fen": "1r6/pp2kppQ/2n1p1n1/3p2P1/5P2/2PqP3/PP1N4/2KR3R b - - 4 27",
            "solution": "c6b4",
            "rating": "1842",
            "theme": "long mate mateIn3 middlegame queensideAttack sacrifice"
        },
        {
            "name": "Lichess Puzzle 0068D",
            "fen": "7r/pppk4/2pN1r2/8/3P2p1/2P5/PP2RPP1/4R1K1 b - - 0 26",
            "solution": "f6h6",
            "rating": "1901",
            "theme": "crushing endgame veryLong"
        },
        {
            "name": "Lichess Puzzle 007eS",
            "fen": "6k1/p4p2/1p5p/4r3/P3B3/1P2KP2/2P3PP/8 b - - 1 29",
            "solution": "f7f5",
            "rating": "1230",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle 009De",
            "fen": "r1q4r/2p1kP2/3p4/2pPp3/p1P1Pb2/P1NB3P/1P3K2/R2Q1R2 b - - 0 23",
            "solution": "c8h3",
            "rating": "2000",
            "theme": "advantage exposedKing middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 00AHY",
            "fen": "r6k/3NR1p1/4n2p/5b1P/p7/6R1/8/6K1 b - - 0 43",
            "solution": "a4a3",
            "rating": "1371",
            "theme": "advantage endgame long"
        },
        {
            "name": "Original Puzzle",
            "fen": "5qk1/2n2ppp/5b2/pp1r4/5Q2/PB2R3/3B1PPP/6K1 w - - 0 1",
            "solution": "f4c7",
            "rating": "1500",
            "theme": "Original Puzzle"
        },
        {
            "name": "New Puzzle - Queen Sacrifice",
            "fen": "5qk1/2Q2ppp/5b2/pp6/8/PB2R3/3r1PPP/6K1 w - - 0 1",
            "solution": "c7f7",
            "rating": "1600",
            "theme": "queen sacrifice"
        },
        {
            "name": "Lichess Puzzle 8",
            "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25",
            "solution": "e6e7",
            "rating": "1736",
            "theme": "crushing hangingPiece long middlegame"
        },
        {
            "name": "Lichess Puzzle 0000D",
            "fen": "5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 3 27",
            "solution": "f8d8",
            "rating": "1513",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle 000Vc",
            "fen": "8/8/4k1p1/2KpP2P/5P2/8/8/8 b - - 0 53",
            "solution": "g6h5",
            "rating": "1495",
            "theme": "crushing endgame long pawnEndgame"
        },
        {
            "name": "Lichess Puzzle 000Zo",
            "fen": "4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 35",
            "solution": "e8e1",
            "rating": "1652",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 000rO",
            "fen": "3R4/8/8/KB2b3/1p6/1P2k3/3p4/8 b - - 0 58",
            "solution": "e5c7",
            "rating": "1039",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 000tp",
            "fen": "4r3/5pk1/1Q3np1/3p3p/2q5/P4N1P/1P3RP1/7K b - - 0 34",
            "solution": "f6e4",
            "rating": "2051",
            "theme": "crushing endgame short trappedPiece"
        },
        {
            "name": "Lichess Puzzle 0018S",
            "fen": "2kr3r/p4p2/1p2p2p/1N1p2p1/3Q4/1P1P4/2q2PPP/5RK1 w - - 0 21",
            "solution": "d4a1",
            "rating": "2652",
            "theme": "advantage endgame pin short"
        },
        {
            "name": "Lichess Puzzle 001Wz",
            "fen": "6k1/5ppp/r1p5/p1n1rP2/8/2P2N1P/2P3P1/3R2K1 w - - 0 22",
            "solution": "d1d8",
            "rating": "1128",
            "theme": "backRankMate endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 001XA",
            "fen": "2r2rk1/pbq1bppp/8/8/2p1N3/P1Bn2P1/2Q2PBP/1R3RK1 w - - 4 24",
            "solution": "b1b7",
            "rating": "1789",
            "theme": "crushing discoveredAttack long master middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 001h8",
            "fen": "2r3k1/2r4p/4p1p1/1p1q1pP1/p2P1P1Q/P6R/4bB2/2R3K1 w - - 6 35",
            "solution": "h4h7",
            "rating": "1801",
            "theme": "crushing deflection kingsideAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 001om",
            "fen": "5r1k/pp4pp/5p2/1BbQp1r1/7K/7P/1PP3P1/3R3R b - - 3 26",
            "solution": "c5f2",
            "rating": "991",
            "theme": "mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle 001u3",
            "fen": "2r3k1/p4pp1/Qq2p2p/b1Np4/2nP1P2/4P1P1/5K1P/2B1N3 w - - 4 34",
            "solution": "a6c8",
            "rating": "2175",
            "theme": "advantage hangingPiece middlegame short"
        },
        {
            "name": "Lichess Puzzle 001w5",
            "fen": "1rb3k1/q4rP1/4p2p/3p3p/3P1P2/2P5/2QK3P/3R2R1 w - - 1 30",
            "solution": "c2h7",
            "rating": "1444",
            "theme": "advancedPawn attraction mate mateIn2 middlegame promotion short"
        },
        {
            "name": "Lichess Puzzle 001xl",
            "fen": "8/4R3/p4kpp/3B4/5q2/8/5P1P/6K1 w - - 6 41",
            "solution": "e7f7",
            "rating": "1250",
            "theme": "advantage endgame master masterVsMaster short skewer superGM"
        },
        {
            "name": "Lichess Puzzle 002Cw",
            "fen": "r7/2p3rk/p2p1q1p/Pp1P4/1P2P3/2PQ4/6R1/R5K1 w - - 3 29",
            "solution": "e4e5",
            "rating": "998",
            "theme": "crushing discoveredAttack endgame short"
        },
        {
            "name": "Lichess Puzzle 2.00E+04",
            "fen": "8/8/kp6/p4pQp/q7/7P/3r2P1/4R2K w - - 0 49",
            "solution": "g5d2",
            "rating": "974",
            "theme": "crushing endgame hangingPiece oneMove"
        },
        {
            "name": "Lichess Puzzle 002VP",
            "fen": "8/6p1/2B2n2/3b2k1/3B4/6K1/4P3/8 w - - 5 45",
            "solution": "d4f6",
            "rating": "1410",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 002uV",
            "fen": "r2r2k1/1p2qppp/2n1p3/5Q2/p2P4/P4N2/BP3PPP/2R1R1K1 b - - 0 20",
            "solution": "e6f5",
            "rating": "1629",
            "theme": "advantage middlegame short"
        },
        {
            "name": "Lichess Puzzle 002xh",
            "fen": "2nk4/8/2PBp3/1pK1P1p1/1P4Pn/8/8/8 w - - 3 43",
            "solution": "c5b5",
            "rating": "2074",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 0039T",
            "fen": "1r5r/p3kp2/4p2p/4P3/R4Pp1/6P1/P1P4P/4K2R b K - 2 25",
            "solution": "b8b1",
            "rating": "1214",
            "theme": "crushing defensiveMove endgame long rookEndgame skewer"
        },
        {
            "name": "Lichess Puzzle 003IX",
            "fen": "8/3pk3/R7/1R2PK1p/2PPn1r1/8/8/8 b - - 0 43",
            "solution": "e4g3",
            "rating": "1628",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle 003Jb",
            "fen": "6k1/Q2bqr1p/2rpp1pR/p7/Pp2P3/1B3P2/1PP3P1/2KR4 b - - 7 22",
            "solution": "e7g5",
            "rating": "1028",
            "theme": "advantage fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 003Tx",
            "fen": "2r5/pR5p/5p1k/4p3/4R3/B4nPP/PP3P2/1K6 b - - 0 27",
            "solution": "f3d2",
            "rating": "1648",
            "theme": "backRankMate endgame fork mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 003eP",
            "fen": "6k1/r1b1q3/2p3p1/2Pp4/1P2p1n1/2B1P3/NQ6/2K4R w - - 2 37",
            "solution": "h1h8",
            "rating": "1097",
            "theme": "crushing exposedKing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle 003jH",
            "fen": "rn3rk1/p5pp/3N4/4np1q/5Q2/1P6/PB1P1KP1/2R4R b - - 1 25",
            "solution": "e5d3",
            "rating": "1099",
            "theme": "crushing fork long middlegame"
        },
        {
            "name": "Lichess Puzzle 003jv",
            "fen": "7R/1p2k2p/p2n2p1/4K3/8/6P1/P6P/8 b - - 11 37",
            "solution": "d6f7",
            "rating": "986",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 003nQ",
            "fen": "6rk/pp6/2n5/3ppn1p/3p4/2P2P1q/PP3QNB/R5RK b - - 3 29",
            "solution": "f5g3",
            "rating": "1307",
            "theme": "crushing kingsideAttack master middlegame pin short"
        },
        # {
        #     "name": "Lichess Puzzle 003wQ",
        #     "fen": "2r2rk1/6pp/3Q1q2/8/3N1B2/6P1/PP1K3P/5R2 b - - 0 24",
        #     "solution": "f6d6",
        #     "rating": "1934",
        #     "theme": "advantage discoveredAttack middlegame pin short"
        # },
        {
            "name": "Lichess Puzzle 0042j",
            "fen": "3r2k1/4nppp/pq3b2/1p2p3/2r2P2/2P1NR2/PP1Q2BP/3R2K1 w - - 0 25",
            "solution": "d2d8",
            "rating": "652",
            "theme": "backRankMate mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle 0048h",
            "fen": "4r3/p5k1/2R4p/2Pp4/1P1pr1P1/P6P/8/3R3K b - - 0 35",
            "solution": "e4e1",
            "rating": "1137",
            "theme": "crushing endgame exposedKing long rookEndgame"
        },
        {
            "name": "Lichess Puzzle 004Ao",
            "fen": "4qk2/1b3R2/p7/1p2Q3/4P2P/P2P3K/2r5/3R4 b - - 0 41",
            "solution": "e8f7",
            "rating": "1735",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle 004LZ",
            "fen": "8/7R/5p2/p7/7P/2p5/3k2N1/1K6 b - - 0 48",
            "solution": "c3c2",
            "rating": "1188",
            "theme": "advancedPawn crushing defensiveMove deflection endgame long promotion"
        },
        {
            "name": "Lichess Puzzle 004Op",
            "fen": "2kr2r1/1bp4n/1pq1p2p/p1P5/1P3B2/P6P/5RP1/RB3QK1 b - - 4 26",
            "solution": "d8d1",
            "rating": "2164",
            "theme": "crushing deflection kingsideAttack middlegame pin sacrifice skewer veryLong"
        },
        {
            "name": "Lichess Puzzle 004RF",
            "fen": "5rk1/5ppp/1p6/1q3P1Q/2pp3P/6R1/6PK/8 w - - 0 31",
            "solution": "g3g7",
            "rating": "1788",
            "theme": "attraction crushing discoveredAttack endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle 004X6",
            "fen": "1r4k1/p4ppp/2Q5/3pq3/8/P6P/2PR1PP1/1R4K1 b - - 0 26",
            "solution": "b8b1",
            "rating": "1176",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 004iZ",
            "fen": "r2r2k1/2q1bpp1/3p3p/1ppn4/1P1BP3/P5Q1/4RPPP/R5K1 w - - 0 21",
            "solution": "g3g7",
            "rating": "1004",
            "theme": "kingsideAttack mate mateIn1 middlegame oneMove"
        },
        {
            "name": "Lichess Puzzle 004nd",
            "fen": "3q2k1/3r4/pp3p1Q/2b1n3/P3N3/2P5/1P4PP/R6K w - - 1 25",
            "solution": "e4f6",
            "rating": "898",
            "theme": "crushing fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 004sg",
            "fen": "6k1/p3b2p/1p1pP3/2P3P1/2np3B/P6P/3Q3K/8 b - - 0 38",
            "solution": "c4d2",
            "rating": "2612",
            "theme": "advantage clearance endgame hangingPiece long quietMove"
        },
        {
            "name": "Lichess Puzzle 004zI",
            "fen": "2q3k1/4br2/6pQ/1p1n2p1/7P/1P4P1/1B2PP2/6K1 w - - 0 28",
            "solution": "h6h8",
            "rating": "1397",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle 0050w",
            "fen": "5rk1/1p2p2p/p2p4/2pPb2R/2P1P3/1P1BKPrR/8/8 w - - 5 31",
            "solution": "h3g3",
            "rating": "853",
            "theme": "crushing endgame fork long"
        },
        {
            "name": "Lichess Puzzle 0055Y",
            "fen": "r1b2rk1/p3pp2/2B5/2Qpq3/3N2pp/4b3/2P2PPP/1R2K2R w K - 0 24",
            "solution": "f2e3",
            "rating": "2089",
            "theme": "advantage defensiveMove middlegame short"
        },
        {
            "name": "Lichess Puzzle 005HF",
            "fen": "3r1rk1/1p4Rp/p2bp3/1q2Np2/3P4/1P5Q/5PPP/4R1K1 b - - 0 27",
            "solution": "g8g7",
            "rating": "1347",
            "theme": "crushing defensiveMove hangingPiece middlegame short"
        },
        {
            "name": "Lichess Puzzle 005N7",
            "fen": "r6k/2q3pp/8/2p5/R1np4/7P/2PB1PP1/6K1 w - - 0 33",
            "solution": "a4a8",
            "rating": "600",
            "theme": "backRankMate endgame hangingPiece mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 005YX",
            "fen": "2rr4/5pk1/p1Q1N1pp/1p4q1/3pP3/1B1P4/PPP3PP/6RK b - - 0 25",
            "solution": "f7e6",
            "rating": "1440",
            "theme": "defensiveMove equality middlegame short"
        },
        {
            "name": "Lichess Puzzle 005ws",
            "fen": "8/8/4Kpp1/7p/3N2kP/8/8/8 b - - 3 62",
            "solution": "g6g5",
            "rating": "2281",
            "theme": "crushing endgame knightEndgame master quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 005xu",
            "fen": "6r1/3k4/1K1P4/2P5/R7/5b2/8/8 w - - 1 69",
            "solution": "a4a7",
            "rating": "1956",
            "theme": "advancedPawn crushing endgame exposedKing long"
        },
        {
            "name": "Lichess Puzzle 0061g",
            "fen": "6k1/pp3pp1/2p1q1Pp/3b4/8/6Q1/PB3Pp1/3r1NK1 w - - 0 28",
            "solution": "g3b8",
            "rating": "773",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 0066C",
            "fen": "r2q1r1k/2p3p1/pb2Q2p/1p1B1n2/8/2P5/PP1B1PPP/3RR1K1 b - - 0 20",
            "solution": "b6f2",
            "rating": "1736",
            "theme": "advantage discoveredAttack kingsideAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 6.00E+01",
            "fen": "5rk1/R4pp1/1p5p/3Q4/1PPp2q1/3P2P1/5P2/4K3 b - - 0 34",
            "solution": "f8e8",
            "rating": "1606",
            "theme": "crushing deflection endgame veryLong"
        },
        {
            "name": "Lichess Puzzle 006HV",
            "fen": "1r6/5k2/2Q1pNp1/p5Pp/1p2P2P/2P4R/KP3P2/3q4 b - - 0 31",
            "solution": "b4b3",
            "rating": "1214",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 006NL",
            "fen": "1r6/k2qn1b1/p1N1p1p1/2PpPpN1/2n2P1P/p4B2/1PP2Q2/1K1R3R b - - 0 32",
            "solution": "e7c6",
            "rating": "2279",
            "theme": "advancedPawn advantage middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle 006RM",
            "fen": "1k1r3r/8/pp1n2p1/2q5/1Q6/3R2P1/PPP2P1P/3R2K1 w - - 5 30",
            "solution": "b4c5",
            "rating": "1556",
            "theme": "crushing long middlegame"
        },
        {
            "name": "Lichess Puzzle 006ia",
            "fen": "1r4k1/4bpp1/1rp2n1B/3p4/3P4/2N3P1/Pq2QPKP/2R1R3 b - - 0 24",
            "solution": "b2e2",
            "rating": "2058",
            "theme": "advantage long middlegame"
        },
        {
            "name": "Lichess Puzzle 006om",
            "fen": "1r3k2/5p1p/2p1pp2/P2n4/r3N3/P4PK1/2R2P1P/2R5 w - - 10 30",
            "solution": "e4c5",
            "rating": "1900",
            "theme": "crushing endgame fork long master"
        },
        {
            "name": "Lichess Puzzle 006wz",
            "fen": "2r5/4ppkp/6p1/1p6/1P6/P3B3/1br2PPP/1R1R2K1 w - - 3 23",
            "solution": "b1b2",
            "rating": "1466",
            "theme": "attraction crushing endgame fork long sacrifice"
        },
        {
            "name": "Lichess Puzzle 006x0",
            "fen": "8/3Q1kr1/8/1P2pB2/2Pp1n2/q2P3P/7K/5R2 b - - 8 49",
            "solution": "f7f6",
            "rating": "1341",
            "theme": "defensiveMove endgame equality short"
        },
        {
            "name": "Lichess Puzzle 006yP",
            "fen": "6R1/8/Kpk1p3/1p1pP3/6P1/PPr5/8/8 w - - 0 41",
            "solution": "g8c8",
            "rating": "806",
            "theme": "crushing endgame master rookEndgame short skewer"
        },
        {
            "name": "Lichess Puzzle 761",
            "fen": "3r2k1/1b4bR/p2P2p1/3p2N1/2p5/2P2N2/PP6/2K5 w - - 0 29",
            "solution": "h7g7",
            "rating": "1490",
            "theme": "attraction crushing endgame exposedKing fork long sacrifice"
        },
        {
            "name": "Lichess Puzzle 007Rn",
            "fen": "4r1k1/p4p1p/1p6/6B1/3P2n1/P4Q2/1P4P1/7K b - - 0 34",
            "solution": "e8e1",
            "rating": "994",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 007gO",
            "fen": "2r3rk/5p2/4p2p/4q3/1Q6/8/1P3PPP/2R2RK1 b - - 1 31",
            "solution": "e5g5",
            "rating": "2177",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 007ku",
            "fen": "r1bq3Q/1np3p1/p5k1/1p1Pp3/1Pn2BP1/2b2P2/P3K3/R4N2 w - - 0 36",
            "solution": "h8h5",
            "rating": "1700",
            "theme": "mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle 0088O",
            "fen": "7Q/2p5/1p2prp1/p4k1p/q4p1P/8/6RK/8 w - - 0 38",
            "solution": "g2g5",
            "rating": "1008",
            "theme": "crushing deflection endgame short"
        },
        {
            "name": "Lichess Puzzle 008GK",
            "fen": "1k6/ppp3p1/8/1P5p/8/P3n2P/2P1r1P1/B2rNRK1 w - - 5 32",
            "solution": "f1f8",
            "rating": "600",
            "theme": "backRankMate endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 008LT",
            "fen": "r4r1k/6p1/b3p1nN/p1pp4/1p3P1q/3P1Q1B/PPP2PK1/R6R w - - 1 27",
            "solution": "h6f7",
            "rating": "1752",
            "theme": "crushing kingsideAttack middlegame pin sacrifice short"
        },
        {
            "name": "Lichess Puzzle 008Nz",
            "fen": "6k1/2p2ppp/pnp5/B7/2P3PP/1P2PPR1/r3b2r/3R2K1 w - - 2 30",
            "solution": "d1d8",
            "rating": "622",
            "theme": "backRankMate mate mateIn1 middlegame oneMove"
        },
        {
            "name": "Lichess Puzzle 008P4",
            "fen": "8/4k3/1p1p4/rP2p1p1/P2nP1P1/3B4/3K4/R7 b - - 1 35",
            "solution": "d4b3",
            "rating": "795",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 008Sk",
            "fen": "8/6pp/3Bp2k/p2pP2P/P3p1PK/8/r4b2/5R2 w - - 3 38",
            "solution": "f1f2",
            "rating": "1950",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 008lc",
            "fen": "7k/pb1qn1rn/1p2R2Q/2p2p2/2Pp4/3B4/PP3P1P/4RK2 w - - 2 28",
            "solution": "h6g7",
            "rating": "1831",
            "theme": "attraction crushing exposedKing fork long middlegame"
        },
        {
            "name": "Lichess Puzzle 008nF",
            "fen": "2rq1rk1/7p/1n4pb/1R2Q3/pPpP1P2/P1B5/3N2PP/2R3K1 b - - 0 31",
            "solution": "f8e8",
            "rating": "2241",
            "theme": "crushing master middlegame short trappedPiece"
        },
        {
            "name": "Lichess Puzzle 008o6",
            "fen": "Q4rk1/p1p3p1/6P1/8/3P4/7P/q3r3/B4RK1 w - - 2 35",
            "solution": ["a8f8", "f1f8"],
            "rating": "1017",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle 008oX",
            "fen": "6k1/2R3pp/2p4q/1p1p4/3P4/P7/1PP2R2/1K1Nr3 w - - 4 33",
            "solution": "c7c8",
            "rating": "974",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 008tL",
            "fen": "8/7k/R6p/3p4/5r2/2P1p2P/P5P1/6K1 b - - 0 40",
            "solution": "e3e2",
            "rating": "1058",
            "theme": "advancedPawn crushing endgame master rookEndgame short"
        },
        {
            "name": "Lichess Puzzle 0092z",
            "fen": "2r3k1/3R1ppp/p1q5/2p2Q2/P7/7P/5PP1/6K1 w - - 4 27",
            "solution": "f5f7",
            "rating": "1088",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 009IO",
            "fen": "3r4/4kp1r/p2Np1p1/3bP3/P2n4/8/1P3RPP/5RK1 w - - 5 26",
            "solution": "f2f7",
            "rating": "1174",
            "theme": "hookMate mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle 009L0",
            "fen": "6k1/pb2r1pN/1n4Bp/3p4/1P2pR2/P7/5PPP/2rR2K1 b - - 3 30",
            "solution": "c1d1",
            "rating": "706",
            "theme": "backRankMate hangingPiece mate mateIn1 middlegame oneMove"
        },
        {
            "name": "Lichess Puzzle 009Os",
            "fen": "r2b2k1/1p3q1p/p2p4/3P2p1/2P1PR1r/6Q1/P2B3P/2R4K b - - 2 29",
            "solution": "h4f4",
            "rating": "1392",
            "theme": "crushing long middlegame"
        },
        {
            "name": "Lichess Puzzle 009f8",
            "fen": "8/1p4p1/pb2pp1p/3n1k2/3P4/P3BN1P/1P2KPP1/8 w - - 1 27",
            "solution": "f3h4",
            "rating": "1183",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 009lk",
            "fen": "1R6/6pk/2p4p/3bP2r/5B1P/2P1RqP1/P4P1Q/6K1 b - - 3 40",
            "solution": "f3d1",
            "rating": "952",
            "theme": "mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle 009oc",
            "fen": "5Q2/pbp3np/1p1pq1pk/1P6/P6P/6K1/8/8 w - - 0 33",
            "solution": "f8f4",
            "rating": "1500",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 009tE",
            "fen": "6k1/6pp/p1N5/1pP2bp1/5P2/8/PPP5/3K4 w - - 0 29",
            "solution": "c6e7",
            "rating": "669",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 009zR",
            "fen": "3Q4/p1p2ppp/4k3/8/5P2/4P3/Prqn2PP/3R1RK1 b - - 0 22",
            "solution": "d2f3",
            "rating": "1591",
            "theme": "discoveredAttack endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle 00A1H",
            "fen": "2r3k1/4brp1/2p3b1/2Pp1qNp/3B3P/2P5/PP3P1K/R2Q2R1 b - - 2 31",
            "solution": "f5f4",
            "rating": "2041",
            "theme": "crushing middlegame short"
        },
        {
            "name": "Lichess Puzzle 00A5v",
            "fen": "4r1k1/pp1qr1p1/7p/2pPR3/2P2p2/1P3P2/P2Q2PP/4R1K1 b - - 4 33",
            "solution": "e7e5",
            "rating": "681",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 00AB1",
            "fen": "8/7Q/3p1kp1/1p6/2b5/2q4P/5PPK/8 w - - 0 37",
            "solution": "h7h8",
            "rating": "1058",
            "theme": "crushing endgame short skewer"
        },
        {
            "name": "Lichess Puzzle 00Aae",
            "fen": "1R6/1P6/4pkp1/5p2/3P4/3KP2p/8/1r6 w - - 0 44",
            "solution": "b8f8",
            "rating": "1019",
            "theme": "advancedPawn clearance crushing endgame long promotion rookEndgame"
        },
    ]

def run_puzzle_test():
    """
    Runs a search test for each puzzle and prints statistics.
    """
    from chess_engine.nnue.engine_types import SearchContext

    # --- Argument Parsing ---
    target_fens = []
    run_failed_only = "--failed-only" in sys.argv
    nnue_only = "--nnue-only" in sys.argv
    
    # Find FEN arg if any
    fen_arg = ""
    for idx, arg in enumerate(sys.argv):
        if arg == "--fen" and idx + 1 < len(sys.argv):
            fen_arg = sys.argv[idx+1]
            break
            
    if nnue_only:
        print("?ïØÔ∏? Mode: [PURE NNUE ONLY] (Depth 1, no search)")
        depth = 1
        time_limit_ms = 999999 
    else:
        print("?? Mode: [FULL SEARCH] (Depth 40, 3s limit)")
        depth = 1
        time_limit_ms = 300000

    if run_failed_only:
        try:
            with open("failed_puzzles.json", "r") as f:
                failed_data = json.load(f)
                target_fens = [p["fen"] for p in failed_data]
                print(f"--- Running {len(target_fens)} previously failed puzzles ---")
        except FileNotFoundError:
            print("--- No failed_puzzles.json found. Running all puzzles. ---")
    elif fen_arg:
        target_fens = [fen_arg]
        print(f"--- Running specific FEN: {fen_arg} ---")


    # 1. Warm-up (triggers JIT compilation)
    print("--- WARMING UP (Compiling JIT functions) ---")
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p_bbs, o_bbs, g_state = parse_fen(fen_start)
    tt = create_transposition_table(16)
    killer_moves = np.zeros(256, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((3, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 12), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16)
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)
    pv_table = np.zeros((128, 128), dtype=np.uint16)
    ctx = SearchContext(tt, killer_moves, pv_table, history_table, butterfly_history, continuation_history, capture_history, pawn_history, pawn_correction_history, minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black)
    
    # Run a quick search
    iterative_deepening_search(p_bbs, o_bbs, g_state, 2, {'optimum_time': 0, 'maximum_time': 0}, ctx)
    print("--- WARM-UP COMPLETE ---\n")


    total_tests = len(puzzles)
    passed_tests = 0
    failed_tests = 0
    total_engine_time = 0
    total_nodes_searched = 0
    total_tt_hits = 0
    total_depth_sum = 0
    failed_puzzles = []

    script_start_time = time.time()

    relevant_puzzles = [p for p in puzzles if not target_fens or p['fen'] in target_fens]
    total_tests_to_run = len(relevant_puzzles)
    
    for i, puzzle in enumerate(puzzles):
        if target_fens and puzzle['fen'] not in target_fens:
            continue
            
        print(f"--- Running Test: {puzzle['name']} ---")
        print(f"FEN: {puzzle['fen']}")

        piece_bbs, occupancy_bbs, game_state = parse_fen(puzzle["fen"])
        
        # killer_moves, pv_table, and history_table are now initialized inside the loop
        killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
        pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
        history_table = np.zeros((12, 64), dtype=np.int32) # Note: history_table size is 12x64 in search.py
        butterfly_history = np.zeros((64, 64), dtype=np.int32)
        continuation_history = np.zeros((3, 12, 64, 12, 64), dtype=np.int16)
        capture_history = np.zeros((12, 64, 12), dtype=np.int32)
        pawn_correction_history = np.zeros(16384, dtype=np.int16)
        minor_correction_history = np.zeros(16384, dtype=np.int16)
        non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
        non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)
        
        # Clear TT before each search
        clear_transposition_table(transposition_table)
        ctx.killer_moves.fill(0)
        ctx.history_table.fill(0)
        
        search_context = SearchContext(
            transposition_table, killer_moves, pv_table, history_table,
            butterfly_history, continuation_history, capture_history, pawn_history,
            pawn_correction_history, minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black
        )

        start_time = time.time()

        # Unpack all 17 return values from the search function
        time_config = {
            'optimum_time': time_limit_ms,
            'maximum_time': time_limit_ms,
        }

        (best_move, best_eval, nodes_searched, quiescence_nodes, tt_hits, last_completed_depth) = iterative_deepening_search(
            piece_bbs, occupancy_bbs, game_state, depth, time_config, search_context, verbose=True
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_engine_time += elapsed_time

        total_nodes = nodes_searched + quiescence_nodes
        total_nodes_searched += total_nodes
        total_tt_hits += tt_hits
        total_depth_sum += last_completed_depth

        nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0
        q_node_percentage = (quiescence_nodes / total_nodes * 100) if total_nodes > 0 else 0
        tt_hit_rate = (tt_hits / total_nodes) * 100 if total_nodes > 0 else 0

        print("--- Search Statistics ---")
        print(f"Total Time: {elapsed_time:.2f}s")
        print(f"Nodes Searched: {total_nodes} ({nps} NPS)")
        print(f"- Quiescence Nodes: {quiescence_nodes} ({q_node_percentage:.1f}%)")
        print(f"Transposition Table Hits: {tt_hits} ({tt_hit_rate:.1f}%)")
        print("-------------------------")
        print("")
        print(f"FEN: {puzzle['fen']}")
        engine_move = move_to_uci(best_move)
        print(f"Best move found: {engine_move}")
        print(f"Score: {best_eval}")
        print(f"rating: {puzzle['rating']}")
        print(f"theme: {puzzle['theme']}")
        print("")

        is_correct = False
        if isinstance(puzzle['solution'], list):
            if engine_move in puzzle['solution']:
                is_correct = True
        elif engine_move == puzzle['solution']:
            is_correct = True

        if is_correct:
            passed_tests += 1
            print(f"Test PASSED: The engine found the correct move ({engine_move}).")
        else:
            failed_tests += 1
            failed_puzzles.append({
                "name": puzzle["name"],
                "fen": puzzle["fen"],
                "solution": puzzle["solution"],
                "engine_move": engine_move,
                "rating": puzzle["rating"],
                "theme": puzzle["theme"]
            })
            print(f"Test FAILED: The engine suggested {engine_move}, but the correct move is {puzzle['solution']}.")

        print(f"{i + 1} tests completed.")
        print("----------------------------------------")

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    avg_nps = int(total_nodes_searched / total_engine_time) if total_engine_time > 0 else 0
    avg_depth = total_depth_sum / total_tests if total_tests > 0 else 0
    tt_usage_percentage = (total_tt_hits / total_nodes_searched * 100) if total_nodes_searched > 0 else 0

    print("\n--- Test Summary ---")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total script execution time: {total_script_time:.2f} seconds")
    print(f"Total engine thinking time: {total_engine_time:.2f} seconds")
    print(f"Total nodes searched: {total_nodes_searched}")
    print(f"Transposition Table Usage: {total_tt_hits}/{total_nodes_searched} ({tt_usage_percentage:.2f}%)")
    print(f"Average NPS: {avg_nps}")
    print(f"Average search depth: {avg_depth:.2f}")

    if failed_puzzles:
        print("\n--- Failed Puzzles ---")
        for fp in failed_puzzles:
            print(f"Name: {fp['name']}")
            print(f"  FEN: {fp['fen']}")
            print(f"  Rating: {fp['rating']}")
            print(f"  Theme: {fp['theme']}")
            print(f"  Correct Answer: {fp['solution']}")
            print(f"  Engine's Answer: {fp['engine_move']}")
            print("-" * 20)

        # Save failed puzzles to JSON for analysis
        with open("failed_puzzles.json", "w") as f:
            json.dump(failed_puzzles, f, indent=4)
        print("Failed puzzles saved to failed_puzzles.json")


if __name__ == "__main__":
    run_puzzle_test()
