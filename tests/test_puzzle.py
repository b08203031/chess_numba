import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import numpy as np
import shutil
from pathlib import Path

# --- Transposition Table Setup ---
from chess_engine.classical.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table
transposition_table = create_transposition_table(TT_SIZE_MB)


def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories under chess_engine/classical/
    to avoid stale Numba JIT cache issues.
    """
    classical_dir = Path(__file__).parent.parent / "chess_engine" / "classical"
    cache_dirs = list(classical_dir.rglob("__pycache__"))

    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)

from chess_engine.classical.debug_utils import log_info

# Clear cache before importing the engine to avoid stale cache issues
clear_numba_cache()


from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.move import move_to_uci
from chess_engine.classical.core import SQUARE_TO_ALGEBRAIC

# Maximum search depth (Ply) for arrays like killer moves
from chess_engine.classical.constants import MAX_PLY

puzzles = [
        {
            "name": "對抗chess.com 2900時遇到的，引擎搜尋不到，但一搜到就知道大優",
            "fen": "4rrk1/p2p1p1p/1p2p1p1/2nPq2P/2P5/4B3/PbB2PP1/1R1Q2KR w - - 2 20",
            "solution": ["b1b2", "h5g6"],
            "rating": "2400",
            "theme": "sacrifice"
        },
        {
            "name": "Lichess Puzzle 6ooAR",
            "fen": "8/1N2k3/PB1b3p/3p2nr/4p3/6P1/1P3P2/5RK1 b - - 3 35",
            "solution": "g5f3",
            "rating": "1227",
            "theme": "endgame hookMate mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle DbecL",
            "fen": "r1b3kr/ppqp2p1/2n1p2p/7Q/3PNn2/2PB4/PP3PPP/R3K2R w KQ - 2 17",
            "solution": "h5e8",
            "rating": "1714",
            "theme": "doubleCheck kingsideAttack mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle Iyr68",
            "fen": "6k1/6pp/1Np3r1/2P1pp1q/1P1p1r2/1R1Q1N1P/5P2/5R1K b - - 0 34",
            "solution": "h5h3",
            "rating": "1835",
            "theme": "kingsideAttack mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle JPRky",
            "fen": "5k2/p1p4Q/8/3r4/1K4P1/2P5/PP3q2/RN6 b - - 2 30",
            "solution": "f2b6",
            "rating": "2237",
            "theme": "endgame mate mateIn2 short"
        },
        {
            "name": "Lichess Puzzle IeUqw",
            "fen": "3k1r2/ppp3Qp/4R3/2brN2n/5P2/5K2/q5PP/4R3 w - - 3 30",
            "solution": "e6d6",
            "rating": "2731",
            "theme": "interference mate mateIn2 middlegame short"
        },
        {
            "name": "Lichess Puzzle Dpv0F",
            "fen": "8/pp3r2/3P4/8/1Q6/P1n1P3/2k3P1/6K1 b - - 0 47",
            "solution": "c3e2",
            "rating": "992",
            "theme": "anastasiaMate endgame long mate mateIn3"
        },
        {
            "name": "Lichess Puzzle 6XBja",
            "fen": "2q4k/6p1/2pP1pP1/1p2nP2/pP2Q3/8/P6K/8 w - - 6 42",
            "solution": "e4h4",
            "rating": "1019",
            "theme": "endgame long mate mateIn3"
        },
        {
            "name": "Lichess Puzzle NJ3j8",
            "fen": "5r2/2Q5/4pqk1/3p2p1/P2P4/2P5/6PP/R4NK1 b - - 0 26",
            "solution": "f6f2",
            "rating": "1244",
            "theme": "backRankMate deflection endgame long mate mateIn3"
        },
        {
            "name": "Lichess Puzzle CBu8c",
            "fen": "4Q3/3R1p2/5Pk1/6p1/8/pP6/1rr3PP/K3R3 b - - 3 38",
            "solution": "b2a2",
            "rating": "1478",
            "theme": "endgame exposedKing long mate mateIn3 queenRookEndgame"
        },
        {
            "name": "Lichess Puzzle CrgsU",
            "fen": "4k3/1p2B2p/p7/7P/1KQ2P2/P4P2/4nq2/R7 b - - 1 32",
            "solution": "f2b6",
            "rating": "1618",
            "theme": "endgame long mate mateIn3 sacrifice"
        },
        {
            "name": "Lichess Puzzle 7CORk",
            "fen": "4r1k1/5p1p/3Q4/p4B2/2P5/8/P1q2PPP/3R1K2 b - - 0 35",
            "solution": "c2e2",
            "rating": "1639",
            "theme": "backRankMate endgame long mate mateIn3 sacrifice"
        },
        {
            "name": "Lichess Puzzle B7CmL",
            "fen": "8/p2q1k1p/1p3ppQ/2p5/2Pr3P/1P4P1/2P1RP2/5K2 b - - 8 34",
            "solution": "d7h3",
            "rating": "1677",
            "theme": "endgame long mate mateIn3"
        },
        {
            "name": "Lichess Puzzle 81kU1",
            "fen": "5k2/1q6/4Rp1p/1P3QpP/3P4/6PK/5P2/2r5 b - - 5 41",
            "solution": "c1h1",
            "rating": "1804",
            "theme": "endgame long mate mateIn3 sacrifice"
        },
        {
            "name": "Lichess Puzzle LtHov",
            "fen": "3R2r1/5p1k/4pQ2/3p3P/3Pb3/2r5/5PP1/6K1 b - - 0 35",
            "solution": "c3c1",
            "rating": "1986",
            "theme": "endgame long mate mateIn3"
        },
        {
            "name": "Lichess Puzzle 0LFwt",
            "fen": "4R3/3Q3p/3p1Npk/3B4/7q/3P4/PPP2n2/2K5 b - - 0 35",
            "solution": "h4g5",
            "rating": "2099",
            "theme": "backRankMate endgame fork long mate mateIn3"
        },
        {
            "name": "Lichess Puzzle MflKL",
            "fen": "r3rnk1/pbq3pp/2p1pp2/1p1nN1N1/3P4/2PQ4/PPB2PPP/R3R1K1 w - - 0 18",
            "solution": "d3h7",
            "rating": "2125",
            "theme": "kingsideAttack long mate mateIn3 middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle Ni7iB",
            "fen": "1Q6/P2b2pk/2p4p/2Pp1p2/3PNq2/6rP/6PK/6R1 b - - 1 37",
            "solution": "g3h3",
            "rating": "2203",
            "theme": "attraction doubleCheck endgame fork long mate mateIn3 sacrifice"
        },
        {
            "name": "Lichess Puzzle DzuQV",
            "fen": "r1b3k1/pppp1Rpp/2n5/8/2B1r1Q1/2b1q3/P5PP/1N5K w - - 0 17",
            "solution": "f7g7",
            "rating": "2205",
            "theme": "doubleCheck kingsideAttack long mate mateIn3 middlegame"
        },
        {
            "name": "Lichess Puzzle 8zhvd",
            "fen": "3rb1kr/2R5/p4R1p/2N3p1/6n1/8/PPP5/2K5 w - - 0 35",
            "solution": "f6f8",
            "rating": "2447",
            "theme": "attraction exposedKing long mate mateIn3 middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle KV0Mh",
            "fen": "rn2kb1r/pp2qpp1/5n2/1N2pP1p/2Qp4/1B6/PPP2PPP/R3K2R w KQkq - 2 14",
            "solution": "c4c8",
            "rating": "2654",
            "theme": "attackingF2F7 attraction deflection discoveredAttack long mate mateIn3 middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 15zqy",
            "fen": "7k/pp3rrP/3pq3/2p1n3/4Pp1Q/2PP2p1/PP6/2K4R w - - 1 36",
            "solution": "h4d8",
            "rating": "744",
            "theme": "advancedPawn doubleCheck endgame exposedKing kingsideAttack mate mateIn4 promotion veryLong"
        },
        {
            "name": "Lichess Puzzle KFx19",
            "fen": "5r1k/pp5p/5r2/3b4/3B2R1/3BR1K1/PP3P2/3q4 w - - 13 34",
            "solution": "d4f6",
            "rating": "1129",
            "theme": "mate mateIn4 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle F7WBO",
            "fen": "8/5pk1/3p2pp/2p1b1q1/4P3/2PP3P/5QP1/r2N1R1K w - - 2 30",
            "solution": "f2f7",
            "rating": "1214",
            "theme": "endgame mate mateIn4 veryLong"
        },
        {
            "name": "Lichess Puzzle 3iXcH",
            "fen": "r1q4r/2p2p1k/2p2Pb1/1p1p2Qp/7R/1P5P/P1P3PK/5R2 w - - 2 35",
            "solution": "h4h5",
            "rating": "1251",
            "theme": "attraction exposedKing mate mateIn4 middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle 5mApw",
            "fen": "1r3rk1/5pp1/2qbp1np/3p2N1/1pnP2N1/2P3PQ/5P1P/1RR3K1 w - - 0 29",
            "solution": "g4h6",
            "rating": "1403",
            "theme": "kingsideAttack mate mateIn4 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle CDpUD",
            "fen": "k1r4r/pp1qbp2/1Bp1p1p1/3nP2p/R2PnP2/1Q4P1/PP2N3/1KR5 w - - 0 29",
            "solution": "a4a7",
            "rating": "1488",
            "theme": "attraction mate mateIn4 middlegame queensideAttack sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 2gOKZ",
            "fen": "1k4r1/p1p5/1pBqp3/1Pn2p2/8/5RP1/Q5K1/8 w - - 2 38",
            "solution": "a2a7",
            "rating": "1513",
            "theme": "attraction endgame mate mateIn4 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle BT9M3",
            "fen": "4k3/4nr2/PQ2P3/3pP1P1/3P4/8/6q1/4KB1r w - - 7 47",
            "solution": "b6b8",
            "rating": "1575",
            "theme": "endgame exposedKing mate mateIn4 veryLong"
        },
        {
            "name": "Lichess Puzzle OTb60",
            "fen": "r3k1r1/1p4Pp/pqb1ppnB/3Q4/6P1/2PP4/P6P/4RR1K w q - 1 23",
            "solution": "e1e6",
            "rating": "1676",
            "theme": "attraction exposedKing mate mateIn4 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle L471p",
            "fen": "6k1/pq6/1p2p3/2p2p2/4P1p1/PP3n2/1B1NQ1N1/3R3K b - - 0 34",
            "solution": "b7h7",
            "rating": "1689",
            "theme": "exposedKing mate mateIn4 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle OboU3",
            "fen": "5k2/5P2/1p2P3/1P3NK1/8/8/7p/7r w - - 2 61",
            "solution": "g5f6",
            "rating": "1717",
            "theme": "advancedPawn endgame exposedKing mate mateIn4 veryLong"
        },
        {
            "name": "Lichess Puzzle KKdBA",
            "fen": "5rk1/ppQ2ppp/2p5/8/8/7P/P1q1r1P1/3R1R1K w - - 2 23",
            "solution": "c7f7",
            "rating": "1800",
            "theme": "endgame mate mateIn4 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 39YwP",
            "fen": "r7/1p2QB2/5r1p/p5k1/P1P5/3bP3/5bPK/1q6 w - - 2 35",
            "solution": "e7e5",
            "rating": "1898",
            "theme": "mate mateIn4 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle LeibE",
            "fen": "r2qrbk1/3n1p1p/p2p2p1/1p1b1PP1/3B3Q/1P1R4/1PP2P1P/2K3R1 w - - 0 22",
            "solution": "h4h7",
            "rating": "1910",
            "theme": "attraction kingsideAttack mate mateIn4 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 901sw",
            "fen": "5rk1/4Q1b1/p5P1/3p4/2bPqpP1/5N2/4pPK1/7R w - - 6 45",
            "solution": "h1h8",
            "rating": "1972",
            "theme": "attraction exposedKing fork mate mateIn4 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle MmHYq",
            "fen": "3q4/8/p2NkbbR/2p3n1/4PQ2/2PPK3/Pr6/7R b - - 4 34",
            "solution": "f6d4",
            "rating": "1982",
            "theme": "attraction mate mateIn4 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle B5JhK",
            "fen": "5r1k/2P3pp/8/3Qp1q1/2P1p3/3P1PPn/P5RP/3R1K2 b - - 0 39",
            "solution": "f8f3",
            "rating": "1993",
            "theme": "attraction endgame exposedKing mate mateIn4 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 5NsNN",
            "fen": "5qk1/p2Q2pp/2p5/3b4/3PB3/5Pb1/P5r1/R2K3R b - - 0 24",
            "solution": "f8f3",
            "rating": "2017",
            "theme": "exposedKing mate mateIn4 middlegame sacrifice veryLong xRayAttack"
        },
        {
            "name": "Lichess Puzzle OkYL2",
            "fen": "3Q4/5p2/4kn2/P2p2R1/4q1P1/4P2K/5P2/8 b - - 0 48",
            "solution": "e4h1",
            "rating": "2230",
            "theme": "endgame fork mate mateIn4 veryLong"
        },
        {
            "name": "Lichess Puzzle 06LQW",
            "fen": "r4Bk1/pp3p2/7p/3N2p1/2BpPbbq/2P5/PPQ3P1/R3R1K1 b - - 2 27",
            "solution": "h4h2",
            "rating": "2291",
            "theme": "attraction kingsideAttack mate mateIn4 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 1v3DI",
            "fen": "1knr3r/3q1ppp/Qp6/3Pp3/8/4P3/6PP/1RR3K1 w - - 0 24",
            "solution": "c1c8",
            "rating": "2291",
            "theme": "exposedKing mate mateIn4 middlegame queensideAttack veryLong"
        },
        {
            "name": "Lichess Puzzle 9UdY7",
            "fen": "1k2q1r1/ppp4p/5RnQ/8/2NP4/2P5/PP4KP/5R2 b - - 0 22",
            "solution": "g6f4",
            "rating": "2337",
            "theme": "doubleCheck fork mate mateIn4 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 6TFPU",
            "fen": "5r2/1R3pp1/r1p4p/4p3/P1k1P3/5P1P/5P2/R5K1 w - - 0 29",
            "solution": "a1c1",
            "rating": "2536",
            "theme": "endgame mate mateIn4 rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle 0KfyN",
            "fen": "k2r1r2/p5p1/BpR5/3p1qn1/N2P2b1/P3P1B1/1P5P/2Q3K1 w - - 7 30",
            "solution": "a6b7",
            "rating": "2637",
            "theme": "attraction clearance mate mateIn4 middlegame queensideAttack sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle IS3Ja",
            "fen": "5b1r/3R1ppp/k1p3n1/3N4/4P3/4B3/P1P1K2P/7r w - - 0 24",
            "solution": "d7a7",
            "rating": "2680",
            "theme": "attraction exposedKing hookMate mate mateIn4 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 4vuBr",
            "fen": "3R2k1/pQp2pp1/6p1/6B1/8/7P/PP2rq2/7K b - - 4 31",
            "solution": "g8h7",
            "rating": "998",
            "theme": "endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle F2th0",
            "fen": "r2q3r/ppp3k1/5pn1/2Q2p2/3P4/2N3p1/PP1BBPP1/R4RK1 b - - 0 23",
            "solution": "h8h1",
            "rating": "1123",
            "theme": "attraction kingsideAttack mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle OxaTE",
            "fen": "8/1R1Q2bk/1pr2q1p/4p3/1B1p3N/3P3P/1P3PP1/6K1 b - - 0 34",
            "solution": "c6c1",
            "rating": "1175",
            "theme": "fork mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 6gs6q",
            "fen": "8/1b4pk/3Q3p/2P1pp2/6P1/7P/4rP2/6K1 b - - 0 31",
            "solution": "e2e1",
            "rating": "1223",
            "theme": "deflection endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle OtlTI",
            "fen": "2k2b1N/ppp3p1/4Pq1p/8/4Q3/8/PPPr1PPP/5RK1 b - - 2 21",
            "solution": "f6f2",
            "rating": "1342",
            "theme": "backRankMate endgame mate mateIn5 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle CUzdk",
            "fen": "1k4r1/1bpR2Bp/p7/np6/5Q2/8/q1P2P1P/2K5 w - - 5 28",
            "solution": "f4c7",
            "rating": "1443",
            "theme": "mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle MDpf8",
            "fen": "5q1k/4b1pp/8/8/4P1Q1/1P1PB3/1Pr3PP/RN4K1 b - - 0 21",
            "solution": "c2c1",
            "rating": "1450",
            "theme": "mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle MS1UE",
            "fen": "r6r/p1pq1pk1/1bppb3/4P1BQ/3PnP2/7P/PP4P1/RN3RK1 w - - 3 19",
            "solution": "g5f6",
            "rating": "1530",
            "theme": "fork mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle Dp7NN",
            "fen": "4q1rr/p1p1kp2/B1p1Pn2/2Np2p1/3P4/2P3p1/PP4P1/R2Q1RK1 b - - 0 21",
            "solution": "h8h1",
            "rating": "1584",
            "theme": "attraction kingsideAttack mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 3CXjK",
            "fen": "5k2/1Bp1p3/p5r1/1P6/P7/r4N2/1RPP1P2/4KR2 b - - 2 34",
            "solution": "g6e6",
            "rating": "1589",
            "theme": "endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle 7aboE",
            "fen": "7r/pp3R2/1kpp4/2n1q3/6Q1/1B5P/P1P5/7K w - - 3 27",
            "solution": "g4b4",
            "rating": "1659",
            "theme": "endgame mate mateIn5 pin veryLong"
        },
        {
            "name": "Lichess Puzzle 55Uom",
            "fen": "2r3k1/1pr3p1/p3p1B1/1b1pP3/7Q/5N1P/P5PK/2q5 w - - 14 31",
            "solution": "h4h7",
            "rating": "1747",
            "theme": "deflection interference mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 8dLuK",
            "fen": "3R4/R4p1k/4p1p1/3bP2p/3Q3P/1r6/1P3PPK/2q5 b - - 9 32",
            "solution": "b3h3",
            "rating": "1781",
            "theme": "attraction endgame master mate mateIn5 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 69SgL",
            "fen": "2rR4/pR4pQ/4p2p/4k1q1/4B3/7P/P5P1/7K b - - 0 31",
            "solution": "c8c1",
            "rating": "1782",
            "theme": "endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle LDyIe",
            "fen": "8/p5kp/3Q1pp1/2pP4/2P2P2/7P/rq4P1/4R1K1 w - - 2 31",
            "solution": "d6e7",
            "rating": "1960",
            "theme": "endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle 1fetJ",
            "fen": "1r4kb/p2QBp1p/2p3pP/3pP1N1/3P4/3R1R2/1rPK1PP1/5q2 b - - 4 24",
            "solution": "b2c2",
            "rating": "2007",
            "theme": "attraction mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 7Px9Z",
            "fen": "2r4k/1b3ppP/p4B2/4p2P/q3P3/1pP2PQ1/1P6/1K1R2R1 b - - 0 29",
            "solution": "a4a2",
            "rating": "2021",
            "theme": "deflection mate mateIn5 middlegame queensideAttack veryLong"
        },
        {
            "name": "Lichess Puzzle AZ4F7",
            "fen": "4r1k1/pp4n1/2p2pK1/3qb3/6P1/1Q6/PP6/7R w - - 2 36",
            "solution": "h1h8",
            "rating": "2026",
            "theme": "attraction endgame exposedKing mate mateIn5 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 2pjFS",
            "fen": "1r1q1rk1/3p2pp/p3P3/n3N2P/5P2/b1pB4/PPQ2P2/1K4Rb w - - 0 22",
            "solution": "d3h7",
            "rating": "2055",
            "theme": "doubleCheck fork mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle Dk4aO",
            "fen": "4r1k1/2Q2pp1/8/7p/1p2P3/3P1P2/Pqn1K1PP/2R2R2 b - - 0 24",
            "solution": "c2d4",
            "rating": "2091",
            "theme": "doubleCheck endgame mate mateIn5 pin sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 6DY3a",
            "fen": "Q7/p1pp4/1pn1k1p1/4P3/3p2K1/2N1q3/PPP4P/5R2 b - - 0 31",
            "solution": "c6e5",
            "rating": "2102",
            "theme": "endgame fork mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle 66EDQ",
            "fen": "r1bq1rk1/pp1nb1p1/2n1p3/2ppP1P1/3P4/2P5/PP1NQPP1/R1B1K2R w KQ - 1 13",
            "solution": "h1h8",
            "rating": "2109",
            "theme": "attraction kingsideAttack mate mateIn5 opening sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle D1YZz",
            "fen": "1Q6/3R1pk1/4p1pp/P3N3/5P2/P5PK/4bq2/8 b - - 6 39",
            "solution": "e2f1",
            "rating": "2119",
            "theme": "deflection endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle 67eP6",
            "fen": "5Bk1/3b2p1/p5K1/1p1P2P1/2p1P3/P1N4r/1P6/1B2R3 b - - 2 38",
            "solution": "h3f3",
            "rating": "2242",
            "theme": "discoveredAttack endgame mate mateIn5 quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle AyNaC",
            "fen": "1k2r3/1Pq2np1/Q3p3/p3np2/P2P4/B6r/5P1N/R4RK1 b - - 2 25",
            "solution": "e5f3",
            "rating": "2247",
            "theme": "attraction discoveredAttack kingsideAttack mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle DX0K6",
            "fen": "r2q1r2/1b1pb1pk/p1n1p3/1p1nP1p1/2pP3P/2P3B1/PP1N1PP1/R2QK2R w KQ - 0 15",
            "solution": "h4g5",
            "rating": "2249",
            "theme": "attraction discoveredAttack mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 74zO6",
            "fen": "r5k1/ppp2p1p/8/3P2p1/2Bp1b2/1P6/P2N1P1q/R2QRK2 b - - 7 28",
            "solution": "h2h3",
            "rating": "2285",
            "theme": "discoveredAttack mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 0UlS3",
            "fen": "q4r2/4bpkp/p1p4N/3p1Qp1/3P4/7P/2n3R1/6K1 w - - 0 31",
            "solution": "g2g5",
            "rating": "2322",
            "theme": "fork mate mateIn5 middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 5z1PF",
            "fen": "r5rk/ppb2R1p/2p5/8/6Q1/7P/PqP3P1/3R3K w - - 4 25",
            "solution": "f7h7",
            "rating": "2363",
            "theme": "attraction fork mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle HApyl",
            "fen": "8/3bqk2/4p1pK/1p1pP2p/pP1P1P1P/P1QB4/8/8 b - - 7 37",
            "solution": "e7f8",
            "rating": "2585",
            "theme": "endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle MZG6H",
            "fen": "1r4k1/p1p2pp1/2p2Pq1/1r1p4/3P4/bPP1P3/P1Q5/3KR2R w - - 3 30",
            "solution": "h1h8",
            "rating": "2643",
            "theme": "advancedPawn kingsideAttack mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 8QYsV",
            "fen": "1nN4R/5kr1/2p1pp2/p5p1/8/p2Q4/1P4q1/1K6 w - - 6 36",
            "solution": "c8d6",
            "rating": "2652",
            "theme": "clearance endgame fork mate mateIn5 sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle PNMjr",
            "fen": "5qk1/8/8/P1NpP3/3P2p1/3K2P1/3Q4/RR1b3r b - - 1 45",
            "solution": "f8f3",
            "rating": "2672",
            "theme": "mate mateIn5 middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle Bx8w7",
            "fen": "8/ppR3p1/4R3/1k1p4/8/1P3q2/PKP2n2/8 w - - 0 33",
            "solution": "c2c4",
            "rating": "2759",
            "theme": "endgame mate mateIn5 veryLong"
        },
        {
            "name": "Lichess Puzzle IAynU",
            "fen": "8/7p/4Q3/3p2pk/p3p3/PqP3P1/5PK1/8 w - - 4 38",
            "solution": "g3g4",
            "rating": "2767",
            "theme": "endgame mate mateIn5 queenEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle 5lCCC",
            "fen": "8/1p3pkp/6p1/8/p1P1B3/5N1P/P4P2/3n3K b - - 0 32",
            "solution": "d1f2",
            "rating": "771",
            "theme": "crushing endgame fork master short"
        },
        {
            "name": "Lichess Puzzle HYPCV",
            "fen": "8/1p2q1pk/6p1/2PQ1p2/PP2PNn1/6P1/6K1/8 b - - 0 46",
            "solution": "g4e3",
            "rating": "782",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle CZBNn",
            "fen": "r4rk1/pp3pp1/3R3p/6n1/5NP1/6K1/PP6/R1B5 b - - 0 27",
            "solution": "g5e4",
            "rating": "826",
            "theme": "crushing fork middlegame short"
        },
        {
            "name": "Lichess Puzzle FZE8A",
            "fen": "6k1/1p1r3p/3B4/1N3R1p/P5nP/1P4P1/5bK1/8 b - - 0 32",
            "solution": "g4e3",
            "rating": "1060",
            "theme": "advantage endgame fork short"
        },
        {
            "name": "Lichess Puzzle 8BPee",
            "fen": "2R5/p5kn/3q3p/5p2/8/6N1/Pb3P1N/6K1 w - - 2 30",
            "solution": "g3f5",
            "rating": "1116",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 0rFTe",
            "fen": "8/1p1nrk2/p1p3p1/3p1p1p/1P1P3P/1PnN1P2/2PB1KP1/2R5 b - - 6 28",
            "solution": "e7e2",
            "rating": "1163",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle MX8wh",
            "fen": "8/6p1/4q1k1/1p3r2/7P/6P1/3Q4/1R5K b - - 2 50",
            "solution": "e6e4",
            "rating": "1243",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 7tAsw",
            "fen": "3r4/pbp2pk1/1p2pnpp/4P1Q1/1qB4P/1P6/P3NPP1/6K1 w - - 0 22",
            "solution": "g5f6",
            "rating": "1263",
            "theme": "crushing fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 5Or40",
            "fen": "8/2Q2nkp/5pp1/p7/Pb1N4/1q4BP/5PP1/6K1 b - - 1 38",
            "solution": "b3d1",
            "rating": "1318",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle 145is",
            "fen": "8/7p/4kP2/3p4/1n6/2K4P/2p5/2R5 b - - 5 46",
            "solution": "b4a2",
            "rating": "1321",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle CLiYU",
            "fen": "3r1r1k/pBp3pp/1p1P2q1/4P3/8/1Q2n1PR/PP1N3P/5R1K b - - 0 24",
            "solution": "e3f1",
            "rating": "1397",
            "theme": "equality fork long middlegame"
        },
        {
            "name": "Lichess Puzzle AdOHR",
            "fen": "8/2p5/p1k5/5B2/r1P1NK2/8/8/8 w - - 1 43",
            "solution": "f5d7",
            "rating": "1404",
            "theme": "attraction crushing endgame exposedKing fork long sacrifice"
        },
        {
            "name": "Lichess Puzzle 920Vn",
            "fen": "r2kq3/pp4R1/2b2p1B/2P2Q2/5P2/2P4P/P7/1K6 b - - 0 30",
            "solution": "c6e4",
            "rating": "1421",
            "theme": "crushing endgame fork master short"
        },
        {
            "name": "Lichess Puzzle 1gDgN",
            "fen": "5k2/4n2p/1p3qp1/p1b3N1/8/7P/6P1/3Q3K w - - 2 50",
            "solution": "g5h7",
            "rating": "1457",
            "theme": "advantage endgame fork short"
        },
        {
            "name": "Lichess Puzzle 4pA9M",
            "fen": "5r1k/p5p1/1p1p3p/4pq2/2P5/4B1PP/P5K1/1R1Q4 b - - 0 37",
            "solution": "f5e4",
            "rating": "1459",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle JzoM7",
            "fen": "3r2k1/b1p3pp/p7/P4p2/1PN5/2P5/4rPPP/4RRK1 b - - 2 24",
            "solution": "a7f2",
            "rating": "1494",
            "theme": "crushing deflection endgame fork short"
        },
        {
            "name": "Lichess Puzzle GCbpD",
            "fen": "2rr2k1/1b3ppp/p7/1p4q1/1P6/P3P1bP/4RQP1/1B1N1RK1 w - - 0 30",
            "solution": "f2f7",
            "rating": "1497",
            "theme": "advantage fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 3UsPU",
            "fen": "8/8/p2Q2qk/1p5p/1P2p3/P1n4P/5PP1/6K1 w - - 5 40",
            "solution": "d6d2",
            "rating": "1516",
            "theme": "advantage endgame fork short"
        },
        {
            "name": "Lichess Puzzle 0SjEJ",
            "fen": "r2q1kr1/pp3p2/3b1P1p/1B1pp3/7Q/8/PPP3PP/R4RK1 b - - 2 21",
            "solution": "d8b6",
            "rating": "1535",
            "theme": "advantage fork long middlegame"
        },
        {
            "name": "Lichess Puzzle MqAOg",
            "fen": "2r3k1/5pp1/5n1p/p3pN2/1qp5/2Q1PP1P/6P1/2R3K1 w - - 2 31",
            "solution": "c3b4",
            "rating": "1572",
            "theme": "advantage endgame fork long"
        },
        {
            "name": "Lichess Puzzle GkhHA",
            "fen": "3Q4/pp3ppk/7p/8/8/P5P1/1rqNRP1P/6K1 b - - 2 31",
            "solution": "c2d1",
            "rating": "1602",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle ICKIj",
            "fen": "r1b2rk1/bp4pp/p6P/2q1p1B1/2NN1n2/6Q1/PP3PP1/2RR2K1 b - - 0 28",
            "solution": "c5d4",
            "rating": "1602",
            "theme": "crushing fork long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 6Pr8W",
            "fen": "1B4k1/3p2pp/5n2/3P1p2/4p3/2p1P1P1/4KPBP/q2Q2NR b - - 1 27",
            "solution": "a1b2",
            "rating": "1622",
            "theme": "advancedPawn advantage fork long middlegame"
        },
        {
            "name": "Lichess Puzzle 1WU4I",
            "fen": "6k1/1p1R4/p2n2p1/6N1/3r4/8/PP4P1/7K w - - 2 38",
            "solution": "d7d8",
            "rating": "1633",
            "theme": "crushing endgame exposedKing fork long"
        },
        {
            "name": "Lichess Puzzle 1e7RI",
            "fen": "r3rk2/pp4b1/4pq2/2n3N1/4Q2P/2P2N2/PP3PP1/2K4R w - - 0 21",
            "solution": "g5h7",
            "rating": "1644",
            "theme": "advantage fork long middlegame"
        },
        {
            "name": "Lichess Puzzle EdcX2",
            "fen": "3r1rk1/ppR3pp/3N1p2/4P3/3Q1n2/5N1P/q4BP1/5RK1 b - - 1 24",
            "solution": "f4e2",
            "rating": "1644",
            "theme": "equality fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 0XEKo",
            "fen": "8/8/pB1Qqk2/1p4p1/5pp1/RPr4P/1r3PK1/8 w - - 1 41",
            "solution": "b6d4",
            "rating": "1645",
            "theme": "endgame equality fork short"
        },
        {
            "name": "Lichess Puzzle LwsCC",
            "fen": "5rk1/2qn3n/r1p1pP2/p1Pp4/3P2pN/P1N3P1/6P1/R2QR1K1 b - - 0 26",
            "solution": "c7g3",
            "rating": "1660",
            "theme": "advantage fork middlegame short"
        },
        {
            "name": "Lichess Puzzle H4tkh",
            "fen": "r6r/pp2k3/q1b1p3/3pP2p/8/5Qn1/P4PPP/R4RK1 w - - 0 22",
            "solution": "f3f6",
            "rating": "1811",
            "theme": "crushing fork interference long middlegame"
        },
        {
            "name": "Lichess Puzzle 2g7vW",
            "fen": "4r1k1/1q2r1pp/p3pp2/2R5/2Pp1Q2/1P1P4/P4PPP/4R1K1 b - - 0 29",
            "solution": "b7b4",
            "rating": "1823",
            "theme": "advantage endgame fork short"
        },
        {
            "name": "Lichess Puzzle F8N4k",
            "fen": "1r1r2k1/3bnppp/4pb2/1pB1q3/4P3/1P1Q4/P3BPPP/3RR1K1 w - - 0 22",
            "solution": "c5d6",
            "rating": "1832",
            "theme": "advantage fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 1omUG",
            "fen": "4r1k1/2p3pb/3b3p/1pB1p3/1P6/1N3P1q/2P3QN/3R2K1 b - - 1 25",
            "solution": "h3g2",
            "rating": "1852",
            "theme": "advantage fork long middlegame"
        },
        {
            "name": "Lichess Puzzle KR2Zs",
            "fen": "r1b1k2r/pp1q1ppp/4pn2/1Qb3B1/3NB3/8/PPP2PPP/R3K2R w KQkq - 3 12",
            "solution": "g5f6",
            "rating": "1875",
            "theme": "crushing fork middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 6R7vJ",
            "fen": "3rr1k1/pp2qppb/1b3n1p/4nN2/1P2p3/P3P1PB/1BRN1P1P/Q4RK1 b - - 2 20",
            "solution": "h7f5",
            "rating": "1876",
            "theme": "advantage attraction fork middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle LmpUv",
            "fen": "3q1r1k/p5p1/4Qpp1/8/3PRb2/N1P2P2/Pr3P1P/R5K1 b - - 1 21",
            "solution": "f6f5",
            "rating": "1889",
            "theme": "advantage clearance fork long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 4HHnY",
            "fen": "6k1/5pp1/1p2p1np/p7/P2Pq1r1/2Q5/5PRP/6RK w - - 0 33",
            "solution": "f2f3",
            "rating": "1924",
            "theme": "crushing endgame fork short"
        },
        {
            "name": "Lichess Puzzle JhaaI",
            "fen": "8/8/7R/1p1p4/P3k1p1/1P2n1Kp/2P5/8 b - - 0 45",
            "solution": "e3f5",
            "rating": "1962",
            "theme": "crushing defensiveMove endgame exposedKing fork quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 5TeaR",
            "fen": "2br1rk1/p1qn1Rbp/1p4p1/2p1p1N1/2P1P3/1P1P2P1/PB2Q1BP/6K1 w - - 1 22",
            "solution": "f7g7",
            "rating": "2047",
            "theme": "attraction crushing fork long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 984dD",
            "fen": "r2qk2r/pp2bppp/2n2n2/1BP1p1B1/3pN3/P4P2/1PP2PPP/R2QK2R b KQkq - 3 10",
            "solution": "d8a5",
            "rating": "2063",
            "theme": "crushing fork long opening"
        },
        {
            "name": "Lichess Puzzle 2VUul",
            "fen": "4r1q1/2p2p1k/p1p1b2p/3p3Q/3P1b2/1PN2R2/P5PP/4R1K1 b - - 1 28",
            "solution": "e6g4",
            "rating": "2070",
            "theme": "crushing fork middlegame short"
        },
        {
            "name": "Lichess Puzzle 2QMj8",
            "fen": "8/4k3/nB5p/2PB2p1/3b1p2/3KpP1P/6P1/8 b - - 6 47",
            "solution": "a6b4",
            "rating": "2083",
            "theme": "advancedPawn crushing endgame fork sacrifice short"
        },
        {
            "name": "Lichess Puzzle 7smmT",
            "fen": "r5k1/p2Q4/6pp/8/1p5b/8/Pq3PP1/4R1K1 w - - 3 30",
            "solution": "d7d5",
            "rating": "2111",
            "theme": "advantage endgame exposedKing fork long"
        },
        {
            "name": "Lichess Puzzle GgxLJ",
            "fen": "r7/1k4p1/1B1Q3p/1pp1p1q1/4n3/6P1/PPP2P1P/6K1 w - - 0 27",
            "solution": "d6c7",
            "rating": "2179",
            "theme": "advantage endgame fork long"
        },
        {
            "name": "Lichess Puzzle 9mXx2",
            "fen": "6k1/2b2p2/6p1/p5P1/3QPp1p/2B4P/q3K3/8 w - - 2 59",
            "solution": "e2f3",
            "rating": "2188",
            "theme": "crushing defensiveMove endgame fork long master"
        },
        {
            "name": "Lichess Puzzle MbeGi",
            "fen": "r6r/2pk1qp1/p3b2p/3BQ1b1/2N5/6P1/PP3P1P/1K5R w - - 1 25",
            "solution": "d5e6",
            "rating": "2192",
            "theme": "crushing fork long middlegame"
        },
        {
            "name": "Lichess Puzzle 9Ca9Y",
            "fen": "r3r1k1/pp1R1pp1/4q2p/2p4Q/2b3N1/2PRP3/1P4PP/2K5 w - - 8 24",
            "solution": "d3d6",
            "rating": "2201",
            "theme": "advantage fork middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle Hpi1M",
            "fen": "6k1/1p6/p2p2p1/2pP4/P1Nb1Pbp/6P1/1P2N1QK/4qB2 b - - 1 32",
            "solution": "h4h3",
            "rating": "2237",
            "theme": "crushing deflection fork long middlegame"
        },
        {
            "name": "Lichess Puzzle A4loo",
            "fen": "rn2qr1k/1b4pp/4Rb2/2p2p2/2B2B2/1NP1PN2/Pp3QPP/1K5R b - - 0 20",
            "solution": "b7e4",
            "rating": "2258",
            "theme": "crushing fork long middlegame xRayAttack"
        },
        {
            "name": "Lichess Puzzle EDgd8",
            "fen": "2R5/p3k1p1/5p2/4q3/7Q/1p4PP/3r1P2/6K1 w - - 0 36",
            "solution": "h4b4",
            "rating": "2300",
            "theme": "crushing endgame fork skewer veryLong"
        },
        {
            "name": "Lichess Puzzle 2iWRI",
            "fen": "3r2k1/p2r1pbp/1qp3p1/1p2P3/3PNQ2/7P/P5P1/3RR2K w - - 5 27",
            "solution": "e4f6",
            "rating": "2321",
            "theme": "advantage fork middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 6EXYV",
            "fen": "6k1/5pp1/1Q5p/8/2Rn2q1/5N2/P4KPP/3r4 b - - 2 31",
            "solution": "d1d2",
            "rating": "2333",
            "theme": "crushing endgame fork long sacrifice"
        },
        {
            "name": "Lichess Puzzle 3C2hv",
            "fen": "7r/Rp3pk1/2b5/2pq2n1/5Q2/1P2N2P/1PP3P1/5RK1 b - - 6 31",
            "solution": "d5g2",
            "rating": "2334",
            "theme": "advantage discoveredAttack fork middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle Or1uH",
            "fen": "4r1k1/2Q3pp/5p2/8/1p6/1qPn3P/5PPB/2R3K1 w - - 6 35",
            "solution": "c7d7",
            "rating": "2344",
            "theme": "advantage endgame fork long"
        },
        {
            "name": "Lichess Puzzle C8x7b",
            "fen": "5rk1/5ppp/pq1p4/1pp2PBQ/4r3/P1Pn4/1P4PP/R4R1K w - - 0 22",
            "solution": "h5f3",
            "rating": "2355",
            "theme": "advantage fork middlegame short"
        },
        {
            "name": "Lichess Puzzle IOjVu",
            "fen": "8/p7/6k1/1P4p1/P3Kn1p/3p4/8/6R1 b - - 0 52",
            "solution": "f4e2",
            "rating": "2454",
            "theme": "crushing endgame fork master veryLong"
        },
        {
            "name": "Lichess Puzzle 3kF4W",
            "fen": "r2k3r/pp3Bbp/4Q1pn/2qNp1N1/3nP3/8/PP3PPP/3R2K1 w - - 6 22",
            "solution": "b2b4",
            "rating": "2467",
            "theme": "advantage exposedKing fork long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle M5EWK",
            "fen": "2r2rk1/pp1b1p2/3b2pp/3Np3/8/P2BP1P1/1P3PP1/2R2RK1 w - - 1 24",
            "solution": "d5f6",
            "rating": "2475",
            "theme": "crushing defensiveMove fork middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 645Ft",
            "fen": "r1b1k2r/pp2q3/2p1p1nn/3pPpNP/3P4/2PB4/PP1N2P1/R2QK2R b KQkq - 0 14",
            "solution": "e7g5",
            "rating": "2517",
            "theme": "advantage exposedKing fork long opening"
        },
        {
            "name": "Lichess Puzzle 24pH4",
            "fen": "6k1/1p3pp1/p3p2q/3pN1Q1/1P1P1K2/4P2P/6P1/8 b - - 3 39",
            "solution": "h6g5",
            "rating": "2547",
            "theme": "attraction crushing endgame fork veryLong"
        },
        {
            "name": "Lichess Puzzle NBqd6",
            "fen": "r5k1/PN3pp1/1B2p1q1/7p/3P3n/1Q2P1P1/5P1P/6K1 b - - 0 31",
            "solution": "h4f3",
            "rating": "2643",
            "theme": "crushing endgame fork veryLong"
        },
        {
            "name": "Lichess Puzzle 3OBKz",
            "fen": "2r3k1/pp4p1/2P3N1/3nP2p/6b1/P2B1q2/1PKQ1P2/4R3 b - - 0 30",
            "solution": "g4f5",
            "rating": "2660",
            "theme": "crushing fork long middlegame pin"
        },
        {
            "name": "Lichess Puzzle NT5SC",
            "fen": "4r2k/p6p/1p5q/5p2/8/2P1p3/PPB3QP/4RK2 b - - 2 29",
            "solution": "e8g8",
            "rating": "2664",
            "theme": "advancedPawn advantage endgame exposedKing fork veryLong"
        },
        {
            "name": "Lichess Puzzle IGdCm",
            "fen": "3r1q1k/1pp3pp/p7/P3p1N1/1n6/3r3P/4QPP1/R2R2K1 w - - 0 27",
            "solution": "d1d3",
            "rating": "2684",
            "theme": "advantage fork middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle IGgNX",
            "fen": "2q5/Q2R1pbk/3p3p/p2N1p1P/4r1p1/8/PP4P1/1K3R2 b - - 0 28",
            "solution": "c8c4",
            "rating": "2693",
            "theme": "attraction crushing fork middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 5St0e",
            "fen": "5bk1/7r/Q3R1p1/3pPb2/5P1q/1P6/P2B1N1P/6K1 b - - 0 33",
            "solution": "h4h2",
            "rating": "2703",
            "theme": "advantage clearance fork middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle EEDYP",
            "fen": "6k1/pQ1P1Rp1/4bpq1/4p2p/7K/5N1P/PP2BPP1/7r b - - 4 31",
            "solution": "g6g4",
            "rating": "1798",
            "theme": "mate mateIn1 middlegame oneMove pin"
        },
        {
            "name": "Lichess Puzzle 89YNa",
            "fen": "8/5k1p/B2p1p2/3np1p1/6P1/P4P2/7P/6K1 w - - 0 36",
            "solution": "a6c4",
            "rating": "844",
            "theme": "crushing endgame pin short"
        },
        {
            "name": "Lichess Puzzle HVqhL",
            "fen": "6k1/7p/8/1r1p4/n1pPp1P1/P1P1P3/4N2K/R7 b - - 2 33",
            "solution": "b5b2",
            "rating": "939",
            "theme": "crushing endgame pin short"
        },
        {
            "name": "Lichess Puzzle GO0uJ",
            "fen": "2kr3r/pp6/3p1R2/7p/8/P6P/1Pn2PP1/4R1K1 w - - 0 24",
            "solution": "e1c1",
            "rating": "1021",
            "theme": "advantage endgame pin short"
        },
        {
            "name": "Lichess Puzzle HqmRH",
            "fen": "8/pp1kn3/n1p5/3q4/6RQ/6P1/PPP5/2KNr3 w - - 9 38",
            "solution": "g4d4",
            "rating": "1102",
            "theme": "advantage middlegame pin short"
        },
        {
            "name": "Lichess Puzzle EkDRS",
            "fen": "5r2/6kp/6p1/1Q2q3/p2p3P/P2B1b2/2P5/1K1N2R1 b - - 0 34",
            "solution": "f8b8",
            "rating": "1312",
            "theme": "crushing master middlegame pin short"
        },
        {
            "name": "Lichess Puzzle 2yFFl",
            "fen": "2k4r/pppb2pp/5n2/8/6q1/1QP5/PP1BN3/R2KR3 b - - 4 30",
            "solution": "d7a4",
            "rating": "1325",
            "theme": "crushing middlegame pin short"
        },
        {
            "name": "Lichess Puzzle NUCpu",
            "fen": "r2r2k1/4ppbp/2N2np1/q1pP4/1n6/P1N5/1P3PPP/R1BQR1K1 b - - 2 17",
            "solution": "b4c6",
            "rating": "1364",
            "theme": "crushing middlegame pin short"
        },
        {
            "name": "Lichess Puzzle JuQ6B",
            "fen": "2r2r2/kb3Q2/p1p4p/1p1pq1p1/1N1Rp3/P3P2P/1PP2P2/2K3R1 w - - 4 27",
            "solution": "b4c6",
            "rating": "1375",
            "theme": "advantage deflection long middlegame pin"
        },
        {
            "name": "Lichess Puzzle EyLAR",
            "fen": "1k1q4/pb6/2p2p2/4p3/1bP1P1PN/8/1PB1QP2/7K b - - 1 34",
            "solution": "d8h8",
            "rating": "1390",
            "theme": "crushing endgame pin short"
        },
        {
            "name": "Lichess Puzzle 5mBZJ",
            "fen": "8/3b3p/3k2p1/1pRPp3/1P1n4/7P/r5PK/4RB2 b - - 3 32",
            "solution": "d4f3",
            "rating": "1429",
            "theme": "crushing endgame pin short"
        },
        {
            "name": "Lichess Puzzle 4JDC4",
            "fen": "r5k1/rpp2ppp/4b3/1P2p3/nb2N3/1p1PPN2/1B2RPPP/1Q3K2 b - - 4 24",
            "solution": "a4b2",
            "rating": "1457",
            "theme": "advantage clearance middlegame pin short"
        },
        {
            "name": "Lichess Puzzle PMfB2",
            "fen": "1r1r2k1/3np3/p4ppb/8/2PP2q1/3KPQ2/PP1N2P1/R1B2R2 b - - 2 21",
            "solution": "d7e5",
            "rating": "1473",
            "theme": "crushing middlegame pin short"
        },
        {
            "name": "Lichess Puzzle ORGFT",
            "fen": "1r4k1/5p1p/RP2p1p1/3n4/8/4B2P/1r3PPK/R7 w - - 1 33",
            "solution": "a6a8",
            "rating": "1474",
            "theme": "crushing endgame long pin"
        },
        {
            "name": "Lichess Puzzle EycOY",
            "fen": "2r4k/p1b2p1p/2pqp1r1/5p2/3P1B1P/1P2NQPb/P4P2/R2R2K1 b - - 3 24",
            "solution": "d6f4",
            "rating": "1555",
            "theme": "crushing master middlegame pin short"
        },
        {
            "name": "Lichess Puzzle 2bMlw",
            "fen": "Rn2k2r/3n1ppp/4p3/1B6/1b1P4/4P3/5PPP/6K1 w k - 0 21",
            "solution": "a8b8",
            "rating": "1564",
            "theme": "advantage endgame pin short skewer"
        },
        {
            "name": "Lichess Puzzle LSYa1",
            "fen": "7k/p2n4/2R5/5p2/5Pp1/8/r3PN2/5K2 w - - 0 36",
            "solution": "c6c8",
            "rating": "1565",
            "theme": "advantage endgame long master pin"
        },
        {
            "name": "Lichess Puzzle Gbst9",
            "fen": "r5k1/1qp2pp1/3bp2p/8/3PB3/2P1P2P/5PP1/Q1R1N1K1 b - - 0 24",
            "solution": "a8a1",
            "rating": "1566",
            "theme": "advantage master middlegame pin short"
        },
        {
            "name": "Lichess Puzzle 1Z2dO",
            "fen": "r3k2r/1pp2pp1/p6p/3nPb1P/1b1PN3/2N2PB1/PP4P1/R3K2R b KQkq - 0 22",
            "solution": "f5e4",
            "rating": "1598",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 6MiNu",
            "fen": "r2qk2r/1b1nbppp/p7/2N1p1P1/8/1N2QP2/1PP4P/2KR1B1R b kq - 0 18",
            "solution": "e7g5",
            "rating": "1625",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle DxwLP",
            "fen": "2r3k1/pp2qpp1/7p/3BbP2/4P1Q1/P5P1/1r5P/3R1RK1 w - - 0 25",
            "solution": "f5f6",
            "rating": "1642",
            "theme": "crushing defensiveMove long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 0DPEV",
            "fen": "2r5/1b2kpQ1/2p1p1Bp/1p2P3/pP1r4/6RP/1P3qPK/8 w - - 0 41",
            "solution": "g3f3",
            "rating": "1649",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle CrRYk",
            "fen": "5rk1/pRp2pp1/4pq1p/6r1/3P4/2P1P3/P1Q2PPP/5RK1 b - - 0 21",
            "solution": "f6f3",
            "rating": "1788",
            "theme": "crushing endgame pin short"
        },
        {
            "name": "Lichess Puzzle B4zGN",
            "fen": "q5k1/5pp1/2p2n1p/8/pPQ1N3/P5PP/3r1PK1/2R5 b - - 4 32",
            "solution": "c6c5",
            "rating": "1816",
            "theme": "crushing endgame long pin"
        },
        {
            "name": "Lichess Puzzle LR7Pf",
            "fen": "7k/1p4r1/p1pP2q1/4Q2p/6P1/5p1P/PP3P1K/8 w - - 0 34",
            "solution": "d6d7",
            "rating": "1872",
            "theme": "advancedPawn crushing endgame pin promotion short"
        },
        {
            "name": "Lichess Puzzle 9DATo",
            "fen": "4r1k1/2N2pbp/p5p1/2p5/1p4nN/1P4P1/P5BP/3R2K1 b - - 4 27",
            "solution": "g7d4",
            "rating": "1880",
            "theme": "crushing intermezzo long middlegame pin sacrifice"
        },
        {
            "name": "Lichess Puzzle Dxwqp",
            "fen": "rn6/pppbRp1k/6pp/8/3N4/2KB4/PPP2q1P/4R3 w - - 0 20",
            "solution": "e1f1",
            "rating": "1890",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle E2pmO",
            "fen": "3rr1k1/pb3ppp/1pq1p3/2n1B3/8/2PB1P2/P1PQ2PP/3RR1K1 w - - 1 20",
            "solution": "d2g5",
            "rating": "1915",
            "theme": "crushing middlegame pin short"
        },
        {
            "name": "Lichess Puzzle 5GkIT",
            "fen": "4r2k/2p4r/p2p1q2/1pnP2Q1/4P1R1/PP3PP1/5K1p/3B3R b - - 0 46",
            "solution": "c5e4",
            "rating": "1963",
            "theme": "advantage deflection long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 8MJpd",
            "fen": "rnbq3r/ppp1kBpp/8/4p2Q/4n3/3P4/PPP2bPP/RNB2RK1 w - - 0 9",
            "solution": "f1f2",
            "rating": "1968",
            "theme": "crushing opening pin short"
        },
        {
            "name": "Lichess Puzzle K4AdU",
            "fen": "8/p5pk/3p3p/2pP4/2P1r3/P3pR1P/3q4/5Q1K w - - 7 36",
            "solution": "f1b1",
            "rating": "1973",
            "theme": "crushing endgame long pin"
        },
        {
            "name": "Lichess Puzzle K2unG",
            "fen": "r4b1r/pp3kp1/2n1p1p1/2P5/2P5/4B3/P4KPq/R2Q1R2 w - - 0 18",
            "solution": "d1d7",
            "rating": "2062",
            "theme": "crushing deflection discoveredAttack middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle 3JKLC",
            "fen": "2rr2k1/1bq2ppp/pp2pb2/8/1PP3R1/P1NR3Q/1B4PP/6K1 w - - 1 23",
            "solution": "c3d5",
            "rating": "2084",
            "theme": "advantage discoveredAttack middlegame pin short"
        },
        {
            "name": "Lichess Puzzle H2FXX",
            "fen": "5rk1/3R1pp1/4p2p/b2pNb1P/3P1B2/2q3Q1/5PP1/6K1 w - - 0 30",
            "solution": "e5f3",
            "rating": "2128",
            "theme": "crushing middlegame pin quietMove short"
        },
        {
            "name": "Lichess Puzzle 8uWsv",
            "fen": "r3n1k1/2p1b1p1/p1q1p2p/1p6/3Q2P1/P1P4P/BP4P1/R4RK1 b - - 0 22",
            "solution": "e7c5",
            "rating": "2131",
            "theme": "advantage defensiveMove long middlegame pin"
        },
        {
            "name": "Lichess Puzzle LCnDN",
            "fen": "1k1rr3/ppp3p1/2P2p2/2Qbq3/1P1N2np/4P1P1/P2N3P/2R1R1K1 w - - 0 24",
            "solution": "c5b5",
            "rating": "2154",
            "theme": "crushing middlegame pin queensideAttack short"
        },
        {
            "name": "Lichess Puzzle 6XMrS",
            "fen": "3r3k/pb4pp/1pR5/3B1p2/1P6/P3B2P/1q3PQ1/6K1 b - - 1 30",
            "solution": "b7c6",
            "rating": "2164",
            "theme": "crushing middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle OOSNG",
            "fen": "1Qb2k2/2P1b3/p4p1q/4pP2/4P2p/P5r1/2R4K/4R3 w - - 3 42",
            "solution": "b8c8",
            "rating": "2186",
            "theme": "crushing hangingPiece middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle 8khMl",
            "fen": "r2r2k1/pb3ppp/1pn1pn2/1N2N3/1bB5/1P2P3/PB2KP1P/3R2R1 w - - 8 17",
            "solution": "e5c6",
            "rating": "2208",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 5eIaX",
            "fen": "1k1r1b1r/p4Qp1/2p2p1p/8/8/2Nb4/Pq3PPP/R2R2K1 w - - 0 21",
            "solution": "a1b1",
            "rating": "2228",
            "theme": "advantage middlegame pin short"
        },
        {
            "name": "Lichess Puzzle 6GE7I",
            "fen": "r2qr1k1/2p2p2/pn1b3p/1p6/3P4/1BP2P2/P2Q1P1P/4RRK1 w - - 0 18",
            "solution": "d2h6",
            "rating": "2284",
            "theme": "crushing kingsideAttack master middlegame pin sacrifice short"
        },
        {
            "name": "Lichess Puzzle 8lLYy",
            "fen": "r6r/p2kb1R1/R7/1N1p3p/4P3/1P1n2P1/1P4P1/6K1 w - - 0 28",
            "solution": "a6d6",
            "rating": "2318",
            "theme": "crushing long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 5DBXS",
            "fen": "2r2rk1/1p4pp/p2q4/3Ppp2/1B2P1n1/P2Q4/1P3PPP/2R2RK1 b - - 1 20",
            "solution": "d6h6",
            "rating": "2319",
            "theme": "crushing middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle GG2Zh",
            "fen": "4r1k1/p6p/1p5P/2p2QP1/8/8/PPP2K2/4r3 b - - 4 42",
            "solution": "e8f8",
            "rating": "2321",
            "theme": "crushing endgame pin queenRookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle 4SBcR",
            "fen": "8/6p1/1P1p1pk1/P2P3p/5P1P/1rp3P1/2R3K1/8 b - - 1 47",
            "solution": "b3b2",
            "rating": "2324",
            "theme": "advancedPawn crushing endgame long pin rookEndgame"
        },
        {
            "name": "Lichess Puzzle 3NSL6",
            "fen": "r4rk1/5p2/p1b2ppb/1p6/2B3Q1/q4N2/6PP/3R1R1K w - - 0 26",
            "solution": "g4g6",
            "rating": "2399",
            "theme": "crushing discoveredAttack long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 8hLrF",
            "fen": "2k4r/pp1r1pp1/4p1p1/2b5/2QB2P1/2P4P/Pq3P2/R3K2R w KQ - 0 21",
            "solution": "e1g1",
            "rating": "2399",
            "theme": "advantage castling master middlegame pin short"
        },
        {
            "name": "Lichess Puzzle 9Bkpb",
            "fen": "r3k3/pbbn1qrp/1pp4Q/3ppP2/7B/3BP3/PPP3PP/R4RK1 w q - 0 20",
            "solution": "d3e2",
            "rating": "2491",
            "theme": "attackingF2F7 crushing discoveredAttack exposedKing middlegame pin quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle DqP8W",
            "fen": "r5k1/p4rpp/npp2p2/8/P2q4/1QN1R3/1P3PPP/R5K1 w - - 0 18",
            "solution": "e3e7",
            "rating": "2512",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle PHPUT",
            "fen": "4r1k1/pppq1pp1/3p1n1p/8/2P5/3Q1R1P/PPP3PN/4rR1K w - - 3 22",
            "solution": "f3f6",
            "rating": "2548",
            "theme": "advantage middlegame pin veryLong"
        },
        {
            "name": "Lichess Puzzle A3U79",
            "fen": "2r4k/2PR2pp/8/p1p1p3/3bQ1P1/5r2/q6P/1R5K w - - 0 34",
            "solution": "d7d8",
            "rating": "2647",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle 96AAS",
            "fen": "8/7p/8/2K1kp2/8/1P3P2/1P6/8 b - - 2 41",
            "solution": "h7h5",
            "rating": "2673",
            "theme": "advancedPawn crushing endgame interference pawnEndgame pin promotion quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 1EuxB",
            "fen": "r2q1r2/pp4pk/5pnp/2p5/3p1R2/1B5P/PPQ2PP1/3R2K1 w - - 0 22",
            "solution": "c2f5",
            "rating": "2692",
            "theme": "advantage long middlegame pin"
        },
        {
            "name": "Lichess Puzzle FeTT7",
            "fen": "6k1/5pp1/1p3q1p/p1b5/2R5/1P2r1PP/P2Q1P2/3R1K2 b - - 1 32",
            "solution": "f6f3",
            "rating": "2695",
            "theme": "crushing endgame pin short"
        },
        {
            "name": "Lichess Puzzle 2Hxyb",
            "fen": "r4rk1/1p1n4/p4q1p/5Np1/P2PpQ2/2P3R1/1P3PK1/4R3 w - - 1 35",
            "solution": "f5h6",
            "rating": "2799",
            "theme": "crushing defensiveMove exposedKing kingsideAttack long middlegame pin"
        },
        {
            "name": "Lichess Puzzle BjUNX",
            "fen": "3r2k1/3r2pp/p1p2pq1/PpP5/1P1PpPPP/4P3/3K4/R2Q3R b - - 0 22",
            "solution": "d7d4",
            "rating": "811",
            "theme": "advantage endgame sacrifice short"
        },
        {
            "name": "Lichess Puzzle IvRvH",
            "fen": "7k/1p4bp/pq4p1/4n3/3rQ3/1B6/PPP3PP/5R1K w - - 1 36",
            "solution": "f1f8",
            "rating": "918",
            "theme": "crushing deflection long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle DU8Tr",
            "fen": "r1b2r2/2q2k1p/p1pp1npQ/3Pp3/2P1P3/2p2P2/PP6/R3KBNR w KQ - 0 18",
            "solution": "h6h7",
            "rating": "1036",
            "theme": "crushing long middlegame sacrifice skewer"
        },
        {
            "name": "Lichess Puzzle MTjzF",
            "fen": "4kr2/pN6/1q2p3/5np1/4Q3/3P4/PPP3PP/4R2K b - - 3 30",
            "solution": "f5g3",
            "rating": "1108",
            "theme": "crushing endgame sacrifice short"
        },
        {
            "name": "Lichess Puzzle 3197C",
            "fen": "3r2k1/1R3pp1/2r1b2p/2N2p2/R2P4/4P1P1/5P1P/6K1 b - - 1 30",
            "solution": "c6c5",
            "rating": "1193",
            "theme": "advantage endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle ICTju",
            "fen": "8/pkpp4/8/5PR1/1n4p1/5b2/3K4/8 w - - 1 43",
            "solution": "f5f6",
            "rating": "1235",
            "theme": "advancedPawn advantage endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle 3FFhY",
            "fen": "4n3/2P1N3/1K1p4/5p2/4kPp1/7p/7B/8 b - - 0 70",
            "solution": "e8c7",
            "rating": "1271",
            "theme": "advantage defensiveMove endgame master sacrifice short"
        },
        {
            "name": "Lichess Puzzle Gt0b3",
            "fen": "4r1k1/2p2pnn/p1q1r1pQ/1pN1p2p/1PPp2P1/P2P1P1P/4PNK1/2R2R2 b - - 6 26",
            "solution": "g6g5",
            "rating": "1384",
            "theme": "advantage middlegame quietMove sacrifice short"
        },
        {
            "name": "Lichess Puzzle 4GSxq",
            "fen": "1r5k/3Q2p1/2R4p/8/8/1pq3PP/6PK/8 b - - 2 37",
            "solution": "c3c6",
            "rating": "1443",
            "theme": "advancedPawn crushing endgame long promotion sacrifice"
        },
        {
            "name": "Lichess Puzzle OMBHB",
            "fen": "5r2/pb1r2k1/6p1/2P1R1P1/5P2/3BK3/P7/5R2 b - - 2 37",
            "solution": "d7d3",
            "rating": "1582",
            "theme": "attraction endgame equality long sacrifice skewer"
        },
        {
            "name": "Lichess Puzzle OAhzX",
            "fen": "2r4r/4q3/2ppk3/3np3/7p/B5p1/P4PPP/2QRR1K1 w - - 0 28",
            "solution": "e1e5",
            "rating": "1593",
            "theme": "advantage attraction middlegame sacrifice short"
        },
        {
            "name": "Lichess Puzzle 25BEk",
            "fen": "r5k1/ppp3p1/6qp/2PPpr1n/6Q1/P1B3PK/1P3P1P/R4R2 b - - 1 20",
            "solution": "h5f4",
            "rating": "1601",
            "theme": "advantage clearance long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle NnDO7",
            "fen": "2r3k1/1p1r2pp/1p2p3/1P5q/3nP3/2NR3P/2P2PP1/2Q2RK1 b - - 2 25",
            "solution": "c8c3",
            "rating": "1613",
            "theme": "advantage middlegame sacrifice short"
        },
        {
            "name": "Lichess Puzzle MuSFN",
            "fen": "3r1rk1/pb2qppp/2n1p3/2pnP3/2N5/2PB3N/P4PPP/2RQR1K1 w - - 1 17",
            "solution": "d3h7",
            "rating": "1618",
            "theme": "advantage attraction kingsideAttack middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 6fC7k",
            "fen": "2r1r2k/p3q1p1/1p2Qn1p/2b5/8/PB3N1P/1P3PP1/3RR1K1 b - - 0 28",
            "solution": "c5f2",
            "rating": "1672",
            "theme": "advantage attraction discoveredAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle D0oSv",
            "fen": "6k1/q4pp1/PrQbp2p/8/3p4/3P2P1/2P2P1P/R5K1 w - - 2 27",
            "solution": "c6b6",
            "rating": "1719",
            "theme": "advancedPawn advantage endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle 0MNGm",
            "fen": "6rk/1pp1Rp1p/p2p2r1/5q2/3P2P1/1PP2P2/P6Q/5RK1 b - - 4 28",
            "solution": "g6g4",
            "rating": "1731",
            "theme": "crushing endgame exposedKing kingsideAttack long sacrifice"
        },
        {
            "name": "Lichess Puzzle 5B9Ud",
            "fen": "b3r3/5pk1/2p4p/1pN1nRpq/8/1BP3P1/PP3PQ1/6K1 b - - 3 42",
            "solution": "e5f3",
            "rating": "1733",
            "theme": "clearance crushing kingsideAttack middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle PC3Tu",
            "fen": "8/1k6/7p/1N1p1Pp1/1r4P1/6PK/8/8 w - - 14 53",
            "solution": "f5f6",
            "rating": "1735",
            "theme": "advancedPawn advantage endgame long promotion sacrifice"
        },
        {
            "name": "Lichess Puzzle DlcIm",
            "fen": "r2r1qk1/1b3ppp/p1npp3/1pbN3Q/4P3/P3B1P1/1PP2PBP/3RR1K1 w - - 6 17",
            "solution": "d5f6",
            "rating": "1773",
            "theme": "crushing kingsideAttack long master middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 7IXCT",
            "fen": "4rr2/1p4qk/p4np1/2PpBbQp/P2P4/7P/1P3R1K/R4B2 b - - 10 35",
            "solution": "e8e5",
            "rating": "1786",
            "theme": "crushing middlegame sacrifice short"
        },
        {
            "name": "Lichess Puzzle 8UE81",
            "fen": "6rk/1R4b1/p5Qp/P4p2/8/5KPP/4r3/8 b - - 4 51",
            "solution": "e2f2",
            "rating": "1812",
            "theme": "attraction crushing discoveredAttack endgame long master sacrifice"
        },
        {
            "name": "Lichess Puzzle I2Jfo",
            "fen": "r3r1k1/pp3ppp/8/5B2/1q6/1P3Q2/P1Pn2PP/3R1R1K w - - 2 22",
            "solution": "f5h7",
            "rating": "1843",
            "theme": "attraction crushing kingsideAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle I8zmA",
            "fen": "1r1bR3/p2P4/2P3p1/1Pkb3p/P4K2/6P1/7P/8 w - - 15 44",
            "solution": "e8d8",
            "rating": "1863",
            "theme": "advancedPawn crushing endgame long promotion sacrifice"
        },
        {
            "name": "Lichess Puzzle 9B1tE",
            "fen": "1k1r3r/2p5/1p2Rp2/p2q1npp/P1Q5/2P5/3B1PPP/3R2K1 w - - 0 25",
            "solution": "e6b6",
            "rating": "1867",
            "theme": "advantage long middlegame queensideAttack sacrifice"
        },
        {
            "name": "Lichess Puzzle 401aN",
            "fen": "b4rk1/5pp1/4p2p/1B1pPq2/P2P4/2Q2N1P/6PK/8 w - - 4 39",
            "solution": "b5d3",
            "rating": "1888",
            "theme": "advantage defensiveMove discoveredAttack endgame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle OrHZ5",
            "fen": "8/8/4b3/2p1k3/2P4P/Pr1B4/1p1K4/1R6 b - - 2 52",
            "solution": "b3d3",
            "rating": "1909",
            "theme": "attraction crushing endgame exposedKing long sacrifice skewer"
        },
        {
            "name": "Lichess Puzzle 3GO12",
            "fen": "8/2qQ1pk1/p1n1p1p1/1pp5/6P1/2P5/PP3P2/3R2K1 b - - 2 30",
            "solution": "c7f4",
            "rating": "1956",
            "theme": "advantage endgame sacrifice short"
        },
        {
            "name": "Lichess Puzzle FzmPM",
            "fen": "4r3/p2rk2q/2Q1pp1P/3p2p1/P7/2P1P3/3K4/1R5R w - - 1 37",
            "solution": "c6d7",
            "rating": "1987",
            "theme": "attraction crushing endgame long sacrifice skewer"
        },
        {
            "name": "Lichess Puzzle A9HM7",
            "fen": "r2q3r/pppn2kB/5pp1/4pbN1/7R/8/PP1Q1PPP/4R1K1 w - - 0 21",
            "solution": "g5e6",
            "rating": "1992",
            "theme": "clearance crushing middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle CqPiZ",
            "fen": "3b1k1r/3R4/3p1p2/p4P1B/Pp1PPP1K/6p1/1P6/8 b - - 1 50",
            "solution": "g3g2",
            "rating": "2062",
            "theme": "advancedPawn crushing endgame sacrifice short"
        },
        {
            "name": "Lichess Puzzle E6y0J",
            "fen": "8/4rkpp/2Q5/p2P4/1PP5/P4Pn1/3p2PP/3q1RK1 w - - 0 38",
            "solution": "c6e6",
            "rating": "2084",
            "theme": "crushing endgame long master sacrifice"
        },
        {
            "name": "Lichess Puzzle LvTfr",
            "fen": "8/1pq4k/2p3pp/p1n1Qn2/8/2P3PP/PPBr4/1R2R1K1 b - - 1 26",
            "solution": "c7b6",
            "rating": "2129",
            "theme": "crushing discoveredAttack exposedKing long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle Ljb97",
            "fen": "8/p7/5pR1/2p1k2P/3n4/r1BP4/P7/5K2 w - - 3 40",
            "solution": "h5h6",
            "rating": "2143",
            "theme": "advancedPawn advantage endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle FlnLF",
            "fen": "2kr3r/1p3ppp/p1pq1n2/4pP2/3P4/2PQ3P/2P3PN/RR4K1 w - - 0 25",
            "solution": "b1b7",
            "rating": "2164",
            "theme": "attraction crushing defensiveMove queensideAttack sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle JjLPQ",
            "fen": "3r2k1/1p3pp1/p1p1b2p/2P1p2q/1P1rP3/P2n1P2/2Q1N1PP/3RRB1K w - - 6 24",
            "solution": "d1d3",
            "rating": "2213",
            "theme": "advantage middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 3CM9Y",
            "fen": "4r3/p5k1/1p3pP1/7p/3PrQ1N/3q3P/5PP1/5RK1 w - - 3 30",
            "solution": "h4f5",
            "rating": "2277",
            "theme": "advantage deflection endgame interference sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle DMOAd",
            "fen": "3r3r/p3kp2/1pQ2q2/2b3p1/8/6P1/PP2PPBP/3R1RK1 b - - 2 18",
            "solution": "c5f2",
            "rating": "2284",
            "theme": "attraction crushing kingsideAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle A8wel",
            "fen": "3rbrk1/1p4b1/p1p1p3/2P3Rp/3P1B1P/4PQ2/PPq2PP1/R3K3 w Q - 9 25",
            "solution": "g5g7",
            "rating": "2286",
            "theme": "attraction crushing deflection discoveredAttack exposedKing kingsideAttack middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 2hEBf",
            "fen": "r1b2rk1/p2p1ppp/4p3/2Qn4/6RP/P7/1PPBq3/2KR4 w - - 0 21",
            "solution": "g4g7",
            "rating": "2370",
            "theme": "attraction crushing kingsideAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle A786p",
            "fen": "r3r1k1/4p1b1/p1np2p1/4q3/1p1N2P1/1P1QB3/P1P1B3/1K5R w - - 2 22",
            "solution": "d3g6",
            "rating": "2375",
            "theme": "crushing exposedKing kingsideAttack long middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 2mDy3",
            "fen": "r5r1/p2np1k1/5np1/1p1Nq2p/8/P1R5/1PPQ2PP/5RK1 w - - 3 24",
            "solution": "c3e3",
            "rating": "2410",
            "theme": "advantage long master middlegame sacrifice"
        },
        {
            "name": "Lichess Puzzle 3yR9L",
            "fen": "2rr3k/1q3p2/p3pN1p/1p2P1p1/2bN2R1/4Q2P/6P1/6K1 w - - 2 37",
            "solution": "g4g5",
            "rating": "2447",
            "theme": "advantage defensiveMove middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 1cMH9",
            "fen": "r4r2/6k1/p5p1/1p2p3/8/q1P5/P1PKQ3/3R2R1 w - - 0 26",
            "solution": "g1g6",
            "rating": "2546",
            "theme": "attraction crushing endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle 7uY1M",
            "fen": "8/5p2/5Pk1/1p1pP3/3P1BKP/1r6/8/8 w - - 0 51",
            "solution": "h4h5",
            "rating": "2557",
            "theme": "advancedPawn clearance crushing endgame exposedKing quietMove sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 3mLih",
            "fen": "3Q4/p4ppk/2N1r3/5Rrp/1P2p2q/7P/P5P1/2R3K1 b - - 3 32",
            "solution": "g5g2",
            "rating": "2658",
            "theme": "attraction crushing middlegame sacrifice short"
        },
        {
            "name": "Lichess Puzzle MvGfh",
            "fen": "8/8/5ppp/5k2/2R2P1r/5K1P/6P1/8 w - - 0 42",
            "solution": "g2g4",
            "rating": "2667",
            "theme": "advancedPawn crushing endgame rookEndgame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle 7zZlh",
            "fen": "6k1/2p2pp1/bP3r1p/8/PQ6/2R1PB1P/3q1PP1/6K1 b - - 0 30",
            "solution": "f6f3",
            "rating": "2692",
            "theme": "advantage endgame long sacrifice"
        },
        {
            "name": "Lichess Puzzle 8Ez7T",
            "fen": "5rk1/pR4pp/1b1p1p2/3p4/3N2q1/1P2QNP1/Pr5P/R5K1 b - - 0 22",
            "solution": "g4h3",
            "rating": "2724",
            "theme": "advantage deflection intermezzo middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle KAk6d",
            "fen": "4Rbk1/3q2r1/p6Q/1p5P/5p2/1PrP1R2/P4PP1/6K1 w - - 1 37",
            "solution": "e8f8",
            "rating": "2730",
            "theme": "attraction crushing exposedKing middlegame sacrifice veryLong"
        },
        {
            "name": "Lichess Puzzle Mg4ba",
            "fen": "8/5p2/p7/1p2k2b/1P5P/P1K1R3/5Br1/8 b - - 12 42",
            "solution": "e5f4",
            "rating": "920",
            "theme": "crushing defensiveMove endgame short"
        },
        {
            "name": "Lichess Puzzle F0k3J",
            "fen": "8/p6p/2Rk3b/1r6/Kp6/1B1n2P1/PP4P1/4R3 b - - 0 41",
            "solution": "d6c6",
            "rating": "956",
            "theme": "advantage defensiveMove endgame hangingPiece master short"
        },
        {
            "name": "Lichess Puzzle 09cHt",
            "fen": "8/2p2k2/1p1p4/1P1P4/3BB3/p2P1K2/1r6/8 w - - 2 55",
            "solution": "d4b2",
            "rating": "1047",
            "theme": "crushing defensiveMove endgame master short"
        },
        {
            "name": "Lichess Puzzle 5I5ZC",
            "fen": "8/p1p5/7p/2p3pP/3k2P1/2P2KP1/PP6/8 b - - 0 34",
            "solution": "d4d3",
            "rating": "1080",
            "theme": "crushing defensiveMove endgame oneMove pawnEndgame"
        },
        {
            "name": "Lichess Puzzle BPuW6",
            "fen": "5k2/pr3pp1/7p/1p6/1n1r3P/5P2/PPP3P1/R2R2K1 w - - 0 29",
            "solution": "d1d4",
            "rating": "1178",
            "theme": "advantage defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle Bxd40",
            "fen": "8/5k2/p3p2p/1ppqP1p1/5p2/P1P2N1P/1P1Q3K/8 b - - 1 32",
            "solution": "d5f3",
            "rating": "1182",
            "theme": "crushing defensiveMove endgame hangingPiece long"
        },
        {
            "name": "Lichess Puzzle 2c0B0",
            "fen": "8/2p2k2/2pB1p2/1p3P1p/3P2PP/2PK4/8/8 b - - 0 44",
            "solution": "c7d6",
            "rating": "1203",
            "theme": "crushing defensiveMove endgame short"
        },
        {
            "name": "Lichess Puzzle NQnYp",
            "fen": "8/8/pp1p1k2/3P4/1PP2K1p/7P/8/8 b - - 3 43",
            "solution": "a6a5",
            "rating": "1252",
            "theme": "crushing defensiveMove endgame long pawnEndgame"
        },
        {
            "name": "Lichess Puzzle DL4u5",
            "fen": "6k1/5pp1/7p/2Q5/1B2b1q1/3N2P1/5P1P/2R3K1 b - - 2 38",
            "solution": "g4f3",
            "rating": "1404",
            "theme": "defensiveMove endgame equality veryLong"
        },
        {
            "name": "Lichess Puzzle Fe3me",
            "fen": "8/p6p/8/5P2/1k6/4K2P/8/8 b - - 0 50",
            "solution": "b4c5",
            "rating": "1437",
            "theme": "crushing defensiveMove endgame pawnEndgame short"
        },
        {
            "name": "Lichess Puzzle 3QfGz",
            "fen": "7r/pp4k1/4N3/8/7R/6P1/PP1K3p/8 b - - 3 42",
            "solution": "g7g8",
            "rating": "1474",
            "theme": "advantage defensiveMove endgame short"
        },
        {
            "name": "Lichess Puzzle 3GTfY",
            "fen": "8/p7/1p1k3p/2p5/PPPK2pP/6P1/8/8 w - - 0 36",
            "solution": "b4c5",
            "rating": "1480",
            "theme": "crushing defensiveMove endgame pawnEndgame short"
        },
        {
            "name": "Lichess Puzzle IsCZw",
            "fen": "r5k1/pq3pp1/7p/1p6/1p1Q4/P2P1PP1/1B1Nr1K1/R7 w - - 1 24",
            "solution": "g2f1",
            "rating": "1513",
            "theme": "crushing defensiveMove middlegame short"
        },
        {
            "name": "Lichess Puzzle NCgrb",
            "fen": "2kr3r/pp1n1p2/2p3p1/4p1qp/4P3/PP2P2B/R1P3PP/3Q1RK1 w - - 0 20",
            "solution": "f1f7",
            "rating": "1594",
            "theme": "advantage defensiveMove middlegame short"
        },
        {
            "name": "Lichess Puzzle HLgWj",
            "fen": "6k1/p5b1/2p3p1/2Np2Pp/P2Pp1b1/4Q2q/5R1K/8 w - - 2 35",
            "solution": "e3h3",
            "rating": "1598",
            "theme": "advantage defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle 2oX3s",
            "fen": "8/p5p1/1pp5/2Pp2k1/3Pp1P1/1P2P1K1/P7/8 w - - 0 47",
            "solution": "c5b6",
            "rating": "1607",
            "theme": "crushing defensiveMove endgame pawnEndgame short"
        },
        {
            "name": "Lichess Puzzle 8INrW",
            "fen": "8/6p1/3b2p1/8/1kpK2P1/4N2P/P7/8 b - - 5 51",
            "solution": "d6c5",
            "rating": "1632",
            "theme": "crushing defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle JQcKb",
            "fen": "b4n2/6k1/3p1p2/1p1P2NQ/1PpPPpP1/2q5/5PK1/3B4 b - - 0 35",
            "solution": "f6g5",
            "rating": "1662",
            "theme": "advantage defensiveMove endgame short"
        },
        {
            "name": "Lichess Puzzle NfONO",
            "fen": "4k3/1bp1b2p/p2r3P/2p1RN2/6p1/1P1P2P1/PBQ2n2/1K5q w - - 2 35",
            "solution": "b2c1",
            "rating": "1689",
            "theme": "advantage defensiveMove exposedKing long middlegame"
        },
        {
            "name": "Lichess Puzzle 11eVa",
            "fen": "8/8/r4k2/2p2r2/7R/4PB1p/5K2/8 w - - 0 49",
            "solution": "h4h6",
            "rating": "1782",
            "theme": "advantage defensiveMove endgame long skewer"
        },
        {
            "name": "Lichess Puzzle 5Rywq",
            "fen": "3rk2r/Q2b2q1/1p2pb2/4p3/3nN3/BP2NpP1/P7/R3R1K1 w k - 3 32",
            "solution": "e4d6",
            "rating": "1784",
            "theme": "advantage defensiveMove discoveredAttack exposedKing middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle GvTEJ",
            "fen": "8/1q6/2p5/2Pp4/1P1P1pk1/3KP3/8/3Q4 b - - 1 62",
            "solution": "f4f3",
            "rating": "1815",
            "theme": "crushing defensiveMove endgame queenEndgame short"
        },
        {
            "name": "Lichess Puzzle GS1gT",
            "fen": "2k2r2/1R4p1/p2r4/7p/P1QP1q2/3B1P1P/6P1/6K1 b - - 3 35",
            "solution": "c8b7",
            "rating": "1833",
            "theme": "crushing defensiveMove endgame hangingPiece short"
        },
        {
            "name": "Lichess Puzzle DyNCf",
            "fen": "4r1k1/p4ppp/1pp2q2/8/1P1NrP1n/P2QP1RP/6P1/4R1K1 b - - 1 26",
            "solution": "e4d4",
            "rating": "1870",
            "theme": "advantage defensiveMove middlegame short"
        },
        {
            "name": "Lichess Puzzle JQv1p",
            "fen": "8/5p2/3kpp2/7p/4K1PP/4P3/5P2/8 b - - 0 40",
            "solution": "h5g4",
            "rating": "1873",
            "theme": "crushing defensiveMove endgame long pawnEndgame quietMove"
        },
        {
            "name": "Lichess Puzzle Dkw1C",
            "fen": "6kr/pp3p1p/4pQpP/3pP3/n2P4/8/Pq2KPP1/7R w - - 0 29",
            "solution": "e2f3",
            "rating": "1898",
            "theme": "crushing defensiveMove endgame master quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle Du1B9",
            "fen": "2kr4/npp5/p3Bp2/2P1r3/PP4p1/1Q2P1P1/5P1q/R3RK2 b - - 7 30",
            "solution": "c8b8",
            "rating": "1905",
            "theme": "advantage defensiveMove long middlegame"
        },
        {
            "name": "Lichess Puzzle 5sG4Q",
            "fen": "5q1k/3R1r1p/4Q2b/1Np1P3/Pp4p1/1P4P1/5PK1/3R4 b - - 0 39",
            "solution": "f7f2",
            "rating": "1937",
            "theme": "crushing defensiveMove middlegame short"
        },
        {
            "name": "Lichess Puzzle J9S1n",
            "fen": "8/p3k3/3r1Rp1/1p4Pp/2p1K2P/P3P3/1P6/8 w - - 6 39",
            "solution": "f6d6",
            "rating": "1967",
            "theme": "crushing defensiveMove endgame long rookEndgame"
        },
        {
            "name": "Lichess Puzzle Fg8W1",
            "fen": "8/5ppp/8/3k4/5P1P/3K2P1/8/8 b - - 0 38",
            "solution": "h7h5",
            "rating": "1981",
            "theme": "crushing defensiveMove endgame pawnEndgame quietMove veryLong zugzwang"
        },
        {
            "name": "Lichess Puzzle Hypok",
            "fen": "Q4r2/8/8/2p5/8/2K5/1PP3k1/R6q b - - 0 66",
            "solution": "f8a8",
            "rating": "2003",
            "theme": "crushing defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle IgTiY",
            "fen": "8/8/6K1/1kb4P/1p6/1P1p4/8/4B3 b - - 0 51",
            "solution": "c5e3",
            "rating": "2008",
            "theme": "advancedPawn bishopEndgame crushing defensiveMove endgame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle GqZRr",
            "fen": "1b3k2/1N1r1pp1/2p2n1p/1pP1p3/1P2P3/2N2P2/2K3PP/R7 w - - 3 27",
            "solution": "b7a5",
            "rating": "2104",
            "theme": "crushing defensiveMove endgame short"
        },
        {
            "name": "Lichess Puzzle 4npFe",
            "fen": "5b2/4pk1r/R4p1p/1P1Q4/1q1P2b1/4P3/1P3PPP/2B1R1K1 b - - 4 22",
            "solution": "e7e6",
            "rating": "2118",
            "theme": "crushing defensiveMove long middlegame"
        },
        {
            "name": "Lichess Puzzle 51ULj",
            "fen": "8/1p4p1/p7/k7/3KP2P/8/7P/8 b - - 1 41",
            "solution": "b7b5",
            "rating": "2184",
            "theme": "crushing defensiveMove endgame long pawnEndgame quietMove"
        },
        {
            "name": "Lichess Puzzle 3xFI7",
            "fen": "1k6/1p4QN/q3p2p/1p1pP3/3n4/1P5P/2P2PP1/6K1 b - - 0 29",
            "solution": "a6a1",
            "rating": "2206",
            "theme": "crushing defensiveMove endgame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 2lBRO",
            "fen": "1q6/pk3P1R/4P3/1p1pK3/2pP4/2P5/PP3b2/8 w - - 3 50",
            "solution": "e5f5",
            "rating": "2353",
            "theme": "crushing defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle Kt0v8",
            "fen": "1r4k1/pq3pp1/3P3p/4Q3/4P3/7P/1p3PP1/1R4K1 w - - 0 29",
            "solution": "d6d7",
            "rating": "2398",
            "theme": "advancedPawn crushing defensiveMove endgame veryLong"
        },
        {
            "name": "Lichess Puzzle NuGQY",
            "fen": "r4rk1/6pp/2PRb3/pB2N3/4n3/1p5P/5P2/4K2R w K - 0 30",
            "solution": "d6e6",
            "rating": "2403",
            "theme": "advantage castling defensiveMove hangingPiece middlegame short"
        },
        {
            "name": "Lichess Puzzle NONmF",
            "fen": "8/1QR3pk/8/3prq2/7p/1P3pPP/P4P1K/5R2 b - - 1 38",
            "solution": "h4g3",
            "rating": "2427",
            "theme": "advancedPawn clearance crushing defensiveMove endgame veryLong"
        },
        {
            "name": "Lichess Puzzle OYIlu",
            "fen": "rn1q3k/ppp2pp1/6Bp/3P2Nb/3P1Q2/8/PP3PPP/4R1K1 b - - 5 20",
            "solution": "h5g6",
            "rating": "2437",
            "theme": "advantage defensiveMove hangingPiece long middlegame"
        },
        {
            "name": "Lichess Puzzle 7wSYp",
            "fen": "6r1/6pk/2Q2p2/4pP2/Np2P2p/1P1P3P/1PP2RK1/4q3 b - - 0 30",
            "solution": "g7g6",
            "rating": "2505",
            "theme": "advantage defensiveMove discoveredAttack endgame long"
        },
        {
            "name": "Lichess Puzzle 5iYWF",
            "fen": "5n2/8/1p6/6P1/1k1N1K2/8/8/8 w - - 2 52",
            "solution": "f4f5",
            "rating": "2515",
            "theme": "crushing defensiveMove endgame knightEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle Er9Ph",
            "fen": "6r1/P7/1K6/2p1k3/2P5/8/1P6/8 w - - 1 51",
            "solution": "b6c5",
            "rating": "2527",
            "theme": "crushing defensiveMove endgame long rookEndgame"
        },
        {
            "name": "Lichess Puzzle NNzvU",
            "fen": "1k1rr3/1p3pp1/p6p/Q1p5/7P/1PBqPP2/P1R2K2/2R5 b - - 0 27",
            "solution": "d3e3",
            "rating": "2543",
            "theme": "crushing defensiveMove master middlegame short"
        },
        {
            "name": "Lichess Puzzle 0WlrM",
            "fen": "6k1/8/p4p2/1pp1r1p1/8/1PP2PP1/P5K1/4R3 w - - 0 38",
            "solution": "e1e5",
            "rating": "2607",
            "theme": "crushing defensiveMove endgame rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle 8oW1H",
            "fen": "1r3R2/4n1P1/6K1/8/6kP/8/8/8 w - - 11 55",
            "solution": "g6f7",
            "rating": "2625",
            "theme": "clearance crushing defensiveMove endgame long"
        },
        {
            "name": "Lichess Puzzle 4acOZ",
            "fen": "8/1p6/p1p2qkp/P1PpQ1p1/3P4/4P1KP/6P1/8 w - - 9 49",
            "solution": "e5e8",
            "rating": "2706",
            "theme": "crushing defensiveMove endgame long queenEndgame"
        },
        {
            "name": "Lichess Puzzle JMp3d",
            "fen": "5rk1/5ppp/P7/2Q5/2P1p2P/P3KbPq/5P2/R3R3 b - - 0 40",
            "solution": "f8d8",
            "rating": "2782",
            "theme": "advantage defensiveMove endgame exposedKing quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle MrFMW",
            "fen": "8/8/2k4P/1p3R2/p5p1/6K1/5P2/1r6 w - - 1 51",
            "solution": "f5h5",
            "rating": "2795",
            "theme": "advancedPawn crushing defensiveMove endgame rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle Ck6DJ",
            "fen": "2r2rk1/pp3p2/4pBB1/n2qP2p/8/P5Q1/1P3PbP/R3R1K1 w - - 5 23",
            "solution": "g6b1",
            "rating": "1702",
            "theme": "discoveredAttack mate mateIn1 middlegame oneMove"
        },
        {
            "name": "Lichess Puzzle 3rQvw",
            "fen": "1k1r4/1p6/1n5p/2p3p1/2PN1n2/1P3K2/P5PP/3R4 w - - 0 31",
            "solution": "d4c6",
            "rating": "728",
            "theme": "advantage discoveredAttack endgame short"
        },
        {
            "name": "Lichess Puzzle Hx76e",
            "fen": "8/1b2k3/7p/3P1pp1/3P4/4KB1P/8/8 w - - 1 40",
            "solution": "d5d6",
            "rating": "847",
            "theme": "bishopEndgame crushing discoveredAttack endgame short"
        },
        {
            "name": "Lichess Puzzle NZq9k",
            "fen": "r2q1rk1/pp1nbppp/2p1pn2/6B1/4N3/3P1B1P/PPPQ1PP1/R4RK1 b - - 7 12",
            "solution": "f6e4",
            "rating": "1078",
            "theme": "advantage discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle Op5SI",
            "fen": "r1b1r1k1/1p3ppp/p2b1n2/3p4/1P1Np3/P1P1BPP1/4P1BP/R3K2R b KQ - 0 21",
            "solution": "e4f3",
            "rating": "1120",
            "theme": "crushing discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle 3S86v",
            "fen": "r4rk1/2Q3pp/p4n2/1pb3N1/2q5/1P6/P4PPP/R1B2RK1 b - - 0 21",
            "solution": "c5f2",
            "rating": "1174",
            "theme": "crushing discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle GbJ67",
            "fen": "8/5kpp/2q2P2/2bp4/8/2Q4P/3N1PP1/6K1 b - - 0 28",
            "solution": "c5f2",
            "rating": "1205",
            "theme": "crushing discoveredAttack endgame short"
        },
        {
            "name": "Lichess Puzzle C3txx",
            "fen": "8/5kp1/7p/2b1p3/3pR3/2P2P1P/1P3NPK/r7 b - - 2 32",
            "solution": "d4c3",
            "rating": "1392",
            "theme": "advantage discoveredAttack endgame master short"
        },
        {
            "name": "Lichess Puzzle Hg6p4",
            "fen": "r4rk1/pp3ppp/2np4/3N4/4P2q/2NB1Q1b/PPP2R2/R6K b - - 5 20",
            "solution": "h3g4",
            "rating": "1406",
            "theme": "crushing discoveredAttack kingsideAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle 7rxcb",
            "fen": "r5r1/pp2np1k/2p1p2P/2PpB3/3Pb3/7R/P3B2K/6R1 w - - 3 39",
            "solution": "g1g7",
            "rating": "1474",
            "theme": "advancedPawn crushing discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle BBV97",
            "fen": "4rr2/p4qpk/1p5p/4p3/1PP1Rn1P/6B1/PQ4P1/5RK1 b - - 2 34",
            "solution": "f4h3",
            "rating": "1475",
            "theme": "crushing deflection discoveredAttack kingsideAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle GsJ5g",
            "fen": "6r1/3q3k/p3p3/1p1b1p1p/5Q1B/3B4/PPP2PP1/R4RK1 b - - 4 28",
            "solution": "g8g2",
            "rating": "1488",
            "theme": "crushing discoveredAttack kingsideAttack long middlegame"
        },
        {
            "name": "Lichess Puzzle JAelV",
            "fen": "2r3k1/3q2pp/2N1p3/8/8/7P/5PP1/2Q3K1 w - - 1 33",
            "solution": "c6e7",
            "rating": "1502",
            "theme": "deflection discoveredAttack endgame equality short"
        },
        {
            "name": "Lichess Puzzle Asafo",
            "fen": "2r2r1k/pp3ppp/2q5/8/3R1P2/4P3/PP3P1P/1KQ4R b - - 4 24",
            "solution": "c6g6",
            "rating": "1555",
            "theme": "advantage discoveredAttack endgame queensideAttack short"
        },
        {
            "name": "Lichess Puzzle 3hq4x",
            "fen": "1rr3k1/p4pp1/2pq3p/3p4/3Pb3/P7/2QNnPBP/R1R4K w - - 0 25",
            "solution": "d2e4",
            "rating": "1574",
            "theme": "advantage discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle 4SfWZ",
            "fen": "5r2/1bp1q1kp/p5p1/1p1rN3/3Q1P2/1P6/P1P5/1K1R2R1 w - - 0 32",
            "solution": "e5g6",
            "rating": "1662",
            "theme": "crushing discoveredAttack long middlegame"
        },
        {
            "name": "Lichess Puzzle HqZPm",
            "fen": "3r2k1/2p2ppp/BQ2p3/4P3/3n4/5N1P/P4PP1/3R2K1 b - - 0 24",
            "solution": "d4f3",
            "rating": "1679",
            "theme": "crushing discoveredAttack endgame short"
        },
        {
            "name": "Lichess Puzzle 5AOWM",
            "fen": "r2qr1k1/2p2ppp/p2b4/np1P1b2/3Q4/1PN2n2/PBPPBPPP/R3R1K1 w - - 0 17",
            "solution": "e2f3",
            "rating": "1701",
            "theme": "advantage discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle 6eXbm",
            "fen": "5rk1/1qrRbpp1/4p2p/8/p1P1N3/5QP1/P4P1P/3R2K1 w - - 9 25",
            "solution": "e4f6",
            "rating": "1775",
            "theme": "advantage discoveredAttack long master middlegame xRayAttack"
        },
        {
            "name": "Lichess Puzzle PFrGV",
            "fen": "5r1r/1p1q1k1p/2p2ppP/4Q3/4P1P1/1P1Pb3/2P3KR/5R2 w - - 0 33",
            "solution": "f1f6",
            "rating": "1837",
            "theme": "attraction crushing discoveredAttack long master middlegame"
        },
        {
            "name": "Lichess Puzzle 2pu0H",
            "fen": "3B4/2R5/1kpp4/4p3/1P2P1B1/K1Pn1P2/1r6/2b5 b - - 0 43",
            "solution": "b2b4",
            "rating": "1928",
            "theme": "crushing discoveredAttack endgame master short"
        },
        {
            "name": "Lichess Puzzle JODDw",
            "fen": "1r1r2k1/R4p2/6pp/1ppb4/4NP2/1R4P1/4K1BP/3n4 w - - 0 35",
            "solution": "e4f6",
            "rating": "1953",
            "theme": "crushing discoveredAttack master middlegame short"
        },
        {
            "name": "Lichess Puzzle LuW01",
            "fen": "r4r1k/1p1q1Bpp/5b2/p6Q/8/5R2/P5PP/5R1K w - - 2 29",
            "solution": "f3h3",
            "rating": "2031",
            "theme": "advantage discoveredAttack middlegame quietMove short"
        },
        {
            "name": "Lichess Puzzle Ls23z",
            "fen": "1qrr2k1/4bppp/b3pn2/1p2N3/1P6/P1N1Q2P/1B3PP1/2R1R1K1 w - - 1 24",
            "solution": "e5f7",
            "rating": "2036",
            "theme": "attraction crushing discoveredAttack kingsideAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle HODMb",
            "fen": "5rk1/1p1Q1pp1/1p1p3p/1Pp1p1q1/2B1P3/r1PP4/7N/5R1K w - - 0 24",
            "solution": "f1f7",
            "rating": "2045",
            "theme": "crushing discoveredAttack middlegame short"
        },
        {
            "name": "Lichess Puzzle 9wFE6",
            "fen": "3r2k1/2p2pp1/2p2n1p/p3r3/4P3/1PP2Q1P/q2N1RP1/3R3K w - - 0 23",
            "solution": "d2f1",
            "rating": "2089",
            "theme": "crushing discoveredAttack middlegame quietMove short"
        },
        {
            "name": "Lichess Puzzle Aiefa",
            "fen": "8/3N2r1/3P1p2/8/4p1k1/8/3R2K1/8 b - - 0 66",
            "solution": "g4f5",
            "rating": "2098",
            "theme": "crushing discoveredAttack endgame intermezzo long"
        },
        {
            "name": "Lichess Puzzle BaDg5",
            "fen": "1k2r2r/pp5p/1P3pp1/3p1n2/8/2PQ1NP1/5qPK/RR6 w - - 0 28",
            "solution": "b6a7",
            "rating": "2100",
            "theme": "advancedPawn crushing discoveredAttack master middlegame queensideAttack short"
        },
        {
            "name": "Lichess Puzzle JhLlW",
            "fen": "2r2rk1/1q2bppp/p1npNn2/1p4B1/4P3/1B3Q1P/PP3PP1/2RR2K1 b - - 0 17",
            "solution": "c6e5",
            "rating": "2110",
            "theme": "advantage discoveredAttack long middlegame"
        },
        {
            "name": "Lichess Puzzle LUPbH",
            "fen": "2kr4/1pp4p/p3r1p1/P1P1bp2/8/1P2B3/5PPP/3RR1K1 w - - 0 25",
            "solution": "d1d8",
            "rating": "2112",
            "theme": "advantage discoveredAttack endgame long"
        },
        {
            "name": "Lichess Puzzle IrlIS",
            "fen": "1k6/pp3b2/2p1qp1b/5Qpp/3PB2P/P2N2P1/1PP5/1K6 b - - 3 28",
            "solution": "e6a2",
            "rating": "2134",
            "theme": "crushing discoveredAttack endgame master short"
        },
        {
            "name": "Lichess Puzzle Lm4Tq",
            "fen": "1kB1r3/p4R1p/1p6/1Q1p4/1b6/1q3P2/1B1P1P1P/2KR4 b - - 0 24",
            "solution": "b4d2",
            "rating": "2223",
            "theme": "crushing discoveredAttack middlegame queensideAttack short"
        },
        {
            "name": "Lichess Puzzle DrUd2",
            "fen": "rn1r2k1/3q3p/1ppB1bp1/2Pb1pN1/1p5P/P4NP1/4PPB1/3Q1RK1 w - - 0 19",
            "solution": "f3e5",
            "rating": "2238",
            "theme": "crushing discoveredAttack interference middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle KZDTb",
            "fen": "2r3k1/pp1b2pp/1q2Pb2/2np4/6B1/2N5/PPP3PP/R3QRK1 b - - 0 18",
            "solution": "c5d3",
            "rating": "2248",
            "theme": "advantage discoveredAttack middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle 4v9FP",
            "fen": "2krr3/1p1q3p/1Qpb2p1/8/P7/2N4P/1P3PP1/R1BR2K1 b - - 2 20",
            "solution": "d6c7",
            "rating": "2270",
            "theme": "advantage discoveredAttack middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle Bxh2b",
            "fen": "7k/1p2r1p1/p1b1p1B1/5nN1/5Pn1/8/1P5P/3R2K1 w - - 2 31",
            "solution": "d1d8",
            "rating": "2291",
            "theme": "crushing discoveredAttack middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle JHTVQ",
            "fen": "7k/6p1/7p/5Q2/3p4/P2Rrp1K/3R3P/4q3 b - - 1 48",
            "solution": "f3f2",
            "rating": "2292",
            "theme": "advancedPawn crushing discoveredAttack endgame exposedKing long promotion"
        },
        {
            "name": "Lichess Puzzle HRDWp",
            "fen": "2r2rk1/2q3pp/p2p1b2/2nP1p2/4pP2/1Q2B3/P3B1PP/2R2RK1 w - - 0 23",
            "solution": "e3c5",
            "rating": "2394",
            "theme": "advantage discoveredAttack long middlegame quietMove"
        },
        {
            "name": "Lichess Puzzle 3tvCo",
            "fen": "1k1r3r/pb3q2/1pnPp2p/4B1p1/6P1/3p4/PP2QPB1/2R2RK1 w - - 0 25",
            "solution": "d6d7",
            "rating": "2459",
            "theme": "advancedPawn crushing discoveredAttack master middlegame short"
        },
        {
            "name": "Lichess Puzzle H0BaI",
            "fen": "1r2r3/p4pkp/3p4/q1pP1bPR/2p2N2/2P2P2/PP1Q4/2KR4 w - - 5 23",
            "solution": "g5g6",
            "rating": "2464",
            "theme": "advantage discoveredAttack middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle NbTsL",
            "fen": "6r1/3kbp2/4q3/pP1pP2Q/P1pP4/5N1P/1P3Pr1/R4R1K b - - 0 33",
            "solution": "g2g6",
            "rating": "2481",
            "theme": "advantage discoveredAttack exposedKing middlegame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle FtD27",
            "fen": "3r2k1/p5pp/1p6/3N4/4pPn1/7P/P1R3P1/6K1 w - - 0 23",
            "solution": "c2d2",
            "rating": "2670",
            "theme": "crushing discoveredAttack endgame long quietMove"
        },
        {
            "name": "Lichess Puzzle PEiyS",
            "fen": "5rk1/pp4p1/2r1pp2/3p4/5Q2/P2BR3/KPP2Pq1/8 w - - 0 26",
            "solution": "f4h4",
            "rating": "2694",
            "theme": "advantage discoveredAttack endgame veryLong"
        },
        {
            "name": "Lichess Puzzle A8TyJ",
            "fen": "4r1k1/pb3pp1/1p1bqn2/2p1N1R1/2Pr3B/7P/PPB2Q2/5RK1 w - - 1 29",
            "solution": "g5g7",
            "rating": "2726",
            "theme": "advantage attraction discoveredAttack kingsideAttack master middlegame veryLong"
        },
        {
            "name": "Lichess Puzzle IvFSs",
            "fen": "2kr3r/1p6/p5p1/P1q2R2/3N2n1/3P2p1/1PB1N1P1/R1BQ2K1 b - - 0 26",
            "solution": "g6f5",
            "rating": "2731",
            "theme": "crushing discoveredAttack middlegame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 0csd3",
            "fen": "8/8/8/r7/6Pk/p4K2/R7/8 w - - 5 63",
            "solution": "a2h2",
            "rating": "943",
            "theme": "crushing endgame long rookEndgame skewer"
        },
        {
            "name": "Lichess Puzzle 8Ca3v",
            "fen": "4R3/8/8/1p3k2/1B2r3/P4rPK/8/8 w - - 11 62",
            "solution": "e8f8",
            "rating": "1068",
            "theme": "crushing endgame short skewer"
        },
        {
            "name": "Lichess Puzzle 5b4sC",
            "fen": "8/1p3r2/p2p4/3Pp3/2P2k2/2KB4/P1P3r1/7R w - - 12 54",
            "solution": "h1f1",
            "rating": "1237",
            "theme": "crushing endgame short skewer"
        },
        {
            "name": "Lichess Puzzle JdoJe",
            "fen": "3rr1k1/p1p2ppp/5n2/n7/q2P4/1pPQB2P/3N1PP1/1R2R1K1 w - - 0 23",
            "solution": "b1a1",
            "rating": "1286",
            "theme": "advantage middlegame short skewer"
        },
        {
            "name": "Lichess Puzzle 5BvES",
            "fen": "5r2/8/3k1p2/3pp2p/5P1P/4K1P1/1B6/8 w - - 1 57",
            "solution": "b2a3",
            "rating": "1391",
            "theme": "crushing endgame short skewer"
        },
        {
            "name": "Lichess Puzzle EblRR",
            "fen": "5rk1/4bpp1/6p1/1Qp5/1nn1q3/5NBP/PP3PP1/3R2K1 w - - 3 27",
            "solution": "d1e1",
            "rating": "1595",
            "theme": "advantage middlegame short skewer"
        },
        {
            "name": "Lichess Puzzle P3cwz",
            "fen": "r1br2k1/5pp1/p2Pp2p/1p3q2/5P2/7R/PP5Q/2K4R w - - 0 27",
            "solution": "h3h6",
            "rating": "1610",
            "theme": "crushing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle OTak5",
            "fen": "r6r/ppb2k2/2p2qp1/2Pp1b2/3PpP1R/4P1P1/P1QB1KB1/7R w - - 0 25",
            "solution": "h4h7",
            "rating": "1625",
            "theme": "crushing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle GbUH8",
            "fen": "4k2r/5ppp/4p2n/4P3/3q4/1P1p4/r4PPP/2RQ1RK1 w k - 0 21",
            "solution": "c1c8",
            "rating": "1634",
            "theme": "equality middlegame short skewer"
        },
        {
            "name": "Lichess Puzzle P80zv",
            "fen": "5b2/p3kp2/1n2p1p1/1B1pP1P1/1r1P1P2/8/R3N1K1/8 w - - 3 37",
            "solution": "a2a7",
            "rating": "1725",
            "theme": "crushing deflection endgame short skewer"
        },
        {
            "name": "Lichess Puzzle 5c1I7",
            "fen": "5rk1/3bp1Bp/3p1np1/8/1p1NP1P1/qP3P2/2PQ4/1K1R3R b - - 0 23",
            "solution": "f8a8",
            "rating": "1728",
            "theme": "crushing middlegame queensideAttack quietMove skewer veryLong"
        },
        {
            "name": "Lichess Puzzle Ad67J",
            "fen": "1r2r3/6kp/1p4p1/p3pp2/n7/q1P2P2/2RN2PP/1R3Q1K w - - 0 32",
            "solution": "b1a1",
            "rating": "1756",
            "theme": "advantage middlegame short skewer"
        },
        {
            "name": "Lichess Puzzle NWv5U",
            "fen": "4r3/2k4p/6p1/3p4/R2Kp2P/8/PPP2PP1/8 b - - 0 38",
            "solution": "e4e3",
            "rating": "1759",
            "theme": "clearance crushing endgame long rookEndgame skewer"
        },
        {
            "name": "Lichess Puzzle NiJIV",
            "fen": "r3k2r/pp3pp1/1q6/2b1pb2/8/1B1PBPpP/PPP1Q1P1/4RR1K b kq - 4 21",
            "solution": "f5h3",
            "rating": "1788",
            "theme": "crushing kingsideAttack master middlegame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle IMpQb",
            "fen": "r5k1/p4p1p/3p2p1/3q4/1Bp2P2/6P1/PPQ4P/3R1K2 b - - 0 25",
            "solution": "d5h1",
            "rating": "1807",
            "theme": "crushing endgame exposedKing interference long skewer"
        },
        {
            "name": "Lichess Puzzle EGhuy",
            "fen": "3q1r1b/r3k3/p5QP/1p2P1N1/2p2n2/2P5/PP3P2/2K3RR w - - 3 28",
            "solution": "g6h7",
            "rating": "1957",
            "theme": "crushing middlegame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle DgWvu",
            "fen": "2k4r/1ppnp1q1/pr6/4p3/6Q1/2PP1P2/PP1RK1P1/4R1B1 b - - 1 30",
            "solution": "b6g6",
            "rating": "1966",
            "theme": "crushing middlegame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle JfcA7",
            "fen": "r3k2r/pp3ppp/3bp3/2p1qb2/1nP5/P2P4/1P2B1PP/RNBQ1R1K w kq - 1 13",
            "solution": "c1f4",
            "rating": "1979",
            "theme": "advantage middlegame short skewer"
        },
        {
            "name": "Lichess Puzzle LwUEz",
            "fen": "8/R7/5p1p/5pkr/8/5K2/8/8 w - - 0 48",
            "solution": "a7g7",
            "rating": "2047",
            "theme": "crushing endgame quietMove rookEndgame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle BL8L8",
            "fen": "5rk1/pp4pp/q3pN2/2b2b2/2P5/2P3P1/P1Q2PP1/1R1B1RK1 b - - 0 23",
            "solution": "f8f6",
            "rating": "2055",
            "theme": "advantage long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle 0ZnjA",
            "fen": "3k4/Q2n2q1/7p/1ppb4/8/2P5/1P4PP/4R1K1 w - - 4 34",
            "solution": "a7a5",
            "rating": "2087",
            "theme": "crushing endgame exposedKing skewer veryLong"
        },
        {
            "name": "Lichess Puzzle JoYvz",
            "fen": "8/5pk1/6p1/8/2B3KP/5Q2/5q2/8 b - - 2 64",
            "solution": "f7f5",
            "rating": "2089",
            "theme": "crushing deflection endgame long skewer"
        },
        {
            "name": "Lichess Puzzle Ozvx3",
            "fen": "2r4k/3nbppP/pp1p4/4pP1Q/PR6/8/2qB3P/5RK1 w - - 0 25",
            "solution": "f1c1",
            "rating": "2130",
            "theme": "advantage middlegame short skewer"
        },
        {
            "name": "Lichess Puzzle Dr4rt",
            "fen": "k2r4/N7/1R2p1p1/1P1p1p2/3P4/3Q1p1q/P4Kn1/6R1 b - - 1 40",
            "solution": "h3h4",
            "rating": "2209",
            "theme": "crushing exposedKing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle 0C3qv",
            "fen": "3r1rk1/ppNR1pp1/2p4p/q3p3/4P1Q1/2P1P3/1P4PP/2K2R2 b - - 2 20",
            "solution": "d8d7",
            "rating": "2211",
            "theme": "crushing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle 1mIKo",
            "fen": "6k1/1pp3p1/p2pprrp/4p3/PP1PP2q/2P1R2P/3N1PQK/5RN1 b - - 0 28",
            "solution": "g6g2",
            "rating": "2224",
            "theme": "advantage attraction long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle F33tm",
            "fen": "2r2rk1/1b4p1/pq2pb1p/1p1pN3/1P1P4/P5NP/3Q1PP1/2RR2K1 b - - 1 26",
            "solution": "c8c1",
            "rating": "2242",
            "theme": "crushing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle CoSMq",
            "fen": "8/pb4k1/1p2pr1p/3qN1pQ/3P4/P1P4P/1PB1K3/2BR4 b - - 1 34",
            "solution": "d5g2",
            "rating": "2430",
            "theme": "crushing deflection exposedKing long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle H820a",
            "fen": "8/8/5b2/3R1P1P/2P3k1/pP6/r5P1/1K3R2 b - - 1 43",
            "solution": "a2a1",
            "rating": "2440",
            "theme": "advancedPawn crushing endgame exposedKing skewer veryLong"
        },
        {
            "name": "Lichess Puzzle FSZUk",
            "fen": "4rrk1/1pb5/p5q1/2Ppnp1Q/3Bp1pR/2P5/4B1PP/5RK1 w - - 2 31",
            "solution": "h5h8",
            "rating": "2452",
            "theme": "crushing exposedKing kingsideAttack master middlegame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle 6S20x",
            "fen": "r3k1r1/pp3p1p/2p1p3/6Qn/3PpP2/2P1P3/P1P4q/R3BK2 w q - 0 19",
            "solution": "g5g8",
            "rating": "2545",
            "theme": "advantage middlegame quietMove skewer veryLong"
        },
        {
            "name": "Lichess Puzzle 8xqqV",
            "fen": "6r1/8/3N2k1/1Pp1p3/p1P5/P1b1K3/5RP1/8 b - - 0 46",
            "solution": "c3d4",
            "rating": "2655",
            "theme": "crushing endgame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle 9nOau",
            "fen": "2r3k1/1Q1nppbp/p2q2p1/P2P4/2r5/2N1B2P/1P3PP1/3RR1K1 b - - 0 26",
            "solution": "c8b8",
            "rating": "2677",
            "theme": "advantage capturingDefender middlegame skewer veryLong"
        },
        {
            "name": "Lichess Puzzle 7ZIwM",
            "fen": "r7/ppp1kpb1/3p4/4pBPr/1NP1Pp1q/3P1P2/PP3R1Q/R5K1 b - - 4 24",
            "solution": "a8h8",
            "rating": "2684",
            "theme": "advantage exposedKing middlegame quietMove skewer veryLong"
        },
        {
            "name": "Lichess Puzzle NSxfP",
            "fen": "r6r/p4pp1/4p1k1/2Q1B2p/4R1n1/3P2Pq/PPP2P2/RN4K1 b - - 4 20",
            "solution": "a8c8",
            "rating": "2770",
            "theme": "advantage long middlegame skewer"
        },
        {
            "name": "Lichess Puzzle 0iODu",
            "fen": "8/8/8/2Kp4/P2P4/1k2n3/8/8 w - - 2 57",
            "solution": "a4a5",
            "rating": "800",
            "theme": "crushing endgame knightEndgame quietMove short"
        },
        {
            "name": "Lichess Puzzle 1v1Rl",
            "fen": "8/1p6/2p5/p2R4/6pP/P5K1/1Pk5/8 b - - 0 37",
            "solution": "c6d5",
            "rating": "1258",
            "theme": "advancedPawn crushing endgame hangingPiece promotion quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle H0NFk",
            "fen": "8/3k4/p2P4/3K1p2/1bP2P2/5P2/P7/8 w - - 3 52",
            "solution": "c4c5",
            "rating": "1290",
            "theme": "bishopEndgame crushing endgame quietMove short"
        },
        {
            "name": "Lichess Puzzle A7tp8",
            "fen": "8/8/2pk2p1/4p3/1bP1PpP1/3K1P2/3N4/8 b - - 1 45",
            "solution": "b4d2",
            "rating": "1328",
            "theme": "crushing endgame quietMove veryLong zugzwang"
        },
        {
            "name": "Lichess Puzzle KNf0L",
            "fen": "8/5pp1/1P5p/2R5/7P/P2k1P2/n1p2P2/6K1 b - - 0 36",
            "solution": "a2c3",
            "rating": "1474",
            "theme": "advancedPawn crushing endgame promotion quietMove short"
        },
        {
            "name": "Lichess Puzzle GLNhu",
            "fen": "4k3/5R2/4p1P1/3b4/3P4/4pN1r/8/4K3 w - - 1 60",
            "solution": "f3e5",
            "rating": "1558",
            "theme": "crushing endgame quietMove short"
        },
        {
            "name": "Lichess Puzzle G971e",
            "fen": "3r1rk1/4ppb1/pp4p1/2pq2N1/P4Q2/2P3N1/1P4PP/5R1K w - - 2 23",
            "solution": "f4h4",
            "rating": "1614",
            "theme": "advantage middlegame quietMove short"
        },
        {
            "name": "Lichess Puzzle DOuva",
            "fen": "8/8/2p1kp2/p2p2p1/1P1P2Pp/2P2P1P/2b1B1KB/8 b - - 0 38",
            "solution": "a5a4",
            "rating": "1632",
            "theme": "bishopEndgame crushing endgame long quietMove"
        },
        {
            "name": "Lichess Puzzle 0LETl",
            "fen": "8/2p5/p1p1k2p/6p1/PP1K2P1/7P/8/8 w - - 3 42",
            "solution": "d4c5",
            "rating": "1633",
            "theme": "crushing endgame long pawnEndgame quietMove zugzwang"
        },
        {
            "name": "Lichess Puzzle 4Uboi",
            "fen": "8/p1p4p/2K1k3/6p1/PPP5/6P1/8/8 b - - 0 34",
            "solution": "h7h5",
            "rating": "1636",
            "theme": "crushing endgame pawnEndgame quietMove short"
        },
        {
            "name": "Lichess Puzzle IV4Rm",
            "fen": "2r2r1k/5p1p/p3pPp1/1p1b4/5Q2/2qB3P/P5P1/3R1R1K w - - 1 27",
            "solution": "f4h6",
            "rating": "1751",
            "theme": "clearance crushing long middlegame quietMove"
        },
        {
            "name": "Lichess Puzzle 6DMSG",
            "fen": "r7/p2k3p/1q1b2b1/2p5/2N5/8/PP1R1PPP/4R1K1 w - - 2 28",
            "solution": "c4b6",
            "rating": "1862",
            "theme": "advantage long middlegame quietMove"
        },
        {
            "name": "Lichess Puzzle 3KKcX",
            "fen": "7k/1rq3p1/2Q4p/1p6/2p1N3/1n3P2/6PP/3R2K1 w - - 2 38",
            "solution": "c6e8",
            "rating": "1995",
            "theme": "crushing endgame long quietMove"
        },
        {
            "name": "Lichess Puzzle Jy29u",
            "fen": "5rk1/pp3p1p/1q6/3ppPp1/8/6Q1/PPP4P/1K5R w - - 0 23",
            "solution": "g3g5",
            "rating": "2038",
            "theme": "crushing endgame long quietMove"
        },
        {
            "name": "Lichess Puzzle Lhz8A",
            "fen": "8/1pp5/8/1P6/2P2kpK/8/7P/8 b - - 0 36",
            "solution": "b7b6",
            "rating": "2098",
            "theme": "crushing endgame pawnEndgame quietMove veryLong zugzwang"
        },
        {
            "name": "Lichess Puzzle GRJ8e",
            "fen": "5k2/1K6/8/P4pp1/1P1b1B1p/8/8/8 w - - 0 57",
            "solution": "f4g5",
            "rating": "2104",
            "theme": "advancedPawn bishopEndgame crushing endgame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle B8Q7R",
            "fen": "5rk1/pp6/1b2p2p/3p1q2/7P/5RB1/2r1Q1P1/1R5K w - - 1 35",
            "solution": "e2f1",
            "rating": "2109",
            "theme": "advantage middlegame quietMove short"
        },
        {
            "name": "Lichess Puzzle NuFMn",
            "fen": "r7/1p1k2p1/pq2p3/3p1p2/1P1B1P2/P3Q1K1/4B2r/R4R2 b - - 0 27",
            "solution": "a8h8",
            "rating": "2175",
            "theme": "crushing long middlegame quietMove"
        },
        {
            "name": "Lichess Puzzle 3uwMh",
            "fen": "r4r1k/1p3pp1/2p4p/p7/3P2R1/1Pq3P1/P1P2PQ1/1K5R w - - 0 28",
            "solution": "g2h3",
            "rating": "2204",
            "theme": "crushing endgame long quietMove"
        },
        {
            "name": "Lichess Puzzle 7IYSi",
            "fen": "5rk1/pp1b1p2/3p1RpQ/8/8/8/Pqr1B1PP/R6K w - - 0 22",
            "solution": "a1f1",
            "rating": "2345",
            "theme": "advantage middlegame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle 7qjgF",
            "fen": "2r3k1/Q2n2p1/1pq1p2p/5p2/3PpP1P/1N1bP1R1/PP4P1/R5K1 b - - 0 22",
            "solution": "c8a8",
            "rating": "2354",
            "theme": "advantage clearance long master middlegame quietMove trappedPiece"
        },
        {
            "name": "Lichess Puzzle 9aQjN",
            "fen": "r4r1k/5Pp1/3q4/p1pn4/R1Bb1pP1/1P6/6PP/3QR2K w - - 1 32",
            "solution": "g4g5",
            "rating": "2453",
            "theme": "advantage middlegame quietMove short"
        },
        {
            "name": "Lichess Puzzle P0Ksu",
            "fen": "8/6p1/4P1kp/8/3R4/7r/5K2/8 w - - 1 35",
            "solution": "d4e4",
            "rating": "2545",
            "theme": "advancedPawn crushing endgame quietMove rookEndgame short"
        },
        {
            "name": "Lichess Puzzle 7kUu3",
            "fen": "8/Q2Rrpbk/1p4p1/8/2P4q/1P4N1/P4PP1/6K1 b - - 0 32",
            "solution": "e7e1",
            "rating": "2633",
            "theme": "crushing endgame quietMove veryLong"
        },
        {
            "name": "Lichess Puzzle LPMRJ",
            "fen": "8/4k1p1/8/p1ppKPP1/1p6/P1P5/1P6/8 b - - 2 37",
            "solution": "d5d4",
            "rating": "2661",
            "theme": "crushing endgame long pawnEndgame quietMove"
        },
        {
            "name": "Lichess Puzzle DBPvQ",
            "fen": "8/8/3K1k2/1b1P4/8/P1p2N2/8/8 b - - 1 65",
            "solution": "c3c2",
            "rating": "813",
            "theme": "advancedPawn crushing endgame promotion short"
        },
        {
            "name": "Lichess Puzzle EYMiA",
            "fen": "8/8/1K6/P3p3/3p4/3Pk3/8/8 w - - 2 65",
            "solution": "a5a6",
            "rating": "1062",
            "theme": "advancedPawn crushing endgame long pawnEndgame promotion"
        },
        {
            "name": "Lichess Puzzle GQ5kJ",
            "fen": "8/B7/2P5/r7/4p3/2K2k2/1P6/8 w - - 0 64",
            "solution": "c6c7",
            "rating": "1223",
            "theme": "advancedPawn crushing endgame promotion short"
        },
        {
            "name": "Lichess Puzzle IrJi1",
            "fen": "6r1/5k2/p2R4/2P2p2/1P3p2/4pK1P/P5P1/8 b - - 5 36",
            "solution": "g8g3",
            "rating": "1399",
            "theme": "advancedPawn advantage endgame long promotion rookEndgame"
        },
        {
            "name": "Lichess Puzzle ErUHk",
            "fen": "q7/7k/8/6pp/1P2N3/6PP/p4PK1/Q7 b - - 5 56",
            "solution": "a8e4",
            "rating": "1471",
            "theme": "advancedPawn crushing endgame hangingPiece long promotion"
        },
        {
            "name": "Lichess Puzzle AnJmu",
            "fen": "8/8/8/1p4p1/1Pp1k1P1/2P2P1P/3K4/8 b - - 0 42",
            "solution": "e4f3",
            "rating": "1504",
            "theme": "advancedPawn crushing endgame pawnEndgame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle NluwE",
            "fen": "8/2k2p2/2P5/p1KPP1p1/1p4P1/8/8/8 b - - 2 39",
            "solution": "b4b3",
            "rating": "1572",
            "theme": "advancedPawn crushing endgame pawnEndgame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle BZ4cD",
            "fen": "8/5K2/p2RP1p1/1k4P1/1q3P2/8/8/8 w - - 0 75",
            "solution": "e6e7",
            "rating": "1664",
            "theme": "advancedPawn crushing endgame promotion queenRookEndgame short"
        },
        {
            "name": "Lichess Puzzle 1Wfp1",
            "fen": "3r3k/R5p1/1p5p/4P3/3p4/7P/PP4PK/8 b - - 0 33",
            "solution": "d4d3",
            "rating": "1696",
            "theme": "advancedPawn advantage endgame promotion rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle PGWln",
            "fen": "8/1b6/1k3P2/1p1p4/3P4/PB4rp/KPP4P/8 w - - 0 37",
            "solution": "h2g3",
            "rating": "1740",
            "theme": "advancedPawn crushing endgame hangingPiece long promotion"
        },
        {
            "name": "Lichess Puzzle JY2jM",
            "fen": "8/7p/p2P2p1/bpP5/3k4/7P/6K1/8 w - - 0 38",
            "solution": "c5c6",
            "rating": "1812",
            "theme": "advancedPawn advantage bishopEndgame endgame long promotion"
        },
        {
            "name": "Lichess Puzzle EkU6K",
            "fen": "8/1p2n1k1/p7/2P5/1P3P2/P2KP2p/4N3/8 b - - 1 35",
            "solution": "e7f5",
            "rating": "1824",
            "theme": "advancedPawn crushing endgame knightEndgame long promotion"
        },
        {
            "name": "Lichess Puzzle JiRXm",
            "fen": "5k2/5pp1/4P3/8/4R1QP/2q5/7r/5K2 w - - 0 37",
            "solution": "e6e7",
            "rating": "1896",
            "theme": "advancedPawn attraction crushing endgame long promotion"
        },
        {
            "name": "Lichess Puzzle PB8p6",
            "fen": "r3r1k1/2p1qp2/1p1p1n2/pPbP1P1p/P4N1P/3Q2B1/6pN/2KRR3 b - - 1 25",
            "solution": "e7e1",
            "rating": "1907",
            "theme": "advancedPawn crushing middlegame promotion short"
        },
        {
            "name": "Lichess Puzzle 9Goup",
            "fen": "2r5/5pkp/3P1pp1/2P5/np6/7P/7P/3R3K w - - 0 34",
            "solution": "d6d7",
            "rating": "1910",
            "theme": "advancedPawn advantage endgame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle Ca9Zx",
            "fen": "8/3k4/1P1r4/P1K5/8/8/8/8 w - - 1 58",
            "solution": "a5a6",
            "rating": "2015",
            "theme": "advancedPawn crushing endgame promotion rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle BVvce",
            "fen": "4k3/8/2PKp3/4P3/8/6R1/2r3p1/8 w - - 1 47",
            "solution": "c6c7",
            "rating": "2035",
            "theme": "advancedPawn crushing deflection endgame long promotion rookEndgame"
        },
        {
            "name": "Lichess Puzzle GidPC",
            "fen": "3k4/8/1KP5/3B4/8/8/7p/6r1 w - - 2 86",
            "solution": "c6c7",
            "rating": "2203",
            "theme": "advancedPawn crushing deflection endgame exposedKing long master promotion"
        },
        {
            "name": "Lichess Puzzle 7PdPh",
            "fen": "2r4k/p5p1/7p/5R2/Pqp2P2/3p1R2/6PP/3Q3K b - - 0 37",
            "solution": "c4c3",
            "rating": "2225",
            "theme": "advancedPawn advantage endgame long promotion"
        },
        {
            "name": "Lichess Puzzle IQtvD",
            "fen": "6k1/6p1/4pBQ1/p4p2/P2P4/6P1/rp2NP1P/5K2 b - - 0 39",
            "solution": "b2b1q",
            "rating": "2276",
            "theme": "advancedPawn crushing endgame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle OIypM",
            "fen": "5k2/8/7R/P1pB1Pp1/P5K1/2B2P2/4r2p/5b2 b - - 1 45",
            "solution": "e2g2",
            "rating": "2282",
            "theme": "advancedPawn crushing endgame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle NQVpp",
            "fen": "8/2B2k1p/4pP2/1P1p2P1/p1nP1P2/3KP3/8/8 b - - 0 50",
            "solution": "a4a3",
            "rating": "2400",
            "theme": "advancedPawn crushing endgame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle 4z7ln",
            "fen": "8/8/2p1r3/p1PPP3/P2K3p/7P/5pk1/5R2 b - - 0 44",
            "solution": "g2f1",
            "rating": "2676",
            "theme": "advancedPawn clearance crushing endgame promotion rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle 5RGK9",
            "fen": "6n1/1q4k1/3P1bp1/7p/2Q5/1pP1N3/1r3PP1/4R1K1 w - - 1 34",
            "solution": "c4c7",
            "rating": "2732",
            "theme": "advancedPawn advantage master middlegame promotion veryLong"
        },
        {
            "name": "Lichess Puzzle 89zlD",
            "fen": "1q4rk/1p2R2p/6r1/p1Pp1pbQ/3P3P/2P5/8/6K1 w - - 0 48",
            "solution": ["h5h7", "e7h7"],
            "rating": "852",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle DZcpn",
            "fen": "8/8/2r1p1k1/RP5p/3n4/7P/P4PP1/5K2 b - - 0 39",
            "solution": "c6c1",
            "rating": "875",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle 0WPrD",
            "fen": "7R/8/8/3q3p/6p1/6Pk/5P1P/5NK1 b - - 6 52",
            "solution": "d5g2",
            "rating": "883",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle BKiIH",
            "fen": "4k3/3q1pQ1/p2p3p/1pnP1N2/2p5/P1P5/2P1rPPP/6K1 w - - 4 28",
            "solution": "g7g8",
            "rating": "1132",
            "theme": "endgame mate mateIn1 oneMove"
        },
        {
            "name": "Lichess Puzzle 4wAxS",
            "fen": "4rk2/1p3ppp/p1pp2q1/8/1PP4r/P2QPPP1/7P/R4RK1 w - - 1 24",
            "solution": "d3g6",
            "rating": "1036",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 534EQ",
            "fen": "3q4/3r2k1/3PBp2/p1Q3p1/1p5p/8/P5P1/5K2 w - - 1 56",
            "solution": "e6d7",
            "rating": "1134",
            "theme": "advancedPawn crushing endgame long"
        },
        {
            "name": "Lichess Puzzle MbUMZ",
            "fen": "6r1/1p5p/p7/4k3/4p3/4K3/PPPR2P1/8 b - - 9 35",
            "solution": "g8g3",
            "rating": "1170",
            "theme": "crushing endgame rookEndgame short"
        },
        {
            "name": "Lichess Puzzle F5qxi",
            "fen": "2r3k1/1p3p1p/p2P1Bp1/4Pb2/5q2/P6P/1PPR2P1/1K1R4 w - - 3 28",
            "solution": "d6d7",
            "rating": "1195",
            "theme": "advancedPawn crushing endgame short"
        },
        {
            "name": "Lichess Puzzle FFtmO",
            "fen": "8/5p2/5kp1/3P3p/7P/3KR1P1/r7/8 b - - 0 51",
            "solution": "a2a3",
            "rating": "1253",
            "theme": "crushing endgame exposedKing long rookEndgame"
        },
        {
            "name": "Lichess Puzzle C625r",
            "fen": "8/8/5pkp/6P1/r6P/5RK1/8/8 w - - 1 43",
            "solution": "f3f6",
            "rating": "1258",
            "theme": "crushing endgame rookEndgame short"
        },
        {
            "name": "Lichess Puzzle 4UAuy",
            "fen": "2r5/8/4k1p1/2PN3p/1P1K1P2/6PP/8/8 b - - 0 54",
            "solution": "c8d8",
            "rating": "1376",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 8YpV2",
            "fen": "8/8/3k2P1/1p6/p7/6K1/5R2/2r5 w - - 3 54",
            "solution": "g6g7",
            "rating": "1416",
            "theme": "advancedPawn crushing endgame master rookEndgame short"
        },
        {
            "name": "Lichess Puzzle NPSts",
            "fen": "8/6kp/5R2/pr1PKP2/8/5P2/8/8 b - - 0 59",
            "solution": "b5d5",
            "rating": "1416",
            "theme": "crushing deflection endgame rookEndgame short"
        },
        {
            "name": "Lichess Puzzle 47qkw",
            "fen": "8/7p/5kp1/3R4/8/3K1P1P/8/7r b - - 0 50",
            "solution": "h1d1",
            "rating": "1452",
            "theme": "crushing endgame rookEndgame veryLong"
        },
        {
            "name": "Lichess Puzzle GpfYo",
            "fen": "5r2/p5pk/3R1nqp/2pP1p2/2P2P1P/2Q5/PB5P/5K2 b - - 2 34",
            "solution": "g6g4",
            "rating": "1480",
            "theme": "advantage endgame oneMove"
        },
        {
            "name": "Lichess Puzzle CwxVB",
            "fen": "8/1p6/2kB3R/2p5/2b5/8/p2K4/8 b - - 1 52",
            "solution": "c6d5",
            "rating": "1483",
            "theme": "endgame equality short"
        },
        {
            "name": "Lichess Puzzle IXBvB",
            "fen": "6k1/p2q1p2/1p1P2p1/3QP1Pp/1Pr4P/8/1R3PK1/8 b - - 0 39",
            "solution": "d7g4",
            "rating": "1496",
            "theme": "crushing endgame oneMove"
        },
        {
            "name": "Lichess Puzzle Ot5FL",
            "fen": "8/3R3p/6pk/1p3pn1/p3b3/P1P4P/1P1Q2PK/5q2 w - - 9 49",
            "solution": "h3h4",
            "rating": "1534",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle 9GR8X",
            "fen": "8/7k/5p2/p3pPr1/1p1p1q2/1P1P2NP/1P3QRK/5r2 w - - 11 46",
            "solution": "f2f1",
            "rating": "1539",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle Da8ZL",
            "fen": "6k1/pp3p2/6p1/4n2p/2qrNQ2/8/5PPP/5RK1 w - - 2 26",
            "solution": "e4f6",
            "rating": "1543",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle ApA4i",
            "fen": "3R4/6r1/1r3pB1/2k2P1P/3p4/5P1K/8/8 w - - 0 57",
            "solution": "h5h6",
            "rating": "1559",
            "theme": "advancedPawn advantage endgame short"
        },
        {
            "name": "Lichess Puzzle H6gt7",
            "fen": "1b1r4/1p4Pp/1P5k/3P1p2/4p3/4P3/1B1KQq2/2R5 b - - 1 39",
            "solution": "d8d5",
            "rating": "1635",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 416vB",
            "fen": "7k/pp4p1/q4r1p/2p1Q3/8/4B2P/PP3P1P/n4RK1 w - - 6 25",
            "solution": "e5e8",
            "rating": "1659",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle 70VoS",
            "fen": "2r2qk1/1p2bp2/p3p1p1/3pP3/3R1N2/P5QP/5PP1/6K1 w - - 1 25",
            "solution": "f4g6",
            "rating": "1719",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 4X3IT",
            "fen": "r3k2r/pp2R3/1q6/1Pp5/3pP1p1/P2P2p1/3QP1KP/2R5 b kq - 0 28",
            "solution": "e8e7",
            "rating": "1727",
            "theme": "crushing endgame hangingPiece short"
        },
        {
            "name": "Lichess Puzzle DVZAF",
            "fen": "8/p4ppk/Q3p2p/1P1pP3/P2q4/7P/5PP1/6K1 b - - 3 30",
            "solution": "d4a1",
            "rating": "1730",
            "theme": "crushing endgame queenEndgame short"
        },
        {
            "name": "Lichess Puzzle 3La2I",
            "fen": "1r6/5Q2/3q2pk/8/5P2/1R5P/1P4P1/6K1 b - - 0 40",
            "solution": "d6d1",
            "rating": "1732",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 6fP7H",
            "fen": "4R3/p4ppp/4pk2/3n4/P1Q5/1P6/2P1rqPP/6RK b - - 6 26",
            "solution": "e2e1",
            "rating": "1775",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 6ysbm",
            "fen": "4r1k1/pR5p/6p1/8/2p1p3/q1P3QP/P5PK/8 w - - 1 34",
            "solution": "g3c7",
            "rating": "1792",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 9FCLC",
            "fen": "6k1/5p2/6p1/3p1q1p/7P/1Qb1P1P1/2rN1P2/3R2K1 w - - 2 40",
            "solution": "e3e4",
            "rating": "1803",
            "theme": "crushing endgame interference short"
        },
        {
            "name": "Lichess Puzzle 73Bqf",
            "fen": "1R4R1/5pkp/6p1/5r2/3K4/5P1P/2r3P1/8 b - - 3 52",
            "solution": "g7h6",
            "rating": "1841",
            "theme": "crushing endgame rookEndgame short"
        },
        {
            "name": "Lichess Puzzle Njv1r",
            "fen": "6k1/5R2/4Kp2/2P1p3/4P3/1P6/6p1/6r1 w - - 2 55",
            "solution": "f7f6",
            "rating": "1864",
            "theme": "crushing endgame exposedKing long master rookEndgame"
        },
        {
            "name": "Lichess Puzzle 2O5CJ",
            "fen": "7k/6p1/7p/2q5/8/p1r2P1P/Q5P1/1R5K w - - 9 47",
            "solution": "b1b8",
            "rating": "1876",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 16IIY",
            "fen": "8/7p/1p4p1/1Kn5/P2N2P1/k6P/8/8 w - - 10 48",
            "solution": "a4a5",
            "rating": "1910",
            "theme": "advantage deflection endgame knightEndgame short"
        },
        {
            "name": "Lichess Puzzle IRBXX",
            "fen": "8/p5p1/5p2/kpp1pP2/2P1P1Kp/2P1P2P/P5P1/8 b - - 1 36",
            "solution": "b5b4",
            "rating": "1917",
            "theme": "crushing endgame long master masterVsMaster pawnEndgame"
        },
        {
            "name": "Lichess Puzzle Cezdf",
            "fen": "5rk1/8/2qN1rp1/2pP4/p2Qp3/P7/1P1R1PPP/6K1 w - - 0 35",
            "solution": "d4f6",
            "rating": "1929",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 6E68K",
            "fen": "5k2/1p3B1p/2p2K2/p7/P7/2P2P1P/1b6/8 b - - 0 35",
            "solution": "b2c3",
            "rating": "1953",
            "theme": "bishopEndgame crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 3XSnG",
            "fen": "8/2R3pk/2p2pqp/3b2r1/P1pP4/5PBP/6PK/4Q3 b - - 0 29",
            "solution": "c4c3",
            "rating": "1984",
            "theme": "crushing deflection endgame master short"
        },
        {
            "name": "Lichess Puzzle BHKdT",
            "fen": "8/8/2k5/p1p2pp1/P1K4P/1P4P1/8/8 b - - 0 41",
            "solution": "f5f4",
            "rating": "1990",
            "theme": "crushing deflection endgame pawnEndgame short"
        },
        {
            "name": "Lichess Puzzle 69zzp",
            "fen": "8/8/p2k4/2pPp1pp/1pP1P3/1P4K1/P4RPP/2r5 b - - 1 37",
            "solution": "c1c3",
            "rating": "1991",
            "theme": "crushing endgame long rookEndgame"
        },
        {
            "name": "Lichess Puzzle JKukc",
            "fen": "1Q1b4/1p1r1pkp/3np1p1/3p4/P2P1PP1/1P1BP3/4K2P/8 b - - 0 31",
            "solution": "d8b6",
            "rating": "2031",
            "theme": "clearance crushing endgame long master trappedPiece"
        },
        {
            "name": "Lichess Puzzle 4AIL6",
            "fen": "8/2R4p/5k2/1P2p3/P3pp2/6P1/1r3P1P/5K2 b - - 1 38",
            "solution": "b2b1",
            "rating": "2065",
            "theme": "crushing endgame master rookEndgame short"
        },
        {
            "name": "Lichess Puzzle Bpmr9",
            "fen": "r7/7p/5k2/1P1RNbp1/8/p4P2/P6P/6K1 b - - 0 34",
            "solution": "f6e6",
            "rating": "2079",
            "theme": "crushing endgame master short"
        },
        {
            "name": "Lichess Puzzle CwDgd",
            "fen": "3r2k1/p5pp/1p2P3/2p2p1Q/2P1p3/1P2P3/P2q2PP/1R4K1 b - - 1 22",
            "solution": "d2e3",
            "rating": "2080",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 7X3e6",
            "fen": "8/8/8/1B1R1pk1/6pp/3P4/5PK1/3r4 b - - 2 65",
            "solution": "h4h3",
            "rating": "2111",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 7M2V7",
            "fen": "6k1/b4ppp/r3p3/p2p4/1Pq5/P1P1QK2/5P2/1R4R1 w - - 6 28",
            "solution": "g1g7",
            "rating": "2130",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle NV6PX",
            "fen": "3q2k1/5p1p/3p2p1/3Pr3/2PQ4/P3BP2/4rP1P/R5K1 b - - 4 25",
            "solution": "e5e3",
            "rating": "2137",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle 2nToq",
            "fen": "b4r2/5Pk1/6p1/3p2Np/5qnP/2P5/2P3P1/5RK1 w - - 0 26",
            "solution": "f1f4",
            "rating": "2144",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle NQA7i",
            "fen": "3rr1k1/pp6/2p2B2/8/1P4P1/P2RPn1p/5P1P/3R3K b - - 2 34",
            "solution": "d8d3",
            "rating": "2205",
            "theme": "crushing endgame master short"
        },
        {
            "name": "Lichess Puzzle 1aPxs",
            "fen": "3k4/ppp5/4p1p1/3p4/5b2/3P4/PPPqN1P1/1K2R2R w - - 0 22",
            "solution": "e2f4",
            "rating": "2207",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle BYnyW",
            "fen": "1r6/4p2p/3p2p1/R1rPk3/5p2/2PK3P/RP3PP1/8 b - - 5 31",
            "solution": "c5a5",
            "rating": "2215",
            "theme": "crushing endgame rookEndgame short"
        },
        {
            "name": "Lichess Puzzle 4ahdQ",
            "fen": "8/8/p7/7K/7P/2k3P1/3N1n2/8 b - - 0 47",
            "solution": "c3d2",
            "rating": "2219",
            "theme": "crushing endgame hangingPiece knightEndgame long"
        },
        {
            "name": "Lichess Puzzle 1EGzr",
            "fen": "3rr1k1/pp4p1/4P3/2p2Pp1/2Pp1q2/5N1Q/PP6/5RK1 w - - 0 29",
            "solution": "h3h5",
            "rating": "2228",
            "theme": "crushing endgame short"
        },
        {
            "name": "Lichess Puzzle OdOzy",
            "fen": "8/8/7p/8/8/3r1B1P/R4rPK/4k3 w - - 19 55",
            "solution": "a2a1",
            "rating": "2240",
            "theme": "crushing endgame long trappedPiece"
        },
        {
            "name": "Lichess Puzzle P3hJA",
            "fen": "8/6R1/3kp2p/2pb4/1r1n4/2NK2P1/7P/R7 b - - 4 54",
            "solution": "d5c4",
            "rating": "2281",
            "theme": "advantage endgame long master"
        },
        {
            "name": "Lichess Puzzle JOOYp",
            "fen": "4r1k1/p5p1/1p2P3/1q4p1/3Q2P1/6P1/r7/2R1R1K1 w - - 0 35",
            "solution": "d4d7",
            "rating": "2338",
            "theme": "advancedPawn crushing endgame short"
        },
        {
            "name": "Lichess Puzzle Hxl9U",
            "fen": "8/1p6/8/p7/P2K4/1P6/5k2/8 w - - 1 49",
            "solution": "d4c5",
            "rating": "2353",
            "theme": "crushing endgame long master pawnEndgame"
        },
        {
            "name": "Lichess Puzzle 6ZCyJ",
            "fen": "1k3r2/1p4p1/2p5/3bP3/5Pqp/2Q3P1/PP5P/R4RK1 b - - 3 26",
            "solution": "g4e2",
            "rating": "2378",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle 6eIxF",
            "fen": "8/5pk1/P3p2p/1Q1p4/3P1KPp/q4P1P/8/8 w - - 6 42",
            "solution": "f4e5",
            "rating": "2402",
            "theme": "crushing endgame master queenEndgame short"
        },
        {
            "name": "Lichess Puzzle FkGNp",
            "fen": "8/5k2/2p5/N4ppp/P3p3/1P1pP1P1/5K2/8 b - - 0 40",
            "solution": "d3d2",
            "rating": "2407",
            "theme": "advancedPawn crushing endgame knightEndgame long"
        },
        {
            "name": "Lichess Puzzle 2MTqq",
            "fen": "r6k/p6P/4p1p1/q2pPp2/1p2r3/5QP1/5PK1/1R5R w - f6 0 28",
            "solution": "e5f6",
            "rating": "2415",
            "theme": "crushing enPassant endgame veryLong"
        },
        {
            "name": "Lichess Puzzle GNuB5",
            "fen": "6r1/pppk2qp/3pR3/8/4Qp2/6PP/PPP2P2/1K6 b - - 1 29",
            "solution": "g7f7",
            "rating": "2436",
            "theme": "advantage endgame short"
        },
        {
            "name": "Lichess Puzzle Np5xi",
            "fen": "6k1/5ppp/5q2/3p1b2/2rP1Q2/4PN2/P2K1PPP/2R5 b - - 1 22",
            "solution": "c4c1",
            "rating": "2455",
            "theme": "attraction crushing endgame veryLong"
        },
        {
            "name": "Lichess Puzzle NCs4i",
            "fen": "8/3R3p/7k/4N3/5PKn/8/7r/8 w - - 0 53",
            "solution": "g4g3",
            "rating": "2617",
            "theme": "crushing endgame long zugzwang"
        },
        {
            "name": "Lichess Puzzle GEmqW",
            "fen": "8/6B1/8/3kp2p/3pp1pP/6P1/2K5/8 b - - 4 64",
            "solution": "d5c4",
            "rating": "2646",
            "theme": "bishopEndgame crushing endgame master short"
        },
        {
            "name": "Lichess Puzzle P3ttZ",
            "fen": "6r1/p1p3p1/2p5/2P3k1/3Qb3/P1P2R1P/4q1PK/8 w - - 2 32",
            "solution": "f3f2",
            "rating": "2677",
            "theme": "advantage endgame long"
        },
        {
            "name": "Lichess Puzzle 2Dh1k",
            "fen": "2rq4/6p1/p3p3/1p1pkn2/8/2NQ4/PPP2P2/2K3R1 w - - 0 28",
            "solution": "g1g6",
            "rating": "2684",
            "theme": "advantage attraction endgame veryLong"
        },
        {
            "name": "Lichess Puzzle B5wzH",
            "fen": "8/p7/8/6p1/3Rn1Nk/Pp5P/1r4P1/6K1 b - - 1 33",
            "solution": "b2e2",
            "rating": "2684",
            "theme": "advancedPawn advantage endgame long"
        },
        {
            "name": "Lichess Puzzle 3zTAN",
            "fen": "4N3/8/1P6/2p1b1p1/6k1/1P1K4/8/8 w - - 1 52",
            "solution": "d3e4",
            "rating": "2760",
            "theme": "crushing endgame long"
        },
        {
            "name": "Lichess Puzzle 3f4EN",
            "fen": "8/8/8/8/6pk/2r5/4KR1P/8 b - - 5 57",
            "solution": "h4h3",
            "rating": "2799",
            "theme": "crushing endgame exposedKing rookEndgame veryLong"
        },
    ]


def run_puzzle_test():
    """
    Runs a search test for each puzzle and prints statistics.
    """
    from chess_engine.classical.engine_types import SearchContext


    # 1. Warm-up (triggers JIT compilation)
    print("--- WARMING UP (Compiling JIT functions) ---")
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p_bbs, o_bbs, g_state = parse_fen(fen_start)
    tt = create_transposition_table(16)
    killer_moves = np.zeros(256, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((4, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 12), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16)
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)
    pv_table = np.zeros((128, 128), dtype=np.uint16)
    ctx = SearchContext(tt, killer_moves, pv_table, history_table, butterfly_history, continuation_history, capture_history, pawn_history, pawn_correction_history, minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black)
    
    # Run a quick search
    iterative_deepening_search(p_bbs, o_bbs, g_state, 8, {'optimum_time': 0, 'maximum_time': 0}, ctx)
    print("--- WARM-UP COMPLETE ---\n")


    depth = 40  # Set a high depth, will be stopped by time
    time_limit_ms = 3000

    total_tests = len(puzzles)
    passed_tests = 0
    failed_tests = 0
    total_engine_time = 0
    total_nodes_searched = 0
    total_tt_hits = 0
    total_depth_sum = 0
    failed_puzzles = []

    script_start_time = time.time()

    

    for i, puzzle in enumerate(puzzles):
        print("New game started. Caches and stats cleared.")
        print(f"--- Running Test: {puzzle['name']} ---")
        print(f"FEN: {puzzle['fen']}")

        piece_bbs, occupancy_bbs, game_state = parse_fen(puzzle["fen"])
        
        # killer_moves, pv_table, and history_table are now initialized inside the loop
        killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
        pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
        history_table = np.zeros((12, 64), dtype=np.int32) # Note: history_table size is 12x64 in search.py
        butterfly_history = np.zeros((64, 64), dtype=np.int32)
        continuation_history = np.zeros((4, 12, 64, 12, 64), dtype=np.int16)
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
            piece_bbs, occupancy_bbs, game_state, depth, time_config, search_context
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
                "theme": puzzle["theme"],
                "score": int(best_eval),
                "completed_depth": int(last_completed_depth),
                "total_time_seconds": float(elapsed_time),
                "total_nodes": int(total_nodes),
                "nps": int(nps),
                "quiescence_nodes": int(quiescence_nodes),
                "quiescence_percentage": float(q_node_percentage),
                "tt_hits": int(tt_hits),
                "tt_hit_rate": float(tt_hit_rate),
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
            print("  --- Search Statistics ---")
            print(f"  Total Time: {fp['total_time_seconds']:.2f}s")
            print(f"  Nodes Searched: {fp['total_nodes']} ({fp['nps']} NPS)")
            print(f"  - Quiescence Nodes: {fp['quiescence_nodes']} ({fp['quiescence_percentage']:.1f}%)")
            print(f"  Transposition Table Hits: {fp['tt_hits']} ({fp['tt_hit_rate']:.1f}%)")
            print(f"  Completed Depth: {fp['completed_depth']}")
            print(f"  Score: {fp['score']}")
            print("-" * 20)

        # Save failed puzzles to JSON for analysis
        import json
        with open("failed_puzzles.json", "w") as f:
            json.dump(failed_puzzles, f, indent=4)
        print("Failed puzzles saved to failed_puzzles.json")


if __name__ == "__main__":
    run_puzzle_test()
