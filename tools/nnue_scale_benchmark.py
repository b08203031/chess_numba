#!/usr/bin/env python3
"""
NNUE static-scale and UCI search/pruning calibration tool.

Single-file calibration suite for internal static NNUE/classical extraction,
UCI depth search, pruning-mode comparisons, and Stockfish reference data.

It is safe to run even when the local Python engine dependencies are missing:
internal static extraction is optional and defaults to auto-detect.

Common examples:
    python nnue_scale_benchmark.py --dry-run
    python nnue_scale_benchmark.py --internal-static on --skip-search
    python nnue_scale_benchmark.py --engine ./your_engine --priorities P1
    python nnue_scale_benchmark.py --engine ./your_engine --stockfish auto --priorities P1,P2
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_POSITIONS = "nnue_search_calibration_fens.txt"
DEFAULT_DEPTHS = [4, 6, 8, 10]
DEFAULT_MODES = ["full_search", "no_nmp_rfp_razor", "no_lmr", "no_see_pruning"]

EMBEDDED_POSITIONS_TSV = """id	priority	bucket	fen	notes
Q01	P1	quiet_static	r2q1rk1/ppp2ppp/2n1bn2/3pp3/3P4/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 1	Italian-like quiet center
Q02	P1	quiet_static	r1bq1rk1/pp2bppp/2n1pn2/2ppN3/3P4/2P1PN2/PP1BBPPP/R2Q1RK1 w - - 0 1	Sicilian developed center
Q03	P1	quiet_static	r2q1rk1/1pp1bppp/p1npbn2/4p3/2BPP3/2N1BN2/PPP2PPP/R2Q1RK1 w - - 0 1	Closed center, active minors
Q04	P1	quiet_static	r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 1	QGD/Carlsbad development
Q05	P1	quiet_static	r2q1rk1/pp2bppp/2n1bn2/2ppp3/3PP3/2PB1N2/PP1N1PPP/R1BQ1RK1 w - - 0 1	French-style locked center
Q06	P1	quiet_static	r2q1rk1/pp1bbppp/2n1pn2/2pp4/2PP4/2NBPN2/PP2BPPP/R2Q1RK1 w - - 0 1	English/QGD quiet tension
Q07	P1	quiet_static	r2q1rk1/pp2bppp/2npbn2/2p1p3/2PPP3/2N1BN2/PP2BPPP/R2Q1RK1 w - - 0 1	KID/Benoni-style closed center
Q08	P1	quiet_static	r1bq1rk1/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP1BBPPP/R2Q1RK1 w - - 0 1	Maroczy-like quiet structure
M01	P2	material_scale	r2q1rk1/pp3ppp/2n1bn2/3pp3/3P4/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 1	White clean extra central pawn
M02	P2	material_scale	r2q1rk1/ppp2ppp/2n1bn2/3pp3/8/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 1	Black clean extra central pawn
M03	P2	material_scale	r2q1rk1/ppp2ppp/2n1bn2/3pp3/3P4/2PBPN2/PP3PPP/R1BQ1RK1 w - - 0 1	White bishop pair vs bishop+knight
M04	P2	material_scale	r2q1rk1/pp2bppp/2n1b3/3pp3/3P4/2N1PN2/PP1B1PPP/R2Q1RK1 w - - 0 1	Black bishop pair vs bishop+knight
M05	P2	material_scale	2bq1rk1/pp3ppp/2n1bn2/3pp3/3P4/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 1	White exchange up, black bishop compensation
M06	P2	material_scale	r2q1rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP1B1PPP/2RQ1RK1 w - - 0 1	White clean minor up
P01	P1	pawn_endgame_rfp	8/8/8/3k4/3P4/3K4/8/8 w - - 0 1	Opposition, white to move
P02	P1	pawn_endgame_rfp	8/8/8/3k4/3P4/3K4/8/8 b - - 0 1	Opposition, black to move
P03	P1	pawn_endgame_rfp	8/8/8/8/2k5/8/2P5/2K5 w - - 0 1	K+P boundary, white to move
P04	P1	pawn_endgame_rfp	8/8/8/8/2k5/8/2P5/2K5 b - - 0 1	K+P boundary, black to move
P05	P1	pawn_endgame_rfp	8/5k2/6p1/5p2/5P2/6P1/5K2/8 w - - 0 1	Passed pawn race
P06	P1	pawn_endgame_rfp	8/6k1/6p1/5p2/P4P2/6P1/6K1/8 w - - 0 1	Outside passer race
N01	P1	low_material_nmp	8/8/8/3k4/3P4/3K1N2/8/8 w - - 0 1	Pawn opposition plus knight; NMP guard candidate
N02	P1	low_material_nmp	8/8/5n2/3k4/3P4/3K4/8/8 b - - 0 1	Pawn opposition plus black knight; NMP guard candidate
N03	P1	low_material_nmp	8/8/8/3k4/3P4/3K4/4B3/8 w - - 0 1	Pawn opposition plus bishop; NMP guard candidate
N04	P1	low_material_nmp	8/8/8/3k4/3P4/3K4/8/4b3 b - - 0 1	Pawn opposition plus black bishop; NMP guard candidate
E01	P2	rook_fortress_endgame	r7/5pkp/6p1/3P4/1R6/6P1/5P1P/6K1 w - - 0 1	Rook endgame with active rook
E02	P2	rook_fortress_endgame	r7/6k1/5pp1/3P4/8/6P1/5P1P/4R1K1 w - - 0 1	Rook behind passer candidate
E03	P2	rook_fortress_endgame	8/8/2k5/2P5/8/2K5/4B3/6b1 w - - 0 1	Opposite-colored bishops, one pawn
E04	P2	rook_fortress_endgame	8/6k1/5pp1/2PP4/8/2K5/4B3/6b1 w - - 0 1	Opposite-colored bishops, multiple pawns
E05	P2	rook_fortress_endgame	8/5k2/5p2/4p3/4P3/5N2/5PPP/6K1 w - - 0 1	Knight endgame material scale
E06	P2	rook_fortress_endgame	8/5k2/5p2/4p3/4P3/5B2/5PPP/6K1 w - - 0 1	Bishop endgame material scale
T01	P1	tactical_search	r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q2/PPPBBPPP/R3K2R w KQkq - 0 1	Kiwipete-like, corrected to 8 black pawns
T02	P1	tactical_search	r2q1rk1/ppp2ppp/2n2n2/2bpp3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1	Central tension capture sequence
T03	P1	tactical_search	r1bq1rk1/ppp2ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 0 1	King-side pressure, uncastled king
T04	P1	tactical_search	r2q1rk1/ppp2ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1	Exposed king / attacking compensation
T05	P1	tactical_search	r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1	Sacrifice-compensation candidate
T06	P1	tactical_search	r1bq1rk1/ppp2ppp/2n2n2/3pp3/1b1PP3/2N1BN2/PPP2PPP/R2QKB1R w KQ - 0 1	Pinned knight / central break candidate
"""

MODE_OPTION_GROUPS: dict[str, list[list[tuple[str, str]]]] = {
    "full_search": [],
    "no_nmp_rfp_razor": [
        [("Enable NMP", "false"), ("ENABLE_NMP", "false"), ("Null Move Pruning", "false"), ("Use Null Move Pruning", "false")],
        [("Enable RFP", "false"), ("ENABLE_RFP", "false"), ("Reverse Futility Pruning", "false"), ("Use Reverse Futility Pruning", "false")],
        [("Enable Razoring", "false"), ("ENABLE_RAZORING", "false"), ("Razoring", "false"), ("Use Razoring", "false")],
    ],
    "no_lmr": [[("Enable LMR", "false"), ("ENABLE_LMR", "false"), ("Late Move Reductions", "false"), ("Use Late Move Reductions", "false"), ("LMR", "false")]],
    "no_see_pruning": [
        [("Enable SEE Pruning", "false"), ("ENABLE_SEE_PRUNING", "false"), ("SEE Pruning", "false")],
        [("Enable Shallow SEE Pruning", "false"), ("ENABLE_SHALLOW_SEE_PRUNING", "false"), ("Shallow SEE Pruning", "false")]
    ],
}

PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0, "p": -100, "n": -320, "b": -330, "r": -500, "q": -900, "k": 0}


@dataclass(frozen=True)
class Position:
    position_id: str
    priority: str
    bucket: str
    fen: str
    notes: str


@dataclass
class UciOption:
    name: str
    option_type: str
    default: str | None = None


class UciError(RuntimeError):
    pass


class UciProcess:
    def __init__(self, command: str, label: str, timeout: float, echo: bool = False):
        self.command = command
        self.label = label
        self.timeout = timeout
        self.echo = echo
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.options: dict[str, UciOption] = {}
        self.option_lookup: dict[str, str] = {}

    def __enter__(self) -> "UciProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(shlex.split(self.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.output_queue = queue.Queue()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        self.send("uci")
        lines = self.read_until(lambda line: line.strip() == "uciok")
        self.options = parse_uci_options(lines)
        self.option_lookup = {normalize_option_name(option.name): option.name for option in self.options.values()}

    def _reader_loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.output_queue.put(line.rstrip("\n"))

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.send("quit")
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        finally:
            self.process = None

    def send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise UciError(f"{self.label}: process is not running")
        if self.echo:
            print(f"[{self.label} <<] {command}", file=sys.stderr)
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, predicate: Callable[[str], bool], timeout: float | None = None) -> list[str]:
        if self.process is None:
            raise UciError(f"{self.label}: process is not running")
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        lines: list[str] = []
        while True:
            if self.process.poll() is not None and self.output_queue.empty():
                raise UciError(f"{self.label}: process exited with code {self.process.returncode}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UciError(f"{self.label}: timed out waiting for UCI output")
            try:
                line = self.output_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise UciError(f"{self.label}: timed out waiting for UCI output") from exc
            if self.echo:
                print(f"[{self.label} >>] {line}", file=sys.stderr)
            lines.append(line)
            if predicate(line):
                return lines

    def is_ready(self) -> None:
        self.send("isready")
        self.read_until(lambda line: line.strip() == "readyok")

    def set_option(self, name: str, value: str | None) -> bool:
        actual_name = self.option_lookup.get(normalize_option_name(name))
        if actual_name is None:
            return False
        option = self.options[actual_name]
        if option.option_type == "button" or value is None:
            self.send(f"setoption name {actual_name}")
        else:
            self.send(f"setoption name {actual_name} value {value}")
        return True

    def configure(self, options: list[tuple[str, str | None]]) -> tuple[list[str], list[str]]:
        applied, missing = [], []
        for name, value in options:
            if self.set_option(name, value):
                applied.append(format_option_pair(name, value))
            else:
                missing.append(format_option_pair(name, value))
        self.is_ready()
        return applied, missing

    def configure_option_groups(self, option_groups: list[list[tuple[str, str]]]) -> tuple[list[str], list[str]]:
        applied, missing = [], []
        for group in option_groups:
            group_missing = []
            for name, value in group:
                if self.set_option(name, value):
                    applied.append(format_option_pair(name, value))
                    break
                group_missing.append(format_option_pair(name, value))
            else:
                missing.append("|".join(group_missing))
        self.is_ready()
        return applied, missing

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.is_ready()

    def eval_position(self, fen: str, eval_timeout: float | None = None) -> tuple[dict[str, int | None], list[str]]:
        self.send(f"position fen {fen}")
        self.send("eval")
        self.send("isready")
        lines = self.read_until(lambda line: line.strip() == "readyok", timeout=eval_timeout)
        return parse_eval_output(lines), lines

    def search(self, fen: str, depths: list[int], search_timeout: float | None = None) -> dict[str, object]:
        max_depth = max(depths)
        self.new_game()
        self.send(f"position fen {fen}")
        self.send(f"go depth {max_depth}")
        lines = self.read_until(lambda line: line.startswith("bestmove"), timeout=search_timeout)
        return parse_search_output(lines, set(depths))


class InternalStaticEvaluator:
    def __init__(self, mode: str):
        self.mode = mode
        self.available = False
        self.error = ""
        self.weights_loaded = False
        self.parse_fen = None
        self.evaluate_classical = None
        self.init_accumulator = None
        self.nnue_forward_incremental = None
        self.np = None
        if mode != "off":
            self._load()

    def _load(self) -> None:
        script_dir = Path(__file__).resolve().parent
        for candidate in (script_dir, script_dir.parent):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
        try:
            import numpy as np
            from chess_engine.classical.fen_parser import parse_fen
            from chess_engine.classical.evaluation import evaluate_position as evaluate_classical
            from chess_engine.nnue.ml_eval.inference import init_accumulator, nnue_forward_incremental, weights_loaded, warmup_numba_kernels
        except Exception as exc:
            self.error = f"internal static imports failed: {exc}"
            if self.mode == "on":
                raise RuntimeError(self.error) from exc
            return
        self.np = np
        self.parse_fen = parse_fen
        self.evaluate_classical = evaluate_classical
        self.init_accumulator = init_accumulator
        self.nnue_forward_incremental = nnue_forward_incremental
        self.weights_loaded = bool(weights_loaded)
        self.available = True
        if self.weights_loaded:
            try:
                warmup_numba_kernels()
            except Exception as exc:
                self.error = f"warmup failed: {exc}"

    def evaluate(self, fen: str) -> dict[str, object]:
        result: dict[str, object] = {
            "internal_static_available": self.available,
            "internal_static_error": self.error,
            "internal_classical_stm_cp": None,
            "internal_classical_white_cp": None,
            "internal_nnue_stm_cp": None,
            "internal_nnue_white_cp": None,
            "internal_piece_count": None,
            "stm": fen.split()[1] if len(fen.split()) > 1 else "",
        }
        if not self.available:
            return result
        assert self.parse_fen is not None and self.evaluate_classical is not None
        assert self.np is not None and self.init_accumulator is not None and self.nnue_forward_incremental is not None
        try:
            piece_bbs, occupancy_bbs, game_state = self.parse_fen(fen)
            side_to_move = int(game_state[0])
            classical_stm = int(self.evaluate_classical(piece_bbs, occupancy_bbs, game_state, lazy=False))
            result["internal_classical_stm_cp"] = classical_stm
            result["internal_classical_white_cp"] = classical_stm if side_to_move == 0 else -classical_stm
            piece_count = 0
            for piece_index in range(12):
                bb = piece_bbs[piece_index]
                while bb:
                    piece_count += 1
                    bb &= bb - self.np.uint64(1)
            result["internal_piece_count"] = piece_count
            if self.weights_loaded:
                accumulator_stack = self.np.zeros((1, 2, 512), dtype=self.np.int32)
                self.init_accumulator(piece_bbs, accumulator_stack)
                nnue_stm = int(self.nnue_forward_incremental(0, side_to_move, accumulator_stack, piece_count))
                result["internal_nnue_stm_cp"] = nnue_stm
                result["internal_nnue_white_cp"] = nnue_stm if side_to_move == 0 else -nnue_stm
            else:
                result["internal_static_error"] = "NNUE weights are not loaded"
        except Exception as exc:
            result["internal_static_error"] = str(exc)
        return result


def normalize_option_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


def format_option_pair(name: str, value: str | None) -> str:
    return name if value is None else f"{name}={value}"


def parse_uci_options(lines: Iterable[str]) -> dict[str, UciOption]:
    options: dict[str, UciOption] = {}
    for line in lines:
        if not line.startswith("option name ") or " type " not in line:
            continue
        rest = line[len("option name "):]
        name, after_type = rest.split(" type ", 1)
        option_type = after_type.split()[0] if after_type.split() else "string"
        options[name] = UciOption(name=name, option_type=option_type, default=parse_uci_option_default(after_type))
    return options


def parse_uci_option_default(after_type: str) -> str | None:
    if " default " not in after_type:
        return None
    value = after_type.split(" default ", 1)[1]
    stops = [value.find(marker) for marker in (" min ", " max ", " var ") if value.find(marker) >= 0]
    return value[:min(stops)].strip() if stops else value.strip()


def parse_option_assignment(raw: str) -> tuple[str, str | None]:
    if "=" not in raw:
        return raw.strip(), None
    name, value = raw.split("=", 1)
    return name.strip(), value.strip()


def parse_mode_option(raw: str) -> tuple[str, str, str | None]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("mode option must be mode:name=value")
    mode, assignment = raw.split(":", 1)
    name, value = parse_option_assignment(assignment)
    return mode.strip(), name, value


def parse_depths(raw: str) -> list[int]:
    depths = sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()})
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError("depths must be positive integers")
    return depths


def parse_list(raw: str | None) -> set[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return {piece.strip() for piece in raw.split(",") if piece.strip()}


def read_positions(path: Path) -> list[Position]:
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = EMBEDDED_POSITIONS_TSV.splitlines()
    positions: list[Position] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "id":
            continue
        if len(parts) != 5:
            source = path if path.exists() else "embedded positions"
            raise ValueError(f"{source}:{line_number}: expected 5 tab-separated fields")
        position = Position(*parts)
        validate_fen(position.fen)
        positions.append(position)
    return positions


def filter_positions(positions: list[Position], priorities: set[str] | None, buckets: set[str] | None, ids: set[str] | None, limit: int | None) -> list[Position]:
    filtered = []
    for position in positions:
        if priorities is not None and position.priority not in priorities:
            continue
        if buckets is not None and position.bucket not in buckets:
            continue
        if ids is not None and position.position_id not in ids:
            continue
        filtered.append(position)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def parse_board(fen: str) -> tuple[dict[int, str], str, str]:
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"FEN must have 6 fields: {fen}")
    board, side_to_move, castling, en_passant, halfmove, fullmove = parts
    if side_to_move not in {"w", "b"}:
        raise ValueError(f"bad side to move in FEN: {fen}")
    if castling != "-" and not set(castling) <= set("KQkq"):
        raise ValueError(f"bad castling field in FEN: {fen}")
    if en_passant != "-" and not re.fullmatch(r"[a-h][36]", en_passant):
        raise ValueError(f"bad en passant field in FEN: {fen}")
    int(halfmove); int(fullmove)
    squares: dict[int, str] = {}
    rank, file_index = 7, 0
    for char in board:
        if char == "/":
            if file_index != 8:
                raise ValueError(f"bad rank width in FEN: {fen}")
            rank, file_index = rank - 1, 0
        elif char.isdigit():
            file_index += int(char)
        elif char in PIECE_VALUES:
            if rank < 0 or file_index >= 8:
                raise ValueError(f"board overflow in FEN: {fen}")
            squares[rank * 8 + file_index] = char
            file_index += 1
        else:
            raise ValueError(f"bad piece char in FEN: {fen}")
    if rank != 0 or file_index != 8:
        raise ValueError(f"bad board shape in FEN: {fen}")
    return squares, side_to_move, castling


def validate_fen(fen: str) -> None:
    squares, _, castling = parse_board(fen)
    counts = Counter(squares.values())
    if counts["K"] != 1 or counts["k"] != 1:
        raise ValueError(f"FEN must contain exactly one white and black king: {fen}")
    if counts["P"] > 8 or counts["p"] > 8:
        raise ValueError(f"FEN has too many pawns: {fen}")
    for square, piece in squares.items():
        if piece in {"P", "p"} and square // 8 in {0, 7}:
            raise ValueError(f"FEN has pawn on first/eighth rank: {fen}")
    white_king = next(square for square, piece in squares.items() if piece == "K")
    black_king = next(square for square, piece in squares.items() if piece == "k")
    if max(abs(white_king % 8 - black_king % 8), abs(white_king // 8 - black_king // 8)) <= 1:
        raise ValueError(f"FEN has adjacent kings: {fen}")
    if is_square_attacked(squares, white_king, by_white=False):
        raise ValueError(f"white king is in check at root: {fen}")
    if is_square_attacked(squares, black_king, by_white=True):
        raise ValueError(f"black king is in check at root: {fen}")
    required = {"K": (4, "K", 7, "R"), "Q": (4, "K", 0, "R"), "k": (60, "k", 63, "r"), "q": (60, "k", 56, "r")}
    if castling != "-":
        for right in castling:
            king_square, king_piece, rook_square, rook_piece = required[right]
            if squares.get(king_square) != king_piece or squares.get(rook_square) != rook_piece:
                raise ValueError(f"FEN castling rights do not match pieces: {fen}")


def is_square_attacked(squares: dict[int, str], square: int, by_white: bool) -> bool:
    rank, file_index = square // 8, square % 8
    pawn_rank, pawn = (rank - 1, "P") if by_white else (rank + 1, "p")
    for delta_file in (-1, 1):
        if on_board(pawn_rank, file_index + delta_file) and squares.get(pawn_rank * 8 + file_index + delta_file) == pawn:
            return True
    knight = "N" if by_white else "n"
    for dr, df in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        if on_board(rank + dr, file_index + df) and squares.get((rank + dr) * 8 + file_index + df) == knight:
            return True
    king = "K" if by_white else "k"
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if (dr or df) and on_board(rank + dr, file_index + df) and squares.get((rank + dr) * 8 + file_index + df) == king:
                return True
    for dr, df, pieces in ((1, 0, "RQ"), (-1, 0, "RQ"), (0, 1, "RQ"), (0, -1, "RQ"), (1, 1, "BQ"), (1, -1, "BQ"), (-1, 1, "BQ"), (-1, -1, "BQ")):
        rr, ff = rank + dr, file_index + df
        while on_board(rr, ff):
            piece = squares.get(rr * 8 + ff)
            if piece:
                if (by_white and piece in pieces) or ((not by_white) and piece in pieces.lower()):
                    return True
                break  # 射線被阻擋，停止此方向的搜尋，繼續檢查下一個方向
            rr, ff = rr + dr, ff + df
    return False


def on_board(rank: int, file_index: int) -> bool:
    return 0 <= rank < 8 and 0 <= file_index < 8


def material_cp(fen: str) -> int:
    squares, _, _ = parse_board(fen)
    return sum(PIECE_VALUES[piece] for piece in squares.values())


def parse_search_output(lines: list[str], wanted_depths: set[int]) -> dict[str, object]:
    depth_info: dict[int, dict[str, object]] = {depth: {} for depth in wanted_depths}
    final_bestmove, final_ponder = "", ""
    for line in lines:
        if line.startswith("bestmove"):
            parts = line.split()
            final_bestmove = parts[1] if len(parts) > 1 else ""
            if "ponder" in parts and parts.index("ponder") + 1 < len(parts):
                final_ponder = parts[parts.index("ponder") + 1]
            continue
        if not line.startswith("info "):
            continue
        depth_match = re.search(r"\bdepth\s+(\d+)", line)
        if not depth_match:
            continue
        depth = int(depth_match.group(1))
        if depth not in wanted_depths:
            continue
        entry = depth_info.setdefault(depth, {})
        score_match = re.search(r"\bscore\s+(cp|mate)\s+(-?\d+)", line)
        if score_match:
            kind, value = score_match.group(1), int(score_match.group(2))
            entry["score_kind"] = kind
            entry["score_value"] = value
            entry["score_cp"] = value if kind == "cp" else None
            entry["score_raw"] = f"{kind}:{value}"
        for key in ("nodes", "nps", "time", "qnodes"):
            match = re.search(rf"\b{key}\s+(\d+)", line)
            if match:
                entry[key] = int(match.group(1))
        pv_match = re.search(r"\bpv\s+(.+)$", line)
        if pv_match:
            pv = pv_match.group(1).strip()
            entry["pv"] = pv
            entry["bestmove"] = pv.split()[0] if pv else ""
    return {"bestmove": final_bestmove, "ponder": final_ponder, "depth_info": depth_info, "raw_lines": lines}


def parse_eval_output(lines: list[str]) -> dict[str, int | None]:
    useful = [line for line in lines if line.strip() and line.strip() != "readyok"]
    return {
        "nnue_cp": find_eval_score(useful, ("nnue",)),
        "classical_cp": find_eval_score(useful, ("classical", "hce")),
        "final_cp": find_eval_score(useful, ("final evaluation", "total evaluation", "final", "total")),
    }


def find_eval_score(lines: list[str], keywords: tuple[str, ...]) -> int | None:
    candidates: list[int] = []
    for line in lines:
        lowered = line.casefold()
        if not any(keyword in lowered for keyword in keywords):
            continue
        if "enabled" in lowered or "file" in lowered or "network" in lowered:
            continue
        numbers = re.findall(r"(?<![A-Za-z0-9_])([+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9_])", line)
        if numbers:
            candidates.append(eval_number_to_cp(float(numbers[-1]), line))
    return candidates[-1] if candidates else None


def eval_number_to_cp(value: float, line: str) -> int:
    lowered = line.casefold()
    return int(round(value)) if " cp" in lowered or lowered.endswith("cp") or abs(value) > 20 else int(round(value * 100))


def score_volatility(depth_info: dict[int, dict[str, object]], depths: list[int]) -> int | None:
    scores = [depth_info.get(depth, {}).get("score_cp") for depth in depths]
    scores = [score for score in scores if isinstance(score, int)]
    if len(scores) < 2:
        return None
    return max(abs(current - previous) for previous, current in zip(scores, scores[1:]))


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def classify_position(sf_eval: dict[str, int | None] | None, volatility: int | None) -> str:
    if volatility is not None and volatility > 120:
        return "unstable"
    if not sf_eval:
        return "unclassified"
    nnue, final = sf_eval.get("nnue_cp"), sf_eval.get("final_cp")
    if nnue is None or final is None:
        return "unclassified"
    diff = abs(final - nnue)
    if diff < 80:
        return "quiet_static"
    if diff >= 150:
        return "search_sensitive"
    return "mixed"


def make_mode_option_groups(mode: str) -> list[list[tuple[str, str]]]:
    return MODE_OPTION_GROUPS.get(mode, [])


def flatten_result_row(position: Position, engine_label: str, mode: str, search_result: dict[str, object] | None, uci_eval: dict[str, int | None] | None, internal_eval: dict[str, object] | None, sf_reference: dict[str, object] | None, depths: list[int], mode_applied: list[str], mode_missing: list[str], error: str = "") -> dict[str, object]:
    depth_info = search_result.get("depth_info", {}) if search_result else {}
    if not isinstance(depth_info, dict):
        depth_info = {}
    sf_eval = sf_reference.get("eval") if sf_reference else None
    sf_search = sf_reference.get("search") if sf_reference else None
    sf_depth_info = sf_search.get("depth_info", {}) if isinstance(sf_search, dict) else {}
    if not isinstance(sf_depth_info, dict):
        sf_depth_info = {}
    internal_eval = internal_eval or {}
    uci_eval = uci_eval or {}
    static_cp = first_int(internal_eval.get("internal_nnue_stm_cp"), uci_eval.get("nnue_cp"), uci_eval.get("final_cp"))
    classical_cp = first_int(internal_eval.get("internal_classical_stm_cp"), uci_eval.get("classical_cp"))
    d8_score = depth_info.get(8, {}).get("score_cp") if isinstance(depth_info.get(8, {}), dict) else None
    gap = d8_score - static_cp if isinstance(d8_score, int) and isinstance(static_cp, int) else None
    volatility = score_volatility(depth_info, depths)
    row: dict[str, object] = {
        "id": position.position_id,
        "priority": position.priority,
        "bucket": position.bucket,
        "engine": engine_label,
        "mode": mode,
        "fen": position.fen,
        "notes": position.notes,
        "material_cp_white_minus_black": material_cp(position.fen),
        "mode_options_applied": ";".join(mode_applied),
        "mode_options_missing": ";".join(mode_missing),
        "our_nnue_static_cp": static_cp,
        "our_classical_cp": classical_cp,
        "our_eval_final_cp": uci_eval.get("final_cp"),
        "internal_static_available": internal_eval.get("internal_static_available"),
        "internal_static_error": internal_eval.get("internal_static_error"),
        "internal_nnue_stm_cp": internal_eval.get("internal_nnue_stm_cp"),
        "internal_nnue_white_cp": internal_eval.get("internal_nnue_white_cp"),
        "internal_classical_stm_cp": internal_eval.get("internal_classical_stm_cp"),
        "internal_classical_white_cp": internal_eval.get("internal_classical_white_cp"),
        "internal_piece_count": internal_eval.get("internal_piece_count"),
        "uci_nnue_cp": uci_eval.get("nnue_cp"),
        "uci_classical_cp": uci_eval.get("classical_cp"),
        "bestmove_final": search_result.get("bestmove", "") if search_result else "",
        "score_volatility_cp": volatility,
        "search_static_gap_d8_cp": gap,
        "suggested_class": classify_position(sf_eval if isinstance(sf_eval, dict) else None, volatility),
        "sf_nnue_cp": sf_eval.get("nnue_cp") if isinstance(sf_eval, dict) else None,
        "sf_final_cp": sf_eval.get("final_cp") if isinstance(sf_eval, dict) else None,
        "sf_classical_cp": sf_eval.get("classical_cp") if isinstance(sf_eval, dict) else None,
        "sf_bestmove_final": sf_search.get("bestmove", "") if isinstance(sf_search, dict) else "",
        "error": error,
    }
    for depth in depths:
        entry = depth_info.get(depth, {}) if isinstance(depth_info.get(depth, {}), dict) else {}
        row[f"d{depth}_score_cp"] = entry.get("score_cp")
        row[f"d{depth}_score_raw"] = entry.get("score_raw")
        row[f"d{depth}_bestmove"] = entry.get("bestmove")
        row[f"d{depth}_nodes"] = entry.get("nodes")
        row[f"d{depth}_qnodes"] = entry.get("qnodes")
        row[f"d{depth}_time_ms"] = entry.get("time")
        sf_entry = sf_depth_info.get(depth, {}) if isinstance(sf_depth_info.get(depth, {}), dict) else {}
        row[f"sf_d{depth}_score_cp"] = sf_entry.get("score_cp")
        row[f"sf_d{depth}_score_raw"] = sf_entry.get("score_raw")
        row[f"sf_d{depth}_bestmove"] = sf_entry.get("bestmove")
        row[f"sf_d{depth}_nodes"] = sf_entry.get("nodes")
    return row


def first_int(*values: object) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
    return None


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("bucket", ""))].append(row)
        by_mode[str(row.get("mode", ""))].append(row)
    return {"rows": len(rows), "by_bucket": {k: summarize_group(v) for k, v in by_bucket.items()}, "by_mode": {k: summarize_group(v) for k, v in by_mode.items()}}


def summarize_group(group: list[dict[str, object]]) -> dict[str, object]:
    gaps = [abs(int(row["search_static_gap_d8_cp"])) for row in group if isinstance(row.get("search_static_gap_d8_cp"), int)]
    volatilities = [int(row["score_volatility_cp"]) for row in group if isinstance(row.get("score_volatility_cp"), int)]
    static_values = [abs(int(row["our_nnue_static_cp"])) for row in group if isinstance(row.get("our_nnue_static_cp"), int)]
    return {"rows": len(group), "errors": sum(1 for row in group if row.get("error")), "static_abs_p50": percentile(static_values, 0.50), "static_abs_p90": percentile(static_values, 0.90), "gap_abs_p50": percentile(gaps, 0.50), "gap_abs_p90": percentile(gaps, 0.90), "volatility_p50": percentile(volatilities, 0.50), "volatility_p90": percentile(volatilities, 0.90)}


def run_stockfish_reference(stockfish_command: str | None, positions: list[Position], depths: list[int], base_options: list[tuple[str, str | None]], timeout: float, eval_timeout: float, search_timeout: float, skip_eval: bool, skip_search: bool, echo: bool) -> dict[str, dict[str, object]]:
    references: dict[str, dict[str, object]] = {}
    if stockfish_command is None:
        return references
    with UciProcess(stockfish_command, "stockfish", timeout=timeout, echo=echo) as proc:
        proc.configure(base_options)
        for position in positions:
            reference: dict[str, object] = {}
            try:
                reference["eval"] = {} if skip_eval else proc.eval_position(position.fen, eval_timeout=eval_timeout)[0]
            except Exception as exc:
                reference["eval_error"] = str(exc)
                reference["eval"] = {}
            try:
                reference["search"] = {} if skip_search else proc.search(position.fen, depths, search_timeout=search_timeout)
            except Exception as exc:
                reference["search_error"] = str(exc)
                reference["search"] = {}
            references[position.position_id] = reference
    return references


def run_engine_modes(engine_command: str, engine_label: str, positions: list[Position], depths: list[int], modes: list[str], base_options: list[tuple[str, str | None]], mode_overrides: dict[str, list[tuple[str, str | None]]], internal_by_id: dict[str, dict[str, object]], sf_references: dict[str, dict[str, object]], timeout: float, eval_timeout: float, search_timeout: float, skip_eval: bool, skip_search: bool, echo: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    active_modes = ["static_only"] if skip_search else modes
    for mode in active_modes:
        mode_option_groups = [] if skip_search else make_mode_option_groups(mode)
        mode_override_options = [] if skip_search else mode_overrides.get(mode, [])
        with UciProcess(engine_command, f"{engine_label}:{mode}", timeout=timeout, echo=echo) as proc:
            base_applied, base_missing = proc.configure(base_options)
            mode_applied, mode_missing = proc.configure_option_groups(mode_option_groups)
            override_applied, override_missing = proc.configure(mode_override_options)
            all_applied = base_applied + mode_applied + override_applied
            all_missing = base_missing + mode_missing + override_missing
            for position in positions:
                uci_eval: dict[str, int | None] | None = None
                search_result: dict[str, object] | None = None
                error = ""
                try:
                    uci_eval = {} if skip_eval else proc.eval_position(position.fen, eval_timeout=eval_timeout)[0]
                    search_result = {} if skip_search else proc.search(position.fen, depths, search_timeout=search_timeout)
                except Exception as exc:
                    error = str(exc)
                rows.append(flatten_result_row(position, engine_label, mode, search_result, uci_eval, internal_by_id.get(position.position_id), sf_references.get(position.position_id), depths, all_applied, all_missing, error))
    return rows


def run_static_only_rows(positions: list[Position], depths: list[int], internal_by_id: dict[str, dict[str, object]], sf_references: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return [flatten_result_row(position, "static_only", "static_only", None, None, internal_by_id.get(position.position_id), sf_references.get(position.position_id), depths, [], []) for position in positions]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_report(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# NNUE / Search Calibration Report", "", f"- Rows: {summary['rows']}", f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "", "## Bucket Summary", "", "| Bucket | Rows | Static Abs P50 | Static Abs P90 | Gap Abs P50 | Gap Abs P90 | Vol P90 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for bucket, stats in summary.get("by_bucket", {}).items():
        lines.append(f"| {bucket} | {stats.get('rows')} | {fmt(stats.get('static_abs_p50'))} | {fmt(stats.get('static_abs_p90'))} | {fmt(stats.get('gap_abs_p50'))} | {fmt(stats.get('gap_abs_p90'))} | {fmt(stats.get('volatility_p90'))} |")
    lines.extend(["", "## Position Results", "", "| ID | Bucket | Engine | Mode | Static | d8 | d10 | Gap d8 | Vol | Best | Notes |", "|---|---|---|---|---:|---:|---:|---:|---:|---|---|"])
    for row in rows:
        lines.append(f"| {row.get('id')} | {row.get('bucket')} | {row.get('engine')} | {row.get('mode')} | {fmt(row.get('our_nnue_static_cp'))} | {fmt(row.get('d8_score_cp'))} | {fmt(row.get('d10_score_cp'))} | {fmt(row.get('search_static_gap_d8_cp'))} | {fmt(row.get('score_volatility_cp'))} | {row.get('bestmove_final') or ''} | {row.get('notes')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def ensure_relative_output(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(f"output path must be relative to the workspace: {path}")
    return path


def resolve_engine_command(raw: str | None, auto_name: str | None = None) -> str | None:
    if raw is None:
        return None
    if raw == "auto" and auto_name:
        found = shutil.which(auto_name)
        if not found:
            raise FileNotFoundError(f"could not find {auto_name!r} on PATH")
        return found
    return raw


def print_dry_run(positions: list[Position], modes: list[str], depths: list[int], internal_static: str) -> None:
    counts = Counter((position.priority, position.bucket) for position in positions)
    print(f"positions: {len(positions)}")
    print(f"depths: {','.join(str(depth) for depth in depths)}")
    print(f"modes: {','.join(modes)}")
    print(f"internal_static: {internal_static}")
    for (priority, bucket), count in sorted(counts.items()):
        print(f"{priority}\t{bucket}\t{count}")
    for position in positions:
        print(f"{position.position_id}\t{position.priority}\t{position.bucket}\t{position.fen}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined NNUE static-scale and UCI search/pruning calibration tool.")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, help="tab-separated FEN file; embedded defaults are used if missing")
    parser.add_argument("--engine", help="UCI executable for your engine")
    parser.add_argument("--engine-label", default="our_engine", help="label used in outputs")
    parser.add_argument("--stockfish", help="Stockfish executable, or 'auto' to search PATH")
    parser.add_argument("--internal-static", choices=["auto", "on", "off"], default="auto", help="use local Python engine imports for raw NNUE/classical static")
    parser.add_argument("--priorities", default="P1", help="comma-separated priorities, e.g. P1 or P1,P2")
    parser.add_argument("--buckets", help="comma-separated bucket filter")
    parser.add_argument("--ids", help="comma-separated id filter")
    parser.add_argument("--limit", type=int, help="limit number of positions after filtering")
    parser.add_argument("--depths", type=parse_depths, default=DEFAULT_DEPTHS, help="comma-separated depths")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="comma-separated mode names")
    parser.add_argument("--engine-option", action="append", default=[], help="base engine option as name=value")
    parser.add_argument("--stockfish-option", action="append", default=[], help="base Stockfish option as name=value")
    parser.add_argument("--mode-option", action="append", default=[], help="extra mode option as mode:name=value")
    parser.add_argument("--timeout", type=float, default=99999.0, help="general UCI timeout seconds")
    parser.add_argument("--eval-timeout", type=float, default=99999.0, help="eval command timeout seconds")
    parser.add_argument("--search-timeout", type=float, default=99999.0, help="go depth timeout seconds")
    parser.add_argument("--skip-eval", action="store_true", help="skip non-standard UCI eval command")
    parser.add_argument("--skip-search", action="store_true", help="only gather static/eval data; no go depth")
    parser.add_argument("--output-csv", default="calibration_results.csv", help="CSV output path")
    parser.add_argument("--output-json", default="calibration_results.json", help="JSON output path")
    parser.add_argument("--report-md", default="calibration_report.md", help="Markdown report path")
    parser.add_argument("--dry-run", action="store_true", help="validate and print planned positions only")
    parser.add_argument("--echo-uci", action="store_true", help="print UCI traffic to stderr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    positions_path = Path(args.positions)
    output_csv = ensure_relative_output(Path(args.output_csv))
    output_json = ensure_relative_output(Path(args.output_json))
    report_md = ensure_relative_output(Path(args.report_md))
    positions = read_positions(positions_path)
    selected_positions = filter_positions(positions, parse_list(args.priorities), parse_list(args.buckets), parse_list(args.ids), args.limit)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown_modes = [mode for mode in modes if mode not in MODE_OPTION_GROUPS]
    if unknown_modes:
        raise ValueError(f"unknown mode(s): {', '.join(unknown_modes)}")
    if args.dry_run:
        print_dry_run(selected_positions, modes, args.depths, args.internal_static)
        return 0
    if not selected_positions:
        raise ValueError("no positions selected")

    internal = InternalStaticEvaluator(args.internal_static)
    internal_by_id = {position.position_id: internal.evaluate(position.fen) for position in selected_positions} if args.internal_static != "off" else {}
    if args.internal_static == "on" and not internal.available:
        raise RuntimeError(internal.error or "internal static evaluator is unavailable")

    engine_command = resolve_engine_command(args.engine)
    stockfish_command = resolve_engine_command(args.stockfish, auto_name="stockfish")
    if engine_command is None and stockfish_command is None and not any(v.get("internal_static_available") for v in internal_by_id.values()):
        raise ValueError("provide --engine, --stockfish, or run on a machine where --internal-static auto/on can import the engine")

    engine_options = [parse_option_assignment(raw) for raw in args.engine_option]
    stockfish_options = [parse_option_assignment(raw) for raw in args.stockfish_option]
    mode_overrides: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for raw in args.mode_option:
        mode, name, value = parse_mode_option(raw)
        mode_overrides[mode].append((name, value))

    sf_references = run_stockfish_reference(stockfish_command, selected_positions, args.depths, stockfish_options, args.timeout, args.eval_timeout, args.search_timeout, args.skip_eval, args.skip_search, args.echo_uci)

    if engine_command is not None:
        rows = run_engine_modes(engine_command, args.engine_label, selected_positions, args.depths, modes, engine_options, mode_overrides, internal_by_id, sf_references, args.timeout, args.eval_timeout, args.search_timeout, args.skip_eval, args.skip_search, args.echo_uci)
    elif stockfish_command is not None and not args.skip_search:
        rows = [flatten_result_row(position, "stockfish_reference_only", "full_search", sf_references.get(position.position_id, {}).get("search"), sf_references.get(position.position_id, {}).get("eval"), internal_by_id.get(position.position_id), sf_references.get(position.position_id), args.depths, [], []) for position in selected_positions]
    else:
        rows = run_static_only_rows(selected_positions, args.depths, internal_by_id, sf_references)

    summary = build_summary(rows)
    payload = {"config": {"positions": str(positions_path), "engine": engine_command, "stockfish": stockfish_command, "internal_static": args.internal_static, "depths": args.depths, "modes": modes, "priorities": args.priorities, "buckets": args.buckets, "ids": args.ids, "skip_search": args.skip_search}, "summary": summary, "results": rows}
    write_csv(output_csv, rows)
    write_json(output_json, payload)
    write_report(report_md, rows, summary)
    print(f"wrote {len(rows)} rows to {output_csv}, {output_json}, and {report_md}")
    if args.internal_static != "off" and not internal.available:
        print(f"note: internal static unavailable: {internal.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
