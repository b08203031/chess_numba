"""Shared match infrastructure for local engine tournaments.

The preferred path keeps the Numba engine in the controller process and gives
each worker an independent SearchContext.  UCI subprocesses remain available
for external engines and as a compatibility fallback.
"""
from __future__ import annotations

import importlib
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import chess
import chess.pgn
import numpy as np


START_FEN = chess.STARTING_FEN
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


class EngineError(RuntimeError):
    """An engine failed to complete the match protocol."""


@dataclass(frozen=True)
class SearchLimit:
    """Exactly one fixed-node, fixed-depth, or movetime limit."""

    nodes: Optional[int] = None
    depth: Optional[int] = None
    time_ms: Optional[int] = None

    def __post_init__(self) -> None:
        values = (self.nodes, self.depth, self.time_ms)
        if sum(value is not None for value in values) != 1:
            raise ValueError("A search limit must specify exactly one of nodes/depth/time_ms")
        if any(value is not None and int(value) <= 0 for value in values):
            raise ValueError("Search limits must be positive integers")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Optional[int]]) -> "SearchLimit":
        if values.get("nodes") is not None:
            return cls(nodes=int(values["nodes"]))
        if values.get("depth") is not None:
            return cls(depth=int(values["depth"]))
        value = values.get("time_ms")
        if value is None:
            value = values.get("time")
        if value is None:
            raise ValueError("Missing search limit")
        return cls(time_ms=int(value))

    def uci_command(self) -> str:
        if self.nodes is not None:
            return f"go nodes {self.nodes}"
        if self.depth is not None:
            return f"go depth {self.depth}"
        return f"go movetime {self.time_ms}"

    def token(self) -> str:
        if self.nodes is not None:
            return f"n{self.nodes}"
        if self.depth is not None:
            return f"d{self.depth}"
        return str(self.time_ms)

    def describe(self) -> str:
        if self.nodes is not None:
            return f"fixed nodes {self.nodes}"
        if self.depth is not None:
            return f"fixed depth {self.depth}"
        return f"movetime {self.time_ms} ms"


def parse_limit_token(token: str) -> SearchLimit:
    token = str(token).strip().lower()
    if not token:
        raise ValueError("Empty search limit")
    if token.startswith("n"):
        return SearchLimit(nodes=int(token[1:]))
    if token.startswith("d"):
        return SearchLimit(depth=int(token[1:]))
    return SearchLimit(time_ms=int(token))


def parse_info_line(line: str) -> str:
    """Convert a UCI info line to a compact PGN comment."""
    if not line:
        return ""
    tokens = line.split()
    info: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"depth", "nodes", "time", "nps", "hashfull"} and i + 1 < len(tokens):
            info[token] = tokens[i + 1]
            i += 2
        elif token == "score" and i + 2 < len(tokens):
            kind, value = tokens[i + 1], tokens[i + 2]
            if kind == "cp":
                try:
                    info["score"] = f"{int(value) / 100:+.2f}"
                except ValueError:
                    info["score"] = f"cp {value}"
            elif kind == "mate":
                info["score"] = f"#{value}"
            i += 3
        elif token == "pv":
            break
        else:
            i += 1
    parts = []
    if "depth" in info:
        parts.append(f"d={info['depth']}")
    if "score" in info:
        parts.append(f"eval={info['score']}")
    if "nodes" in info:
        parts.append(f"n={info['nodes']}")
    if "time" in info:
        parts.append(f"t={info['time']}ms")
    if "nps" in info:
        parts.append(f"nps={info['nps']}")
    if "hashfull" in info:
        parts.append(f"hashfull={info['hashfull']}")
    return ", ".join(parts)


class UciEngine:
    """Long-lived UCI subprocess with a non-blocking stdout reader."""

    def __init__(
        self,
        name: str,
        path: str,
        *,
        options: Optional[Mapping[str, object]] = None,
        env_extra: Optional[Mapping[str, object]] = None,
        verbose: bool = False,
        search_timeout_s: float = 300.0,
    ) -> None:
        self.name = name
        self.path = os.path.abspath(path)
        self.options = dict(options or {})
        self.env_extra = dict(env_extra or {})
        self.verbose = verbose
        self.search_timeout_s = float(search_timeout_s)
        self.process: Optional[subprocess.Popen[str]] = None
        self._lines: queue.Queue[Optional[str]] = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._moves: list[str] = []
        self._start_fen: Optional[str] = None
        self._cache_dir: Optional[str] = None

    def start(self) -> None:
        if self.process is not None:
            return
        is_python = self.path.lower().endswith(".py")
        command = [sys.executable, "-u", self.path] if is_python else [self.path]
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.env_extra.items()})
        # UCI fallback processes cannot share the recursive search compilation.
        # Isolate their writable cache directories to avoid cross-worker locks.
        if is_python:
            self._cache_dir = tempfile.mkdtemp(prefix=f"numba_match_{self.name}_")
            env["NUMBA_CACHE_DIR"] = self._cache_dir
        env.setdefault("NUMBA_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(self.path) or None,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._send("uci")
        self._wait_for("uciok", timeout_s=30.0, phase="UCI initialization")
        for option, value in self.options.items():
            self._send(f"setoption name {option} value {str(value).lower() if isinstance(value, bool) else value}")
        self._ready(timeout_s=max(30.0, self.search_timeout_s))

    def _reader_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for raw in self.process.stdout:
                self._lines.put(raw.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise EngineError(f"{self.name} is not running")
        if self.process.poll() is not None:
            raise EngineError(f"{self.name} exited with code {self.process.returncode}")
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError(f"Lost connection to {self.name}") from exc

    def _next_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError from exc
        if line is None:
            code = self.process.poll() if self.process is not None else None
            raise EngineError(f"{self.name} closed its output (exit code {code})")
        return line

    def _wait_for(self, expected: str, *, timeout_s: float, phase: str) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                line = self._next_line(deadline)
            except TimeoutError as exc:
                raise EngineError(f"{self.name} timed out during {phase}") from exc
            if self.verbose and line:
                print(f"[{self.name} {phase}] {line}", flush=True)
            if line == expected:
                return

    def _ready(self, *, timeout_s: float) -> None:
        self._send("isready")
        self._wait_for("readyok", timeout_s=timeout_s, phase="ready")

    def new_game(self, start_fen: Optional[str]) -> None:
        self._send("ucinewgame")
        self._ready(timeout_s=max(30.0, self.search_timeout_s))
        self._start_fen = start_fen
        self._moves = []

    def push(self, move_uci: str) -> None:
        self._moves.append(move_uci)

    def get_move(
        self, limit: SearchLimit, *, show_info: bool = False
    ) -> tuple[chess.Move, str]:
        command = f"position fen {self._start_fen}" if self._start_fen else "position startpos"
        if self._moves:
            command += " moves " + " ".join(self._moves)
        self._send(command)
        self._send(limit.uci_command())

        if limit.time_ms is not None:
            timeout_s = max(self.search_timeout_s, limit.time_ms / 1000.0 * 5.0 + 5.0)
        else:
            timeout_s = self.search_timeout_s
        deadline = time.monotonic() + timeout_s
        last_info = ""
        stopped = False
        while True:
            try:
                line = self._next_line(deadline)
            except TimeoutError:
                if not stopped:
                    self._send("stop")
                    stopped = True
                    deadline = time.monotonic() + 5.0
                    continue
                raise EngineError(f"{self.name} exceeded the search timeout")
            if line.startswith("info ") and " depth " in f" {line} " and " score " in f" {line} ":
                if show_info:
                    print(f"[warmup {self.name}] {line}", flush=True)
                last_info = line
            if not line.startswith("bestmove"):
                continue
            parts = line.split()
            if len(parts) < 2 or parts[1] in {"(none)", "0000"}:
                raise EngineError(f"{self.name} returned no best move")
            try:
                move = chess.Move.from_uci(parts[1])
            except ValueError as exc:
                raise EngineError(f"{self.name} returned invalid UCI move {parts[1]!r}") from exc
            return move, parse_info_line(last_info)

    def terminate(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                process.wait(timeout=3.0)
            except Exception:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3.0)
        if self._cache_dir:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None


class UciEngineFactory:
    shared_jit = False

    def __init__(self, name: str, path: str, **kwargs: object) -> None:
        self.name = name
        self.path = path
        self.kwargs = kwargs
        self.requires_per_instance_warmup = str(path).lower().endswith(".py")

    def create(self, worker_id: int) -> UciEngine:
        kwargs = dict(self.kwargs)
        kwargs.setdefault(
            "verbose",
            worker_id == 0 and str(self.path).lower().endswith(".py"),
        )
        return UciEngine(self.name, self.path, **kwargs)

    def prepare(self, engine: UciEngine, worker_id: int) -> None:
        """Restore the historical Kiwipete d14 warm-up for Python UCI engines."""
        if not self.requires_per_instance_warmup:
            return
        print(
            f"[warmup] worker {worker_id + 1}: {self.name} Kiwipete depth=14...",
            flush=True,
        )
        started = time.perf_counter()
        engine.new_game(KIWIPETE_FEN)
        engine.get_move(SearchLimit(depth=14), show_info=(worker_id == 0))
        print(
            f"[warmup] worker {worker_id + 1}: {self.name} Kiwipete complete "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

_ENV_LOCK = threading.Lock()


def _reset_context(ctx: object, clear_tt: Optional[Callable[[object], None]] = None) -> None:
    tt = ctx.transposition_table
    if clear_tt is not None:
        clear_tt(tt)
    else:
        try:
            tt.view(np.uint8).fill(0)
            tt["static_eval"] = np.int16(32767)
        except Exception:
            tt["key"] = np.uint32(0)
            tt["score"] = np.int16(0)
            tt["static_eval"] = np.int16(32767)
            tt["depth"] = np.uint8(0)
            tt["flag"] = np.uint8(0)
            tt["generation"] = np.uint8(0)
            tt["best_move"] = np.uint16(0)
            tt["is_pv"] = False
    for field in (
        "history_table", "butterfly_history", "continuation_history",
        "capture_history", "pawn_correction_history", "minor_correction_history",
        "non_pawn_correction_history_white", "non_pawn_correction_history_black",
        "continuation_correction_history", "killer_moves", "counter_moves",
        "low_ply_history", "pv_table", "diag_stats",
    ):
        getattr(ctx, field)[:] = 0
    ctx.pawn_history[:] = -1238
    ctx.pawn_table_keys[:] = np.uint64(0xFFFFFFFFFFFFFFFF)
    ctx.game_history_count = 0
    ctx.tt_generation = 0
    ctx.nodes_searched = np.uint64(0)
    ctx.nodes_searched_array[0] = 0
    ctx.stop_flag[0] = False


class InProcessEngine:
    """One independent context using an already imported Numba backend."""

    def __init__(self, factory: "InProcessEngineFactory", context: object) -> None:
        self.factory = factory
        self.name = factory.name
        self.context = context
        self.piece_bbs = None
        self.occupancy_bbs = None
        self.game_state = None
        self.game_history: list[int] = []
        self.tt_generation = 0
        self._last_search_move_uci: Optional[str] = None
        self._last_search_move_encoded: Optional[int] = None

    def start(self) -> None:
        return

    def new_game(self, start_fen: Optional[str]) -> None:
        _reset_context(self.context, getattr(self.factory, "clear_tt", None))
        fen = start_fen or START_FEN
        self.piece_bbs, self.occupancy_bbs, self.game_state = self.factory.parse_fen(fen)
        self.game_history = [int(self.game_state[4])]
        self.tt_generation = 0
        self._last_search_move_uci = None
        self._last_search_move_encoded = None

    def push(self, move_uci: str) -> None:
        if self.game_state is None:
            raise EngineError(f"{self.name} has no active game")

        # Fast path 1: mover played this exact move in its last search
        if self._last_search_move_uci == move_uci and self._last_search_move_encoded is not None:
            encoded = self._last_search_move_encoded
            self._last_search_move_encoded = None
            self._last_search_move_uci = None
            self.factory.make_move(self.piece_bbs, self.occupancy_bbs, self.game_state, encoded)
            self.game_history.append(int(self.game_state[4]))
            return

        self._last_search_move_encoded = None
        self._last_search_move_uci = None

        # Fast path 2: match candidate move by square indices without string allocation
        encoded = None
        candidates = self.factory.generate_legal_moves(
            self.piece_bbs, self.occupancy_bbs, self.game_state
        )
        if len(move_uci) >= 4:
            target_from = (ord(move_uci[0]) - 97) + (ord(move_uci[1]) - 49) * 8
            target_to = (ord(move_uci[2]) - 97) + (ord(move_uci[3]) - 49) * 8
            promo_char = move_uci[4] if len(move_uci) > 4 else None
            promo_expected = {"n": 0, "b": 1, "r": 2, "q": 3}.get(promo_char) if promo_char else None

            for candidate in candidates:
                cand_int = int(candidate)
                if (cand_int & 0x3F) == target_from and ((cand_int >> 6) & 0x3F) == target_to:
                    if promo_char is not None:
                        flag = (cand_int >> 14) & 0x3
                        if flag == 1:  # SPECIAL_MOVE_FLAG_PROMOTION
                            promo = (cand_int >> 12) & 0x3
                            if promo == promo_expected:
                                encoded = candidate
                                break
                    else:
                        encoded = candidate
                        break

        # Fallback to full string comparison if needed
        if encoded is None:
            for candidate in candidates:
                if self.factory.move_to_uci(candidate) == move_uci:
                    encoded = candidate
                    break

        if encoded is None:
            raise EngineError(f"{self.name} backend rejected legal referee move {move_uci}")
        self.factory.make_move(self.piece_bbs, self.occupancy_bbs, self.game_state, encoded)
        self.game_history.append(int(self.game_state[4]))

    def get_move(
        self, limit: SearchLimit, *, show_info: bool = False
    ) -> tuple[chess.Move, str]:
        if self.game_state is None:
            raise EngineError(f"{self.name} has no active game")
        max_depth = self.factory.max_ply
        time_config: dict[str, int] = {"optimum_time": 0, "maximum_time": 0}
        if limit.nodes is not None:
            time_config["nodes_limit"] = int(limit.nodes)
        elif limit.depth is not None:
            max_depth = int(limit.depth)
        else:
            time_config = {
                "optimum_time": int(limit.time_ms),
                "maximum_time": int(limit.time_ms),
            }
        self.tt_generation = (self.tt_generation + 1) & 0xFF
        started = time.perf_counter()
        result = self.factory.search(
            self.piece_bbs,
            self.occupancy_bbs,
            self.game_state,
            max_depth,
            time_config,
            self.context,
            game_history_list=list(self.game_history),
            tt_generation=self.tt_generation,
            verbose=show_info,
        )
        elapsed_ms = max(1, int((time.perf_counter() - started) * 1000.0))
        if not isinstance(result, (tuple, list)) or len(result) < 6:
            raise EngineError(f"{self.name} returned an invalid search result")
        encoded, score, nodes, _qnodes, _tt_hits, depth = result[:6]
        encoded = int(encoded)
        if encoded == 0 or encoded == self.factory.no_move:
            raise EngineError(f"{self.name} returned no best move")
        uci = self.factory.move_to_uci(np.uint16(encoded))
        self._last_search_move_encoded = encoded
        self._last_search_move_uci = uci
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise EngineError(f"{self.name} produced invalid internal move {uci!r}") from exc
        nodes_i = int(nodes)
        nps = int(nodes_i * 1000 / elapsed_ms)
        info = (
            f"d={int(depth)}, eval={int(score) / 100:+.2f}, n={nodes_i}, "
            f"t={elapsed_ms}ms, nps={nps}"
        )
        return move, info

    def terminate(self) -> None:
        return


class InProcessEngineFactory:
    """Import one engine namespace and allocate independent SearchContexts."""

    shared_jit = True
    requires_per_instance_warmup = False

    def __init__(
        self,
        name: str,
        namespace: str,
        *,
        tt_mb: int = 128,
        env_extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.name = name
        self.namespace = namespace.rstrip(".")
        self.tt_mb = int(tt_mb)
        self.env_extra = dict(env_extra or {})
        constants = importlib.import_module(f"{self.namespace}.constants")
        engine_types = importlib.import_module(f"{self.namespace}.engine_types")
        fen_parser = importlib.import_module(f"{self.namespace}.fen_parser")
        core = importlib.import_module(f"{self.namespace}.core")
        board_ops = importlib.import_module(f"{self.namespace}.board_operations")
        move = importlib.import_module(f"{self.namespace}.move")
        search = importlib.import_module(f"{self.namespace}.search")
        tt = importlib.import_module(f"{self.namespace}.transposition_table")
        self.max_ply = int(constants.MAX_PLY)
        self.constants = constants
        self.no_move = int(constants.NO_MOVE)
        self.SearchContext = engine_types.SearchContext
        self.parse_fen = fen_parser.parse_fen
        self.generate_legal_moves = core.generate_legal_moves
        self.make_move = board_ops.make_move
        self.move_to_uci = move.move_to_uci
        self.search = search.iterative_deepening_search
        self.create_tt = tt.create_transposition_table
        self.clear_tt = getattr(tt, "clear_transposition_table", None)

    def _make_context(self) -> object:
        max_ply = self.max_ply
        args = (
            self.create_tt(self.tt_mb),
            np.zeros(max_ply * 2, dtype=np.uint16),
            np.zeros((max_ply, max_ply), dtype=np.uint16),
            np.zeros((12, 64), dtype=np.int32),
            np.zeros((64, 64), dtype=np.int32),
            np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
            np.zeros((12, 64, 6), dtype=np.int32),
            np.full((8192, 12, 64), -1238, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros((12, 64, 12, 64), dtype=np.int16),
        )
        if not self.env_extra:
            context = self.SearchContext(*args)
            self._set_namespace_eval_weights(context)
            return context
        # SearchContext currently reads optional runtime tuning overrides from
        # os.environ. Context allocation is sequential, so temporarily applying
        # the side-specific values is safe and does not leak to worker threads.
        with _ENV_LOCK:
            previous = {key: os.environ.get(key) for key in self.env_extra}
            try:
                os.environ.update({str(key): str(value) for key, value in self.env_extra.items()})
                context = self.SearchContext(*args)
                self._set_namespace_eval_weights(context)
                return context
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def _set_namespace_eval_weights(self, context: object) -> None:
        """Keep current/old runtime HCE weights tied to their own constants."""
        try:
            weights = importlib.import_module("tune_eval_match.eval_weights")
            context.eval_weights[:] = weights.default_eval_weights(self.constants)
        except (AttributeError, ImportError):
            # Backends predating runtime eval weights remain usable.
            return

    def create(self, worker_id: int) -> InProcessEngine:
        return InProcessEngine(self, self._make_context())

    def prepare(self, engine: InProcessEngine, worker_id: int) -> None:
        """Compile/fault-in this namespace with the historical Kiwipete d14 search."""
        if worker_id != 0:
            return
        print(f"[warmup] {self.name}: Kiwipete depth=14...", flush=True)
        started = time.perf_counter()
        engine.new_game(KIWIPETE_FEN)
        engine.get_move(SearchLimit(depth=14), show_info=(worker_id == 0))
        print(
            f"[warmup] {self.name}: Kiwipete complete in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

@dataclass
class EnginePair:
    first: object
    second: object

    def terminate(self) -> None:
        self.first.terminate()
        self.second.terminate()


def create_engine_pairs(factory1: object, factory2: object, concurrency: int) -> list[EnginePair]:
    """Create workers and run both Kiwipete-d14 and startpos-d1 warm-ups.

    Shared in-process namespaces need one complete warm-up game.  A Python UCI
    process has its own Numba runtime, so every worker containing one gets its
    own warm-up game. Native external engines do not force extra games.
    """
    pairs: list[EnginePair] = []
    try:
        for worker_id in range(max(1, int(concurrency))):
            print(f"[pool] allocate worker {worker_id + 1}/{max(1, int(concurrency))}", flush=True)
            first = factory1.create(worker_id)
            second = factory2.create(worker_id)
            first.start()
            second.start()
            factory1.prepare(first, worker_id)
            factory2.prepare(second, worker_id)
            pair = EnginePair(first, second)
            pairs.append(pair)
            needs_warmup = (
                worker_id == 0
                or bool(getattr(factory1, "requires_per_instance_warmup", False))
                or bool(getattr(factory2, "requires_per_instance_warmup", False))
            )
            if needs_warmup:
                run_depth1_warmup_game(pair, worker_id=worker_id)
        return pairs
    except Exception:
        for pair in pairs:
            pair.terminate()
        for engine in (locals().get("first"), locals().get("second")):
            if engine is not None:
                try:
                    engine.terminate()
                except Exception:
                    pass
        raise


def close_engine_pairs(pairs: Sequence[EnginePair]) -> None:
    for pair in pairs:
        pair.terminate()


@dataclass
class GameResult:
    result: str
    pgn: chess.pgn.Game
    log: str


def _forfeit_result(side_to_move: chess.Color) -> str:
    return "0-1" if side_to_move == chess.WHITE else "1-0"


def _failure_header(engine: object, exc: BaseException) -> str:
    """Return a bounded, PGN-safe diagnostic for an engine failure."""
    detail = " ".join(f"{type(exc).__name__}: {exc}".split())
    return f"{getattr(engine, 'name', 'engine')}: {detail}"[:240]


def play_game(
    white_engine: object,
    black_engine: object,
    white_limit: SearchLimit,
    black_limit: SearchLimit,
    game_number: int,
    *,
    start_fen: Optional[str] = None,
    event: str = "Engine Match",
    max_plies: int = 400,
) -> GameResult:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    game = chess.pgn.Game()
    if start_fen:
        game.setup(board)
    game.headers.update({
        "Event": event,
        "Site": "Local",
        "Date": datetime.now().strftime("%Y.%m.%d"),
        "Round": str(game_number),
        "White": white_engine.name,
        "Black": black_engine.name,
    })
    logs = [f"Game {game_number}: {white_engine.name} (White) vs {black_engine.name} (Black)"]
    try:
        white_engine.new_game(start_fen)
        black_engine.new_game(start_fen)
    except Exception as exc:
        raise EngineError(f"Could not initialize game {game_number}: {exc}") from exc

    node = game
    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        mover = white_engine if board.turn == chess.WHITE else black_engine
        limit = white_limit if board.turn == chess.WHITE else black_limit
        try:
            move, info = mover.get_move(limit)
        except Exception as exc:
            result = _forfeit_result(board.turn)
            logs.append(f"{mover.name} forfeits: {exc}")
            game.headers["Termination"] = "engine failure"
            game.headers["Failure"] = _failure_header(mover, exc)
            game.headers["FailurePly"] = (
                f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
            )
            game.headers["Result"] = result
            return GameResult(result, game, "\n".join(logs) + "\n")
        if move not in board.legal_moves:
            result = _forfeit_result(board.turn)
            logs.append(f"{mover.name} forfeits: illegal move {move.uci()}")
            game.headers["Termination"] = "illegal move"
            game.headers["Result"] = result
            return GameResult(result, game, "\n".join(logs) + "\n")

        move_uci = move.uci()
        board.push(move)
        node = node.add_variation(move)
        if info:
            node.comment = info
        for engine, loses_as_white in ((white_engine, True), (black_engine, False)):
            try:
                engine.push(move_uci)
            except Exception as exc:
                result = "0-1" if loses_as_white else "1-0"
                logs.append(f"{engine.name} forfeits while synchronizing {move_uci}: {exc}")
                game.headers["Termination"] = "engine state synchronization failure"
                game.headers["Failure"] = _failure_header(engine, exc)
                game.headers["FailurePly"] = move_uci
                game.headers["Result"] = result
                return GameResult(result, game, "\n".join(logs) + "\n")
    else:
        result = "1/2-1/2"
        game.headers["Termination"] = f"adjudicated draw at {max_plies} plies"
        game.headers["Result"] = result
        logs.append(f"Result: {result} (maximum plies)")
        return GameResult(result, game, "\n".join(logs) + "\n")

    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    logs.append(f"Result: {result}")
    return GameResult(result, game, "\n".join(logs) + "\n")


def run_depth1_warmup_game(engines: EnginePair, *, worker_id: int = 0) -> GameResult:
    """Play and discard one complete start-position game at depth 1."""
    limit = SearchLimit(depth=1)
    print(
        f"[warmup] worker {worker_id + 1}: startpos, depth=1, "
        f"{engines.first.name} vs {engines.second.name}...",
        flush=True,
    )
    started = time.perf_counter()
    result = play_game(
        engines.first,
        engines.second,
        limit,
        limit,
        0,
        start_fen=None,
        event="Tournament Warm-up (discarded)",
        max_plies=400,
    )
    termination = result.pgn.headers.get("Termination", "")
    failure_markers = (
        "engine failure",
        "illegal move",
        "engine state synchronization failure",
    )
    if termination in failure_markers:
        detail = result.log.strip().splitlines()[-1] if result.log.strip() else termination
        raise EngineError(f"Depth-1 warm-up failed: {detail}")
    plies = sum(1 for _ in result.pgn.mainline_moves())
    print(
        f"[warmup] worker {worker_id + 1}: complete, result={result.result}, "
        f"plies={plies}, elapsed={time.perf_counter() - started:.1f}s (discarded)",
        flush=True,
    )
    return result


@dataclass
class PairResult:
    pair_number: int
    game_a: GameResult
    game_b: GameResult


def play_pair(
    engines: EnginePair,
    pair_number: int,
    opening_fen: Optional[str],
    first_limit: SearchLimit,
    second_limit: SearchLimit,
    *,
    event: str,
    max_plies: int,
) -> PairResult:
    game_a = play_game(
        engines.first, engines.second, first_limit, second_limit,
        2 * pair_number - 1, start_fen=opening_fen, event=event, max_plies=max_plies,
    )
    game_b = play_game(
        engines.second, engines.first, second_limit, first_limit,
        2 * pair_number, start_fen=opening_fen, event=event, max_plies=max_plies,
    )
    return PairResult(pair_number, game_a, game_b)


def score_for_first(result: str, first_is_white: bool) -> float:
    if result == "1-0":
        return 1.0 if first_is_white else 0.0
    if result == "0-1":
        return 0.0 if first_is_white else 1.0
    return 0.5


class SPRTTest:
    def __init__(self, elo0: float = 0.0, elo1: float = 10.0, alpha: float = 0.05, beta: float = 0.05) -> None:
        if elo1 <= elo0:
            raise ValueError("SPRT elo1 must be greater than elo0")
        self.elo0 = float(elo0)
        self.elo1 = float(elo1)
        self.a = math.log((1.0 - beta) / alpha)
        self.b = math.log(beta / (1.0 - alpha))
        self.pair_scores: list[float] = []

    def add_pair_result(self, score: float) -> None:
        self.pair_scores.append(float(score))

    def check_status(self) -> tuple[str, float]:
        n = len(self.pair_scores)
        if n < 5:
            return "Continue", 0.0
        mean = sum(self.pair_scores) / n
        variance = sum(value * value for value in self.pair_scores) / n - mean * mean
        if variance <= 1e-10:
            return "Continue", 0.0
        mu0 = 2.0 / (1.0 + 10.0 ** (-self.elo0 / 400.0))
        mu1 = 2.0 / (1.0 + 10.0 ** (-self.elo1 / 400.0))
        llr = (mu1 - mu0) / variance * (
            sum(self.pair_scores) - n * (mu0 + mu1) / 2.0
        )
        if llr >= self.a:
            return "H1 Accepted (Engine improved)", llr
        if llr <= self.b:
            return "H0 Accepted (No improvement)", llr
        return "Continue", llr


def calculate_elo_statistics(wins: int, draws: int, losses: int) -> tuple[float, float]:
    games = wins + draws + losses
    if games <= 0:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / games
    if score <= 0.0:
        return -math.inf, 0.0
    if score >= 1.0:
        return math.inf, 0.0
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    variance = max(0.0, (wins + 0.25 * draws) / games - score * score)
    se = math.sqrt(variance) / math.sqrt(games)
    gradient = 400.0 / (math.log(10.0) * score * (1.0 - score))
    return elo, 1.96 * se * gradient


@dataclass
class StageResult:
    games: int
    wins: int
    draws: int
    losses: int
    points: float
    status: str
    llr: float
    pgn_path: str


def run_stage(
    engine_pairs: Sequence[EnginePair],
    openings: Sequence[Optional[str]],
    *,
    games: int,
    first_limit: SearchLimit,
    second_limit: SearchLimit,
    first_name: str,
    second_name: str,
    pgn_path: str,
    event: str,
    use_sprt: bool = True,
    sprt_elo0: float = 0.0,
    sprt_elo1: float = 10.0,
    early_trash: bool = False,
    early_trash_min_games: int = 80,
    early_trash_max_score_pct: float = 40.0,
    max_plies: int = 400,
) -> StageResult:
    if games < 2 or games % 2:
        raise ValueError("Games must be an even integer >= 2")
    if not engine_pairs:
        raise ValueError("No engine workers were created")
    openings = list(openings) or [None]
    pgn = Path(pgn_path).resolve()
    pgn.parent.mkdir(parents=True, exist_ok=True)
    pgn.write_text("", encoding="utf-8")

    workers: queue.Queue[EnginePair] = queue.Queue()
    active = list(engine_pairs[: min(len(engine_pairs), games // 2)])
    for pair in active:
        workers.put(pair)

    def task(pair_number: int) -> PairResult:
        worker = workers.get()
        try:
            opening = openings[(pair_number - 1) % len(openings)]
            return play_pair(
                worker, pair_number, opening, first_limit, second_limit,
                event=event, max_plies=max_plies,
            )
        finally:
            workers.put(worker)

    sprt = SPRTTest(sprt_elo0, sprt_elo1)
    wins = draws = losses = 0
    points = 0.0
    played = 0
    status = "Continue" if use_sprt else "Disabled (full games)"
    llr = 0.0
    pair_count = games // 2
    next_pair = 1
    executor = ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="match")
    pending = set()
    try:
        while next_pair <= pair_count and len(pending) < len(active):
            pending.add(executor.submit(task, next_pair))
            next_pair += 1
        stop = False
        with pgn.open("a", encoding="utf-8", newline="\n") as pgn_stream:
            while pending and not stop:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pair = future.result()  # Never hide worker failures.
                    a = score_for_first(pair.game_a.result, True)
                    b = score_for_first(pair.game_b.result, False)
                    pair_score = a + b
                    sprt.add_pair_result(pair_score)
                    points += pair_score
                    played += 2
                    for value in (a, b):
                        if value == 1.0:
                            wins += 1
                        elif value == 0.0:
                            losses += 1
                        else:
                            draws += 1
                    pgn_stream.write(str(pair.game_a.pgn) + "\n\n")
                    pgn_stream.write(str(pair.game_b.pgn) + "\n\n")
                    pgn_stream.flush()
                    print(pair.game_a.log + pair.game_b.log, end="", flush=True)

                    score_pct = 100.0 * points / played
                    elo, error = calculate_elo_statistics(wins, draws, losses)
                    if use_sprt:
                        status, llr = sprt.check_status()
                    print(
                        f"After {played // 2} pairs ({played} games): "
                        f"{first_name} {points:.1f} ({wins}W {draws}D {losses}L), "
                        f"score={score_pct:.1f}%, Elo={elo:+.1f} ±{error:.1f}",
                        flush=True,
                    )
                    if use_sprt:
                        print(
                            f"SPRT LLR={llr:.2f}, bounds=[{sprt.b:.2f}, {sprt.a:.2f}], {status}",
                            flush=True,
                        )
                    trash_hit = (
                        early_trash
                        and played >= early_trash_min_games
                        and score_pct <= early_trash_max_score_pct
                    )
                    if trash_hit:
                        status = (
                            f"Early trash (games>={early_trash_min_games}, "
                            f"score<={early_trash_max_score_pct:.1f}%)"
                        )
                    if trash_hit or (use_sprt and status != "Continue"):
                        stop = True
                        break
                    if next_pair <= pair_count:
                        pending.add(executor.submit(task, next_pair))
                        next_pair += 1
            if stop:
                for future in pending:
                    future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return StageResult(played, wins, draws, losses, points, status, llr, str(pgn))


def load_openings(
    *,
    epd_path: Optional[str] = None,
    pgn_path: Optional[str] = None,
    pgn_plies: int = 16,
    enabled: bool = True,
) -> list[Optional[str]]:
    if not enabled:
        return [None]
    positions: list[Optional[str]] = []
    seen: set[str] = set()
    if pgn_path:
        with open(pgn_path, "r", encoding="utf-8", errors="replace") as stream:
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= pgn_plies:
                        break
                    board.push(move)
                fen = board.fen()
                key = " ".join(fen.split()[:4])
                if key not in seen:
                    seen.add(key)
                    positions.append(fen)
    elif epd_path and os.path.exists(epd_path):
        with open(epd_path, "r", encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    fen = chess.Board(line).fen()
                except ValueError:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    fen = chess.Board(" ".join(parts[:4]) + " 0 1").fen()
                key = " ".join(fen.split()[:4])
                if key not in seen:
                    seen.add(key)
                    positions.append(fen)
    return positions or [None]


def clear_numba_cache(root_dir: str) -> None:
    """Explicit maintenance action; normal match runs preserve disk caches."""
    root = Path(root_dir).resolve()
    print(f"Clearing Numba __pycache__ directories under {root}...", flush=True)
    for cache_dir in root.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
