"""
In-process self-play for eval weight A/B.

Design goals:
  - One Numba JIT compile for the whole SPSA run
  - Long-lived SearchContexts (θ+ / θ−); mutate eval_weights in-place
  - Fixed-node searches; no subprocess / no UCI spawn per game
  - Optional ThreadPool concurrency (-c N): N independent ctx pairs,
    Numba releases GIL — still ONE process / ONE JIT (no ProcessPool)
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from chess_engine.classical.board_operations import make_move
from chess_engine.classical.constants import MAX_PLY, NO_MOVE
from chess_engine.classical.engine_types import SearchContext
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move_generator import (
    generate_legal_moves,
    has_sufficient_material,
    is_in_check,
)
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.transposition_table import create_transposition_table

from tune_eval_match.eval_weights import apply_param_dict

ROOT = Path(__file__).resolve().parents[1]


def make_search_context(tt_mb: int = 16) -> SearchContext:
    """Build a SearchContext with default eval_weights (same path as UCI)."""
    tt = create_transposition_table(tt_mb)
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((5, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 6), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16)
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)
    continuation_correction_history = np.zeros((12, 64, 12, 64), dtype=np.int16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    return SearchContext(
        tt, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history, capture_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black,
        continuation_correction_history,
    )


def set_eval_params(ctx: SearchContext, params: dict) -> None:
    """In-place update of live eval_weights array (visible to Numba)."""
    apply_param_dict(ctx.eval_weights, params)


def reset_ctx_tables(ctx: SearchContext) -> None:
    """Clear TT / history between games (keep eval_weights)."""
    tt = ctx.transposition_table
    tt["key"] = np.uint32(0)
    tt["flag"] = np.uint8(0)
    tt["depth"] = np.uint8(0)
    tt["score"] = np.int16(0)
    tt["static_eval"] = np.int16(32767)
    ctx.history_table[:] = 0
    ctx.butterfly_history[:] = 0
    ctx.capture_history[:] = 0
    ctx.continuation_history[:] = 0
    ctx.low_ply_history[:] = 0
    ctx.pawn_history[:] = -1238
    ctx.pawn_correction_history[:] = 0
    ctx.minor_correction_history[:] = 0
    ctx.non_pawn_correction_history_white[:] = 0
    ctx.non_pawn_correction_history_black[:] = 0
    ctx.continuation_correction_history[:] = 0
    ctx.killer_moves[:] = 0
    ctx.counter_moves[:] = 0
    ctx.game_history_count = 0
    ctx.tt_generation = 0
    ctx.nodes_searched = np.uint64(0)
    ctx.nodes_searched_array[0] = 0
    ctx.stop_flag[0] = False
    ctx.diag_stats[:] = 0
    # Clear pawn structure TT
    ctx.pawn_table_keys[:] = np.uint64(0xFFFFFFFFFFFFFFFF)


def load_openings(path: Optional[Path], limit: int) -> List[str]:
    if path is None or not path.exists():
        # Prefer openings EPD, then puzzles
        for candidate in (
            ROOT / "data" / "hist_diag_openings_500.epd",
            ROOT / "data" / "openings.epd",
            ROOT / "data" / "hist_diag_puzzles_500.epd",
        ):
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        ][:limit]

    fens: List[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # EPD/FEN: first 4–6 tokens
        parts = line.split()
        if len(parts) >= 4:
            fen = " ".join(parts[:4])
            if len(parts) >= 6 and parts[4].isdigit():
                fen = " ".join(parts[:6])
            fens.append(fen)
        if len(fens) >= limit:
            break
    return fens or ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]


def _count_repetitions(history: Sequence[int], key: int) -> int:
    n = 0
    for h in history:
        if h == key:
            n += 1
    return n


def _terminal_score_white(
    piece_bbs, occupancy_bbs, game_state, pos_history: list
) -> Optional[float]:
    """
    Return white-perspective game score if terminal else None.
    1.0 white win, 0.0 black win, 0.5 draw.
    """
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    if len(moves) == 0:
        if is_in_check(piece_bbs, occupancy_bbs, game_state):
            # side to move is checkmated
            stm = int(game_state[0])
            return 0.0 if stm == 0 else 1.0
        return 0.5

    halfmove = int(game_state[3])
    if halfmove >= 100:
        return 0.5
    if not has_sufficient_material(piece_bbs):
        return 0.5
    key = int(game_state[4])
    # 3-fold: current position already appears twice in history (third now)
    if _count_repetitions(pos_history, key) >= 2:
        return 0.5
    return None


def search_best_move(
    piece_bbs,
    occupancy_bbs,
    game_state,
    ctx: SearchContext,
    nodes: int,
    game_history_list: Optional[list] = None,
    tt_generation: int = 0,
) -> int:
    result = iterative_deepening_search(
        piece_bbs,
        occupancy_bbs,
        game_state,
        64,
        {"nodes_limit": int(nodes), "maximum_time": 0, "optimum_time": 0},
        ctx,
        game_history_list=game_history_list,
        tt_generation=tt_generation,
        verbose=False,
    )
    # Return: (best_move, score, nodes, q_nodes, tt_hits, depth)
    best = NO_MOVE
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        best = int(result[0])
    if best == 0 or best == int(NO_MOVE):
        # Fallback: first legal move
        moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
        if len(moves) > 0:
            best = int(moves[0])
    return int(best)


def search_best_move_scored(
    piece_bbs,
    occupancy_bbs,
    game_state,
    ctx: SearchContext,
    nodes: int,
    game_history_list: Optional[list] = None,
    tt_generation: int = 0,
) -> Tuple[int, int]:
    """Like :func:`search_best_move` but also returns the root score (stm POV, cp)."""
    result = iterative_deepening_search(
        piece_bbs,
        occupancy_bbs,
        game_state,
        64,
        {"nodes_limit": int(nodes), "maximum_time": 0, "optimum_time": 0},
        ctx,
        game_history_list=game_history_list,
        tt_generation=tt_generation,
        verbose=False,
    )
    best = NO_MOVE
    score = 0
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        best = int(result[0])
        score = int(result[1])
    if best == 0 or best == int(NO_MOVE):
        moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
        if len(moves) > 0:
            best = int(moves[0])
    return int(best), score


from tune_search.stats import DEFAULT_ADJUDICATION, _Adjudicator


def play_game(
    fen: str,
    ctx_white: SearchContext,
    ctx_black: SearchContext,
    nodes: int,
    max_plies: int = 400,
    adjudicate: Optional[dict] = None,
) -> float:
    """
    Play one game. Returns score for WHITE (1/0.5/0).
    ctx_white / ctx_black already have desired eval_weights.

    ``adjudicate``: ``None`` (default) plays to the end; a dict (possibly empty,
    see :data:`DEFAULT_ADJUDICATION`) enables score-based win/draw adjudication.
    """
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    # Fresh tables each game (weights preserved)
    reset_ctx_tables(ctx_white)
    reset_ctx_tables(ctx_black)

    pos_history: list = []  # zobrist keys before each move (for 3-fold)
    game_zobrist_path: list = [int(game_state[4])]
    tt_gen_w = 0
    tt_gen_b = 0
    adjudicator = _Adjudicator(adjudicate) if adjudicate is not None else None

    for ply in range(max_plies):
        term = _terminal_score_white(piece_bbs, occupancy_bbs, game_state, pos_history)
        if term is not None:
            return term

        stm = int(game_state[0])
        ctx = ctx_white if stm == 0 else ctx_black
        # Pass full path for repetition awareness inside search
        if adjudicator is None:
            best = search_best_move(
                piece_bbs,
                occupancy_bbs,
                game_state,
                ctx,
                nodes=nodes,
                game_history_list=game_zobrist_path,
                tt_generation=tt_gen_w if stm == 0 else tt_gen_b,
            )
        else:
            best, stm_score = search_best_move_scored(
                piece_bbs,
                occupancy_bbs,
                game_state,
                ctx,
                nodes=nodes,
                game_history_list=game_zobrist_path,
                tt_generation=tt_gen_w if stm == 0 else tt_gen_b,
            )
            verdict = adjudicator.update(ply, stm_score if stm == 0 else -stm_score)
            if verdict is not None:
                return verdict
        if best == 0 or best == int(NO_MOVE):
            # No move → treat as terminal again
            term = _terminal_score_white(piece_bbs, occupancy_bbs, game_state, pos_history)
            return 0.5 if term is None else term

        pos_history.append(int(game_state[4]))
        make_move(piece_bbs, occupancy_bbs, game_state, np.uint16(best))
        game_zobrist_path.append(int(game_state[4]))
        if stm == 0:
            tt_gen_w = (tt_gen_w + 1) & 0xFF
        else:
            tt_gen_b = (tt_gen_b + 1) & 0xFF

    return 0.5  # long game adjudicate draw


def play_match_pair(
    fen: str,
    ctx_a: SearchContext,
    ctx_b: SearchContext,
    nodes: int,
    adjudicate: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Two games (color swap). Returns points for A and B over the pair (each 0..2).
    """
    # Game 1: A white, B black
    s_w = play_game(fen, ctx_a, ctx_b, nodes=nodes, adjudicate=adjudicate)
    # Game 2: B white, A black
    s_w2 = play_game(fen, ctx_b, ctx_a, nodes=nodes, adjudicate=adjudicate)
    pts_a = s_w + (1.0 - s_w2)
    pts_b = (1.0 - s_w) + s_w2
    return pts_a, pts_b


def make_worker_pool(concurrency: int, tt_mb: int = 16) -> List[Tuple[SearchContext, SearchContext]]:
    """
    Build `concurrency` independent (A,B) SearchContext pairs.
    Call once after first JIT warmup so workers only pay allocation cost.
    """
    n = max(1, int(concurrency))
    pool: List[Tuple[SearchContext, SearchContext]] = []
    for i in range(n):
        print(f"[pool] allocate worker pair {i + 1}/{n} (tt={tt_mb}MB)...", flush=True)
        pool.append((make_search_context(tt_mb=tt_mb), make_search_context(tt_mb=tt_mb)))
    return pool


def expand_worker_pool(
    pool: List[Tuple[SearchContext, SearchContext]],
    new_concurrency: int,
    tt_mb: int = 4,
    warmup_nodes: int = 800,
) -> List[Tuple[SearchContext, SearchContext]]:
    """
    Grow pool to new_concurrency without re-JIT (same process).
    Extra pairs get a light touch search to fault-in pages.
    """
    n = max(1, int(new_concurrency))
    while len(pool) < n:
        i = len(pool)
        print(f"[pool] expand pair {i + 1}/{n} (tt={tt_mb}MB)...", flush=True)
        wa, wb = make_search_context(tt_mb=tt_mb), make_search_context(tt_mb=tt_mb)
        warmup_jit(wa, nodes=warmup_nodes)
        warmup_jit(wb, nodes=warmup_nodes)
        pool.append((wa, wb))
    return pool


def run_match(
    params_a: dict,
    params_b: dict,
    openings: Sequence[str],
    nodes: int,
    games: int,
    log_every: int = 10,
    worker_pool: Optional[List[Tuple[SearchContext, SearchContext]]] = None,
    ctx_a: Optional[SearchContext] = None,
    ctx_b: Optional[SearchContext] = None,
    concurrency: int = 1,
    early_trash_min_games: int = 0,
    early_trash_max_score: float = 0.45,
    param_setter=None,
    opening_offset: int = 0,
    adjudicate: Optional[dict] = None,
) -> dict:
    """
    Play `games` games (paired color swap). Score = fraction of points for A.

    Concurrency (-c N):
      ThreadPool of N independent ctx pairs. Numba nopython releases GIL →
      true parallel search in ONE process (no re-JIT). Do NOT use ProcessPool.

    Early trash (optional, for L2):
      If early_trash_min_games > 0 and after that many games score_A < early_trash_max_score,
      stop the match and return early_trash=True (A is hopeless vs B).

    opening_offset:
      Pair ``i`` plays ``openings[(opening_offset + i) % len(openings)]``.  SPSA
      callers must advance this every batch; with the default 0 every batch
      replays the same first ``games // 2`` openings (a systematic overfit).

    The result also carries ``penta``: counts of pair outcomes for A in
      points (0, 0.5, 1, 1.5, 2) -- the pentanomial statistic used for SPRT/Elo.
    """
    if worker_pool is None:
        if ctx_a is None or ctx_b is None:
            raise ValueError("run_match needs worker_pool or (ctx_a, ctx_b)")
        worker_pool = [(ctx_a, ctx_b)]
    concurrency = max(1, min(int(concurrency), len(worker_pool)))

    # Push current θ onto every worker (in-place array mutate; live under Numba).
    # Search-parameter tuning reuses this match engine with a different setter
    # that writes SearchContext.tune[] instead of eval_weights.
    if param_setter is None:
        param_setter = set_eval_params
    for wa, wb in worker_pool[:concurrency]:
        param_setter(wa, params_a)
        param_setter(wb, params_b)

    n_pairs = max(1, games // 2)
    penta = [0, 0, 0, 0, 0]
    opening_offset = int(opening_offset)
    pts_a = 0.0
    pts_b = 0.0
    played = 0
    t0 = time.time()
    print_lock = threading.Lock()
    early_trash = False
    stop_submit = threading.Event()

    # Free-list of worker pairs (thread-safe borrow)
    free_q: queue.Queue = queue.Queue()
    for pair in worker_pool[:concurrency]:
        free_q.put(pair)

    def _one_pair(pair_idx: int) -> Tuple[float, float]:
        if stop_submit.is_set():
            return 0.0, 0.0
        fen = openings[(opening_offset + pair_idx) % len(openings)]
        wa, wb = free_q.get()
        try:
            if stop_submit.is_set():
                return 0.0, 0.0
            return play_match_pair(fen, wa, wb, nodes=nodes, adjudicate=adjudicate)
        finally:
            free_q.put((wa, wb))

    def _score() -> float:
        total = pts_a + pts_b
        return pts_a / total if total > 0 else 0.5

    def _log_progress(tag: str = "") -> None:
        if not log_every and not tag:
            return
        score = _score()
        elapsed = time.time() - t0
        extra = f"  {tag}" if tag else ""
        print(
            f"  [match] {played}/{n_pairs * 2} games  "
            f"A={pts_a:.1f} B={pts_b:.1f} score_A={score:.3f}  "
            f"{elapsed:.1f}s  c={concurrency}{extra}",
            flush=True,
        )

    def _check_early_trash() -> bool:
        """Return True if A should be trashed (hopeless)."""
        nonlocal early_trash
        if early_trash_min_games <= 0:
            return False
        if played < early_trash_min_games:
            return False
        if _score() < early_trash_max_score:
            early_trash = True
            stop_submit.set()
            _log_progress(
                tag=f"EARLY_TRASH score<{early_trash_max_score} @>={early_trash_min_games}g"
            )
            return True
        return False

    if concurrency <= 1:
        for i in range(n_pairs):
            a, b = _one_pair(i)
            pts_a += a
            pts_b += b
            penta[min(4, max(0, int(round(a * 2))))] += 1
            played += 2
            if log_every and (i + 1) % max(1, log_every // 2) == 0:
                _log_progress()
            if _check_early_trash():
                break
    else:
        # Parallel over opening pairs; cancel remaining work on early trash.
        done_pairs = 0
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_one_pair, i): i for i in range(n_pairs)}
            for fut in as_completed(futs):
                if stop_submit.is_set() and fut.cancelled():
                    continue
                try:
                    a, b = fut.result()
                except Exception:
                    continue
                # Ignore zero-point phantom results from cancelled late workers
                with print_lock:
                    if early_trash and a == 0.0 and b == 0.0:
                        continue
                    pts_a += a
                    pts_b += b
                    penta[min(4, max(0, int(round(a * 2))))] += 1
                    played += 2
                    done_pairs += 1
                    if log_every and done_pairs % max(1, log_every // 2) == 0:
                        _log_progress()
                    if _check_early_trash():
                        for other in futs:
                            other.cancel()
                        # Best-effort: don't wait forever on in-flight pair games
                        break

    total = pts_a + pts_b
    score_a = pts_a / total if total > 0 else 0.5
    return {
        "score_a": score_a,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "penta": penta,
        "opening_offset": opening_offset,
        "games": played,
        "elapsed_s": time.time() - t0,
        "concurrency": concurrency,
        "early_trash": early_trash,
        "early_trash_min_games": early_trash_min_games,
        "early_trash_max_score": early_trash_max_score,
    }


def warmup_jit(ctx: SearchContext, nodes: int = 3000) -> None:
    """Single shallow search to force Numba compile once."""
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    print(f"[warmup] JIT nodes={nodes} (may take several minutes first time)...", flush=True)
    t0 = time.time()
    iterative_deepening_search(
        piece_bbs, occupancy_bbs, game_state, 64,
        {"nodes_limit": nodes, "maximum_time": 0, "optimum_time": 0},
        ctx, verbose=False,
    )
    print(f"[warmup] done in {time.time() - t0:.1f}s", flush=True)


def verify_live_weights(ctx: SearchContext) -> bool:
    """
    Sanity: mutating eval_weights must change eval score (StructRef live read).
    """
    from chess_engine.classical.evaluation import _evaluate_position_jit

    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

    ew = ctx.eval_weights
    old = int(ew[0])  # TEMPO
    ew[0] = np.int32(30)
    s1, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, ctx)
    ew[0] = np.int32(120)
    s2, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, ctx)
    ew[0] = np.int32(old)
    ok = int(s1) != int(s2)
    print(f"[verify] tempo 30→score={int(s1)}  120→score={int(s2)}  live={ok}", flush=True)
    return ok
