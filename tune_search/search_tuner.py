
import argparse
import json
import subprocess
import os
import sys
import random
import time
import math
import importlib
import threading
import concurrent.futures
import traceback
from pathlib import Path
from tune_search.param_config import get_default_params
from tune_search.search_param_registry import (
    DEFAULT_PROFILE,
    PROFILE_NAMES,
    get_spec,
    profile_names,
    quantize_integer_value,
    quantize_value,
    resolve_names,
)

# Engine command template
ENGINE_CMD_TEMPLATE = "python tune_search/engine_adapter.py"
chess = None
ROOT = Path(__file__).resolve().parents[1]
# v5 follows the classic Fishtest SPSA schedule.  Older checkpoints
# intentionally remain incompatible: their one-step/capped updates do not
# carry the canonical pair-count iteration and c/a/r schedules.
SPSA_UPDATE_MODE = "classic_fishtest_v5"


def _ensure_chess():
    """Load python-chess only when a match is actually requested.

    Registry inspection (``--list-params``) should not need the optional UCI
    frontend dependency, and this also keeps light registry tests independent
    from the match runner.
    """
    global chess
    if chess is None:
        chess = importlib.import_module("chess")
        importlib.import_module("chess.engine")
        importlib.import_module("chess.pgn")
    return chess

def get_engine_cmd_list(params):
    """
    Constructs the command line list for the engine with specific parameters.
    params: dict of {PARAM_NAME: value}
    """
    cmd = [sys.executable, "-u", "tune_search/engine_adapter.py"]
    for k, v in params.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    return cmd


def _find_uci_opening_book():
    """Find the legacy PGN/polyglot source used by the UCI backend."""
    book_path = "chess_engine/polyglot.bin"
    if os.path.exists(book_path):
        return book_path
    for candidate in (
        "polyglot.bin",
        "tuner/data/pgn/twic1613.pgn",
        "twic1613.pgn",
        "openings.pgn",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _build_inprocess_runner(args, param_names):
    """Construct one persistent fixed-node worker pool for runtime axes."""
    from tune_search.inprocess_match import (
        load_openings,
        make_worker_pool,
        run_match,
        warmup_jit,
    )

    compile_time = [name for name in param_names if not get_spec(name).is_runtime]
    if compile_time:
        raise ValueError(
            "inprocess backend only accepts runtime parameters; compile-time: "
            + ", ".join(compile_time)
        )

    if args.openings:
        openings_path = Path(args.openings)
    else:
        openings_path = ROOT / "data" / "hist_diag_openings_500.epd"
    openings = load_openings(openings_path, limit=max(500, args.games * 2))
    print(
        f"[inprocess] openings={len(openings)} nodes={args.nodes} "
        f"concurrency={args.concurrency} tt={args.tt_mb}MB",
        flush=True,
    )

    worker_pool = make_worker_pool(args.concurrency, tt_mb=args.tt_mb)
    warmup_nodes = min(5000, max(1000, int(args.nodes)))
    warmup_jit(worker_pool[0][0], nodes=warmup_nodes)
    warmup_jit(worker_pool[0][1], nodes=min(1500, warmup_nodes))
    for i, (ctx_a, ctx_b) in enumerate(worker_pool[1:], start=1):
        print(f"[inprocess] touch worker {i + 1}/{len(worker_pool)}", flush=True)
        warmup_jit(ctx_a, nodes=min(800, warmup_nodes))
        warmup_jit(ctx_b, nodes=min(800, warmup_nodes))

    def _match(params_base, params_test):
        result = run_match(
            params_base,
            params_test,
            openings=openings,
            nodes=int(args.nodes),
            games=int(args.games),
            log_every=max(1, int(args.log_every)),
            worker_pool=worker_pool,
            concurrency=int(args.concurrency),
            scale_mode=args.scale_mode,
        )
        # ``SPSAOptimizer.step`` passes (theta_minus, theta_plus) and expects
        # the score of the second argument (plus).  The generic in-process
        # matcher reports the first argument (A), so invert it here rather
        # than silently reversing every update direction.  Keep the match
        # metadata for the JSONL audit log.
        return {
            "score": 1.0 - float(result["score_a"]),
            "match": result,
        }

    return _match


def _save_state(path: Path, optimizer, *, backend, profile, param_names, args, log_path=None):
    """Atomically checkpoint SPSA state so a long campaign can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "iteration": int(optimizer.iteration),
        "theta": {name: float(value) for name, value in optimizer.theta.items()},
        "current_params": dict(optimizer.current_params),
        "backend": backend,
        "profile": profile,
        "params": list(param_names),
        "games": int(args.games),
        "nodes": int(args.nodes),
        "concurrency": int(args.concurrency),
        "log": str(log_path) if log_path is not None else None,
        "match_games": int(optimizer.match_games),
        "spsa_update_mode": SPSA_UPDATE_MODE,
        "spsa_schedule": "fishtest_classic",
        "spsa_iteration": int(optimizer.spsa_iteration),
        "schedule_total_batches": int(optimizer.total_batches),
        "match_pairs": int(optimizer.match_pairs),
        "total_pairs": int(optimizer.total_pairs),
        "A_ratio": float(optimizer.A_ratio),
        "A": float(optimizer.A),
        "alpha": float(optimizer.alpha),
        "gamma": float(optimizer.gamma),
        "scale_mode": optimizer.scale_mode,
        # ``random`` drives both SPSA Bernoulli signs and opening selection.
        # Saving it prevents a resumed campaign from silently replaying a
        # different/randomly shifted sequence of perturbations.
        "random_state": _jsonable_random_state(random.getstate()),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one flushed JSON record so a long run is auditable live."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _default_log_path(state_path: Path | None, resume: bool) -> Path:
    """Reuse a checkpoint's log on resume, otherwise create a timestamped log."""
    if resume and state_path is not None and state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle).get("log")
            if saved:
                return Path(saved)
        except (OSError, ValueError, TypeError):
            # The normal state loader will report a malformed checkpoint later.
            pass
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("tune_search") / "logs" / f"search_spsa_{stamp}.jsonl"


def _jsonable_random_state(value):
    """Convert ``random.getstate()``'s nested tuples to JSON containers."""
    if isinstance(value, tuple):
        return [_jsonable_random_state(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_random_state(item) for item in value]
    return value


def _tuple_random_state(value):
    """Rebuild the tuple shape expected by ``random.setstate``."""
    if isinstance(value, list):
        return tuple(_tuple_random_state(item) for item in value)
    return value


def _load_state(
    path: Path,
    optimizer,
    param_names,
    *,
    expected_backend=None,
    expected_profile=None,
    expected_games=None,
    expected_nodes=None,
    expected_concurrency=None,
    expected_scale_mode=None,
    expected_schedule_total_batches=None,
    expected_a_ratio=None,
):
    """Restore a compatible state file; never silently mix campaigns."""
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if tuple(state.get("params", ())) != tuple(param_names):
        raise ValueError("checkpoint parameter list differs; choose a new --state path")
    if expected_backend is not None and state.get("spsa_update_mode") != SPSA_UPDATE_MODE:
        raise ValueError(
            "checkpoint uses an older SPSA update scale; choose a new --state path"
        )
    if expected_scale_mode is not None and state.get("scale_mode", "registry") != expected_scale_mode:
        raise ValueError(
            "checkpoint uses a different SPSA scale mode; choose a new --state path"
        )
    if state.get("spsa_schedule") != "fishtest_classic":
        raise ValueError(
            "checkpoint uses a non-canonical SPSA schedule; choose a new --state path"
        )
    if expected_schedule_total_batches is not None:
        saved_batches = int(state.get("schedule_total_batches", 0))
        if saved_batches != int(expected_schedule_total_batches):
            raise ValueError(
                "checkpoint uses a different SPSA schedule length; choose a new --state path"
            )
    if expected_a_ratio is not None:
        saved_ratio = float(state.get("A_ratio", float("nan")))
        if not math.isclose(saved_ratio, float(expected_a_ratio), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "checkpoint uses a different SPSA A ratio; choose a new --state path"
            )
    checks = (
        ("backend", expected_backend),
        ("profile", expected_profile),
        ("games", expected_games),
        ("nodes", expected_nodes),
        ("concurrency", expected_concurrency),
        ("match_games", expected_games),
    )
    for key, expected in checks:
        if expected is not None and key in state and state[key] != expected:
            raise ValueError(
                f"checkpoint {key}={state[key]!r} differs from requested {expected!r}; "
                "choose a new --state path"
            )
    for name in param_names:
        optimizer.theta[name] = float(state["theta"][name])
        optimizer.current_params[name] = int(state["current_params"][name])
    optimizer.iteration = int(state.get("iteration", 0))
    optimizer.spsa_iteration = int(state.get("spsa_iteration", 0))
    # Older checkpoints did not contain a random state; keep them resumable,
    # but new checkpoints resume the exact SPSA/opening stream.
    if "random_state" in state:
        random.setstate(_tuple_random_state(state["random_state"]))
    return state

def warmup_engine(engine, name="Engine"):
    """
    Runs a shallow search to trigger Numba JIT compilation.
    This prevents the first actual game move from timing out.
    """
    try:
        # Search Kiwipete position for depth 1 to trigger JIT compilation on more complex positions
        print(f"[{name}] Warming up with Kiwipete depth 1... (May take ~1 min)")
        kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        engine.play(chess.Board(kiwipete), chess.engine.Limit(depth=1))
        print(f"[{name}] Warmup complete.")
    except Exception as e:
        print(f"[{name}] Warmup failed: {e}")

def play_game_reused(engine1, engine2, time_limit, increment, opening_fen=None):
    """
    Plays a single game between two ALREADY RUNNING engines.
    engine1: White
    engine2: Black
    Returns score for Engine 1 (1.0 win, 0.5 draw, 0.0 loss).
    """
    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    
    # Reset engines for new game
    # python-chess engine.configure() fails for 'ucinewgame' because main.py doesn't report it as an option.
    # We send it directly as a command to ensure internal state (Hash, History) is cleared.
    try:
        if engine1.protocol: engine1.protocol.send_line("ucinewgame")
        if engine2.protocol: engine2.protocol.send_line("ucinewgame")
        # Small sleep to ensure engines process it, though UCI doesn't strictly require wait
        time.sleep(0.05)
    except Exception as e:
        print(f"Error sending ucinewgame command: {e}")

    # Parse time control
    # Note: 'time' in Limit is generally "time limit for this move" if used this way,
    # or base time if handling clock. python-chess SimpleEngine.play with Limit(time=X) 
    # usually treats X as movetime (seconds per move).
    # Increment is not a standard argument for fixed-movetime Limit constructor.
    # If the user provides "0.1+0.01", we treat it as movetime=0.1s and ignore increment for now
    # to avoid TypeError.
    limit = chess.engine.Limit(time=time_limit)
    
    # Game Loop
    try:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = engine1.play(board, limit)
            else:
                result = engine2.play(board, limit)
                
            if result.move is None:
                print(f"Engine ({'White' if board.turn == chess.WHITE else 'Black'}) returned no move. Claiming loss.")
                return 0.0 if board.turn == chess.WHITE else 1.0
                    
            board.push(result.move)
    except Exception as e:
        # Check if it's a timeout error (wrapped by concurrent.futures or asyncio)
        msg = str(e)
        if "TimeoutError" in msg or "timeout" in msg.lower():
            # If engine times out, it loses.
            print(f"Engine ({'White' if board.turn == chess.WHITE else 'Black'}) timed out (Loss)")
            if board.turn == chess.WHITE:
                return 0.0 # White timed out, Black wins (Score for Engine 1 is 0.0)
            else:
                return 1.0 # Black timed out, White wins (Score for Engine 1 is 1.0)
        
        print(f"Game error: {e}")
        traceback.print_exc()
        return 0.5

    outcome = board.outcome()
    if outcome.winner == chess.WHITE:
        return 1.0
    elif outcome.winner == chess.BLACK:
        return 0.0
    else:
        return 0.5

def run_match_python(base_params, test_params, num_games=10, tc="0.1+0.01", concurrency=1, opening_book="tuner/data/pgn/twic1613.pgn"):
    """
    Runs a match using python-chess, REUSING engine processes to avoid recompilation overhead.
    """
    _ensure_chess()
    
    # Parse TC
    if '+' in tc:
        parts = tc.split('+')
        base_time = float(parts[0])
        inc = float(parts[1])
    else:
        base_time = float(tc)
        inc = 0.0
        
    # Load Openings
    openings = []
    if opening_book and os.path.exists(opening_book):
        try:
            _, ext = os.path.splitext(opening_book)
            if ext == '.bin':
                import chess.polyglot as pg
                print(f"Generating openings from Polyglot book: {opening_book}")
                with pg.open_reader(opening_book) as reader:
                    for _ in range(num_games):
                        board = chess.Board()
                        for _ in range(8):
                            try:
                                entry = reader.weighted_choice(board)
                                board.push(entry.move)
                            except IndexError:
                                break
                        openings.append(board.fen())
            else:
                with open(opening_book) as f:
                    for _ in range(num_games * 2):
                        game = chess.pgn.read_game(f)
                        if game is None: break
                        board = game.board()
                        for i, move in enumerate(game.mainline_moves()):
                            if i >= 8: break
                            board.push(move)
                        openings.append(board.fen())
        except Exception as e:
            print(f"Error reading opening book: {e}")
    
    if not openings:
        openings = [chess.STARTING_FEN]

    # --- Start Engines ONCE ---
    # We create two engines: 'Base' and 'Test'.
    # Note: If concurrency > 1, we would need a pool of engines.
    # For now, let's assume concurrency=1 or create multiple pairs.
    
    # To support concurrency properly with reused engines, we need a list of engine pairs.
    # num_pairs = concurrency
    
    engine_pairs = []
    
    print(f"Starting {concurrency} engine pair(s)...")
    
    # Use ThreadPoolExecutor to start engines and warmup in parallel
    # This avoids sequential waiting for each engine to compile
    
    def start_engine_pair(index):
        try:
            # Base Engine
            cmd_base = get_engine_cmd_list(base_params)
            engine_base = chess.engine.SimpleEngine.popen_uci(cmd_base, timeout=120.0)
            
            # Test Engine
            cmd_test = get_engine_cmd_list(test_params)
            engine_test = chess.engine.SimpleEngine.popen_uci(cmd_test, timeout=120.0)
            
            # Warmup
            # Warmup both engines in this thread (still sequential per pair, but pairs are parallel)
            warmup_engine(engine_base, f"Base-{index}")
            warmup_engine(engine_test, f"Test-{index}")
            
            return (engine_base, engine_test)
        except Exception as e:
            print(f"Error starting engine pair {index}: {e}")
            return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(start_engine_pair, i) for i in range(concurrency)]
            
            for future in concurrent.futures.as_completed(futures):
                pair = future.result()
                if pair:
                    engine_pairs.append(pair)
                else:
                    raise Exception("Failed to start one or more engine pairs")
            
    except Exception as e:
        print(f"Error starting engines: {e}")
        for b, t in engine_pairs:
            b.quit()
            t.quit()
        return 0.5

    score_test = 0.0
    games_played = 0
    results = {"win": 0, "loss": 0, "draw": 0}
    
    # We distribute the 'pairs of games' (A vs B, B vs A) among the engine workers
    total_pairs = num_games // 2
    
    # Queue of work: each item is an opening FEN
    work_queue = [random.choice(openings) for _ in range(total_pairs)]
    
    lock = threading.Lock()
    
    def worker(pair_idx):
        """
        Worker function using a specific engine pair (engine_base, engine_test).
        Process games from the queue until empty.
        """
        my_base, my_test = engine_pairs[pair_idx]
        
        while True:
            try:
                # Pop work
                with lock:
                    if not work_queue:
                        return
                    opening = work_queue.pop()
            except IndexError:
                return

            # Play Game 1: Test (White) vs Base (Black)
            # Use 'my_test' as White, 'my_base' as Black
            s1 = play_game_reused(my_test, my_base, base_time, inc, opening)
            
            res_str1 = "Draw"
            if s1 == 1.0: res_str1 = "Test Won"
            elif s1 == 0.0: res_str1 = "Base Won"
            
            # Play Game 2: Base (White) vs Test (Black)
            # Use 'my_base' as White, 'my_test' as Black
            s2 = play_game_reused(my_base, my_test, base_time, inc, opening)
            
            # s2 is score for Base (White). Test score is 1.0 - s2
            test_score_g2 = 1.0 - s2
            
            res_str2 = "Draw"
            if s2 == 1.0: res_str2 = "Base Won"
            elif s2 == 0.0: res_str2 = "Test Won"

            pair_score = s1 + test_score_g2
            
            with lock:
                nonlocal score_test, games_played
                score_test += pair_score
                games_played += 2
                
                # Update stats
                for s in [s1, test_score_g2]:
                    if s == 1.0: results["win"] += 1
                    elif s == 0.0: results["loss"] += 1
                    else: results["draw"] += 1
                
                print(f"Game Pair Finished: {res_str1} | {res_str2} (Test Score: {score_test}/{games_played})")

    # Start threads
    threads = []
    # import threading # Removed as it's already imported at top
    for i in range(concurrency):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()

    # Cleanup Engines
    print("Closing engines...")
    for b, t in engine_pairs:
        try:
            b.quit()
        except Exception:
            b.close()
        try:
            t.quit()
        except Exception:
            t.close()

    final_score = score_test / games_played if games_played > 0 else 0.5
    print(f"Match Finished. Test Score: {final_score:.3f} ({results['win']} W - {results['loss']} L - {results['draw']} D)")
    return final_score

class SPSAOptimizer:
    """Classic Fishtest SPSA over a batched fixed-node match.

    A call to :meth:`step` evaluates one plus/minus match batch.  Fishtest
    counts the batch as ``games // 2`` paired games, so ``spsa_iteration`` is
    advanced by that many pairs while the human-facing ``iteration`` remains
    the number of completed batches.  ``c_end`` is the registry probe radius;
    ``canonical_r_end`` supplies the final ``a_end / c_end**2`` ratio.
    """

    def __init__(
        self,
        param_names,
        initial_params,
        *,
        total_batches=1,
        A_ratio=0.1,
        alpha=0.602,
        gamma=0.101,
        match_games=1,
        scale_mode="registry",
    ):
        if scale_mode not in ("registry", "integer"):
            raise ValueError(f"unknown SPSA scale mode: {scale_mode!r}")
        if float(A_ratio) < 0.0:
            raise ValueError("A_ratio must be non-negative")
        if float(alpha) <= 0.0 or float(gamma) <= 0.0:
            raise ValueError("alpha and gamma must be positive")
        self.param_names = tuple(param_names)
        self.specs = {name: get_spec(name) for name in self.param_names}
        self.scale_mode = str(scale_mode)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.match_games = max(2, int(match_games))
        self.match_pairs = max(1, self.match_games // 2)
        self.total_batches = max(1, int(total_batches))
        self.total_pairs = max(self.match_pairs, self.total_batches * self.match_pairs)
        self.A_ratio = float(A_ratio)
        self.A = self.A_ratio * float(self.total_pairs)
        self.theta = {
            name: float(self._quantize(name, initial_params[name]))
            for name in self.param_names
        }
        self.current_params = {
            name: int(self.theta[name]) for name in self.param_names
        }
        self.iteration = 0
        self.spsa_iteration = 0

    def _quantize(self, name, value):
        # Integer mode is only a boundary representation.  The SPSA theta
        # remains floating point so sub-unit evidence is not discarded.
        if self.scale_mode == "integer":
            return quantize_integer_value(name, value)
        return quantize_value(name, value)

    def _c_end(self, name):
        return float(self.specs[name].step)

    def _r_end(self, name):
        return float(self.specs[name].canonical_r_end)

    def _c_k(self, name, k):
        # Parse-compatible form: c_k = c_end * (N ** gamma) / k ** gamma,
        # where N is the planned total number of paired games.  Therefore the
        # last scheduled pair uses exactly the configured c_end.
        k = max(1.0, float(k))
        return self._c_end(name) * (self.total_pairs / k) ** self.gamma

    def _a_k(self, name, k):
        k = max(1.0, float(k))
        a_end = self._r_end(name) * self._c_end(name) ** 2
        a_base = a_end * (self.A + self.total_pairs) ** self.alpha
        return a_base / (self.A + k) ** self.alpha

    def _r_k(self, name, k):
        ck = self._c_k(name, k)
        return self._a_k(name, k) / (ck * ck)

    def step(self, match_runner_func):
        self.iteration += 1
        self.spsa_iteration += self.match_pairs
        k = self.spsa_iteration

        delta = {
            name: 1 if random.random() < 0.5 else -1
            for name in self.param_names
        }
        theta_plus = {}
        theta_minus = {}
        for name in self.param_names:
            perturbation = self._c_k(name, k) * delta[name]
            theta_plus[name] = self._quantize(name, self.theta[name] + perturbation)
            theta_minus[name] = self._quantize(name, self.theta[name] - perturbation)

        print(f"Iter {self.iteration} (SPSA pair {k}/{self.total_pairs}): Playing Match (Plus vs Minus)...")
        raw_match = match_runner_func(theta_minus, theta_plus)
        if isinstance(raw_match, dict):
            score_plus_vs_minus = float(
                raw_match.get("score", raw_match.get("score_plus", 0.5))
            )
            match_metadata = raw_match.get("match", {})
        else:
            score_plus_vs_minus = float(raw_match)
            match_metadata = {}
        result_score = max(0.0, min(1.0, score_plus_vs_minus))
        win_signal = result_score - 0.5
        # With a paired match score, 2 * games * (score - .5) is exactly
        # wins - losses (draws contribute zero), matching Fishtest workers.
        sample_signal = 2.0 * self.match_games * win_signal
        print(
            f"Iter {self.iteration}: Score(Plus vs Minus) = {result_score:.3f}. "
            f"Signal = {win_signal:.3f} (W-L={sample_signal:.3f})"
        )

        raw_updates = {}
        applied_updates = {}
        for name in self.param_names:
            ck = self._c_k(name, k)
            # Fishtest's worker applies ``R * c * (wins-losses) * flip``.
            # Since R=a/c² and Bernoulli ``flip`` is its own reciprocal, this
            # is exactly a/c * sample_signal * delta.  Do not divide by two:
            # the paired match signal above is already the W-L count used by
            # the canonical worker.
            raw_update = self._a_k(name, k) / ck * sample_signal * delta[name]
            self.theta[name] += raw_update
            self.theta[name] = float(
                max(self.specs[name].minimum,
                    min(self.specs[name].maximum, self.theta[name]))
            )
            self.current_params[name] = self._quantize(name, self.theta[name])
            raw_updates[name] = float(raw_update)
            applied_updates[name] = float(raw_update)

        print(f"Updated Params: {self.current_params}")
        return {
            "iteration": int(self.iteration),
            "spsa_iteration": int(k),
            "match_pairs": int(self.match_pairs),
            "total_pairs": int(self.total_pairs),
            "A_ratio": float(self.A_ratio),
            "A": float(self.A),
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "delta": {name: int(value) for name, value in delta.items()},
            "c_end": {name: float(self._c_end(name)) for name in self.param_names},
            "r_end": {name: float(self._r_end(name)) for name in self.param_names},
            "c_k": {name: float(self._c_k(name, k)) for name in self.param_names},
            "a_k": {name: float(self._a_k(name, k)) for name in self.param_names},
            "r_k": {name: float(self._r_k(name, k)) for name in self.param_names},
            "theta_minus": {name: int(value) for name, value in theta_minus.items()},
            "theta_plus": {name: int(value) for name, value in theta_plus.items()},
            "score_plus": float(result_score),
            "signal": float(win_signal),
            "sample_signal": float(sample_signal),
            "match_games": int(self.match_games),
            "scale_mode": self.scale_mode,
            "raw_updates": raw_updates,
            "applied_updates": applied_updates,
            "match": match_metadata,
            "updated_params": dict(self.current_params),
        }


def _print_registry(profile=None, names=None):
    """Print a compact registry view without starting any engine process."""
    selected = tuple(names) if names is not None else profile_names(profile or DEFAULT_PROFILE)
    label = "explicit" if names is not None else (profile or DEFAULT_PROFILE)
    print(f"profile={label} parameters={len(selected)}")
    for name in selected:
        spec = get_spec(name)
        runtime = f"slot={spec.runtime_index}" if spec.is_runtime else "compile-time"
        print(
            f"{name:32s} default={spec.default:6d} "
            f"range=[{spec.minimum:6d},{spec.maximum:6d}] step={spec.step:5d} "
            f"kind={spec.kind:10s} {runtime} group={spec.group}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10, help="Games per iteration")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency")
    parser.add_argument("--tc", type=str, default="0.1+0.01", help="Time control (e.g. 0.1+0.01)")
    parser.add_argument(
        "--nodes",
        type=int,
        default=30000,
        help="fixed nodes for inprocess backend (L1 default; use 200000 for L2)",
    )
    parser.add_argument("--tt-mb", type=int, default=16, help="TT size per inprocess context")
    parser.add_argument(
        "--backend",
        choices=("auto", "inprocess", "uci"),
        default="auto",
        help="auto=inprocess for runtime-only profiles, UCI for compile-time axes",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("integer", "registry"),
        default="integer",
        help=(
            "SPSA value representation: integer rounds only at the engine "
            "boundary; registry preserves the coarse registry grid"
        ),
    )
    parser.add_argument(
        "--schedule-iterations",
        type=int,
        default=None,
        help=(
            "planned SPSA batches used to set the Fishtest decay horizon; "
            "defaults to --iter on a fresh campaign and the saved value on resume"
        ),
    )
    parser.add_argument(
        "--spsa-a-ratio",
        type=float,
        default=0.1,
        help="Fishtest A ratio (A = ratio * total paired games; default: 0.1)",
    )
    parser.add_argument("--openings", type=str, default=None, help="EPD/FEN source for inprocess matches")
    parser.add_argument("--log-every", type=int, default=10, help="match progress interval")
    parser.add_argument("--iter", type=int, default=100, help="Iterations")
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=DEFAULT_PROFILE,
        help="staged parameter group; runtime (34 axes) is the safe default",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="comma-separated explicit parameter names; overrides --profile",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="print the selected registry entries and exit",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for SPSA deltas/openings")
    parser.add_argument(
        "--state",
        type=str,
        default="tune_search/search_spsa_state.json",
        help="checkpoint path (set empty to disable)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="JSONL iteration log (default: timestamped tune_search/logs/ path; resume reuses it)",
    )
    parser.add_argument("--resume", action="store_true", help="resume a compatible SPSA checkpoint")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        param_names = resolve_names(
            profile=args.profile,
            names=[args.params] if args.params else None,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_params:
        _print_registry(args.profile, names=param_names if args.params else None)
        return

    if not param_names:
        parser.error("the selected search profile contains no parameters")

    runtime_only = all(get_spec(name).is_runtime for name in param_names)
    backend = args.backend
    if backend == "auto":
        backend = "inprocess" if runtime_only else "uci"
    if backend == "inprocess" and not runtime_only:
        parser.error("--backend inprocess requires runtime-only parameters")

    state_path = Path(args.state) if args.state else None
    schedule_batches = args.schedule_iterations
    if args.resume and schedule_batches is None and state_path is not None:
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                saved_schedule = json.load(handle).get("schedule_total_batches")
            if saved_schedule is not None:
                schedule_batches = int(saved_schedule)
        except (OSError, ValueError, TypeError):
            # The full state loader below emits the actionable checkpoint
            # error.  Keep argument parsing focused on the common path.
            schedule_batches = None
    if schedule_batches is None:
        schedule_batches = int(args.iter)
    if int(schedule_batches) <= 0:
        parser.error("--schedule-iterations must be positive")
    if float(args.spsa_a_ratio) < 0.0:
        parser.error("--spsa-a-ratio must be non-negative")

    initial_params = get_default_params(names=param_names)
    optimizer = SPSAOptimizer(
        param_names,
        initial_params,
        total_batches=int(schedule_batches),
        A_ratio=float(args.spsa_a_ratio),
        match_games=int(args.games),
        scale_mode=args.scale_mode,
    )
    log_path = Path(args.log) if args.log else _default_log_path(state_path, args.resume)
    if args.resume:
        if state_path is None or not state_path.exists():
            parser.error("--resume requires an existing --state file")
        try:
            restored = _load_state(
                state_path,
                optimizer,
                param_names,
                expected_backend=backend,
                expected_profile=args.profile,
                expected_games=int(args.games),
                expected_nodes=int(args.nodes),
                expected_concurrency=int(args.concurrency),
                expected_scale_mode=args.scale_mode,
                expected_schedule_total_batches=int(schedule_batches),
                expected_a_ratio=float(args.spsa_a_ratio),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"cannot resume checkpoint: {exc}")
        print(f"[resume] iteration={optimizer.iteration} from {state_path} ({restored.get('backend', '?')})")

    # Compile-time candidates still use the legacy UCI process adapter.  The
    # runtime profile uses one persistent SearchContext worker pool instead.
    if backend == "inprocess":
        match_runner = _build_inprocess_runner(args, param_names)
        print("[backend] inprocess fixed-node; one JIT for the campaign", flush=True)
    else:
        book_path = _find_uci_opening_book()
        if book_path:
            print(f"Using UCI opening book: {book_path}")
        else:
            print("Warning: no PGN/polyglot book found; UCI matches start from startpos.")

        def match_runner(params_base, params_test):
            return run_match_python(
                params_base,
                params_test,
                num_games=args.games,
                tc=args.tc,
                concurrency=args.concurrency,
                opening_book=book_path or "tuner/data/pgn/twic1613.pgn",
            )

    print(
        f"Starting tuning for {args.iter} batches "
        f"(schedule={optimizer.total_batches}, pairs={optimizer.total_pairs})."
    )
    print(f"Params to tune ({len(param_names)}): {param_names}")
    print(
        f"SPSA schedule={SPSA_UPDATE_MODE} scale={optimizer.scale_mode} "
        f"A_ratio={optimizer.A_ratio} A={optimizer.A} "
        f"alpha={optimizer.alpha} gamma={optimizer.gamma}"
    )
    print(f"Iteration log: {log_path}")
    _append_jsonl(
        log_path,
        {
            "event": "resume" if args.resume else "start",
            "timestamp": time.time(),
            "iteration": int(optimizer.iteration),
            "backend": backend,
            "profile": args.profile,
            "params": list(param_names),
            "games": int(args.games),
            "nodes": int(args.nodes),
            "concurrency": int(args.concurrency),
            "scale_mode": args.scale_mode,
            "spsa_update_mode": SPSA_UPDATE_MODE,
            "spsa_schedule": "fishtest_classic",
            "schedule_total_batches": int(optimizer.total_batches),
            "spsa_iteration": int(optimizer.spsa_iteration),
            "match_pairs": int(optimizer.match_pairs),
            "total_pairs": int(optimizer.total_pairs),
            "A_ratio": float(optimizer.A_ratio),
            "A": float(optimizer.A),
            "alpha": float(optimizer.alpha),
            "gamma": float(optimizer.gamma),
            "tt_mb": int(args.tt_mb),
            "state": str(state_path) if state_path is not None else None,
            "log": str(log_path),
            "current_params": dict(optimizer.current_params),
        },
    )

    target_iterations = optimizer.iteration + max(0, int(args.iter))
    while optimizer.iteration < target_iterations:
        step_record = optimizer.step(match_runner)
        _append_jsonl(
            log_path,
            {
                "event": "iteration",
                "timestamp": time.time(),
                "backend": backend,
                "profile": args.profile,
                "params": list(param_names),
                "games": int(args.games),
                "nodes": int(args.nodes),
                "concurrency": int(args.concurrency),
                **step_record,
            },
        )
        # Save current best/state
        with open("tune_search/current_params.txt", "w") as f:
            f.write(str(optimizer.current_params))
        if state_path is not None:
            _save_state(
                state_path,
                optimizer,
                backend=backend,
                profile=args.profile,
                param_names=param_names,
                args=args,
                log_path=log_path,
            )

if __name__ == "__main__":
    main()
