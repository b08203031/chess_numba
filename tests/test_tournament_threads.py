from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import chess
import chess.pgn
import numpy as np

from tools import match_core, match_runner, tournament


def _make_fake_backend_modules(
    namespace: str,
    *,
    smp_search_signature: bool = True,
    provide_create_helper: bool = True,
    provide_reset_helper: bool = True,
    provide_warmup_jit: bool = True,
    env_tune_key: str | None = None,
) -> tuple[dict[str, types.ModuleType], dict[str, object]]:
    """Register an isolated fake engine package in sys.modules (no Numba/classical import)."""
    records: dict[str, object] = {
        "search_calls": [],
        "clear_tt_calls": 0,
        "reset_helper_calls": [],
        "warmup_calls": [],
        "created_contexts": [],
        "created_helpers": [],
    }

    pkg = types.ModuleType(namespace)
    pkg.__path__ = []  # type: ignore[attr-defined]

    constants = types.ModuleType(f"{namespace}.constants")
    constants.MAX_PLY = 128
    constants.NO_MOVE = 0
    constants.EVAL_WEIGHT_COUNT = 4

    tt_mod = types.ModuleType(f"{namespace}.transposition_table")

    def create_transposition_table(tt_mb: int):
        return {
            "mb": tt_mb,
            "key": np.zeros(4, dtype=np.uint32),
            "score": np.zeros(4, dtype=np.int16),
            "static_eval": np.full(4, 32767, dtype=np.int16),
            "depth": np.zeros(4, dtype=np.uint8),
            "flag": np.zeros(4, dtype=np.uint8),
            "generation": np.zeros(4, dtype=np.uint8),
            "best_move": np.zeros(4, dtype=np.uint16),
            "is_pv": np.zeros(4, dtype=np.bool_),
        }

    def clear_transposition_table(tt):
        records["clear_tt_calls"] = int(records["clear_tt_calls"]) + 1
        tt["key"][:] = 0
        tt["static_eval"][:] = 32767

    tt_mod.create_transposition_table = create_transposition_table
    tt_mod.clear_transposition_table = clear_transposition_table

    class FakeContext:
        def __init__(
            self,
            transposition_table,
            killer_moves,
            pv_table,
            history_table,
            butterfly_history,
            continuation_history,
            capture_history,
            pawn_history,
            pawn_correction_history,
            minor_correction_history,
            non_pawn_correction_history_white,
            non_pawn_correction_history_black,
            continuation_correction_history=None,
            *,
            stop_flag=None,
        ):
            self.transposition_table = transposition_table
            self.killer_moves = killer_moves
            self.pv_table = pv_table
            self.history_table = history_table
            self.butterfly_history = butterfly_history
            self.continuation_history = continuation_history
            self.capture_history = capture_history
            self.pawn_history = pawn_history
            self.pawn_correction_history = pawn_correction_history
            self.minor_correction_history = minor_correction_history
            self.non_pawn_correction_history_white = non_pawn_correction_history_white
            self.non_pawn_correction_history_black = non_pawn_correction_history_black
            self.continuation_correction_history = (
                continuation_correction_history
                if continuation_correction_history is not None
                else np.zeros((12, 64, 12, 64), dtype=np.int16)
            )
            self.counter_moves = np.zeros((12, 64), dtype=np.uint16)
            self.low_ply_history = np.zeros((5, 64), dtype=np.int16)
            self.diag_stats = np.zeros(8, dtype=np.uint64)
            self.pawn_table_keys = np.full(8, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
            self.pawn_table_w_king_sq = np.full(8, -1, dtype=np.int8)
            self.pawn_table_b_king_sq = np.full(8, -1, dtype=np.int8)
            self.game_history_count = 0
            self.tt_generation = 0
            self.nodes_searched = np.uint64(0)
            self.nodes_searched_array = np.zeros(2, dtype=np.uint64)
            self.stop_flag = stop_flag if stop_flag is not None else np.array([False], dtype=np.bool_)
            self.tune = np.zeros(8, dtype=np.int32)
            if env_tune_key and env_tune_key in os.environ:
                self.tune[0] = int(os.environ[env_tune_key])
            self.eval_weights = np.array([100, 200, 300, 400], dtype=np.int32)
            self.enable_nmp = True
            self.enable_rfp = True
            self.enable_razoring = True
            self.enable_lmr = True
            self.enable_see_pruning = True
            self.enable_lmp = True
            self.enable_fp = True
            self.enable_probcut = True
            self.enable_multicut = False
            records["created_contexts"].append(self)

    engine_types = types.ModuleType(f"{namespace}.engine_types")
    engine_types.SearchContext = FakeContext

    if provide_create_helper:
        def create_smp_helper_context(main_ctx):
            helper = FakeContext(
                main_ctx.transposition_table,
                np.zeros_like(main_ctx.killer_moves),
                np.zeros_like(main_ctx.pv_table),
                np.zeros_like(main_ctx.history_table),
                np.zeros_like(main_ctx.butterfly_history),
                np.zeros_like(main_ctx.continuation_history),
                np.zeros_like(main_ctx.capture_history),
                np.full_like(main_ctx.pawn_history, -1238),
                np.zeros_like(main_ctx.pawn_correction_history),
                np.zeros_like(main_ctx.minor_correction_history),
                np.zeros_like(main_ctx.non_pawn_correction_history_white),
                np.zeros_like(main_ctx.non_pawn_correction_history_black),
                np.zeros_like(main_ctx.continuation_correction_history),
                stop_flag=main_ctx.stop_flag,
            )
            helper.tune[:] = main_ctx.tune[:]
            helper.eval_weights[:] = main_ctx.eval_weights[:]
            helper.enable_nmp = bool(main_ctx.enable_nmp)
            helper.enable_rfp = bool(main_ctx.enable_rfp)
            helper.enable_razoring = bool(main_ctx.enable_razoring)
            helper.enable_lmr = bool(main_ctx.enable_lmr)
            helper.enable_see_pruning = bool(main_ctx.enable_see_pruning)
            helper.enable_lmp = bool(main_ctx.enable_lmp)
            helper.enable_fp = bool(main_ctx.enable_fp)
            helper.enable_probcut = bool(main_ctx.enable_probcut)
            helper.enable_multicut = bool(main_ctx.enable_multicut)
            records["created_helpers"].append(helper)
            return helper

        engine_types.create_smp_helper_context = create_smp_helper_context

    if provide_reset_helper:
        def reset_search_context(ctx):
            records["reset_helper_calls"].append(ctx)
            ctx.history_table[:] = 0
            ctx.pawn_history[:] = -1238
            ctx.stop_flag[0] = False

        engine_types.reset_search_context = reset_search_context

    fen_parser = types.ModuleType(f"{namespace}.fen_parser")

    def parse_fen(fen: str):
        board = chess.Board(fen)
        state = [board, 0, 0, 0, hash(board.fen()) & 0xFFFFFFFFFFFFFFFF]
        return np.zeros(12, dtype=np.uint64), np.zeros(3, dtype=np.uint64), state

    fen_parser.parse_fen = parse_fen

    def _encode_move(move: chess.Move) -> int:
        promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
        val = move.from_square | (move.to_square << 6)
        if move.promotion is not None:
            val |= (promo_map[move.promotion] << 12) | (1 << 14)
        return val

    def _decode_move(encoded: int) -> chess.Move:
        from_sq = int(encoded) & 0x3F
        to_sq = (int(encoded) >> 6) & 0x3F
        flag = (int(encoded) >> 14) & 0x3
        promo = None
        if flag == 1:
            promo = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][(int(encoded) >> 12) & 0x3]
        return chess.Move(from_sq, to_sq, promotion=promo)

    core = types.ModuleType(f"{namespace}.core")
    core.generate_legal_moves = lambda p_bbs, o_bbs, g_state: [
        _encode_move(mv) for mv in g_state[0].legal_moves
    ]

    board_ops = types.ModuleType(f"{namespace}.board_operations")

    def make_move(p_bbs, o_bbs, g_state, encoded):
        g_state[0].push(_decode_move(int(encoded)))
        g_state[4] = hash(g_state[0].fen()) & 0xFFFFFFFFFFFFFFFF

    board_ops.make_move = make_move

    move_mod = types.ModuleType(f"{namespace}.move")
    move_mod.move_to_uci = lambda encoded: _decode_move(int(encoded)).uci()

    search_mod = types.ModuleType(f"{namespace}.search")
    if smp_search_signature:
        def iterative_deepening_search(
            piece_bbs,
            occupancy_bbs,
            game_state,
            max_depth,
            time_config,
            search_context,
            game_history_list=None,
            tt_generation=0,
            verbose=True,
            num_threads=1,
            helper_contexts=None,
        ):
            records["search_calls"].append({
                "max_depth": max_depth,
                "time_config": dict(time_config),
                "search_context": search_context,
                "tt_generation": tt_generation,
                "verbose": verbose,
                "num_threads": num_threads,
                "helper_contexts": helper_contexts,
            })
            mv = next(iter(game_state[0].legal_moves))
            return (_encode_move(mv), 25, 1000, 200, 10, min(max_depth, 4))
    else:
        def iterative_deepening_search(
            piece_bbs,
            occupancy_bbs,
            game_state,
            max_depth,
            time_config,
            search_context,
            game_history_list=None,
            tt_generation=0,
            verbose=True,
        ):
            records["search_calls"].append({
                "max_depth": max_depth,
                "time_config": dict(time_config),
                "search_context": search_context,
                "tt_generation": tt_generation,
                "verbose": verbose,
            })
            mv = next(iter(game_state[0].legal_moves))
            return (_encode_move(mv), 25, 1000, 200, 10, min(max_depth, 4))

    search_mod.iterative_deepening_search = iterative_deepening_search
    if provide_warmup_jit:
        def warmup_search_jit(search_context=None):
            records["warmup_calls"].append(search_context)
            return True

        search_mod.warmup_search_jit = warmup_search_jit

    modules = {
        namespace: pkg,
        f"{namespace}.constants": constants,
        f"{namespace}.transposition_table": tt_mod,
        f"{namespace}.engine_types": engine_types,
        f"{namespace}.fen_parser": fen_parser,
        f"{namespace}.core": core,
        f"{namespace}.board_operations": board_ops,
        f"{namespace}.move": move_mod,
        f"{namespace}.search": search_mod,
    }
    return modules, records


def _write_dummy_uci_script(
    path: Path,
    log_path: Path,
    *,
    advertise_threads: bool = True,
    emit_sf19_bounds: bool = True,
) -> Path:
    script = f'''import json
import sys
import chess

log_file = {repr(str(log_path))}
board = chess.Board()
received_options = []

def record(event):
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\\n")

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    tokens = line.split()
    cmd = tokens[0]
    if cmd == "uci":
        print("id name DummyUCI")
        print("id author Test")
        print("option name Hash type spin default 16 min 1 max 4096")
        print("option name OwnBook type check default true")
        if {advertise_threads!r}:
            print("option name Threads type spin default 1 min 1 max 64")
        print("uciok", flush=True)
    elif cmd == "setoption":
        name_idx = tokens.index("name")
        val_idx = tokens.index("value")
        opt_name = " ".join(tokens[name_idx + 1:val_idx])
        opt_val = " ".join(tokens[val_idx + 1:])
        received_options.append([opt_name, opt_val])
        record({{"cmd": "setoption", "name": opt_name, "value": opt_val}})
    elif cmd == "isready":
        print("readyok", flush=True)
    elif cmd == "ucinewgame":
        board = chess.Board()
    elif cmd == "position":
        if "startpos" in tokens:
            board = chess.Board()
        elif "fen" in tokens:
            fen_start = tokens.index("fen") + 1
            fen_end = tokens.index("moves") if "moves" in tokens else len(tokens)
            board = chess.Board(" ".join(tokens[fen_start:fen_end]))
        if "moves" in tokens:
            for mv in tokens[tokens.index("moves") + 1:]:
                board.push_uci(mv)
    elif cmd == "go":
        mv = next(iter(board.legal_moves))
        if {emit_sf19_bounds!r}:
            print("info string debug depth 99 score cp 9999", flush=True)
            print("info depth 6 seldepth 10 multipv 1 score cp 42 lowerbound nodes 1200 nps 60000 hashfull 5 tbhits 0 time 20 pv " + mv.uci(), flush=True)
            print("info depth 6 seldepth 10 multipv 2 score cp -900 nodes 1200 nps 60000 time 20 pv " + mv.uci(), flush=True)
            print("info depth 7 seldepth 11 multipv 1 score cp -35 upperbound nodes 2400 nps 80000 hashfull 8 tbhits 0 time 30 pv " + mv.uci(), flush=True)
        else:
            print("info depth 1 score cp 15 nodes 50 nps 5000 time 10 pv " + mv.uci(), flush=True)
        print("bestmove " + mv.uci(), flush=True)
    elif cmd == "quit":
        break
'''
    path.write_text(script, encoding="utf-8", newline="\n")
    return path


class TestTournamentAndMatchThreads(unittest.TestCase):
    def test_tournament_cli_threads_and_asymmetric_limits_parsing(self) -> None:
        parser = tournament.build_parser()

        args_default = parser.parse_args([])
        self.assertIsNone(args_default.threads)
        self.assertEqual(args_default.threads1, 1)
        self.assertEqual(args_default.threads2, 1)

        args_shared = parser.parse_args(["--threads", "4"])
        self.assertEqual(args_shared.threads, 4)
        self.assertEqual(args_shared.threads1, 4)
        self.assertEqual(args_shared.threads2, 4)

        args_override_after = parser.parse_args(["--threads", "4", "--threads2", "1"])
        self.assertEqual(args_override_after.threads1, 4)
        self.assertEqual(args_override_after.threads2, 1)

        args_override_before = parser.parse_args(["--threads2", "1", "--threads", "4"])
        self.assertEqual(args_override_before.threads1, 4)
        self.assertEqual(args_override_before.threads2, 1)

        args_both_explicit = parser.parse_args(["--threads1", "2", "--threads2", "3", "--threads", "8"])
        self.assertEqual(args_both_explicit.threads1, 2)
        self.assertEqual(args_both_explicit.threads2, 3)

        stages = tournament.parse_stages("10:n50000:n100000,6:d8")
        self.assertEqual(stages[0]["limit1"], match_core.SearchLimit(nodes=50000))
        self.assertEqual(stages[0]["limit2"], match_core.SearchLimit(nodes=100000))
        self.assertEqual(stages[1]["limit1"], match_core.SearchLimit(depth=8))
        self.assertEqual(stages[1]["limit2"], match_core.SearchLimit(depth=8))

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tournament.main(["--stages", "10:n1000", "--nodes1", "5000"])

        self.assertEqual(
            tournament._select_backend(
                "auto",
                str(tournament.DEFAULT_ENGINE1),
                str(tournament.DEFAULT_ENGINE1),
                False,
                False,
            ),
            "uci",
        )
        self.assertEqual(
            tournament._select_backend(
                "auto",
                str(tournament.DEFAULT_ENGINE1),
                str(tournament.DEFAULT_ENGINE2),
                False,
                False,
            ),
            "inprocess",
        )

        for bad_argv in (
            ["--threads", "0"],
            ["--threads1", "0"],
            ["--threads2", "-2"],
            ["--threads", "0", "--threads1", "2", "--threads2", "2"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(tournament.main(bad_argv), 2)

    def test_match_runner_cli_threads_and_adjudication_flags(self) -> None:
        parser = match_runner.build_parser()

        args_default = parser.parse_args([])
        self.assertEqual(args_default.engine_threads, 1)
        self.assertEqual(args_default.sf_threads, 1)
        self.assertTrue(args_default.adjudicate)
        self.assertEqual(args_default.adjudicate_win_score, 500)
        self.assertEqual(args_default.adjudicate_win_moves, 5)
        self.assertEqual(args_default.adjudicate_draw_score, 8)
        self.assertEqual(args_default.adjudicate_draw_min_move, 40)
        self.assertEqual(args_default.adjudicate_draw_moves, 6)

        args_custom = parser.parse_args([
            "--engine-threads", "4",
            "--sf-threads", "2",
            "--no-adjudicate",
            "--adjudicate-win-score", "650",
            "--adjudicate-win-moves", "4",
            "--adjudicate-draw-score", "12",
            "--adjudicate-draw-min-move", "35",
            "--adjudicate-draw-moves", "7",
        ])
        self.assertEqual(args_custom.engine_threads, 4)
        self.assertEqual(args_custom.sf_threads, 2)
        self.assertFalse(args_custom.adjudicate)
        self.assertEqual(args_custom.adjudicate_win_score, 650)
        self.assertEqual(args_custom.adjudicate_win_moves, 4)
        self.assertEqual(args_custom.adjudicate_draw_score, 12)
        self.assertEqual(args_custom.adjudicate_draw_min_move, 35)
        self.assertEqual(args_custom.adjudicate_draw_moves, 7)

        for bad_argv in (["--engine-threads", "0"], ["--sf-threads", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(match_runner.main(bad_argv), 2)

    def test_inprocess_factory_smp_allocation_reset_forwarding_and_validation(self) -> None:
        with self.assertRaises(ValueError):
            match_core.InProcessEngineFactory("Bad", "unused", threads=0)
        with self.assertRaises(ValueError):
            match_core.InProcessEngineFactory("BadBool", "unused", threads=True)  # type: ignore[arg-type]

        # 1) Legacy/NNUE-like backend (no SMP signature or helper functions):
        #    works at threads=1, rejects threads > 1 with clear ValueError.
        legacy_mods, legacy_records = _make_fake_backend_modules(
            "_fake_legacy_engine",
            smp_search_signature=False,
            provide_create_helper=False,
            provide_reset_helper=False,
            provide_warmup_jit=False,
        )
        with mock.patch.dict(sys.modules, legacy_mods):
            f_legacy = match_core.InProcessEngineFactory("Legacy", "_fake_legacy_engine", tt_mb=16, threads=1)
            e_legacy = f_legacy.create(0)
            self.assertEqual(e_legacy.threads, 1)
            self.assertEqual(e_legacy.helper_contexts, [])
            e_legacy.new_game(None)
            mv, _ = e_legacy.get_move(match_core.SearchLimit(depth=2))
            self.assertIsInstance(mv, chess.Move)
            call_kwargs = legacy_records["search_calls"][-1]
            self.assertNotIn("num_threads", call_kwargs)
            self.assertNotIn("helper_contexts", call_kwargs)

            with self.assertRaisesRegex(ValueError, "missing keyword parameter.*num_threads.*helper_contexts"):
                match_core.InProcessEngineFactory("Legacy", "_fake_legacy_engine", threads=2)

        # 2) Search supports num_threads/helper_contexts, but engine_types lacks helper funcs.
        missing_et_mods, _ = _make_fake_backend_modules(
            "_fake_missing_et_engine",
            smp_search_signature=True,
            provide_create_helper=False,
            provide_reset_helper=False,
        )
        with mock.patch.dict(sys.modules, missing_et_mods):
            with self.assertRaisesRegex(ValueError, "create_smp_helper_context.*reset_search_context"):
                match_core.InProcessEngineFactory("MissingET", "_fake_missing_et_engine", threads=2)

        # 3) Full SMP-capable mock backend with env_extra + runtime sync + helper reset + prepare.
        smp_mods, smp_records = _make_fake_backend_modules(
            "_fake_smp_engine",
            smp_search_signature=True,
            provide_create_helper=True,
            provide_reset_helper=True,
            provide_warmup_jit=True,
            env_tune_key="_FAKE_TUNE_SCALE",
        )
        with mock.patch.dict(sys.modules, smp_mods):
            factory = match_core.InProcessEngineFactory(
                "SMP",
                "_fake_smp_engine",
                tt_mb=32,
                env_extra={"_FAKE_TUNE_SCALE": "77"},
                threads=4,
            )
            engine = factory.create(0)
            self.assertEqual(engine.threads, 4)
            self.assertEqual(len(engine.helper_contexts), 3)
            self.assertNotIn("_FAKE_TUNE_SCALE", os.environ)
            self.assertEqual(int(engine.context.tune[0]), 77)
            for helper in engine.helper_contexts:
                self.assertIs(helper.transposition_table, engine.context.transposition_table)
                self.assertIs(helper.stop_flag, engine.context.stop_flag)
                self.assertEqual(int(helper.tune[0]), 77)
                np.testing.assert_array_equal(helper.eval_weights, engine.context.eval_weights)

            # Mutate helper state before new_game() and verify single TT clear + helper reset.
            engine.helper_contexts[0].history_table[0, 0] = 99
            engine.new_game(None)
            self.assertEqual(smp_records["clear_tt_calls"], 1)
            self.assertEqual(smp_records["reset_helper_calls"], engine.helper_contexts)
            self.assertEqual(int(engine.helper_contexts[0].history_table[0, 0]), 0)

            # Verify get_move forwards num_threads and helper_contexts.
            engine.get_move(match_core.SearchLimit(nodes=5000))
            last_search = smp_records["search_calls"][-1]
            self.assertEqual(last_search["num_threads"], 4)
            self.assertEqual(last_search["helper_contexts"], engine.helper_contexts)

            # Verify prepare() warms up SMP path on worker 0 and skips worker > 0.
            with contextlib.redirect_stdout(io.StringIO()):
                factory.prepare(engine, 1)
                self.assertEqual(len(smp_records["warmup_calls"]), 0)
                factory.prepare(engine, 0)
            self.assertEqual(smp_records["warmup_calls"], [engine.context])
            self.assertEqual(smp_records["search_calls"][-1]["max_depth"], 14)
            self.assertEqual(smp_records["search_calls"][-1]["num_threads"], 4)

    def test_uci_engine_threads_hash_and_sf19_bound_parsing(self) -> None:
        with self.assertRaises(ValueError):
            match_core.UciEngine("Bad", "dummy.py", options={"Threads": 0})
        with self.assertRaises(ValueError):
            match_core.UciEngineFactory("Bad", "dummy.py", threads=0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            log_path = tmp_dir / "uci_log.jsonl"
            script_path = _write_dummy_uci_script(
                tmp_dir / "dummy_sf19.py",
                log_path,
                advertise_threads=True,
                emit_sf19_bounds=True,
            )
            factory = match_core.UciEngineFactory(
                "DummySF19",
                str(script_path),
                options={"OwnBook": False, "Threads": 4, "Hash": 64},
            )
            engine = factory.create(0)
            with contextlib.redirect_stdout(io.StringIO()):
                engine.start()
            try:
                engine.new_game(None)
                move, info = engine.get_move(match_core.SearchLimit(depth=7))
            finally:
                engine.terminate()

            self.assertIsInstance(move, chess.Move)
            self.assertEqual(engine.last_score_cp, -35)
            self.assertIn("d=7", info)
            self.assertIn("eval=-0.35", info)
            self.assertIn("n=2400", info)

            # Hash must be sent before Threads so UCI engines allocate TT once before SMP helpers.
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            import json
            setopts = [json.loads(line) for line in lines]
            names = [entry["name"] for entry in setopts]
            self.assertEqual(set(names), {"OwnBook", "Hash", "Threads"})
            self.assertLess(names.index("Hash"), names.index("Threads"))
            by_name = {entry["name"]: entry["value"] for entry in setopts}
            self.assertEqual(by_name["Hash"], "64")
            self.assertEqual(by_name["Threads"], "4")
            self.assertEqual(by_name["OwnBook"], "false")

            # Engine advertising options without Threads emits non-fatal warning when Threads > 1.
            log_no_threads = tmp_dir / "no_threads_log.jsonl"
            no_threads_script = _write_dummy_uci_script(
                tmp_dir / "dummy_no_threads.py",
                log_no_threads,
                advertise_threads=False,
                emit_sf19_bounds=False,
            )
            no_threads_engine = match_core.UciEngine(
                "NoThreads",
                str(no_threads_script),
                options={"Threads": 3},
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                no_threads_engine.start()
            no_threads_engine.terminate()
            self.assertIn("did not advertise UCI option 'Threads'", buf.getvalue())

    def test_tournament_wiring_oversubscription_and_same_engine_uci_match(self) -> None:
        # 1) Mocked wiring check for inprocess same-module + oversubscription warning.
        fake_mods, _ = _make_fake_backend_modules("_fake_wiring_engine")
        captured: dict[str, object] = {}

        def fake_create_pairs(f1, f2, conc):
            captured["f1"] = f1
            captured["f2"] = f2
            captured["conc"] = conc
            return [mock.MagicMock()]

        def fake_run_stage(*args, **kwargs):
            captured["run_stage_kwargs"] = kwargs
            return match_core.StageResult(2, 1, 1, 0, 1.5, "Disabled", 0.0, kwargs["pgn_path"])

        with tempfile.TemporaryDirectory() as tmp:
            pgn_out = Path(tmp) / "wired.pgn"
            buf = io.StringIO()
            with (
                mock.patch.dict(sys.modules, fake_mods),
                mock.patch.object(tournament, "create_engine_pairs", side_effect=fake_create_pairs),
                mock.patch.object(tournament, "close_engine_pairs"),
                mock.patch.object(tournament, "run_stage", side_effect=fake_run_stage),
                mock.patch.object(tournament.os, "cpu_count", return_value=4),
                contextlib.redirect_stdout(buf),
            ):
                tournament.run_tournament(
                    "unused1.py",
                    "unused2.py",
                    games_count=4,
                    time_ms=None,
                    engine1_name="SMP4",
                    engine2_name="SMP1",
                    nodes=1000,
                    concurrency=2,
                    pgn_file=str(pgn_out),
                    use_openings=False,
                    use_sprt=False,
                    backend="inprocess",
                    engine1_module="_fake_wiring_engine",
                    engine2_module="_fake_wiring_engine",
                    tt_mb=32,
                    threads1=4,
                    threads2=1,
                )

        out = buf.getvalue()
        self.assertEqual(captured["f1"].threads, 4)
        self.assertEqual(captured["f2"].threads, 1)
        self.assertIn("SMP4:4 vs SMP1:1", out)
        self.assertIn("[warning]", out)
        self.assertIn("exceeds CPU count (4)", out)

        # 2) End-to-end dummy UCI same-engine match (main.py vs main.py pattern).
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            log_path = tmp_dir / "same_engine.jsonl"
            dummy_uci = _write_dummy_uci_script(
                tmp_dir / "dummy_main.py",
                log_path,
                advertise_threads=True,
                emit_sf19_bounds=False,
            )
            pgn_path = tmp_dir / "same_engine.pgn"
            buf = io.StringIO()
            with (
                mock.patch.object(tournament.os, "cpu_count", return_value=16),
                contextlib.redirect_stdout(buf),
            ):
                rc = tournament.main([
                    "--engine1", str(dummy_uci),
                    "--engine2", str(dummy_uci),
                    "--name1", "SMP4",
                    "--name2", "SMP1",
                    "--threads1", "4",
                    "--threads2", "1",
                    "--tt-mb", "64",
                    "--games", "2",
                    "--depth", "1",
                    "--no-openings",
                    "--no-sprt",
                    "--pgn", str(pgn_path),
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(pgn_path.exists())
            with pgn_path.open("r", encoding="utf-8") as fh:
                g1 = chess.pgn.read_game(fh)
                g2 = chess.pgn.read_game(fh)
            assert g1 is not None and g2 is not None
            self.assertEqual(g1.headers["White"], "SMP4")
            self.assertEqual(g1.headers["WhiteThreads"], "4")
            self.assertEqual(g1.headers["BlackThreads"], "1")
            self.assertEqual(g2.headers["Black"], "SMP4")
            self.assertEqual(g2.headers["WhiteThreads"], "1")
            self.assertEqual(g2.headers["BlackThreads"], "4")

    def test_match_runner_threads_adjudication_and_uci_wiring(self) -> None:
        # 1) Mocked inprocess wiring + adjudication forwarding + oversubscription warning.
        fake_mods, _ = _make_fake_backend_modules("_fake_runner_engine")
        captured: dict[str, object] = {}

        def fake_create_pairs(f1, f2, conc):
            captured["engine_factory"] = f1
            captured["sf_factory"] = f2
            captured["conc"] = conc
            return [mock.MagicMock()]

        def fake_run_stage(*args, **kwargs):
            captured["run_stage_kwargs"] = kwargs
            return match_core.StageResult(2, 0, 2, 0, 1.0, "Disabled", 0.0, kwargs["pgn_path"])

        with tempfile.TemporaryDirectory() as tmp:
            sf_dummy = Path(tmp) / "sf.exe"
            sf_dummy.write_text("", encoding="utf-8")
            pgn_path = Path(tmp) / "sf_match.pgn"
            buf = io.StringIO()
            with (
                mock.patch.dict(sys.modules, fake_mods),
                mock.patch.object(match_runner, "create_engine_pairs", side_effect=fake_create_pairs),
                mock.patch.object(match_runner, "close_engine_pairs"),
                mock.patch.object(match_runner, "run_stage", side_effect=fake_run_stage),
                mock.patch.object(match_runner.os, "cpu_count", return_value=4),
                contextlib.redirect_stdout(buf),
            ):
                match_runner.run_match_stages(
                    str(match_runner.DEFAULT_ENGINE),
                    str(sf_dummy),
                    [{"games": 4, "engine_limit": match_core.SearchLimit(nodes=5000), "sf_limit": match_core.SearchLimit(nodes=5000)}],
                    concurrency=2,
                    pgn_base=str(pgn_path),
                    no_sprt=True,
                    engine_backend="inprocess",
                    engine_module="_fake_runner_engine",
                    tt_mb=64,
                    sf_hash_mb=32,
                    engine_threads=3,
                    sf_threads=2,
                    use_openings=False,
                    adjudicate=False,
                    adjudicate_win_score=600,
                    adjudicate_win_plies=8,
                    adjudicate_draw_score=10,
                    adjudicate_draw_min_move=30,
                    adjudicate_draw_plies=14,
                )

        out = buf.getvalue()
        self.assertEqual(captured["engine_factory"].threads, 3)
        self.assertEqual(captured["sf_factory"].kwargs["options"], {"Threads": 2, "Ponder": False, "Hash": 32})
        self.assertFalse(captured["run_stage_kwargs"]["adjudicate"])
        self.assertEqual(captured["run_stage_kwargs"]["adjudicate_win_score"], 600)
        self.assertEqual(captured["run_stage_kwargs"]["adjudicate_win_plies"], 8)
        self.assertEqual(captured["run_stage_kwargs"]["adjudicate_draw_score"], 10)
        self.assertEqual(captured["run_stage_kwargs"]["adjudicate_draw_min_move"], 30)
        self.assertEqual(captured["run_stage_kwargs"]["adjudicate_draw_plies"], 14)
        self.assertIn("Threads=3", out)
        self.assertIn("Adjudication: disabled", out)
        self.assertIn("[warning]", out)

        # 2) End-to-end dummy UCI match_runner test verifying Hash & Threads options.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            eng_log = tmp_dir / "eng.jsonl"
            sf_log = tmp_dir / "sf.jsonl"
            eng_script = _write_dummy_uci_script(tmp_dir / "eng.py", eng_log, emit_sf19_bounds=False)
            sf_script = _write_dummy_uci_script(tmp_dir / "sf.py", sf_log, emit_sf19_bounds=True)
            pgn_path = tmp_dir / "runner_uci.pgn"
            buf = io.StringIO()
            with (
                mock.patch.object(match_runner.os, "cpu_count", return_value=16),
                contextlib.redirect_stdout(buf),
            ):
                rc = match_runner.main([
                    "--engine", str(eng_script),
                    "--sf", str(sf_script),
                    "--engine-backend", "uci",
                    "--engine-threads", "3",
                    "--sf-threads", "2",
                    "--tt-mb", "64",
                    "--sf-hash-mb", "32",
                    "--games", "2",
                    "--engine-depth", "1",
                    "--sf-depth", "1",
                    "--no-openings",
                    "--no-sprt",
                    "--adjudicate-win-score", "550",
                    "--pgn", str(pgn_path),
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(pgn_path.exists())
            import json
            eng_opts = {
                json.loads(line)["name"]: json.loads(line)["value"]
                for line in eng_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            sf_opts = {
                json.loads(line)["name"]: json.loads(line)["value"]
                for line in sf_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            self.assertEqual(eng_opts, {"OwnBook": "false", "Hash": "64", "Threads": "3"})
            self.assertEqual(sf_opts, {"Threads": "2", "Ponder": "false", "Hash": "32"})

    def test_sf19_mate_bounds_and_tournament_asymmetric_limit_wiring(self) -> None:
        self.assertEqual(
            match_core.parse_info_line(
                "info depth 12 seldepth 16 multipv 1 score mate 3 lowerbound nodes 5000 nps 100000 time 50 pv e2e4"
            ),
            "d=12, eval=#3, n=5000, t=50ms, nps=100000",
        )
        self.assertEqual(
            match_core.extract_score_from_info(
                match_core.parse_info_line(
                    "info depth 12 multipv 1 score mate -2 upperbound nodes 5000 time 50 pv e2e4"
                )
            ),
            -30000,
        )

        fake_mods, _ = _make_fake_backend_modules("_fake_asym_engine")
        captured: dict[str, object] = {}

        def fake_run_stage(*args, **kwargs):
            captured["first_limit"] = kwargs["first_limit"]
            captured["second_limit"] = kwargs["second_limit"]
            return match_core.StageResult(2, 1, 1, 0, 1.5, "Disabled", 0.0, kwargs["pgn_path"])

        with tempfile.TemporaryDirectory() as tmp:
            pgn_path = Path(tmp) / "asym.pgn"
            with (
                mock.patch.dict(sys.modules, fake_mods),
                mock.patch.object(tournament, "create_engine_pairs", return_value=[mock.MagicMock()]),
                mock.patch.object(tournament, "close_engine_pairs"),
                mock.patch.object(tournament, "run_stage", side_effect=fake_run_stage),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc = tournament.main([
                    "--backend", "inprocess",
                    "--engine1-module", "_fake_asym_engine",
                    "--engine2-module", "_fake_asym_engine",
                    "--threads1", "4",
                    "--threads2", "1",
                    "--nodes1", "2000",
                    "--nodes2", "8000",
                    "--games", "2",
                    "--no-openings",
                    "--no-sprt",
                    "--pgn", str(pgn_path),
                ])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["first_limit"], match_core.SearchLimit(nodes=2000))
        self.assertEqual(captured["second_limit"], match_core.SearchLimit(nodes=8000))


if __name__ == "__main__":
    unittest.main()
