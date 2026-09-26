#!/usr/bin/env python3
"""
History / search diagnostics bench for classical search experiments.

Runs fixed-node searches on FENs and aggregates:
  - DIAG_HIST_* counters (prune rate, bonus/malus, updates)
  - cut_first / LMR research
  - history table energy / saturation

Supports dual-set mode: 500 puzzles + 500 openings with separate and combined stats.
Parallel workers (-c N) use ThreadPool + independent SearchContext/TT per job
(Numba nogil) — same stats as serial, no per-process re-JIT.

Usage (repo root):
  python tools/hist_diag_bench.py
  python tools/hist_diag_bench.py --nodes 30000 -c 4 --tag puzzles_openings_30k
  python tools/hist_diag_bench.py --puzzles data/hist_diag_puzzles_500.epd \\
      --openings data/hist_diag_openings_500.epd --nodes 30000 -c 4
  python tools/hist_diag_bench.py --nodes 30000 --positions 100 -c 4 \\
      --sequence-plies 3 --paired-cold --tag warm_history_sequence

Protocol: run BEFORE and AFTER a single search change at 30k nodes; paste summary into SEARCH_EXP_LOG.
``--param NAME=VALUE`` applies runtime-only values to diagnostic contexts and
does not modify ``constants.py`` or the production UCI engine.
Sequence mode clears TT between plies by default so warm/fresh differences
measure persisted heuristic/correction history rather than the previous subtree.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ThreadPool + Numba nogil: one JIT compile, parallel searches with independent ctx.
# ProcessPool would re-JIT per worker on Windows and is much slower for this engine.
_print_lock = threading.Lock()

from chess_engine.classical.constants import (  # noqa: E402
    HISTORY_MAX_BUTTERFLY,
    HISTORY_MAX_CAPTURE,
    HISTORY_MAX_CONTINUATION,
    HISTORY_MAX_MAIN,
    HISTORY_MAX_PAWN,
    HISTORY_COMPAT_1A,
    LMP_SCALE_PERCENT,
    LMR_BASE_OFFSET,
    LMR_HISTORY_SCALE,
    MAX_PLY,
    NO_MOVE,
    PRUNING_HISTORY_THRESHOLD,
)
from chess_engine.classical.board_operations import make_move  # noqa: E402
from chess_engine.classical.engine_types import SearchContext  # noqa: E402
from chess_engine.classical.fen_parser import parse_fen  # noqa: E402
from chess_engine.classical.move import move_to_uci  # noqa: E402
from chess_engine.classical.search import (  # noqa: E402
    collect_hist_diag_dict,
    format_search_diagnostics,
    iterative_deepening_search,
)
from chess_engine.classical.transposition_table import create_transposition_table  # noqa: E402
from tune_search.runtime_apply import apply_runtime_params  # noqa: E402
from tune_search.search_param_registry import get_spec  # noqa: E402


DEFAULT_PUZZLES = ROOT / "data" / "hist_diag_puzzles_500.epd"
DEFAULT_OPENINGS = ROOT / "data" / "hist_diag_openings_500.epd"
DEFAULT_OUT_DIR = ROOT / "tournament_analysis"
_HP_LAYER_NAMES = (
    "main", "butterfly", "pawn", "cont1", "cont2", "cont3", "cont4", "cont6",
)

# Optional per-run overrides are applied only to freshly-created diagnostic
# contexts.  The production constants and the UCI engine are untouched.
_RUNTIME_OVERRIDES: dict[str, int] = {}
_RUNTIME_OVERRIDE_SCALE = "integer"


def load_fens(path: Path, limit: int) -> list[str]:
    lines = []
    if not path.exists():
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ][:limit]
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fen = line.strip()
        if not fen or fen.startswith("#"):
            continue
        lines.append(fen)
        if len(lines) >= limit:
            break
    return lines


def mean_dict_list(dicts: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d and np.isfinite(d[k])]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def _make_ctx() -> SearchContext:
    tt = create_transposition_table(16)
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
    ctx = SearchContext(
        tt, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history, capture_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black,
        continuation_correction_history,
        enable_diagnostics=True,
    )
    if _RUNTIME_OVERRIDES:
        apply_runtime_params(
            ctx,
            _RUNTIME_OVERRIDES,
            scale_mode=_RUNTIME_OVERRIDE_SCALE,
        )
    return ctx


def _search_board(
    ctx: SearchContext,
    p_bbs: np.ndarray,
    o_bbs: np.ndarray,
    g_state: np.ndarray,
    nodes: int,
    idx: int,
    set_name: str,
    fen_label: str = "",
    game_history: list[int] | None = None,
    tt_generation: int = 0,
) -> tuple[dict, tuple]:
    """Search one already-parsed board and return diagnostics plus raw result."""
    t0 = time.time()
    result = iterative_deepening_search(
        p_bbs, o_bbs, g_state, 64,
        {"nodes_limit": nodes, "maximum_time": 0, "optimum_time": 0},
        ctx, game_history_list=game_history, tt_generation=tt_generation,
        verbose=False,
    )
    elapsed = time.time() - t0
    total_nodes = nodes
    search_nodes = 0
    q_nodes = 0
    depth_done = 0
    if isinstance(result, (tuple, list)) and len(result) >= 6:
        search_nodes = int(result[2])
        q_nodes = int(result[3])
        measured_total = search_nodes + q_nodes
        total_nodes = measured_total if measured_total > 0 else nodes
        depth_done = int(result[5])
    elif isinstance(result, (tuple, list)) and len(result) >= 3:
        search_nodes = int(result[2])
        total_nodes = search_nodes if search_nodes > 0 else nodes

    d = collect_hist_diag_dict(ctx, total_nodes=total_nodes)
    d["fen"] = fen_label[:80]
    d["elapsed_s"] = elapsed
    d["depth"] = depth_done
    d["search_nodes"] = search_nodes
    d["q_nodes"] = q_nodes
    d["idx"] = idx
    d["set"] = set_name
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        d["best_move"] = int(result[0])
        d["best_move_uci"] = (
            move_to_uci(np.uint16(result[0])) if int(result[0]) != int(NO_MOVE) else "0000"
        )
        d["score"] = int(result[1])
    d["_sample_text"] = format_search_diagnostics(ctx, total_nodes=total_nodes, q_nodes=q_nodes)
    return d, result


def _search_one(fen: str, nodes: int, idx: int, set_name: str) -> dict:
    """Independent cold-context search."""
    ctx = _make_ctx()
    p_bbs, o_bbs, g_state = parse_fen(fen)
    d, _ = _search_board(
        ctx, p_bbs, o_bbs, g_state, nodes, idx, set_name, fen_label=fen,
    )
    return d


def _search_sequence_seed(
    fen: str,
    nodes: int,
    seed_idx: int,
    set_name: str,
    sequence_plies: int,
    paired_cold: bool,
    keep_sequence_tt: bool,
) -> list[dict]:
    """Follow warm-context best moves and optionally pair each ply with cold history."""
    warm_ctx = _make_ctx()
    p_bbs, o_bbs, g_state = parse_fen(fen)
    game_history = [int(g_state[4])]
    rows: list[dict] = []

    for sequence_ply in range(sequence_plies):
        cold_d = None
        if paired_cold:
            cold_ctx = _make_ctx()
            cold_d, _ = _search_board(
                cold_ctx,
                p_bbs.copy(), o_bbs.copy(), g_state.copy(),
                nodes, seed_idx * sequence_plies + sequence_ply, set_name,
                fen_label=f"{fen} | sequence_ply={sequence_ply}",
                game_history=game_history, tt_generation=sequence_ply + 1,
            )

        d, result = _search_board(
            warm_ctx, p_bbs, o_bbs, g_state,
            nodes, seed_idx * sequence_plies + sequence_ply, set_name,
            fen_label=f"{fen} | sequence_ply={sequence_ply}",
            game_history=game_history, tt_generation=sequence_ply + 1,
        )
        d["seed_idx"] = seed_idx
        d["seed_fen"] = fen[:80]
        d["sequence_ply"] = sequence_ply
        d["history_state"] = "warm" if sequence_ply > 0 else "fresh"

        if cold_d is not None:
            for key in (
                "nodes", "search_nodes", "q_nodes", "depth", "score",
                "best_move", "best_move_uci", "hist_prune", "lmr_try",
                "lmr_research", "cut_first_pct",
            ):
                d[f"cold_{key}"] = cold_d.get(key, 0)
            for key, value in cold_d.items():
                if key.startswith("hist_prune_"):
                    d[f"cold_{key}"] = value
            d["nodes_delta"] = d["nodes"] - cold_d["nodes"]
            d["depth_delta"] = d["depth"] - cold_d["depth"]
            d["score_delta"] = d.get("score", 0) - cold_d.get("score", 0)
            d["best_move_same"] = int(
                d.get("best_move", 0) == cold_d.get("best_move", 0)
            )

        rows.append(d)
        if not isinstance(result, (tuple, list)) or not result:
            break
        best_move = np.uint16(result[0])
        if int(best_move) == int(NO_MOVE):
            break
        make_move(p_bbs, o_bbs, g_state, best_move)
        game_history.append(int(g_state[4]))
        if not keep_sequence_tt:
            # The next root is the previous search's direct child.  Retaining
            # TT would dominate the comparison; default sequence mode isolates
            # persisted heuristic/correction history instead.
            warm_ctx.transposition_table.fill(0)

    return rows


def run_bench_set(
    fens: list[str],
    nodes: int,
    set_name: str,
    workers: int,
    warmup: bool,
) -> tuple[list[dict], str]:
    if not fens:
        return [], ""

    # Warm JIT once (serial). ThreadPool reuses compiled code; each job has own ctx/TT.
    if warmup:
        print(f"  [{set_name}] warmup JIT ...", flush=True)
        _search_one(fens[0], min(5000, nodes), -1, set_name)
        print(f"  [{set_name}] warmup done", flush=True)

    per: list[dict | None] = [None] * len(fens)
    sample_text = ""
    done = 0
    total = len(fens)

    def _finish(d: dict) -> None:
        nonlocal done, sample_text
        i = d["idx"]
        if i == 0:
            sample_text = d.pop("_sample_text", "")
        else:
            d.pop("_sample_text", None)
        per[i] = d
        done += 1
        with _print_lock:
            print(
                f"  [{set_name} {done}/{total}] nodes={d['nodes']} depth={d['depth']} "
                f"prune/1k={d['hist_prune_per_1k_nodes']:.2f} "
                f"cut1={d['cut_first_pct']:.1f}% {d['elapsed_s']:.1f}s",
                flush=True,
            )

    if workers <= 1:
        for i, fen in enumerate(fens):
            _finish(_search_one(fen, nodes, i, set_name))
    else:
        # ThreadPool: Numba search is nogil → true parallel; independent ctx per job.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_search_one, fen, nodes, i, set_name)
                for i, fen in enumerate(fens)
            ]
            for fut in as_completed(futs):
                _finish(fut.result())

    return [d for d in per if d is not None], sample_text


def run_sequence_bench_set(
    fens: list[str],
    nodes: int,
    set_name: str,
    workers: int,
    warmup: bool,
    sequence_plies: int,
    paired_cold: bool,
    keep_sequence_tt: bool,
) -> tuple[list[dict], str]:
    """Run deterministic per-seed game continuations with one warm context."""
    if not fens:
        return [], ""
    if warmup:
        print(f"  [{set_name}] warmup JIT ...", flush=True)
        _search_one(fens[0], min(5000, nodes), -1, set_name)
        print(f"  [{set_name}] warmup done", flush=True)

    per: list[dict] = []
    sample_text = ""
    done = 0
    total = len(fens)

    def _finish(rows: list[dict]) -> None:
        nonlocal done, sample_text
        if rows and rows[0].get("seed_idx") == 0:
            sample_text = rows[0].pop("_sample_text", "")
        for d in rows:
            d.pop("_sample_text", None)
        per.extend(rows)
        done += 1
        last = rows[-1] if rows else {}
        with _print_lock:
            print(
                f"  [{set_name} seed {done}/{total}] plies={len(rows)} "
                f"last_depth={last.get('depth', 0)} "
                f"cold_delta={last.get('depth_delta', 0):+d}",
                flush=True,
            )

    if workers <= 1:
        for i, fen in enumerate(fens):
            _finish(_search_sequence_seed(
                fen, nodes, i, set_name, sequence_plies, paired_cold,
                keep_sequence_tt,
            ))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _search_sequence_seed,
                    fen, nodes, i, set_name, sequence_plies, paired_cold,
                    keep_sequence_tt,
                )
                for i, fen in enumerate(fens)
            ]
            for fut in as_completed(futs):
                _finish(fut.result())

    per.sort(key=lambda d: (int(d.get("seed_idx", 0)), int(d.get("sequence_ply", 0))))
    return per, sample_text


def summarize(per: list[dict]) -> dict:
    keys = [
        "nodes", "cut_nodes", "cut_first", "cut_first_pct",
        "lmr_try", "lmr_research", "lmr_research_pct",
        # P4 move-ordering source precision and LMR verification quality.
        "order_tt_try", "order_tt_cut", "order_tt_cut_pct",
        "order_good_capture_try", "order_good_capture_cut",
        "order_good_capture_cut_pct",
        "order_good_quiet_try", "order_good_quiet_cut",
        "order_good_quiet_cut_pct",
        "order_bad_capture_try", "order_bad_capture_cut",
        "order_bad_capture_cut_pct",
        "order_bad_quiet_try", "order_bad_quiet_cut",
        "order_bad_quiet_cut_pct",
        "order_other_try", "order_other_cut", "order_other_cut_pct",
        "order_killer1_try", "order_killer1_cut", "order_killer1_cut_pct",
        "order_killer2_try", "order_killer2_cut", "order_killer2_cut_pct",
        "order_counter_try", "order_counter_cut", "order_counter_cut_pct",
        "order_quiet_cut", "order_quiet_cut_rank_sum",
        "order_quiet_cut_avg_rank", "order_nodes_before_cut_sum",
        "order_avg_nodes_before_cut", "order_source_try_total",
        "order_source_cut_total", "order_source_cut_coverage_pct",
        "lmr_r1_try", "lmr_r2_try", "lmr_r3p_try",
        "lmr_r1_fail_high", "lmr_r2_fail_high", "lmr_r3p_fail_high",
        "lmr_r1_fail_high_pct", "lmr_r2_fail_high_pct",
        "lmr_r3p_fail_high_pct",
        "lmr_quiet_try", "lmr_quiet_fail_high", "lmr_quiet_fail_high_pct",
        "lmr_capture_try", "lmr_capture_fail_high",
        "lmr_capture_fail_high_pct",
        "lmr_research_keep", "lmr_research_reject",
        "lmr_research_reject_pct", "lmr_research_deeper",
        "lmr_research_shallower", "lmr_fail_high_unverified",
        "post_lmr_bonus_sample", "post_lmr_malus_sample",
        "post_lmr_unverified_sample",
        "post_lmr_bonus_butterfly_nonneg",
        "post_lmr_malus_butterfly_nonneg",
        "post_lmr_bonus_stat_nonneg", "post_lmr_malus_stat_nonneg",
        "post_lmr_bonus_butterfly_nonneg_pct",
        "post_lmr_malus_butterfly_nonneg_pct",
        "post_lmr_bonus_stat_nonneg_pct",
        "post_lmr_malus_stat_nonneg_pct",
        # Mid / shallow prune rates (per 1k nodes) + raw counts
        "lmp_skip", "lmp_skip_per_1k_nodes",
        "fp_skip", "fp_skip_per_1k_nodes",
        "rfp_cut", "rfp_cut_per_1k_nodes",
        "razor_cut", "razor_cut_per_1k_nodes",
        "nmp_cut", "nmp_cut_per_1k_nodes",
        "tt_cut", "tt_cut_per_1k_nodes",
        "probcut", "probcut_per_1k_nodes",
        "nmp_eligible", "nmp_gate_pass", "nmp_null_try", "nmp_null_fh",
        "nmp_gate_pct", "nmp_fh_pct", "nmp_vfail_pct",
        "hist_quiet_bonus", "hist_quiet_malus", "hist_bonus_malus_ratio",
        "hist_prune", "hist_prune_per_1k_nodes",
        "hist_cap_bonus", "hist_cap_malus",
        "hist_cont_upd", "hist_tt_hit", "hist_fail_low",
        "hist_piece_to_upd", "hist_pawn_upd", "hist_lph_upd",
        # P5 history-prune attribution (raw counts/sums are reweighted below).
        "hist_prune_cont_coalition_decisive",
        "hist_prune_cont_coalition_decisive_pct",
        "hist_prune_noncont_already_prunes",
        "hist_prune_noncont_already_prunes_pct",
        "hist_prune_lmr_eligible", "hist_prune_lmr_eligible_pct",
        "hist_prune_main_gate_near_zero",
        "hist_prune_main_gate_near_zero_pct",
        "hist_prune_cont1_cap_hit", "hist_prune_cont1_cap_rescue",
        "hist_prune_cont1_cap_rescue_pct",
        "hist_prune_cont1_cap_relief_sum",
        "hist_prune_cont1_cap_relief_mean",
        "hist_prune_margin_near", "hist_prune_margin_mid",
        "hist_prune_margin_far", "hist_prune_margin_sum",
        "hist_prune_margin_mean", "hist_prune_depth_sum",
        "hist_prune_depth_mean", "hist_prune_move_index_sum",
        "hist_prune_move_index_mean",
        # P1 main quiet SEE / legality / geometry classification
        "main_quiet_see_precheck", "main_quiet_pre_pin_illegal",
        "main_quiet_pre_king_illegal", "main_quiet_see_pass",
        "main_quiet_post_pin_illegal", "main_quiet_post_king_illegal",
        "main_quiet_fallback_try", "main_quiet_fallback_reject",
        "main_geometry_node", "main_score_geometry_rebuild",
        "main_quiet_pre_illegal_per_see", "main_quiet_post_illegal_per_see_pass",
        "main_quiet_fallback_reject_pct", "main_score_geometry_per_node",
        # P3 qsearch cap / main in-check / TT deep-verify diagnostics.
        "q_cap_noncheck", "q_cap_check", "q_cap_total", "q_cap_forcing",
        "q_cap_forcing_moves", "q_cap_forcing_pct",
        "q_cap_forcing_moves_per_hit",
        "main_in_check_nodes", "main_in_check_pick_try",
        "main_in_check_pre_illegal", "main_in_check_pre_pass",
        "main_in_check_fallback", "main_in_check_pre_illegal_pct",
        "main_in_check_fallback_pct",
        "tt_verify_try", "tt_verify_pass", "tt_verify_reject",
        "tt_verify_saved_cut", "tt_verify_skip", "tt_verify_pass_pct",
        "tt_verify_reject_pct", "tt_verify_skip_pct",
        "tt_verify_saved_cut_pct",
        # P2 aspiration-window diagnostics.
        "aspiration_iterations", "aspiration_fail_low", "aspiration_fail_high",
        "aspiration_research", "aspiration_max_delta", "aspiration_wasted_nodes",
        "aspiration_fail_low_per_iteration", "aspiration_fail_high_per_iteration",
        "aspiration_research_per_iteration", "aspiration_research_search_pct",
        "aspiration_wasted_nodes_pct",
        "depth", "elapsed_s",
    ]
    for layer in _HP_LAYER_NAMES:
        keys.extend([
            f"hist_prune_{layer}_negative",
            f"hist_prune_{layer}_negative_pct",
            f"hist_prune_{layer}_negative_abs_sum",
            f"hist_prune_{layer}_negative_mean_abs",
            f"hist_prune_{layer}_decisive",
            f"hist_prune_{layer}_decisive_pct",
            f"hist_prune_{layer}_dominant",
            f"hist_prune_{layer}_dominant_pct",
        ])
    summary = mean_dict_list(per, keys)

    # Percentage fields emitted by one search are useful for its text trace, but
    # averaging those percentages would give a low-event position the same weight
    # as a high-event position.  Rebuild aggregate rates from the mean raw counts;
    # with one result per position, mean(num) / mean(den) == sum(num) / sum(den).
    def ratio_pct(num_key: str, den_key: str) -> float:
        den = summary.get(den_key, 0.0)
        return 100.0 * summary.get(num_key, 0.0) / den if den else 0.0

    summary["cut_first_pct"] = ratio_pct("cut_first", "cut_nodes")
    summary["lmr_research_pct"] = ratio_pct("lmr_research", "lmr_try")

    for source in (
        "tt", "good_capture", "good_quiet", "bad_capture", "bad_quiet", "other",
    ):
        summary[f"order_{source}_cut_pct"] = ratio_pct(
            f"order_{source}_cut", f"order_{source}_try"
        )
    for source in ("killer1", "killer2", "counter"):
        summary[f"order_{source}_cut_pct"] = ratio_pct(
            f"order_{source}_cut", f"order_{source}_try"
        )
    summary["order_source_cut_coverage_pct"] = ratio_pct(
        "order_source_cut_total", "cut_nodes"
    )
    quiet_cuts = summary.get("order_quiet_cut", 0.0)
    summary["order_quiet_cut_avg_rank"] = (
        summary.get("order_quiet_cut_rank_sum", 0.0) / quiet_cuts
        if quiet_cuts else 0.0
    )
    cut_nodes = summary.get("cut_nodes", 0.0)
    summary["order_avg_nodes_before_cut"] = (
        summary.get("order_nodes_before_cut_sum", 0.0) / cut_nodes
        if cut_nodes else 0.0
    )

    for bucket in ("r1", "r2", "r3p"):
        summary[f"lmr_{bucket}_fail_high_pct"] = ratio_pct(
            f"lmr_{bucket}_fail_high", f"lmr_{bucket}_try"
        )
    summary["lmr_quiet_fail_high_pct"] = ratio_pct(
        "lmr_quiet_fail_high", "lmr_quiet_try"
    )
    summary["lmr_capture_fail_high_pct"] = ratio_pct(
        "lmr_capture_fail_high", "lmr_capture_try"
    )
    verified_research = (
        summary.get("lmr_research_keep", 0.0)
        + summary.get("lmr_research_reject", 0.0)
    )
    summary["lmr_research_reject_pct"] = (
        100.0 * summary.get("lmr_research_reject", 0.0) / verified_research
        if verified_research else 0.0
    )

    for outcome in ("bonus", "malus"):
        samples = f"post_lmr_{outcome}_sample"
        summary[f"post_lmr_{outcome}_butterfly_nonneg_pct"] = ratio_pct(
            f"post_lmr_{outcome}_butterfly_nonneg", samples
        )
        summary[f"post_lmr_{outcome}_stat_nonneg_pct"] = ratio_pct(
            f"post_lmr_{outcome}_stat_nonneg", samples
        )

    prune_mean = summary.get("hist_prune", 0.0)
    for layer in _HP_LAYER_NAMES:
        neg = summary.get(f"hist_prune_{layer}_negative", 0.0)
        summary[f"hist_prune_{layer}_negative_pct"] = (
            100.0 * neg / prune_mean if prune_mean else 0.0
        )
        summary[f"hist_prune_{layer}_negative_mean_abs"] = (
            summary.get(f"hist_prune_{layer}_negative_abs_sum", 0.0) / neg
            if neg else 0.0
        )
        for suffix in ("decisive", "dominant"):
            raw = summary.get(f"hist_prune_{layer}_{suffix}", 0.0)
            summary[f"hist_prune_{layer}_{suffix}_pct"] = (
                100.0 * raw / prune_mean if prune_mean else 0.0
            )
    for raw_key in (
        "hist_prune_cont_coalition_decisive",
        "hist_prune_noncont_already_prunes",
        "hist_prune_lmr_eligible",
        "hist_prune_main_gate_near_zero",
    ):
        summary[f"{raw_key}_pct"] = (
            100.0 * summary.get(raw_key, 0.0) / prune_mean if prune_mean else 0.0
        )
    summary["hist_prune_cont1_cap_rescue_pct"] = ratio_pct(
        "hist_prune_cont1_cap_rescue", "hist_prune_cont1_cap_hit"
    )
    cap_hits = summary.get("hist_prune_cont1_cap_hit", 0.0)
    summary["hist_prune_cont1_cap_relief_mean"] = (
        summary.get("hist_prune_cont1_cap_relief_sum", 0.0) / cap_hits
        if cap_hits else 0.0
    )
    for mean_key, sum_key in (
        ("hist_prune_margin_mean", "hist_prune_margin_sum"),
        ("hist_prune_depth_mean", "hist_prune_depth_sum"),
        ("hist_prune_move_index_mean", "hist_prune_move_index_sum"),
    ):
        summary[mean_key] = (
            summary.get(sum_key, 0.0) / prune_mean if prune_mean else 0.0
        )

    for tname in ("piece_to", "butterfly", "capture", "continuation", "pawn"):
        mean_abs = [d["tables"][tname]["mean_abs"] for d in per if "tables" in d]
        sat = [d["tables"][tname]["sat_pct"] for d in per if "tables" in d]
        summary[f"table_{tname}_mean_abs"] = float(np.mean(mean_abs)) if mean_abs else 0.0
        summary[f"table_{tname}_sat_pct"] = float(np.mean(sat)) if sat else 0.0
    summary["n_positions"] = float(len(per))
    return summary


def summarize_sequence_pairs(per: list[dict]) -> dict:
    """Aggregate warm-vs-cold pairs by continuation ply."""
    out = {}
    plies = sorted({int(d["sequence_ply"]) for d in per if "cold_depth" in d})
    for ply in plies:
        rows = [d for d in per if int(d.get("sequence_ply", -1)) == ply and "cold_depth" in d]
        if not rows:
            continue
        warm_prunes = sum(int(d.get("hist_prune", 0)) for d in rows)
        cold_prunes = sum(int(d.get("cold_hist_prune", 0)) for d in rows)
        warm_cont = sum(int(d.get("hist_prune_cont_coalition_decisive", 0)) for d in rows)
        cold_cont = sum(int(d.get("cold_hist_prune_cont_coalition_decisive", 0)) for d in rows)
        depth_delta = [int(d.get("depth_delta", 0)) for d in rows]
        out[str(ply)] = {
            "n_positions": len(rows),
            "warm_nodes": float(np.mean([d["nodes"] for d in rows])),
            "cold_nodes": float(np.mean([d["cold_nodes"] for d in rows])),
            "warm_depth": float(np.mean([d["depth"] for d in rows])),
            "cold_depth": float(np.mean([d["cold_depth"] for d in rows])),
            "depth_up": sum(v > 0 for v in depth_delta),
            "depth_same": sum(v == 0 for v in depth_delta),
            "depth_down": sum(v < 0 for v in depth_delta),
            "best_move_same_pct": 100.0 * float(np.mean([
                int(d.get("best_move_same", 0)) for d in rows
            ])),
            "score_abs_delta_mean": float(np.mean([
                abs(int(d.get("score_delta", 0))) for d in rows
            ])),
            "warm_hist_prune": warm_prunes,
            "cold_hist_prune": cold_prunes,
            "warm_cont_required_pct": 100.0 * warm_cont / warm_prunes if warm_prunes else 0.0,
            "cold_cont_required_pct": 100.0 * cold_cont / cold_prunes if cold_prunes else 0.0,
        }
    return out


def format_summary_block(title: str, summary: dict) -> list[str]:
    lines = [f"## {title} (n={int(summary.get('n_positions', 0))})", "  Aggregate (mean over positions)"]
    for k, v in summary.items():
        if k == "n_positions":
            continue
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}" if abs(v) < 1000 else f"  {k}: {v:.1f}")
        else:
            lines.append(f"  {k}: {v}")
    return lines


def format_compact_rows(per: list[dict], limit: int = 20) -> list[str]:
    lines = []
    show = per if len(per) <= limit else per[:limit]
    for d in show:
        seq = (
            f" seed={d.get('seed_idx')} ply={d.get('sequence_ply')}"
            if "sequence_ply" in d else ""
        )
        pair = (
            f" cold_d={d.get('cold_depth', 0)} delta={d.get('depth_delta', 0):+d}"
            if "cold_depth" in d else ""
        )
        lines.append(
            f"  [{d.get('set', '?')}:{d.get('idx', '?')}{seq}] "
            f"hprune/1k={d['hist_prune_per_1k_nodes']:.2f} "
            f"lmp/1k={d.get('lmp_skip_per_1k_nodes', 0):.1f} "
            f"rfp/1k={d.get('rfp_cut_per_1k_nodes', 0):.2f} "
            f"fp/1k={d.get('fp_skip_per_1k_nodes', 0):.2f} "
            f"nmp/1k={d.get('nmp_cut_per_1k_nodes', 0):.2f} "
            f"cut1={d['cut_first_pct']:.1f}% lmr_re={d['lmr_research_pct']:.1f}% "
            f"depth={d.get('depth', 0)}{pair}"
        )
    if len(per) > limit:
        lines.append(f"  ... ({len(per) - limit} more omitted; see JSON)")
    return lines


def main():
    global _RUNTIME_OVERRIDES, _RUNTIME_OVERRIDE_SCALE
    ap = argparse.ArgumentParser(description="Classical history diagnostics bench")
    ap.add_argument("--nodes", type=int, default=30000,
                    help="Fixed nodes per position (default: 30000; formal search protocol)")
    ap.add_argument("--positions", type=int, default=500,
                    help="Max positions per set (default 500)")
    ap.add_argument("--puzzles", type=Path, default=DEFAULT_PUZZLES)
    ap.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    ap.add_argument("--single", type=Path, default=None,
                    help="Legacy: single EPD only (no dual split)")
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("-c", "--workers", type=int, default=1,
                    help="Parallel workers (independent ctx per pos; default 1)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument(
        "--sequence-plies", type=int, default=1,
        help="Follow each seed's best move for N plies with one reused context (default 1/cold)",
    )
    ap.add_argument(
        "--paired-cold", action="store_true",
        help="In sequence mode, search every reached position again with a fresh context",
    )
    ap.add_argument(
        "--keep-sequence-tt", action="store_true",
        help="Retain TT between sequence plies (default clears TT to isolate heuristic history)",
    )
    ap.add_argument("--json", action="store_true", help="Also write JSON next to text report")
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "runtime-only override for diagnostic contexts; repeat for each "
            "parameter (does not modify constants.py)"
        ),
    )
    ap.add_argument(
        "--param-scale-mode",
        choices=("integer", "registry"),
        default="integer",
        help="grid used for --param values (default: integer)",
    )
    args = ap.parse_args()
    if args.sequence_plies < 1:
        ap.error("--sequence-plies must be >= 1")

    overrides: dict[str, int] = {}
    for raw in args.param:
        if "=" not in raw:
            ap.error(f"--param must use NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            ap.error(f"--param has an empty name: {raw!r}")
        if name in overrides:
            ap.error(f"duplicate --param override: {name}")
        try:
            spec = get_spec(name)
        except KeyError:
            ap.error(f"unknown --param override: {name}")
        if not spec.is_runtime:
            ap.error(f"--param override must be runtime-backed: {name}")
        try:
            overrides[name] = int(value.strip())
        except ValueError:
            ap.error(f"--param value must be an integer: {raw!r}")
    _RUNTIME_OVERRIDES = overrides
    _RUNTIME_OVERRIDE_SCALE = args.param_scale_mode

    out = args.out or (DEFAULT_OUT_DIR / f"hist_diag_{args.tag}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"hist_diag_bench: nodes={args.nodes} positions/set={args.positions} "
        f"workers={args.workers} sequence_plies={args.sequence_plies} "
        f"paired_cold={args.paired_cold} keep_sequence_tt={args.keep_sequence_tt} "
        f"tag={args.tag} overrides={_RUNTIME_OVERRIDES or 'none'}",
        flush=True,
    )
    print(
        f"constants: HISTORY_MODE={'PHASE1A_DIV2' if HISTORY_COMPAT_1A else 'FULL'} "
        f"BF_MAX={HISTORY_MAX_BUTTERFLY} CAP={HISTORY_MAX_CAPTURE} CONT={HISTORY_MAX_CONTINUATION} "
        f"PRUNE_THR={PRUNING_HISTORY_THRESHOLD} LMR_BASE={LMR_BASE_OFFSET} "
        f"LMR_HIST={LMR_HISTORY_SCALE} LMP={LMP_SCALE_PERCENT}",
        flush=True,
    )

    t_all = time.time()
    results: dict[str, list[dict]] = {}
    samples: dict[str, str] = {}

    def run_selected(fens: list[str], set_name: str, warmup: bool):
        if args.sequence_plies > 1 or args.paired_cold:
            return run_sequence_bench_set(
                fens, args.nodes, set_name, args.workers,
                warmup=warmup, sequence_plies=args.sequence_plies,
                paired_cold=args.paired_cold,
                keep_sequence_tt=args.keep_sequence_tt,
            )
        return run_bench_set(
            fens, args.nodes, set_name, args.workers, warmup=warmup,
        )

    if args.single is not None:
        fens = load_fens(args.single, args.positions)
        print(f"\n=== single ({len(fens)}) from {args.single} ===", flush=True)
        per, sample = run_selected(fens, "single", warmup=not args.no_warmup)
        results["single"] = per
        samples["single"] = sample
    else:
        p_fens = load_fens(args.puzzles, args.positions)
        o_fens = load_fens(args.openings, args.positions)
        print(f"\n=== puzzles ({len(p_fens)}) from {args.puzzles} ===", flush=True)
        p_per, p_sample = run_selected(
            p_fens, "puzzle", warmup=not args.no_warmup,
        )
        results["puzzle"] = p_per
        samples["puzzle"] = p_sample

        print(f"\n=== openings ({len(o_fens)}) from {args.openings} ===", flush=True)
        o_per, o_sample = run_selected(o_fens, "opening", warmup=False)
        results["opening"] = o_per
        samples["opening"] = o_sample

        results["combined"] = results["puzzle"] + results["opening"]

    summaries = {name: summarize(per) for name, per in results.items()}
    sequence_summaries = {
        name: summarize_sequence_pairs(per)
        for name, per in results.items()
        if name != "combined" and any("sequence_ply" in d for d in per)
    }
    elapsed_all = time.time() - t_all

    lines = [
        f"# History diagnostics bench  tag={args.tag}",
        f"nodes={args.nodes}  workers={args.workers}  wall_s={elapsed_all:.1f}",
        f"sequence_plies={args.sequence_plies}  paired_cold={args.paired_cold}  "
        f"keep_sequence_tt={args.keep_sequence_tt}",
        f"puzzles={args.puzzles if args.single is None else 'n/a'}  "
        f"openings={args.openings if args.single is None else args.single}",
        f"constants: HISTORY_MODE={'PHASE1A_DIV2' if HISTORY_COMPAT_1A else 'FULL'} "
        f"HISTORY_MAX_BUTTERFLY={HISTORY_MAX_BUTTERFLY}",
        f"  PRUNING_HISTORY_THRESHOLD={PRUNING_HISTORY_THRESHOLD} LMR_BASE_OFFSET={LMR_BASE_OFFSET}",
        f"  LMR_HISTORY_SCALE={LMR_HISTORY_SCALE} LMP_SCALE_PERCENT={LMP_SCALE_PERCENT}",
        f"  HISTORY_MAX_MAIN={HISTORY_MAX_MAIN} CAP={HISTORY_MAX_CAPTURE} "
        f"CONT={HISTORY_MAX_CONTINUATION} PAWN={HISTORY_MAX_PAWN}",
        f"  runtime_overrides={_RUNTIME_OVERRIDES or 'none'} "
        f"scale={_RUNTIME_OVERRIDE_SCALE}",
        "",
    ]

    # Prefer order: puzzle, opening, combined (or single)
    order = [k for k in ("puzzle", "opening", "combined", "single") if k in summaries]
    for name in order:
        title = {
            "puzzle": "PUZZLES only",
            "opening": "OPENINGS only",
            "combined": "COMBINED (puzzles + openings)",
            "single": "SINGLE set",
        }[name]
        lines.extend(format_summary_block(title, summaries[name]))
        lines.append("")

    if sequence_summaries:
        lines.append("## Warm-sequence vs fresh-context pairs")
        for name, by_ply in sequence_summaries.items():
            lines.append(f"### {name}")
            for ply, row in by_ply.items():
                lines.append(
                    f"  ply={ply} n={row['n_positions']} "
                    f"depth warm/cold={row['warm_depth']:.3f}/{row['cold_depth']:.3f} "
                    f"up/same/down={row['depth_up']}/{row['depth_same']}/{row['depth_down']} "
                    f"best_same={row['best_move_same_pct']:.1f}% "
                    f"score_abs_delta={row['score_abs_delta_mean']:.2f} "
                    f"hist_prune warm/cold={row['warm_hist_prune']}/{row['cold_hist_prune']} "
                    f"cont_required={row['warm_cont_required_pct']:.1f}%/"
                    f"{row['cold_cont_required_pct']:.1f}%"
                )
        lines.append("")

    lines.extend([
        "## Interpretation hints",
        "  hist_prune_per_1k_nodes: higher → more history quiet pruning (E3/E12/E21/E23)",
        "  lmp_skip_per_1k_nodes: higher → later moves skipped earlier (LMP scale axis)",
        "  rfp_cut_per_1k_nodes: reverse futility fail-highs (shallow static margin)",
        "  fp_skip_per_1k_nodes: futility quiet skips (shallow)",
        "  razor_cut_per_1k_nodes: razoring (depth==1 only; should stay rare deeper)",
        "  nmp_cut / nmp_gate_pct / nmp_fh_pct: null-move funnel health",
        "  probcut_per_1k_nodes: ProbCut rarity; near 0 is normal on HCE",
        "  tt_cut_per_1k_nodes: TT cutoff density",
        "  table_*_sat_pct: high → consider raising that ceiling (E10 axis)",
        "  table_*_mean_abs low after many updates → aging too aggressive (E11 axis)",
        "  hist_bonus_malus_ratio: extreme → asymmetric update scales",
        "  lmr_research_pct high → LMR (incl. hist scale) too aggressive",
        "  cut_first_pct low → ordering weak (weights / ceiling)",
        "  order_*_cut_pct → source precision; compare killer/counter with ordinary quiets",
        "  lmr_research_reject_pct → reduced fail-highs overturned by verification",
        "  post_lmr_*_sample → counterfactual verified continuation-history learning events",
        "  butterfly_nonneg bonus vs malus gap → whether from-to history adds LMR discrimination",
        "  hist_prune_*_decisive_pct → removing that one layer would rescue the move",
        "  hist_prune_cont_coalition_decisive_pct → all continuation terms jointly caused the prune",
        "  sequence ply>0 warm/cold deltas isolate persisted search-context history",
        "  aspiration_research / iteration high → initial window too narrow; inspect by-depth JSON",
        "  aspiration_wasted_nodes_pct → nodes spent on failed aspiration windows (re-search overhead)",
        "  q_cap_forcing_pct high → cap often truncates pseudo captures/evasions; confirm legality before widening",
        "  main_in_check_pre_illegal_pct high → picker had avoidable pseudo evasions (P3 pre-score compaction)",
        "  tt_verify_reject / skip nonzero → inspect TT deep verification before changing SF18 protection",
        "  Compare puzzle vs opening for mid/shallow prune mix (n10k regime)",
        "",
    ])

    for name in order:
        if name in samples and samples[name]:
            lines.append(f"## Sample ({name}) first FEN full dump")
            lines.append(samples[name])
            lines.append("")

    lines.append("## Per-position compact (first 20 per set)")
    for name in order:
        if name == "combined":
            continue
        lines.append(f"### {name}")
        lines.extend(format_compact_rows(results[name], limit=20))
        lines.append("")

    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"Wrote {out}  (wall {elapsed_all:.1f}s)", flush=True)

    if args.json:
        jpath = out.with_suffix(".json")
        payload = {
            "summaries": summaries,
            "sequence_summaries": sequence_summaries,
            "per_position": {k: v for k, v in results.items() if k != "combined"},
            "meta": {
                "nodes": args.nodes,
                "workers": args.workers,
                "sequence_plies": args.sequence_plies,
                "paired_cold": args.paired_cold,
                "keep_sequence_tt": args.keep_sequence_tt,
                "tag": args.tag,
                "runtime_overrides": dict(_RUNTIME_OVERRIDES),
                "runtime_override_scale": _RUNTIME_OVERRIDE_SCALE,
                "wall_s": elapsed_all,
            },
        }
        jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {jpath}", flush=True)


if __name__ == "__main__":
    main()
