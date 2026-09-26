#!/usr/bin/env python3
"""
Parse per-move search stats from tournament PGN comments and compare two engines.

Comment formats accepted (field order is flexible; extra fields are ignored):
  { d=11, eval=+0.07, n=332609, t=528ms }
  { d=11, eval=+0.07, n=332609, t=528ms, nps=629941, hashfull=12 }
  Mate: eval=#3 or eval=#-2

Main stats (depth / nodes / time / NPS) exclude:
  1. Moves with mate score (eval starts with '#') — reported separately
  2. In drawn games (Result 1/2-1/2), moves with depth >= draw_depth_cap (default 40)

Game-level WDL / Elo: 統計數據.py

Usage (repo root):
  python tournament_analysis/parse_search_stats.py
  python tournament_analysis/parse_search_stats.py --name1 New --name2 Old
  python tournament_analysis/parse_search_stats.py --draw-depth-cap 40
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats

HERE = Path(__file__).resolve().parent
DEFAULT_PGN = HERE / "tournament_results.pgn"
DEFAULT_DRAW_DEPTH_CAP = 40

# Match the move/comment envelope only. Search fields are parsed separately so
# adding nps/hashfull or changing field order cannot invalidate the whole move.
MOVE_COMMENT_RE = re.compile(
    r"(?<![\d.])(?P<move_num>\d+)\.(?P<black>\.\.)?\s+"
    r"(?P<san>[^\s{}]+)\s*\{(?P<comment>[^{}]*)\}"
)
COMMENT_FIELD_RE = re.compile(
    r"(?:^|,)\s*(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(?P<value>[^,}]*)"
)


def _comment_int(value: str, suffix: str = "") -> Optional[int]:
    text = (value or "").strip().lower().replace("_", "")
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)].strip()
    # Thousands separators conflict with the comment field delimiter and are
    # therefore intentionally unsupported.
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def parse_search_comment(comment: str) -> Optional[Tuple[int, str, int, int, Optional[int]]]:
    """Return depth/eval/nodes/time_ms/nps from one PGN comment.

    Required fields accept their compact tournament names and readable aliases.
    Unknown fields are ignored. NPS is optional and is later derived from
    nodes/time for legacy PGNs.
    """
    fields = {
        match.group("key").strip().lower(): match.group("value").strip()
        for match in COMMENT_FIELD_RE.finditer(comment)
    }
    depth = _comment_int(fields.get("d", fields.get("depth", "")))
    nodes = _comment_int(fields.get("n", fields.get("nodes", "")))
    time_ms = _comment_int(fields.get("t", fields.get("time", "")), suffix="ms")
    evaluation = fields.get("eval", fields.get("score", "")).strip()
    nps = _comment_int(fields.get("nps", "")) if "nps" in fields else None
    if depth is None or nodes is None or time_ms is None or not evaluation:
        return None
    return depth, evaluation, nodes, time_ms, nps


def empty_stats() -> Dict[str, List]:
    return {
        "depth": [], "nodes": [], "time": [], "nps": [],
        "move_num": [], "color": [],
    }


def empty_counts() -> Dict[str, int]:
    return {
        "comments_seen": 0,
        "malformed_comments": 0,
        "raw_moves": 0,
        "main_moves": 0,
        "mate_moves": 0,
        "draw_deep_excluded": 0,
    }


def is_mate_eval(ev: str) -> bool:
    s = (ev or "").strip()
    return s.startswith("#")


def parse_pgn(
    file_path: str | Path,
    name1: str,
    name2: str,
    draw_depth_cap: int = DEFAULT_DRAW_DEPTH_CAP,
):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: PGN file not found: {path}")
        return None

    content = path.read_text(encoding="utf-8", errors="replace")
    game_blocks = content.split('[Event "')
    if len(game_blocks) <= 1:
        game_blocks = re.split(r"\[Event ", content)

    new_stats = empty_stats()
    old_stats = empty_stats()
    new_mate = empty_stats()
    old_mate = empty_stats()
    by_color = defaultdict(lambda: {"depth": [], "nodes": [], "time": [], "nps": []})
    counts = {
        name1: empty_counts(),
        name2: empty_counts(),
    }

    games_parsed = 0
    draw_games = 0

    def record_main(
        eng: str,
        col: str,
        depth: int,
        nodes: int,
        time_ms: int,
        move_num: int,
        nps: Optional[int],
    ) -> None:
        effective_nps = nps if nps is not None and nps > 0 else (
            nodes / (time_ms / 1000.0) if time_ms > 0 else 0.0
        )
        bucket = new_stats if eng == name1 else old_stats
        bucket["depth"].append(depth)
        bucket["nodes"].append(nodes)
        bucket["time"].append(time_ms)
        bucket["nps"].append(effective_nps)
        bucket["move_num"].append(move_num)
        bucket["color"].append(col)
        bc = by_color[(eng, col)]
        bc["depth"].append(depth)
        bc["nodes"].append(nodes)
        bc["time"].append(time_ms)
        if effective_nps > 0:
            bc["nps"].append(effective_nps)
        counts[eng]["main_moves"] += 1

    def record_mate(
        eng: str,
        col: str,
        depth: int,
        nodes: int,
        time_ms: int,
        move_num: int,
        nps: Optional[int],
    ) -> None:
        bucket = new_mate if eng == name1 else old_mate
        bucket["depth"].append(depth)
        bucket["nodes"].append(nodes)
        bucket["time"].append(time_ms)
        bucket["nps"].append(
            nps if nps is not None and nps > 0 else (
                nodes / (time_ms / 1000.0) if time_ms > 0 else 0.0
            )
        )
        bucket["move_num"].append(move_num)
        bucket["color"].append(col)
        counts[eng]["mate_moves"] += 1

    for block in game_blocks:
        if not block.strip():
            continue
        white_match = re.search(r'\[White\s+"([^"]+)"\]', block)
        black_match = re.search(r'\[Black\s+"([^"]+)"\]', block)
        if not white_match or not black_match:
            continue
        white_player = white_match.group(1)
        black_player = black_match.group(1)
        if white_player not in (name1, name2) or black_player not in (name1, name2):
            continue
        games_parsed += 1

        result_m = re.search(r'\[Result\s+"([^"]+)"\]', block)
        result = result_m.group(1) if result_m else "*"
        is_draw = result == "1/2-1/2"
        if is_draw:
            draw_games += 1

        moves: List[Tuple[str, str, int, str, int, int, int, Optional[int]]] = []
        for move_match in MOVE_COMMENT_RE.finditer(block):
            is_black = move_match.group("black") is not None
            eng = black_player if is_black else white_player
            col = "B" if is_black else "W"
            if eng not in counts:
                continue
            counts[eng]["comments_seen"] += 1
            parsed = parse_search_comment(move_match.group("comment"))
            if parsed is None:
                counts[eng]["malformed_comments"] += 1
                continue
            depth, ev, nodes, time_ms, nps = parsed
            moves.append(
                (
                    eng,
                    col,
                    int(move_match.group("move_num")),
                    ev,
                    depth,
                    nodes,
                    time_ms,
                    nps,
                )
            )

        for eng, col, move_num, ev, depth, nodes, time_ms, nps in moves:
            if eng not in counts:
                continue
            counts[eng]["raw_moves"] += 1

            if is_mate_eval(ev):
                record_mate(eng, col, depth, nodes, time_ms, move_num, nps)
                continue

            if is_draw and depth >= draw_depth_cap:
                counts[eng]["draw_deep_excluded"] += 1
                continue

            record_main(eng, col, depth, nodes, time_ms, move_num, nps)

    print(
        f"Successfully parsed {games_parsed} games "
        f"(draws={draw_games}, draw_depth_cap={draw_depth_cap})."
    )
    meta = {
        "games_parsed": games_parsed,
        "draw_games": draw_games,
        "draw_depth_cap": draw_depth_cap,
        "counts": counts,
    }
    return new_stats, old_stats, by_color, new_mate, old_mate, meta


def plot_line_metric(
    new_moves, new_means, new_sems, old_moves, old_means, old_sems,
    title, xlabel, ylabel, filename, name1, name2, use_log=False,
):
    plt.figure(figsize=(10, 5.5))
    plt.plot(new_moves, new_means, color="#1f77b4", label=f"{name1} (Avg)", linewidth=2.0, marker="o", markersize=4)
    if len(new_sems) > 0:
        plt.fill_between(new_moves, new_means - new_sems, new_means + new_sems, color="#1f77b4", alpha=0.15, label=f"{name1} SEM")
    plt.plot(old_moves, old_means, color="#ff7f0e", label=f"{name2} (Avg)", linewidth=2.0, marker="s", markersize=4)
    if len(old_sems) > 0:
        plt.fill_between(old_moves, old_means - old_sems, old_means + old_sems, color="#ff7f0e", alpha=0.15, label=f"{name2} SEM")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    if use_log:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def get_metric_per_move(move_nums, values, max_move=100, min_samples=5):
    if len(move_nums) == 0:
        return np.array([]), np.array([]), np.array([])
    unique_moves = np.unique(move_nums)
    unique_moves = unique_moves[unique_moves <= max_move]
    m_list, mean_list, sem_list = [], [], []
    for m in unique_moves:
        samples = values[move_nums == m]
        if len(samples) >= min_samples:
            m_list.append(m)
            mean_list.append(np.mean(samples))
            sem_list.append(np.std(samples, ddof=1) / np.sqrt(len(samples)) if len(samples) > 1 else 0.0)
    return np.array(m_list), np.array(mean_list), np.array(sem_list)


def append_exclusion_and_mate_section(
    report: List[str],
    meta: Dict[str, Any],
    new_mate: Dict[str, List],
    old_mate: Dict[str, List],
    name1: str,
    name2: str,
) -> None:
    cap = meta["draw_depth_cap"]
    counts = meta["counts"]
    report.append("【過濾規則】")
    report.append("  * 主統計排除: eval 為將殺 (#…) 的著法")
    report.append(f"  * 主統計排除: 和棋局 (1/2-1/2) 且 depth>={cap} 的著法")
    report.append(
        f"  * 對局: {meta['games_parsed']}  (其中和棋 {meta['draw_games']})\n"
    )

    report.append("【排除與將殺樣本】")
    for eng in (name1, name2):
        c = counts.get(eng, empty_counts())
        report.append(
            f"  * {eng}: comments={c['comments_seen']} malformed={c['malformed_comments']}  "
            f"raw={c['raw_moves']}  main={c['main_moves']}  "
            f"mate={c['mate_moves']}  draw_deep(d>={cap})={c['draw_deep_excluded']}"
        )
    report.append("")

    report.append("【將殺著法額外統計 (eval=#…)】")
    for eng, mate in ((name1, new_mate), (name2, old_mate)):
        d = np.array(mate["depth"], dtype=float) if mate["depth"] else np.array([])
        n = np.array(mate["nodes"], dtype=float) if mate["nodes"] else np.array([])
        t = np.array(mate["time"], dtype=float) if mate["time"] else np.array([])
        if len(d) == 0:
            report.append(f"  * {eng}: (no mate moves)")
            continue
        report.append(
            f"  * {eng}: n={len(d)}  mean_d={d.mean():.2f} med_d={np.median(d):.0f} "
            f"max_d={d.max():.0f}  mean_n={n.mean():.0f} med_n={np.median(n):.0f}  "
            f"mean_t={t.mean():.1f}ms"
        )
        deep_mate = float(np.mean(d >= 100) * 100)
        report.append(f"      depth>=100: {deep_mate:.1f}%  depth>=40: {float(np.mean(d >= 40)*100):.1f}%")
    report.append("")


def append_by_color_section(report: List[str], by_color, name1: str, name2: str) -> None:
    report.append("【5. 分執色搜尋統計 (engine × color, 主統計樣本)】")
    for eng in (name1, name2):
        for col in ("W", "B"):
            s = by_color.get((eng, col))
            if not s or not s["depth"]:
                report.append(f"  {eng} {col}: (no moves)")
                continue
            d = np.array(s["depth"], dtype=float)
            n = np.array(s["nodes"], dtype=float)
            nps = np.array(s["nps"], dtype=float) if s["nps"] else np.array([])
            deep = float(np.mean(d >= 14) * 100)
            line = (
                f"  {eng} {col}: moves={len(d)} mean_d={d.mean():.2f} med_d={np.median(d):.0f} "
                f"d>=14={deep:.1f}% mean_n={n.mean():.0f}"
            )
            if len(nps):
                line += f" mean_nps={nps.mean():.0f}"
            report.append(line)
    report.append("")


def append_depth_hist_and_ratios(report: List[str], new_depths, new_nodes, old_depths, old_nodes, name1, name2) -> None:
    report.append("【6. 深度直方圖 (%, 主統計樣本)】")
    bins = [(1, 10), (11, 12), (13, 13), (14, 15), (16, 20), (21, 39), (40, 128)]
    for eng, darr in ((name1, new_depths), (name2, old_depths)):
        if len(darr) == 0:
            continue
        parts = []
        for a, b in bins:
            parts.append(f"{a}-{b}:{(np.mean((darr >= a) & (darr <= b)) * 100):.1f}%")
        report.append(f"  {eng}: " + " ".join(parts))
        report.append(
            f"    mean_d={darr.mean():.2f} med={np.median(darr):.0f} "
            f"d>=14={(darr >= 14).mean() * 100:.1f}%"
        )
    report.append("")

    report.append("【7. 同深度平均節點比 (min 20 samples each)】")
    any_row = False
    for d in range(6, 25):
        nn = new_nodes[new_depths == d]
        on = old_nodes[old_depths == d]
        if len(nn) >= 20 and len(on) >= 20:
            any_row = True
            ratio = nn.mean() / on.mean() if on.mean() > 0 else float("nan")
            report.append(
                f"  d={d}: {name1} {nn.mean():.0f} (n={len(nn)})  "
                f"{name2} {on.mean():.0f} (n={len(on)})  ratio={ratio:.3f}"
            )
    if not any_row:
        report.append("  (insufficient samples)")
    report.append("")


def analyze_and_plot(
    new_stats,
    old_stats,
    by_color,
    new_mate,
    old_mate,
    meta,
    output_dir,
    name1,
    name2,
):
    os.makedirs(output_dir, exist_ok=True)

    new_depths = np.array(new_stats["depth"])
    new_nodes = np.array(new_stats["nodes"])
    new_times = np.array(new_stats["time"])
    new_move_nums = np.array(new_stats.get("move_num", []))

    old_depths = np.array(old_stats["depth"])
    old_nodes = np.array(old_stats["nodes"])
    old_times = np.array(old_stats["time"])
    old_move_nums = np.array(old_stats.get("move_num", []))

    if len(new_depths) == 0 or len(old_depths) == 0:
        print(
            f"Error: Not enough move data after filters. "
            f"{name1} main moves: {len(new_depths)}, {name2} main moves: {len(old_depths)}"
        )
        return

    new_nps = np.array(new_stats.get("nps", []), dtype=float)
    old_nps = np.array(old_stats.get("nps", []), dtype=float)
    if len(new_nps) != len(new_nodes):
        new_nps = np.where(new_times > 0, new_nodes / (new_times / 1000.0), 0)
    if len(old_nps) != len(old_nodes):
        old_nps = np.where(old_times > 0, old_nodes / (old_times / 1000.0), 0)
    new_nps_clean = new_nps[new_nps > 0]
    old_nps_clean = old_nps[old_nps > 0]

    def sample_std(a):
        return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    t_depth, p_depth = sp_stats.ttest_ind(new_depths, old_depths, equal_var=False)
    t_node, p_node = sp_stats.ttest_ind(new_nodes, old_nodes, equal_var=False)
    t_time, p_time = sp_stats.ttest_ind(new_times, old_times, equal_var=False)
    if len(new_nps_clean) and len(old_nps_clean):
        t_nps, p_nps = sp_stats.ttest_ind(new_nps_clean, old_nps_clean, equal_var=False)
    else:
        t_nps, p_nps = 0.0, 1.0

    report: List[str] = []
    report.append("========== 搜尋效能與深度對比統計報告 ==========\n")
    append_exclusion_and_mate_section(report, meta, new_mate, old_mate, name1, name2)

    report.append("【樣本 (主統計)】")
    report.append(f"  * {name1} 著步數: {len(new_depths)}")
    report.append(f"  * {name2} 著步數: {len(old_depths)}\n")

    report.append("【1. 搜尋深度 (Depth)】")
    report.append(
        f"  * {name1}: mean={np.mean(new_depths):.2f} σ={sample_std(new_depths):.2f} "
        f"med={np.median(new_depths):.0f} max={np.max(new_depths)}"
    )
    report.append(
        f"  * {name2}: mean={np.mean(old_depths):.2f} σ={sample_std(old_depths):.2f} "
        f"med={np.median(old_depths):.0f} max={np.max(old_depths)}"
    )
    report.append(f"  * 差異 ({name1}-{name2}): {np.mean(new_depths) - np.mean(old_depths):+.2f}")
    report.append(f"  * Welch t={t_depth:+.3f}, p={p_depth:.4e}  {'[顯著]' if p_depth < 0.05 else '[不顯著]'}")
    report.append(
        f"  * depth>=14: {name1} {np.mean(new_depths >= 14)*100:.1f}% vs "
        f"{name2} {np.mean(old_depths >= 14)*100:.1f}%\n"
    )

    report.append("【2. 節點數 (Nodes)】")
    report.append(f"  * {name1}: mean={np.mean(new_nodes):.0f} σ={sample_std(new_nodes):.0f} med={np.median(new_nodes):.0f}")
    report.append(f"  * {name2}: mean={np.mean(old_nodes):.0f} σ={sample_std(old_nodes):.0f} med={np.median(old_nodes):.0f}")
    if np.mean(old_nodes) > 0:
        report.append(f"  * 差異: {(np.mean(new_nodes) - np.mean(old_nodes)) / np.mean(old_nodes) * 100:+.1f}%")
    report.append(f"  * Welch t={t_node:+.3f}, p={p_node:.4e}  {'[顯著]' if p_node < 0.05 else '[不顯著]'}\n")

    report.append("【3. 每步耗時 (ms)】")
    report.append(f"  * {name1}: mean={np.mean(new_times):.1f} σ={sample_std(new_times):.1f}")
    report.append(f"  * {name2}: mean={np.mean(old_times):.1f} σ={sample_std(old_times):.1f}")
    report.append(f"  * Welch t={t_time:+.3f}, p={p_time:.4e}  {'[顯著]' if p_time < 0.05 else '[不顯著]'}\n")

    report.append("【4. NPS】")
    if len(new_nps_clean) and len(old_nps_clean):
        report.append(
            f"  * {name1}: mean={np.mean(new_nps_clean):.0f} σ={sample_std(new_nps_clean):.0f} med={np.median(new_nps_clean):.0f}"
        )
        report.append(
            f"  * {name2}: mean={np.mean(old_nps_clean):.0f} σ={sample_std(old_nps_clean):.0f} med={np.median(old_nps_clean):.0f}"
        )
        report.append(
            f"  * 差異: {(np.mean(new_nps_clean) - np.mean(old_nps_clean)) / np.mean(old_nps_clean) * 100:+.1f}%"
        )
        report.append(f"  * Welch t={t_nps:+.3f}, p={p_nps:.4e}  {'[顯著]' if p_nps < 0.05 else '[不顯著]'}\n")
    else:
        report.append("  * 無法計算 NPS\n")

    append_by_color_section(report, by_color, name1, name2)
    append_depth_hist_and_ratios(report, new_depths, new_nodes, old_depths, old_nodes, name1, name2)
    report.append("=========================================\n")

    report_text = "\n".join(report)
    print(report_text)
    with open(os.path.join(output_dir, "search_stats_report.txt"), "w", encoding="utf-8") as rf:
        rf.write(report_text)

    for old_file in ("depth_distribution.png", "nps_distribution.png"):
        fp = os.path.join(output_dir, old_file)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError as e:
                print(f"Warning: could not remove {fp}: {e}")

    new_m_depth, new_mean_depth, new_sem_depth = get_metric_per_move(new_move_nums, new_depths)
    old_m_depth, old_mean_depth, old_sem_depth = get_metric_per_move(old_move_nums, old_depths)
    new_m_nodes, new_mean_nodes, new_sem_nodes = get_metric_per_move(new_move_nums, new_nodes)
    old_m_nodes, old_mean_nodes, old_sem_nodes = get_metric_per_move(old_move_nums, old_nodes)
    new_m_time, new_mean_time, new_sem_time = get_metric_per_move(new_move_nums, new_times)
    old_m_time, old_mean_time, old_sem_time = get_metric_per_move(old_move_nums, old_times)

    if len(new_m_depth) and len(old_m_depth):
        plot_line_metric(
            new_m_depth, new_mean_depth, new_sem_depth,
            old_m_depth, old_mean_depth, old_sem_depth,
            title=f"Average Search Depth per Move ({name1} vs {name2})",
            xlabel="Move Number", ylabel="Average Depth (Plies)",
            filename=os.path.join(output_dir, "depth_per_move.png"),
            name1=name1, name2=name2,
        )
    if len(new_m_nodes) and len(old_m_nodes):
        plot_line_metric(
            new_m_nodes, new_mean_nodes, new_sem_nodes,
            old_m_nodes, old_mean_nodes, old_sem_nodes,
            title=f"Average Nodes Searched per Move ({name1} vs {name2})",
            xlabel="Move Number", ylabel="Average Nodes Searched",
            filename=os.path.join(output_dir, "nodes_per_move.png"),
            name1=name1, name2=name2, use_log=True,
        )
    if len(new_m_time) and len(old_m_time):
        plot_line_metric(
            new_m_time, new_mean_time, new_sem_time,
            old_m_time, old_mean_time, old_sem_time,
            title=f"Average Search Time per Move ({name1} vs {name2})",
            xlabel="Move Number", ylabel="Average Search Time (ms)",
            filename=os.path.join(output_dir, "time_per_move.png"),
            name1=name1, name2=name2,
        )

    common_depths = sorted(set(new_depths.tolist()).intersection(set(old_depths.tolist())))
    common_depths = [d for d in common_depths if d < 30]
    if common_depths:
        plt.figure(figsize=(10, 5))
        new_avg_nodes = [np.mean(new_nodes[new_depths == d]) for d in common_depths]
        old_avg_nodes = [np.mean(old_nodes[old_depths == d]) for d in common_depths]
        plt.plot(common_depths, new_avg_nodes, marker="o", label=f"{name1}", color="#1f77b4", linewidth=2)
        plt.plot(common_depths, old_avg_nodes, marker="s", label=f"{name2}", color="#ff7f0e", linewidth=2)
        plt.title(f"Average Nodes vs Depth ({name1} vs {name2})")
        plt.xlabel("Search Depth (Plies)")
        plt.ylabel("Average Nodes Searched")
        plt.yscale("log")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avg_nodes_vs_depth.png"), dpi=150)
        plt.close()

    saved: List[str] = [
        "depth_per_move.png", "nodes_per_move.png", "time_per_move.png", "avg_nodes_vs_depth.png",
    ]

    bins = [(1, 10), (11, 12), (13, 13), (14, 15), (16, 20), (21, 39), (40, 128)]
    bin_labels = [f"{a}-{b}" for a, b in bins]
    new_pct = [float(np.mean((new_depths >= a) & (new_depths <= b)) * 100) for a, b in bins]
    old_pct = [float(np.mean((old_depths >= a) & (old_depths <= b)) * 100) for a, b in bins]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bin_labels))
    w = 0.38
    ax.bar(x - w / 2, new_pct, w, label=name1, color="#1f77b4")
    ax.bar(x + w / 2, old_pct, w, label=name2, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("% of moves")
    ax.set_xlabel("Depth bin")
    ax.set_title(f"Depth Histogram ({name1} vs {name2})")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "depth_histogram.png"), dpi=150)
    plt.close(fig)
    saved.append("depth_histogram.png")

    if len(new_nps_clean) and len(old_nps_clean):
        fig, ax = plt.subplots(figsize=(10, 5))
        hi = float(np.percentile(np.concatenate([new_nps_clean, old_nps_clean]), 99))
        ax.hist(new_nps_clean[new_nps_clean <= hi], bins=40, alpha=0.55, label=name1, color="#1f77b4", density=True)
        ax.hist(old_nps_clean[old_nps_clean <= hi], bins=40, alpha=0.55, label=name2, color="#ff7f0e", density=True)
        ax.set_xlabel("NPS")
        ax.set_ylabel("Density")
        ax.set_title(f"NPS Distribution ({name1} vs {name2})")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "nps_histogram.png"), dpi=150)
        plt.close(fig)
        saved.append("nps_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    hi_n = float(np.percentile(np.concatenate([new_nodes, old_nodes]), 99))
    ax.hist(new_nodes[new_nodes <= hi_n], bins=40, alpha=0.55, label=name1, color="#1f77b4", density=True)
    ax.hist(old_nodes[old_nodes <= hi_n], bins=40, alpha=0.55, label=name2, color="#ff7f0e", density=True)
    ax.set_xlabel("Nodes / move")
    ax.set_ylabel("Density")
    ax.set_title(f"Nodes Distribution ({name1} vs {name2})")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nodes_histogram.png"), dpi=150)
    plt.close(fig)
    saved.append("nodes_histogram.png")

    if len(new_move_nums) and len(old_move_nums) and len(new_nps) and len(old_nps):
        nm, nmean, nsem = get_metric_per_move(new_move_nums, new_nps)
        om, omean, osem = get_metric_per_move(old_move_nums, old_nps)
        if len(nm) and len(om):
            plot_line_metric(
                nm, nmean, nsem, om, omean, osem,
                title=f"Average NPS per Move ({name1} vs {name2})",
                xlabel="Move Number", ylabel="Average NPS",
                filename=os.path.join(output_dir, "nps_per_move.png"),
                name1=name1, name2=name2,
            )
            saved.append("nps_per_move.png")

    if by_color:
        cats, means, cols = [], [], []
        for eng, color in ((name1, "#1f77b4"), (name2, "#ff7f0e")):
            for col in ("W", "B"):
                s = by_color.get((eng, col))
                if s and s["depth"]:
                    cats.append(f"{eng}\n{col}")
                    means.append(float(np.mean(s["depth"])))
                    cols.append(color)
        if cats:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(cats, means, color=cols, edgecolor="white")
            ax.set_ylabel("Mean depth")
            ax.set_title(f"Mean Depth by Engine × Color ({name1} vs {name2})")
            for i, v in enumerate(means):
                ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "depth_by_color.png"), dpi=150)
            plt.close(fig)
            saved.append("depth_by_color.png")

    ratio_ds, ratios = [], []
    for d in range(6, 21):
        nn = new_nodes[new_depths == d]
        on = old_nodes[old_depths == d]
        if len(nn) >= 20 and len(on) >= 20 and on.mean() > 0:
            ratio_ds.append(d)
            ratios.append(float(nn.mean() / on.mean()))
    if ratio_ds:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = ["#2ca02c" if r >= 1.0 else "#d62728" for r in ratios]
        ax.bar([str(d) for d in ratio_ds], ratios, color=bar_colors, edgecolor="white")
        ax.axhline(1.0, color="#888888", linestyle="--")
        ax.set_xlabel("Depth")
        ax.set_ylabel(f"Mean nodes ratio ({name1}/{name2})")
        ax.set_title(f"Same-Depth Node Ratio ({name1} / {name2})")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "same_depth_node_ratio.png"), dpi=150)
        plt.close(fig)
        saved.append("same_depth_node_ratio.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for arr, lab, col in ((new_depths, name1, "#1f77b4"), (old_depths, name2, "#ff7f0e")):
        s = np.sort(arr)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, label=lab, color=col, linewidth=2)
    ax.set_xlabel("Depth")
    ax.set_ylabel("CDF")
    ax.set_title(f"Depth CDF (main sample) ({name1} vs {name2})")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "depth_cdf.png"), dpi=150)
    plt.close(fig)
    saved.append("depth_cdf.png")

    print(f"Visualizations saved to {output_dir}")
    print("Charts: " + ", ".join(saved))


def auto_names(pgn_path: Path) -> Tuple[str, str]:
    content = pgn_path.read_text(encoding="utf-8", errors="replace")
    white_names = re.findall(r'\[White\s+"([^"]+)"\]', content)
    black_names = re.findall(r'\[Black\s+"([^"]+)"\]', content)
    unique_names = list(set(white_names + black_names))
    if "New" in unique_names and "Old" in unique_names:
        return "New", "Old"
    if "MyEngine" in unique_names and "Stockfish" in unique_names:
        return "MyEngine", "Stockfish"
    if len(unique_names) >= 2:
        unique_names.sort()
        return unique_names[0], unique_names[1]
    if len(unique_names) == 1:
        return unique_names[0], "Unknown"
    return "New", "Old"


def main():
    parser = argparse.ArgumentParser(description="Parse search stats from tournament PGN")
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--name1", default=None, help="First engine (e.g. New)")
    parser.add_argument("--name2", default=None, help="Second engine (e.g. Old)")
    parser.add_argument(
        "--draw-depth-cap",
        type=int,
        default=DEFAULT_DRAW_DEPTH_CAP,
        help=f"In drawn games, exclude moves with depth >= this (default {DEFAULT_DRAW_DEPTH_CAP})",
    )
    args = parser.parse_args()

    if not args.pgn.exists():
        print(f"Error: PGN not found: {args.pgn}")
        return 1

    name1 = args.name1
    name2 = args.name2
    if name1 is None or name2 is None:
        a1, a2 = auto_names(args.pgn)
        name1 = name1 or a1
        name2 = name2 or a2

    print(
        f"Parsing stats: '{name1}' vs '{name2}'  from {args.pgn}  "
        f"(draw_depth_cap={args.draw_depth_cap})"
    )
    stats = parse_pgn(args.pgn, name1, name2, draw_depth_cap=args.draw_depth_cap)
    if not stats:
        return 1
    new_stats, old_stats, by_color, new_mate, old_mate, meta = stats
    analyze_and_plot(
        new_stats, old_stats, by_color, new_mate, old_mate, meta,
        str(args.out_dir), name1, name2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
