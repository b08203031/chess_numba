#!/usr/bin/env python3
"""
Parse per-move search stats from tournament PGN comments and compare two engines.

Comment format expected:
  { d=11, eval=+0.07, n=332609, t=528ms }

Consolidates former _extra_analysis.py search portions (by-color depth/NPS,
depth histogram, same-depth node ratios).

Game-level WDL / Elo: 統計數據.py

Usage (repo root):
  python tournament_analysis/parse_search_stats.py
  python tournament_analysis/parse_search_stats.py --name1 New --name2 Old
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats

HERE = Path(__file__).resolve().parent
DEFAULT_PGN = HERE / "tournament_results.pgn"

WHITE_MOVE_RE = re.compile(
    r"\b(?<!\.)\b(\d+)\.\s+(\S+)\s*\{\s*d=(\d+),\s*eval=([^,]+),\s*n=(\d+),\s*t=(\d+)ms\s*\}"
)
BLACK_MOVE_RE = re.compile(
    r"\b(\d+)\.\.\.\s+(\S+)\s*\{\s*d=(\d+),\s*eval=([^,]+),\s*n=(\d+),\s*t=(\d+)ms\s*\}"
)


def empty_stats() -> Dict[str, List]:
    return {"depth": [], "nodes": [], "time": [], "move_num": [], "color": []}


def parse_pgn(file_path: str | Path, name1: str, name2: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: PGN file not found: {path}")
        return None

    content = path.read_text(encoding="utf-8", errors="replace")
    game_blocks = content.split('[Event "')
    # Also accept [Event without quote variants from split on [Event
    if len(game_blocks) <= 1:
        game_blocks = re.split(r"\[Event ", content)

    new_stats = empty_stats()
    old_stats = empty_stats()
    # (engine, color) -> lists
    by_color = defaultdict(lambda: {"depth": [], "nodes": [], "time": [], "nps": []})

    games_parsed = 0
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

        for move in WHITE_MOVE_RE.findall(block):
            move_num, _san, depth_s, _ev, nodes_s, time_s = move
            depth, nodes, time_ms = int(depth_s), int(nodes_s), int(time_s)
            eng = white_player
            col = "W"
            bucket = new_stats if eng == name1 else old_stats
            bucket["depth"].append(depth)
            bucket["nodes"].append(nodes)
            bucket["time"].append(time_ms)
            bucket["move_num"].append(int(move_num))
            bucket["color"].append(col)
            bc = by_color[(eng, col)]
            bc["depth"].append(depth)
            bc["nodes"].append(nodes)
            bc["time"].append(time_ms)
            if time_ms > 0:
                bc["nps"].append(nodes / (time_ms / 1000.0))

        for move in BLACK_MOVE_RE.findall(block):
            move_num, _san, depth_s, _ev, nodes_s, time_s = move
            depth, nodes, time_ms = int(depth_s), int(nodes_s), int(time_s)
            eng = black_player
            col = "B"
            bucket = new_stats if eng == name1 else old_stats
            bucket["depth"].append(depth)
            bucket["nodes"].append(nodes)
            bucket["time"].append(time_ms)
            bucket["move_num"].append(int(move_num))
            bucket["color"].append(col)
            bc = by_color[(eng, col)]
            bc["depth"].append(depth)
            bc["nodes"].append(nodes)
            bc["time"].append(time_ms)
            if time_ms > 0:
                bc["nps"].append(nodes / (time_ms / 1000.0))

    print(f"Successfully parsed {games_parsed} games.")
    return new_stats, old_stats, by_color


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


def append_by_color_section(report: List[str], by_color, name1: str, name2: str) -> None:
    report.append("【5. 分執色搜尋統計 (engine × color)】")
    for eng in (name1, name2):
        for col in ("W", "B"):
            s = by_color.get((eng, col))
            if not s or not s["depth"]:
                report.append(f"  {eng} {col}: (no moves)")
                continue
            d = np.array(s["depth"], dtype=float)
            n = np.array(s["nodes"], dtype=float)
            nps = np.array(s["nps"], dtype=float) if s["nps"] else np.array([])
            d_nomate = d[d < 100]
            deep = float(np.mean(d >= 14) * 100)
            line = (
                f"  {eng} {col}: moves={len(d)} mean_d={d.mean():.2f} med_d={np.median(d):.0f} "
                f"d>=14={deep:.1f}% mean_n={n.mean():.0f}"
            )
            if len(nps):
                line += f" mean_nps={nps.mean():.0f}"
            if len(d_nomate):
                line += f" mean_d_nomate={d_nomate.mean():.2f}"
            report.append(line)
    report.append("")


def append_depth_hist_and_ratios(report: List[str], new_depths, new_nodes, old_depths, old_nodes, name1, name2) -> None:
    report.append("【6. 深度直方圖 (%)】")
    bins = [(1, 10), (11, 12), (13, 13), (14, 15), (16, 20), (21, 50), (51, 128)]
    for eng, darr in ((name1, new_depths), (name2, old_depths)):
        if len(darr) == 0:
            continue
        parts = []
        for a, b in bins:
            parts.append(f"{a}-{b}:{(np.mean((darr >= a) & (darr <= b)) * 100):.1f}%")
        report.append(f"  {eng}: " + " ".join(parts))
        dnm = darr[darr < 100]
        if len(dnm):
            report.append(
                f"    no-mate mean_d={dnm.mean():.2f} med={np.median(dnm):.0f} "
                f"d>=14={(dnm >= 14).mean() * 100:.1f}%"
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


def analyze_and_plot(new_stats, old_stats, by_color, output_dir, name1, name2):
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
            f"Error: Not enough move data. {name1} moves: {len(new_depths)}, {name2} moves: {len(old_depths)}"
        )
        return

    new_nps = np.where(new_times > 0, new_nodes / (new_times / 1000.0), 0)
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
    report.append("【樣本】")
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

    print(f"Visualizations saved to {output_dir}")


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

    print(f"Parsing stats: '{name1}' vs '{name2}'  from {args.pgn}")
    stats = parse_pgn(args.pgn, name1, name2)
    if not stats:
        return 1
    new_stats, old_stats, by_color = stats
    analyze_and_plot(new_stats, old_stats, by_color, str(args.out_dir), name1, name2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
