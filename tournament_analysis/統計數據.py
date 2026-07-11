#!/usr/bin/env python3
"""
Tournament game-level statistics (WDL, Elo, length, pairs, openings, …).

Consolidates former one-off scripts:
  - scratch/analyze_tournament_phase_{a,b,b4}.py
  - tournament_analysis/_extra_analysis.py (game / pair portion)
  - scratch/inspect_pgn.py (use --peek)

Search depth/nodes/NPS charts live in parse_search_stats.py.

Usage (from repo root):
  python tournament_analysis/統計數據.py
  python tournament_analysis/統計數據.py --pgn tournament_analysis/tournament_results.pgn --target New
  python tournament_analysis/統計數據.py --peek
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

HERE = Path(__file__).resolve().parent
DEFAULT_PGN = HERE / "tournament_results.pgn"

LENGTH_BUCKETS = [
    ("short(<=25)", lambda m: m <= 25),
    ("medium(26-50)", lambda m: 26 <= m <= 50),
    ("long(51-80)", lambda m: 51 <= m <= 80),
    ("very_long(>80)", lambda m: m > 80),
]


def to_elo(s: float) -> float:
    if s <= 0.0:
        return float("-inf")
    if s >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / s - 1.0)


def detect_engines(pgn_path: Path) -> List[str]:
    text = pgn_path.read_text(encoding="utf-8", errors="replace")
    names = re.findall(r'\[(?:White|Black)\s+"([^"]+)"\]', text)
    # Preserve frequency order
    counts = Counter(names)
    return [n for n, _ in counts.most_common()]


def peek_pgn(pgn_path: Path, head_lines: int = 50) -> None:
    text = pgn_path.read_text(encoding="utf-8", errors="replace")
    n_games = len(re.findall(r"\[Event ", text))
    whites = Counter(re.findall(r'\[White "(.+?)"\]', text))
    blacks = Counter(re.findall(r'\[Black "(.+?)"\]', text))
    dates = Counter(re.findall(r'\[Date "(.+?)"\]', text))
    events = Counter(re.findall(r'\[Event "(.+?)"\]', text)).most_common(5)
    print(f"file: {pgn_path}")
    print(f"games: {n_games}")
    print(f"White: {whites}")
    print(f"Black: {blacks}")
    print(f"Date:  {dates}")
    print(f"Event: {events}")
    print("--- head ---")
    print("\n".join(text.splitlines()[:head_lines]))


class ChessEngineAnalyzer:
    def __init__(self, target_engine: str = "New"):
        self.target_engine = target_engine
        self.w_wins = self.w_draws = self.w_losses = 0
        self.b_wins = self.b_draws = self.b_losses = 0
        self.lengths: List[int] = []
        self.lengths_w: List[int] = []
        self.lengths_l: List[int] = []
        self.lengths_d: List[int] = []
        self.openings_white: Dict[str, Dict[str, int]] = {}
        self.openings_black: Dict[str, Dict[str, int]] = {}
        self.score_sequence: List[float] = []
        self.games: List[Dict[str, Any]] = []
        self.pair_outcomes = {
            "WW": 0, "WD": 0, "WL": 0,
            "DW": 0, "DD": 0, "DL": 0,
            "LW": 0, "LD": 0, "LL": 0,
        }
        self.total_pairs = 0
        self.consecutive_pair_counts: Counter = Counter()
        self.fen_pair_counts: Counter = Counter()

    def parse_pgn(self, file_path: str | Path) -> None:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace")
        games_raw = re.split(r"\[Event ", content)[1:]
        self.games.clear()
        self.score_sequence.clear()
        self.lengths.clear()
        self.lengths_w.clear()
        self.lengths_l.clear()
        self.lengths_d.clear()
        self.openings_white.clear()
        self.openings_black.clear()
        self.w_wins = self.w_draws = self.w_losses = 0
        self.b_wins = self.b_draws = self.b_losses = 0

        for game in games_raw:
            white_m = re.search(r'\[White "(.*?)"\]', game)
            black_m = re.search(r'\[Black "(.*?)"\]', game)
            result_m = re.search(r'\[Result "(.*?)"\]', game)
            if not (white_m and black_m and result_m):
                continue

            white = white_m.group(1)
            black = black_m.group(1)
            result = result_m.group(1)
            if self.target_engine not in (white, black):
                continue

            rnd_m = re.search(r'\[Round "(.*?)"\]', game)
            date_m = re.search(r'\[Date "(.*?)"\]', game)
            fen_m = re.search(r'\[FEN "(.*?)"\]', game)

            moves_m = re.search(r"\]\s*\n\s*\n(.*?)$", game, re.DOTALL)
            moves_text = moves_m.group(1) if moves_m else ""
            move_numbers = re.findall(r"\b(\d+)\.", moves_text)
            num_moves = max(int(m) for m in move_numbers) if move_numbers else 0
            first_m = re.search(r"1\.\s+([a-zA-Z0-9\-=+#]+)", moves_text)
            first_move = first_m.group(1) if first_m else "Unknown"
            promos = len(re.findall(r"=[QRBN]", moves_text))
            mate_flag = bool(re.search(r"eval=#-?\d+", moves_text))

            is_target_white = white == self.target_engine
            if result == "1-0":
                score = 1.0 if is_target_white else 0.0
                res_type = "W" if is_target_white else "L"
            elif result == "0-1":
                score = 0.0 if is_target_white else 1.0
                res_type = "L" if is_target_white else "W"
            else:
                score = 0.5
                res_type = "D"

            if num_moves <= 25:
                length_bucket = "short(<=25)"
            elif num_moves <= 50:
                length_bucket = "medium(26-50)"
            elif num_moves <= 80:
                length_bucket = "long(51-80)"
            else:
                length_bucket = "very_long(>80)"

            self.games.append(
                {
                    "round": int(rnd_m.group(1)) if rnd_m and rnd_m.group(1).isdigit() else 0,
                    "date": date_m.group(1) if date_m else "?",
                    "white": white,
                    "black": black,
                    "result": result,
                    "score": score,
                    "rt": res_type,
                    "nmoves": num_moves,
                    "length_bucket": length_bucket,
                    "is_target_white": is_target_white,
                    "promos": promos,
                    "mate_flag": mate_flag,
                    "fen": fen_m.group(1) if fen_m else "",
                    "first_move": first_move,
                }
            )

            self.score_sequence.append(score)
            self.lengths.append(num_moves)
            if res_type == "W":
                self.lengths_w.append(num_moves)
            elif res_type == "L":
                self.lengths_l.append(num_moves)
            else:
                self.lengths_d.append(num_moves)

            if is_target_white:
                if res_type == "W":
                    self.w_wins += 1
                elif res_type == "D":
                    self.w_draws += 1
                else:
                    self.w_losses += 1
                bucket = self.openings_white.setdefault(
                    first_move, {"W": 0, "D": 0, "L": 0, "Total": 0}
                )
                bucket[res_type] += 1
                bucket["Total"] += 1
            else:
                if res_type == "W":
                    self.b_wins += 1
                elif res_type == "D":
                    self.b_draws += 1
                else:
                    self.b_losses += 1
                bucket = self.openings_black.setdefault(
                    first_move, {"W": 0, "D": 0, "L": 0, "Total": 0}
                )
                bucket[res_type] += 1
                bucket["Total"] += 1

        self._calculate_pair_stats()
        self._calculate_fen_pairs()

    def _calculate_pair_stats(self) -> None:
        """Consecutive match-pairs in file order (game i, i+1) from target W/D/L."""
        self.pair_outcomes = {k: 0 for k in self.pair_outcomes}
        self.consecutive_pair_counts = Counter()
        n = len(self.score_sequence)
        self.total_pairs = n // 2

        for i in range(0, self.total_pairs * 2, 2):
            a = self.games[i]["rt"]
            b = self.games[i + 1]["rt"]
            key = a + b
            self.consecutive_pair_counts[key] += 1
            if key in self.pair_outcomes:
                self.pair_outcomes[key] += 1
            else:
                # unexpected keys still counted in consecutive_pair_counts
                pass

            # Also 9-grid keyed by color order when pair is color-swapped for target
            g0, g1 = self.games[i], self.games[i + 1]
            if g0["is_target_white"] and not g1["is_target_white"]:
                # first game target white, second target black
                pass  # pair_outcomes key already a+b from target POV

    def _calculate_fen_pairs(self) -> None:
        """Same opening FEN, two games with colors swapped (true match-pair)."""
        by_fen: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for g in self.games:
            if g["fen"]:
                by_fen[g["fen"]].append(g)

        self.fen_pair_counts = Counter()
        for fen, gl in by_fen.items():
            if len(gl) != 2:
                continue
            g_w = next((g for g in gl if g["is_target_white"]), None)
            g_b = next((g for g in gl if not g["is_target_white"]), None)
            if not g_w or not g_b:
                continue
            self.fen_pair_counts[g_w["rt"] + g_b["rt"]] += 1

    def calculate_elo_and_ci(self) -> Optional[Dict[str, Any]]:
        total_w = self.w_wins + self.b_wins
        total_d = self.w_draws + self.b_draws
        total_l = self.w_losses + self.b_losses
        n = total_w + total_d + total_l
        if n == 0:
            return None

        score_expected = (total_w + 0.5 * total_d) / n
        elo_diff = to_elo(score_expected)
        sum_sq = (
            total_w * (1 - score_expected) ** 2
            + total_d * (0.5 - score_expected) ** 2
            + total_l * (0 - score_expected) ** 2
        )
        std_dev = math.sqrt(sum_sq / (n - 1)) if n > 1 else 0.0
        std_err = std_dev / math.sqrt(n) if n > 0 else 0.0
        z_val = 1.96
        lo = max(1e-12, min(1 - 1e-12, score_expected - z_val * std_err))
        hi = max(1e-12, min(1 - 1e-12, score_expected + z_val * std_err))
        elo_min = to_elo(lo)
        elo_max = to_elo(hi)
        return {
            "Total": n,
            "W": total_w,
            "D": total_d,
            "L": total_l,
            "Expected_Score": score_expected,
            "Elo_Diff": elo_diff,
            "CI_95": (elo_min, elo_max),
            "Margin_of_Error": (elo_max - elo_min) / 2 if math.isfinite(elo_max) and math.isfinite(elo_min) else float("nan"),
            "Draw_Rate": total_d / n,
        }

    def plot_cumulative_wins(self, output_file: str | Path) -> None:
        wins_seq = [1 if s == 1.0 else 0 for s in self.score_sequence]
        if not wins_seq:
            return
        cumulative = np.cumsum(wins_seq)
        total_games = len(wins_seq)
        avg_win_rate = cumulative[-1] / total_games
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, total_games + 1), cumulative, label="Cumulative Wins", color="#1f77b4", linewidth=2)
        plt.plot(
            [1, total_games],
            [avg_win_rate, cumulative[-1]],
            linestyle="--",
            color="#d62728",
            label=f"Ideal Trend (Slope: {avg_win_rate:.3f})",
        )
        plt.xlabel("Total Games (N)")
        plt.ylabel("Cumulative Wins (W)")
        plt.title(f"Cumulative Win Trajectory for {self.target_engine}")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

    def plot_all_charts(self, output_dir: Path) -> List[Path]:
        """Generate charts matching the richer game-level report sections."""
        out_paths: List[Path] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.games:
            return out_paths

        # 1) Cumulative wins (existing)
        p = output_dir / "cumulative_wins.png"
        self.plot_cumulative_wins(p)
        out_paths.append(p)

        # 2) Cumulative score rate + rolling score
        scores = np.array(self.score_sequence, dtype=float)
        n = len(scores)
        x = np.arange(1, n + 1)
        cum_pts = np.cumsum(scores)
        cum_rate = cum_pts / x
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(x, cum_pts, color="#1f77b4", linewidth=2, label="Cumulative points")
        axes[0].plot(x, 0.5 * x, linestyle="--", color="#888888", label="Equal (0.5/game)")
        axes[0].set_ylabel("Points")
        axes[0].set_title(f"Cumulative Score — {self.target_engine}")
        axes[0].legend()
        axes[0].grid(True, linestyle=":", alpha=0.6)
        window = 20 if n >= 40 else max(5, n // 5)
        if n >= window:
            roll = np.convolve(scores, np.ones(window) / window, mode="valid")
            axes[1].plot(np.arange(window, n + 1), roll, color="#2ca02c", linewidth=2, label=f"Rolling score (w={window})")
        axes[1].axhline(0.5, color="#888888", linestyle="--", label="0.50")
        axes[1].set_xlabel("Games")
        axes[1].set_ylabel("Score rate")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        p = output_dir / "cumulative_score.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

        # 3) WDL by color (grouped bars)
        labels = ["White", "Black"]
        w_counts = [self.w_wins, self.b_wins]
        d_counts = [self.w_draws, self.b_draws]
        l_counts = [self.w_losses, self.b_losses]
        fig, ax = plt.subplots(figsize=(8, 5))
        xpos = np.arange(len(labels))
        width = 0.25
        ax.bar(xpos - width, w_counts, width, label="Win", color="#2ca02c")
        ax.bar(xpos, d_counts, width, label="Draw", color="#7f7f7f")
        ax.bar(xpos + width, l_counts, width, label="Loss", color="#d62728")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Games")
        ax.set_title(f"WDL by Color — {self.target_engine}")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        p = output_dir / "wdl_by_color.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

        # 4) Score by length bucket
        by_len: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for g in self.games:
            by_len[g["length_bucket"]].append(g)
        names = [name for name, _ in LENGTH_BUCKETS if by_len.get(name)]
        if names:
            rates, ns, colors = [], [], []
            for name in names:
                sub = by_len[name]
                w = sum(1 for g in sub if g["rt"] == "W")
                d = sum(1 for g in sub if g["rt"] == "D")
                rates.append((w + 0.5 * d) / len(sub))
                ns.append(len(sub))
                colors.append("#1f77b4" if rates[-1] >= 0.5 else "#ff7f0e")
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(range(len(names)), rates, color=colors, edgecolor="white")
            ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=15, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Score rate")
            ax.set_title(f"Score by Game Length — {self.target_engine}")
            for bar, rate, cnt in zip(bars, rates, ns):
                ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.2f}\nn={cnt}",
                        ha="center", va="bottom", fontsize=8)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            p = output_dir / "score_by_length.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out_paths.append(p)

        # 5) Length distribution by result
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.arange(0, max(self.lengths + [1]) + 10, 10)
        if self.lengths_w:
            ax.hist(self.lengths_w, bins=bins, alpha=0.55, label="Wins", color="#2ca02c")
        if self.lengths_l:
            ax.hist(self.lengths_l, bins=bins, alpha=0.55, label="Losses", color="#d62728")
        if self.lengths_d:
            ax.hist(self.lengths_d, bins=bins, alpha=0.45, label="Draws", color="#7f7f7f")
        ax.set_xlabel("Full-moves")
        ax.set_ylabel("Count")
        ax.set_title(f"Game Length Distribution by Result — {self.target_engine}")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        p = output_dir / "length_by_result.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

        # 6) Pair outcomes (FEN pairs preferred, else consecutive)
        pair_src = self.fen_pair_counts if self.fen_pair_counts else self.consecutive_pair_counts
        pair_label = "FEN match-pairs" if self.fen_pair_counts else "Consecutive pairs"
        if pair_src:
            keys = ["WW", "WD", "WL", "DW", "DD", "DL", "LW", "LD", "LL"]
            vals = [pair_src.get(k, 0) for k in keys]
            fig, ax = plt.subplots(figsize=(10, 5))
            colors9 = ["#2ca02c" if k[0] == "W" else ("#7f7f7f" if k[0] == "D" else "#d62728") for k in keys]
            ax.bar(keys, vals, color=colors9, edgecolor="white")
            ax.set_ylabel("Pairs")
            ax.set_title(f"Pair Outcomes ({pair_label}) — {self.target_engine}")
            tot = sum(vals) or 1
            for i, v in enumerate(vals):
                if v:
                    ax.text(i, v + max(vals) * 0.01, f"{v}\n{v / tot:.0%}", ha="center", va="bottom", fontsize=8)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            p = output_dir / "pair_outcomes.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out_paths.append(p)

        # 7) Elo + 95% CI
        elo_data = self.calculate_elo_and_ci()
        if elo_data:
            elo = elo_data["Elo_Diff"]
            lo, hi = elo_data["CI_95"]
            if math.isfinite(elo) and math.isfinite(lo) and math.isfinite(hi):
                fig, ax = plt.subplots(figsize=(8, 3.5))
                yerr_lo = max(0.0, elo - lo)
                yerr_hi = max(0.0, hi - elo)
                ax.errorbar([0], [elo], yerr=[[yerr_lo], [yerr_hi]], fmt="o", color="#1f77b4",
                            capsize=8, markersize=10, linewidth=2)
                ax.axhline(0, color="#888888", linestyle="--")
                ax.set_xticks([0])
                ax.set_xticklabels([self.target_engine])
                ax.set_ylabel("Elo vs opponent")
                ax.set_title(f"Elo Difference ± 95% CI  (score={elo_data['Expected_Score']:.3f}, n={elo_data['Total']})")
                ax.grid(True, axis="y", linestyle=":", alpha=0.5)
                fig.tight_layout()
                p = output_dir / "elo_ci.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                out_paths.append(p)

        # 8) Top openings score (white + black)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, title, op_dict in (
            (axes[0], "As White (first move)", self.openings_white),
            (axes[1], "As Black (vs first move)", self.openings_black),
        ):
            items = sorted(op_dict.items(), key=lambda x: x[1]["Total"], reverse=True)[:8]
            if not items:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
                continue
            labels = [k for k, _ in items]
            rates = [(d["W"] + 0.5 * d["D"]) / d["Total"] for _, d in items]
            ns = [d["Total"] for _, d in items]
            y = np.arange(len(labels))
            ax.barh(y, rates, color="#1f77b4", alpha=0.85)
            ax.axvline(0.5, color="#888888", linestyle="--")
            ax.set_yticks(y)
            ax.set_yticklabels([f"{lab} (n={c})" for lab, c in zip(labels, ns)])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Score rate")
            ax.set_title(title)
            ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        fig.suptitle(f"Opening Score Rates — {self.target_engine}", fontweight="bold")
        fig.tight_layout()
        p = output_dir / "openings_score.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

        # 9) Promotions by result
        promo_means = []
        promo_labs = []
        for lab, color in (("W", "#2ca02c"), ("D", "#7f7f7f"), ("L", "#d62728")):
            sub = [g["promos"] for g in self.games if g["rt"] == lab]
            if sub:
                promo_labs.append(lab)
                promo_means.append(float(np.mean(sub)))
        if promo_labs:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(promo_labs, promo_means, color=["#2ca02c", "#7f7f7f", "#d62728"][: len(promo_labs)])
            ax.set_ylabel("Avg promotions / game")
            ax.set_title(f"Promotions by Result — {self.target_engine}")
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            p = output_dir / "promotions_by_result.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out_paths.append(p)

        # 10) First half vs second half score
        if n >= 20:
            h = n // 2
            s1 = float(np.mean(scores[:h]))
            s2 = float(np.mean(scores[h:]))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["First half", "Second half"], [s1, s2], color=["#1f77b4", "#ff7f0e"])
            ax.axhline(0.5, color="#888888", linestyle="--")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score rate")
            ax.set_title(f"Score Stability (halves) — {self.target_engine}")
            for i, v in enumerate((s1, s2)):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            p = output_dir / "score_halves.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out_paths.append(p)

        return out_paths

    def analyze_color_bias(self) -> Optional[Dict[str, Any]]:
        obs = np.array(
            [
                [self.w_wins, self.w_draws, self.w_losses],
                [self.b_wins, self.b_draws, self.b_losses],
            ]
        )
        if np.sum(obs[0]) == 0 or np.sum(obs[1]) == 0:
            return None
        chi2, p_val, _, _ = stats.chi2_contingency(obs)
        w_rate = (self.w_wins + 0.5 * self.w_draws) / np.sum(obs[0])
        b_rate = (self.b_wins + 0.5 * self.b_draws) / np.sum(obs[1])
        white_perspective_w = self.w_wins + self.b_losses
        white_perspective_d = self.w_draws + self.b_draws
        white_perspective_l = self.w_losses + self.b_wins
        total_games = white_perspective_w + white_perspective_d + white_perspective_l
        white_overall_wr = (
            (white_perspective_w + 0.5 * white_perspective_d) / total_games if total_games else 0.0
        )
        return {
            "W_Rate": w_rate,
            "B_Rate": b_rate,
            "Chi2": chi2,
            "p_value": p_val,
            "White_Overall_WR": white_overall_wr,
        }

    def _analyze_opening_dict(self, openings_dict):
        sorted_ops = sorted(openings_dict.items(), key=lambda x: x[1]["Total"], reverse=True)
        if len(sorted_ops) < 2:
            return sorted_ops, None
        op1_name, op1_data = sorted_ops[0]
        op2_name, op2_data = sorted_ops[1]
        n1, n2 = op1_data["Total"], op2_data["Total"]
        p1 = (op1_data["W"] + 0.5 * op1_data["D"]) / n1
        p2 = (op2_data["W"] + 0.5 * op2_data["D"]) / n2
        p_pool = (
            op1_data["W"] + 0.5 * op1_data["D"] + op2_data["W"] + 0.5 * op2_data["D"]
        ) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if n1 > 0 and n2 > 0 else 0
        z_score = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        return sorted_ops, {"op1": op1_name, "op2": op2_name, "Z": z_score, "p_value": p_val}

    def analyze_openings(self):
        w_ops, w_z = self._analyze_opening_dict(self.openings_white)
        b_ops, b_z = self._analyze_opening_dict(self.openings_black)
        return {"White": {"Ops": w_ops, "Z": w_z}, "Black": {"Ops": b_ops, "Z": b_z}}

    def analyze_lengths(self):
        if not self.lengths:
            return None
        return {
            "Overall": (float(np.mean(self.lengths)), float(np.std(self.lengths))),
            "Wins": (float(np.mean(self.lengths_w)), float(np.std(self.lengths_w))) if self.lengths_w else (0.0, 0.0),
            "Losses": (float(np.mean(self.lengths_l)), float(np.std(self.lengths_l))) if self.lengths_l else (0.0, 0.0),
            "Draws": (float(np.mean(self.lengths_d)), float(np.std(self.lengths_d))) if self.lengths_d else (0.0, 0.0),
        }

    def analyze_sequence(self):
        if len(self.score_sequence) < 2:
            return None
        seq = np.array(self.score_sequence)
        autocorr = (
            np.corrcoef(seq[:-1], seq[1:])[0, 1]
            if np.std(seq[:-1]) > 0 and np.std(seq[1:]) > 0
            else 0.0
        )
        median_score = np.median(seq)
        binary_seq = (seq > median_score).astype(int)
        runs = 1 + sum(1 for i in range(1, len(binary_seq)) if binary_seq[i] != binary_seq[i - 1])
        n1 = int(np.sum(binary_seq))
        n0 = len(binary_seq) - n1
        if n0 == 0 or n1 == 0:
            return {"Autocorrelation": float(autocorr), "Runs_Z": 0.0, "Runs_P": 1.0}
        expected_runs = 2 * n0 * n1 / len(binary_seq) + 1
        var_runs = (expected_runs - 1) * (expected_runs - 2) / (len(binary_seq) - 1)
        z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0.0
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
        return {"Autocorrelation": float(autocorr), "Runs_Z": float(z_runs), "Runs_P": float(p_runs)}

    def analyze_length_buckets(self) -> List[str]:
        lines = ["[7. 對局長度分桶 WDL (target 視角)]"]
        by_len: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for g in self.games:
            by_len[g["length_bucket"]].append(g)
        for name, _ in LENGTH_BUCKETS:
            sub = by_len.get(name, [])
            if not sub:
                lines.append(f"  {name}: n=0")
                continue
            w = sum(1 for g in sub if g["rt"] == "W")
            d = sum(1 for g in sub if g["rt"] == "D")
            l = sum(1 for g in sub if g["rt"] == "L")
            avg_m = sum(g["nmoves"] for g in sub) / len(sub)
            avg_p = sum(g["promos"] for g in sub) / len(sub)
            lines.append(
                f"  {name}: n={len(sub):3d}  WDL={w}-{d}-{l}  "
                f"score={(w + 0.5 * d) / len(sub):.3f}  avg_moves={avg_m:.1f}  avg_promo={avg_p:.2f}"
            )
        # long / very long lists for decisive games
        long_g = [g for g in self.games if g["nmoves"] >= 50 and g["rt"] != "D"]
        if long_g:
            lines.append(f"  Decisive games with moves>=50: {len(long_g)}")
            for g in long_g[:20]:
                lines.append(
                    f"    R{g['round']}: {g['rt']} {g['nmoves']}m "
                    f"color={'W' if g['is_target_white'] else 'B'} result={g['result']}"
                )
            if len(long_g) > 20:
                lines.append(f"    … ({len(long_g) - 20} more)")
        return lines

    def analyze_promos(self) -> List[str]:
        lines = ["[8. 升變次數 by result]"]
        for lab in ("W", "D", "L"):
            sub = [g for g in self.games if g["rt"] == lab]
            if not sub:
                lines.append(f"  {lab}: n=0")
                continue
            lines.append(
                f"  {lab}: n={len(sub)} avg_promo={np.mean([g['promos'] for g in sub]):.2f} "
                f"avg_moves={np.mean([g['nmoves'] for g in sub]):.1f}"
            )
        return lines

    def analyze_halves_and_cumulative(self) -> List[str]:
        lines = ["[9. 前後半場與累積得分 (每 10 局)]"]
        n = len(self.games)
        if n == 0:
            lines.append("  (no games)")
            return lines
        h = n // 2
        if h > 0:
            s1 = sum(g["score"] for g in self.games[:h]) / h
            s2 = sum(g["score"] for g in self.games[h:]) / (n - h)
            lines.append(f"  first  {h:3d} games score={s1:.3f}")
            lines.append(f"  second {n - h:3d} games score={s2:.3f}")
        cum = wins = 0.0
        for i, g in enumerate(self.games, 1):
            cum += g["score"]
            if g["rt"] == "W":
                wins += 1
            if i % 10 == 0 or i == n:
                lines.append(f"  after {i:3d}: wins={int(wins):3d} pts={cum:.1f} score={cum / i:.3f}")
        return lines

    def analyze_mates(self) -> List[str]:
        lines = ["[10. 註解含 mate eval (#N) 的決勝局]"]
        mate_games = [g for g in self.games if g["mate_flag"] and g["rt"] != "D"]
        if not mate_games:
            lines.append("  (none)")
            return lines
        for g in mate_games[:30]:
            lines.append(
                f"  R{g['round']}: {g['rt']} in {g['nmoves']}m "
                f"({'white' if g['is_target_white'] else 'black'}) result={g['result']}"
            )
        if len(mate_games) > 30:
            lines.append(f"  … ({len(mate_games) - 30} more)")
        return lines

    def generate_report(self, output_dir: Path, write_file: bool = True) -> str:
        lines: List[str] = []

        def out(s: str = "") -> None:
            lines.append(s)

        out(f"========== 引擎對戰深度分析報告 ({self.target_engine}) ==========\n")
        if self.games:
            dates = Counter(g["date"] for g in self.games)
            out(f"日期分佈: {dict(dates)}")
            out(f"White 名稱: {Counter(g['white'] for g in self.games)}")
            out(f"Black 名稱: {Counter(g['black'] for g in self.games)}\n")

        elo_data = self.calculate_elo_and_ci()
        if not elo_data:
            out("沒有足夠的對局數據可供分析。")
            text = "\n".join(lines)
            print(text)
            return text

        out("[1. 總體戰績與 Elo 表現]")
        out(f"總對局數 : {elo_data['Total']} 局")
        out(f"戰績統計 : 勝 {elo_data['W']} | 和 {elo_data['D']} | 負 {elo_data['L']}")
        out(f"勝率(含和) : {elo_data['Expected_Score']:.2%}")
        out(f"和局率 : {elo_data['Draw_Rate']:.2%}")
        out(f"相對 Elo 變化 : {elo_data['Elo_Diff']:.2f}")
        out(f"95% 信賴區間 : [{elo_data['CI_95'][0]:.2f}, {elo_data['CI_95'][1]:.2f}]")
        lo, hi = elo_data["CI_95"]
        if lo > 0:
            out("  ✅ 95% 信心：明顯強於對手（CI 下界 > 0）")
        elif hi < 0:
            out("  ❌ 95% 信心：明顯弱於對手（CI 上界 < 0）")
        else:
            out("  ⚖️ 尚無顯著差距（CI 含 0）")
        out("\n" + "=" * 50 + "\n")

        color_data = self.analyze_color_bias()
        if color_data:
            out("[2. 先手優勢與執色分析]")
            out(
                f"持白勝率 : {color_data['W_Rate']:.2%} "
                f"(勝:{self.w_wins} 和:{self.w_draws} 負:{self.w_losses})"
            )
            out(
                f"持黑勝率 : {color_data['B_Rate']:.2%} "
                f"(勝:{self.b_wins} 和:{self.b_draws} 負:{self.b_losses})"
            )
            out(f"賽局總體白方勝率 : {color_data['White_Overall_WR']:.2%}")
            out(f"卡方 p-value : {color_data['p_value']:.4e}")
            out("\n" + "=" * 50 + "\n")

        op_data = self.analyze_openings()
        out("[3. 開局策略分析 (Top 5)]")
        out("【持白主流開局】")
        for op, data in op_data["White"]["Ops"][:5]:
            rate = (data["W"] + 0.5 * data["D"]) / data["Total"]
            out(f"  - {op}: n={data['Total']} 勝率 {rate:.2%} (W{data['W']} D{data['D']} L{data['L']})")
        out("【持黑面對的主流開局】")
        for op, data in op_data["Black"]["Ops"][:5]:
            rate = (data["W"] + 0.5 * data["D"]) / data["Total"]
            out(f"  - 面對 {op}: n={data['Total']} 勝率 {rate:.2%} (W{data['W']} D{data['D']} L{data['L']})")
        out("\n" + "=" * 50 + "\n")

        len_data = self.analyze_lengths()
        if len_data:
            out("[4. 對局長度 (Full-Moves)]")
            out(f"整體平均 : {len_data['Overall'][0]:.1f} (σ={len_data['Overall'][1]:.1f})")
            out(f"獲勝平均 : {len_data['Wins'][0]:.1f} | 落敗 : {len_data['Losses'][0]:.1f} | 和局 : {len_data['Draws'][0]:.1f}")
            out("\n" + "=" * 50 + "\n")

        seq_data = self.analyze_sequence()
        if seq_data:
            out("[5. 賽果序列]")
            out(f"Lag-1 自相關 : {seq_data['Autocorrelation']:.4f}")
            out(f"游程檢定 p : {seq_data['Runs_P']:.4f}")
            out("\n" + "=" * 50 + "\n")

        out("[6. 對局對結果]")
        out(f"連續配對 (檔案順序 game 2k,2k+1) 對數 : {self.total_pairs}")
        if self.consecutive_pair_counts:
            for k, v in self.consecutive_pair_counts.most_common():
                out(f"  {k}: {v} ({v / self.total_pairs:.1%})" if self.total_pairs else f"  {k}: {v}")
        if self.fen_pair_counts:
            tot_f = sum(self.fen_pair_counts.values())
            out(f"同 FEN 換先配對 : {tot_f}")
            for k, v in self.fen_pair_counts.most_common():
                out(f"  (W then B for target) {k}: {v} ({v / tot_f:.1%})")
        out("  鍵 = 連續兩局 target 的 W/D/L；FEN 對則為 target 持白結果 + 持黑結果")
        out("\n" + "=" * 50 + "\n")

        for block in (
            self.analyze_length_buckets(),
            self.analyze_promos(),
            self.analyze_halves_and_cumulative(),
            self.analyze_mates(),
        ):
            for line in block:
                out(line)
            out("\n" + "=" * 50 + "\n")

        chart_paths = self.plot_all_charts(Path(output_dir))
        out("📈 圖表產出:")
        for cp in chart_paths:
            out(f"  - {cp.name}")
        out("\n提示: 搜尋深度/節點/NPS 請跑 parse_search_stats.py")
        out("================ 分析結束 ================\n")

        text = "\n".join(lines)
        print(text)
        if write_file:
            report_path = Path(output_dir) / "game_stats_report.txt"
            report_path.write_text(text, encoding="utf-8")
            print(f"報告已寫入 {report_path}")
        return text


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tournament game-level statistics")
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN, help="PGN path")
    parser.add_argument("--target", default=None, help="Target engine name (default: auto / New)")
    parser.add_argument("--out-dir", type=Path, default=HERE, help="Output directory for plot/report")
    parser.add_argument("--peek", action="store_true", help="Only print PGN header summary")
    parser.add_argument("--no-report-file", action="store_true", help="Do not write game_stats_report.txt")
    args = parser.parse_args(argv)

    if not args.pgn.exists():
        print(f"Error: PGN not found: {args.pgn}", file=sys.stderr)
        return 1

    if args.peek:
        peek_pgn(args.pgn)
        return 0

    engines = detect_engines(args.pgn)
    target = args.target
    if target is None:
        if "New" in engines:
            target = "New"
        elif engines:
            target = engines[0]
        else:
            target = "New"

    print(f"PGN: {args.pgn}")
    print(f"Detected engines: {engines}")
    print(f"Target: {target}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    analyzer = ChessEngineAnalyzer(target_engine=target)
    analyzer.parse_pgn(args.pgn)
    analyzer.generate_report(args.out_dir, write_file=not args.no_report_file)
    return 0


if __name__ == "__main__":
    # Allow running with cwd = repo root or tournament_analysis/
    if not DEFAULT_PGN.exists():
        alt = Path("tournament_results.pgn")
        if alt.exists():
            # re-bind for local runs inside the folder
            pass
    raise SystemExit(main())
