import re
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class ChessEngineAnalyzer:
    def __init__(self, target_engine="chess_engine"):
        self.target_engine = target_engine
        # 基本計數
        self.w_wins, self.w_draws, self.w_losses = 0, 0, 0
        self.b_wins, self.b_draws, self.b_losses = 0, 0, 0
        
        # 對局特徵
        self.lengths = []
        self.lengths_w = []
        self.lengths_l = []
        self.lengths_d = []
        
        # 開局統計 (分持白與持黑)
        self.openings_white = {}
        self.openings_black = {}
        
        # 賽果序列 (1.0=勝, 0.5=和, 0.0=負)
        self.score_sequence = []
        
        # 對局對分析 (v2 視角: 白勝-黑勝, 白勝-黑和, ...)
        # 1.WW, 2.WD, 3.WL, 4.DW, 5.DD, 6.DL, 7.LW, 8.LD, 9.LL
        self.pair_outcomes = {
            "WW": 0, "WD": 0, "WL": 0,
            "DW": 0, "DD": 0, "DL": 0,
            "LW": 0, "LD": 0, "LL": 0
        }
        self.total_pairs = 0

    def parse_pgn(self, file_path):
        """讀取並解析 PGN 檔案"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 以 [Event 切割每局 (忽略第一個空元素)
        games_raw = re.split(r'\[Event ', content)[1:]

        for game in games_raw:
            white_match = re.search(r'\[White "(.*?)"\]', game)
            black_match = re.search(r'\[Black "(.*?)"\]', game)
            result_match = re.search(r'\[Result "(.*?)"\]', game)
            
            if not (white_match and black_match and result_match):
                continue
            
            white = white_match.group(1)
            black = black_match.group(1)
            result = result_match.group(1)
            
            # 提取走法文本
            moves_match = re.search(r'\]\s*\n\s*\n(.*?)$', game, re.DOTALL)
            moves_text = moves_match.group(1) if moves_match else ""
            
            # 提取總步數 (Full-moves)
            move_numbers = re.findall(r'\b(\d+)\.', moves_text)
            num_moves = max([int(m) for m in move_numbers]) if move_numbers else 0
            
            # 提取第一步開局
            first_move_match = re.search(r'1\.\s+([a-zA-Z0-9]+)', moves_text)
            first_move = first_move_match.group(1) if first_move_match else "Unknown"
            
            is_target_white = (white == self.target_engine)
            
            # 決定相對賽果
            if result == "1-0":
                score = 1.0 if is_target_white else 0.0
                res_type = "W" if is_target_white else "L"
            elif result == "0-1":
                score = 0.0 if is_target_white else 1.0
                res_type = "L" if is_target_white else "W"
            else:
                score = 0.5
                res_type = "D"
                
            self.score_sequence.append(score)
            self.lengths.append(num_moves)
            
            if res_type == "W": self.lengths_w.append(num_moves)
            elif res_type == "L": self.lengths_l.append(num_moves)
            else: self.lengths_d.append(num_moves)
                
            if is_target_white:
                if res_type == "W": self.w_wins += 1
                elif res_type == "D": self.w_draws += 1
                else: self.w_losses += 1
                
                if first_move not in self.openings_white:
                    self.openings_white[first_move] = {"W": 0, "D": 0, "L": 0, "Total": 0}
                self.openings_white[first_move][res_type] += 1
                self.openings_white[first_move]["Total"] += 1
            elif black == self.target_engine: # 計算持黑時的開局與賽果
                if res_type == "W": self.b_wins += 1
                elif res_type == "D": self.b_draws += 1
                else: self.b_losses += 1
                
                if first_move not in self.openings_black:
                    self.openings_black[first_move] = {"W": 0, "D": 0, "L": 0, "Total": 0}
                self.openings_black[first_move][res_type] += 1
                self.openings_black[first_move]["Total"] += 1

        # 對局對分類分析 (假設 PGN 中每兩局為一組，且同一開局交換顏色)
        self._calculate_pair_stats()

    def _calculate_pair_stats(self):
        """分析九種組合比例 (假設偶數局為執白，奇數局為執黑)"""
        self.pair_outcomes = {
            "WW": 0, "WD": 0, "WL": 0,
            "DW": 0, "DD": 0, "DL": 0,
            "LW": 0, "LD": 0, "LL": 0
        }
        self.total_pairs = len(self.score_sequence) // 2
        
        for i in range(0, self.total_pairs * 2, 2):
            s_white = self.score_sequence[i]  # 偶數盤 (v2執白)
            s_black = self.score_sequence[i+1] # 奇數盤 (v2執黑)
            
            if s_white == 1.0:
                if s_black == 1.0: self.pair_outcomes["WW"] += 1
                elif s_black == 0.5: self.pair_outcomes["WD"] += 1
                else: self.pair_outcomes["WL"] += 1
            elif s_white == 0.5:
                if s_black == 1.0: self.pair_outcomes["DW"] += 1
                elif s_black == 0.5: self.pair_outcomes["DD"] += 1
                else: self.pair_outcomes["DL"] += 1
            else: # s_white == 0.0
                if s_black == 1.0: self.pair_outcomes["LW"] += 1
                elif s_black == 0.5: self.pair_outcomes["LD"] += 1
                else: self.pair_outcomes["LL"] += 1

    def calculate_elo_and_ci(self):
        """計算 Elo 差距與 95% 信賴區間"""
        total_w = self.w_wins + self.b_wins
        total_d = self.w_draws + self.b_draws
        total_l = self.w_losses + self.b_losses
        n = total_w + total_d + total_l
        
        if n == 0: return None
        
        score_expected = (total_w + 0.5 * total_d) / n
        
        def to_elo(s):
            if s >= 1.0: return float('inf')
            if s <= 0.0: return float('-inf')
            return -400 * math.log10(1/s - 1)
            
        elo_diff = to_elo(score_expected)
        
        # 計算樣本變異數與標準誤
        sum_sq = total_w * (1 - score_expected)**2 + total_d * (0.5 - score_expected)**2 + total_l * (0 - score_expected)**2
        std_dev = math.sqrt(sum_sq / (n - 1)) if n > 1 else 0
        std_err = std_dev / math.sqrt(n)
        
        z_val = 1.96 # 95% CI
        elo_min = to_elo(score_expected - z_val * std_err)
        elo_max = to_elo(score_expected + z_val * std_err)
        
        return {
            "Total": n, "W": total_w, "D": total_d, "L": total_l,
            "Expected_Score": score_expected,
            "Elo_Diff": elo_diff,
            "CI_95": (elo_min, elo_max),
            "Margin_of_Error": (elo_max - elo_min) / 2,
            "Draw_Rate": total_d / n if n > 0 else 0
        }

    def plot_cumulative_wins(self, output_file="tournament_analysis/cumulative_wins.png"):
        """繪製累積勝率圖與理論期望線"""
        wins_seq = [1 if s == 1.0 else 0 for s in self.score_sequence]
        cumulative = np.cumsum(wins_seq)
        
        total_games = len(wins_seq)
        if total_games == 0: return
        
        avg_win_rate = cumulative[-1] / total_games
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, total_games + 1), cumulative, label="Cumulative Wins", color="#1f77b4", linewidth=2)
        plt.plot([1, total_games], [avg_win_rate, cumulative[-1]], linestyle="--", color="#d62728", 
                 label=f"Ideal Trend (Slope: {avg_win_rate:.3f})")
                 
        plt.xlabel("Total Games ($N$)", fontsize=12)
        plt.ylabel("Cumulative Wins ($W$)", fontsize=12)
        plt.title(f"Cumulative Win Trajectory for {self.target_engine}", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def analyze_color_bias(self):
        """利用卡方檢定分析先手優勢"""
        obs = np.array([
            [self.w_wins, self.w_draws, self.w_losses],
            [self.b_wins, self.b_draws, self.b_losses]
        ])
        
        # 若任何一列皆為 0，則無法檢定
        if np.sum(obs[0]) == 0 or np.sum(obs[1]) == 0:
            return None
            
        chi2, p_val, dof, expected = stats.chi2_contingency(obs)
        w_rate = (self.w_wins + 0.5 * self.w_draws) / np.sum(obs[0])
        b_rate = (self.b_wins + 0.5 * self.b_draws) / np.sum(obs[1])
        
        # 總體白黑勝率 (以全局白方視角看)
        white_perspective_w = self.w_wins + self.b_losses
        white_perspective_d = self.w_draws + self.b_draws
        white_perspective_l = self.w_losses + self.b_wins
        total_games = white_perspective_w + white_perspective_d + white_perspective_l
        
        white_overall_wr = (white_perspective_w + 0.5 * white_perspective_d) / total_games if total_games > 0 else 0
        
        return {
            "W_Rate": w_rate, "B_Rate": b_rate, 
            "Chi2": chi2, "p_value": p_val,
            "White_Overall_WR": white_overall_wr
        }

    def _analyze_opening_dict(self, openings_dict):
        sorted_ops = sorted(openings_dict.items(), key=lambda x: x[1]["Total"], reverse=True)
        if len(sorted_ops) < 2: return sorted_ops, None
        
        op1_name, op1_data = sorted_ops[0]
        op2_name, op2_data = sorted_ops[1]
        
        n1, n2 = op1_data["Total"], op2_data["Total"]
        p1 = (op1_data["W"] + 0.5 * op1_data["D"]) / n1
        p2 = (op2_data["W"] + 0.5 * op2_data["D"]) / n2
        
        p_pool = (op1_data["W"] + 0.5 * op1_data["D"] + op2_data["W"] + 0.5 * op2_data["D"]) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if n1 > 0 and n2 > 0 else 0
        z_score = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return sorted_ops, {"op1": op1_name, "op2": op2_name, "Z": z_score, "p_value": p_val}

    def analyze_openings(self):
        """分析主流開局的效率與雙比例 Z-test"""
        w_ops, w_z = self._analyze_opening_dict(self.openings_white)
        b_ops, b_z = self._analyze_opening_dict(self.openings_black)
        
        return {"White": {"Ops": w_ops, "Z": w_z}, "Black": {"Ops": b_ops, "Z": b_z}}

    def analyze_lengths(self):
        """分析對局長度的動差"""
        if not self.lengths:
            return None
        return {
            "Overall": (np.mean(self.lengths), np.std(self.lengths)),
            "Wins": (np.mean(self.lengths_w), np.std(self.lengths_w)) if self.lengths_w else (0, 0),
            "Losses": (np.mean(self.lengths_l), np.std(self.lengths_l)) if self.lengths_l else (0, 0),
            "Draws": (np.mean(self.lengths_d), np.std(self.lengths_d)) if self.lengths_d else (0, 0)
        }

    def analyze_sequence(self):
        """分析賽果序列的自相關性與馬可夫性質"""
        if len(self.score_sequence) < 2: return None
        seq = np.array(self.score_sequence)
        
        autocorr = np.corrcoef(seq[:-1], seq[1:])[0, 1] if np.std(seq[:-1]) > 0 and np.std(seq[1:]) > 0 else 0
        
        median_score = np.median(seq)
        binary_seq = (seq > median_score).astype(int)
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] != binary_seq[i-1]:
                runs += 1
                
        n1 = np.sum(binary_seq)
        n0 = len(binary_seq) - n1
        
        if n0 == 0 or n1 == 0:
            return {"Autocorrelation": autocorr, "Runs_Z": 0, "Runs_P": 1.0}
            
        expected_runs = 2 * n0 * n1 / len(binary_seq) + 1
        var_runs = (expected_runs - 1) * (expected_runs - 2) / (len(binary_seq) - 1)
        z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
        
        return {"Autocorrelation": autocorr, "Runs_Z": z_runs, "Runs_P": p_runs}

    def generate_report(self):
        """列印完整且易懂的統計報告"""
        print(f"========== 引擎對戰深度分析報告 ({self.target_engine}) ==========\n")
        
        elo_data = self.calculate_elo_and_ci()
        if not elo_data:
            print("沒有足夠的對局數據可供分析。")
            return
            
        print("[1. 總體戰績與 Elo 表現]")
        print(f"總對局數 : {elo_data['Total']} 局")
        print(f"戰績統計 : 勝 {elo_data['W']} | 和 {elo_data['D']} | 負 {elo_data['L']}")
        print(f"勝率(含和) : {elo_data['Expected_Score']:.2%}")
        print(f"和局率 : {elo_data['Draw_Rate']:.2%}")
        print(f"相對 Elo 變化 : {elo_data['Elo_Diff']:.2f} (大於0代表較強，小於0代表較弱)")
        print(f"95% 信賴區間 : [{elo_data['CI_95'][0]:.2f}, {elo_data['CI_95'][1]:.2f}]")
        print("💡【數據意義】")
        if elo_data['CI_95'][0] > 0:
            print("  ✅ 您的引擎有 95% 信心水準證實【明顯強於對手】（信賴區間下界 > 0）。")
        elif elo_data['CI_95'][1] < 0:
            print("  ❌ 您的引擎有 95% 信心水準證實【明顯弱於對手】（信賴區間上界 < 0）。")
        else:
            print("  ⚖️ 目前對局數不足以證明雙方有顯著實力差距（信賴區間包含 0）。需更多對局。")
        print("\n" + "="*50 + "\n")
        
        
        color_data = self.analyze_color_bias()
        if color_data:
            print("[2. 先手優勢與執色分析]")
            print(f"持白時勝率 : {color_data['W_Rate']:.2%} (勝:{self.w_wins} 和:{self.w_draws} 負:{self.w_losses})")
            print(f"持黑時勝率 : {color_data['B_Rate']:.2%} (勝:{self.b_wins} 和:{self.b_draws} 負:{self.b_losses})")
            print(f"賽局總體白方勝率 : {color_data['White_Overall_WR']:.2%} (包含雙方，可看出該賽制下白方是否占優)")
            print(f"卡方 p-value : {color_data['p_value']:.4e} (用於檢定您的引擎持白與持黑勝率是否有顯著差異)")
            print("💡【數據意義】")
            if color_data['p_value'] < 0.05:
                if color_data['W_Rate'] > color_data['B_Rate']:
                    print("  📈 【具有顯著先手優勢】：您的引擎持白時的表現顯著優於持黑時的表現。")
                else:
                    print("  ⚠️ 【反常現象】：您的引擎持黑時的表現居然顯著優於持白。")
                    print("      這暗示您的引擎可能存在後手反擊優勢，或持白時開局庫/搜索存在缺陷。")
            else:
                print("  ⚖️ 【無顯著執色差異】：統計上來說，您的引擎持白或持黑的表現差異不大，")
                print("      或樣本數尚不足以證明差異（p-value > 0.05）。")
            print("\n" + "="*50 + "\n")
        
        op_data = self.analyze_openings()
        print("[3. 開局策略分析 (Top 3)]")
        print("【您的引擎持白時的主流開局】")
        for op, data in op_data["White"]["Ops"][:3]:
            rate = (data['W'] + 0.5 * data['D']) / data['Total']
            print(f"  - 第一步 {op} : 共 {data['Total']} 局 | 勝率 {rate:.2%} (勝:{data['W']} 和:{data['D']} 負:{data['L']})")
        
        w_zt = op_data["White"]["Z"]
        if w_zt:
            print("  💡 雙比例 Z 檢定 (比較第一與第二常用開局的勝率差異):")
            print(f"     {w_zt['op1']} vs {w_zt['op2']} -> p-value = {w_zt['p_value']:.4f}")
            if w_zt['p_value'] < 0.05:
                print("     (這兩個開局的期望勝率有統計上的顯著差異！)")
        
        print("\n【您的引擎持黑時面對的主流開局】")
        for op, data in op_data["Black"]["Ops"][:3]:
            rate = (data['W'] + 0.5 * data['D']) / data['Total']
            print(f"  - 面對第一步 {op} : 共 {data['Total']} 局 | 勝率 {rate:.2%} (勝:{data['W']} 和:{data['D']} 負:{data['L']})")
            
        print("\n" + "="*50 + "\n")
            
        len_data = self.analyze_lengths()
        if len_data:
            print("[4. 對局長度分析 (Full-Moves)]")
            print(f"整體平均步數 : {len_data['Overall'][0]:.1f} 步 (標準差: {len_data['Overall'][1]:.1f})")
            print(f"獲勝平均步數 : {len_data['Wins'][0]:.1f} 步")
            print(f"落敗平均步數 : {len_data['Losses'][0]:.1f} 步")
            print(f"和局平均步數 : {len_data['Draws'][0]:.1f} 步")
            print("💡【數據意義】")
            print("  了解引擎通常在什麼時候建立優勢或被擊敗。")
            print("  - 若和局步數很長，代表引擎在殘局有強韌抵抗力，或難以轉化微小優勢。")
            print("  - 若落敗步數很短，可能暗示有開局陷阱或快速崩盤的盲點。")
            print("\n" + "="*50 + "\n")
        
        seq_data = self.analyze_sequence()
        if seq_data:
            print("[5. 狀態序列分析 (戰績波動與連勝/連敗)]")
            print(f"Lag-1 自相關係數 : {seq_data['Autocorrelation']:.4f}")
            print(f"游程檢定 p-value : {seq_data['Runs_P']:.4f}")
            print("💡【數據意義】")
            if seq_data['Runs_P'] < 0.05:
                if seq_data['Autocorrelation'] > 0:
                    print("  ⚠️ 【動量效應】：引擎的狀態容易受到上一局影響，出現明顯的一波連勝 or 連敗。")
                else:
                    print("  ⚠️ 【劇烈震盪】：引擎的輸贏容易出現交替震盪的情況。")
            else:
                print("  ✅ 【獨立性良好】：每局對弈的結果近似於相互獨立，沒有明顯的連續波動干擾。")
            print("\n" + "="*50 + "\n")

        if self.total_pairs > 0:
            print("[6. 對局對 (Pairs) 結果分析 - 9宮格]")
            print(f"總對局對數 : {self.total_pairs}")
            
            outcomes = [
                ("白勝-黑勝 (WW)", "WW"), ("白勝-黑和 (WD)", "WD"), ("白勝-黑負 (WL)", "WL"),
                ("白和-黑勝 (DW)", "DW"), ("白和-黑和 (DD)", "DD"), ("白和-黑負 (DL)", "DL"),
                ("白負-黑勝 (LW)", "LW"), ("白負-黑和 (LD)", "LD"), ("白負-黑負 (LL)", "LL")
            ]
            
            for label, key in outcomes:
                count = self.pair_outcomes[key]
                percent = count / self.total_pairs
                print(f"  - {label:<16} : {count:>3} 盤 | {percent:.2%}")
            
            print("\n💡【數據意義】")
            print("  - WW / LL: 實力代差，完全不受執色影響。")
            print("  - WD / DW: 具備優勢，但存在某一執色下的技術瓶頸。")
            print("  - WL: 極端的白方開局優勢 (White Bias)；LW: 極端的黑方反擊優勢 (Black Bias)。")
            print("  - DD: 完美的勢均力敵，防禦體系極其成熟。")
            print("\n" + "="*50 + "\n")
        
        self.plot_cumulative_wins()
        print(f"📈 已生成累積勝率圖，儲存為 cumulative_wins.png")
        print("\n================ 分析結束 ================\n")

if __name__ == "__main__":
    # 確保當前目錄存在 tournament_results.pgn 檔案
    target_file = "tournament_analysis/tournament_results.pgn"
    try:
        analyzer = ChessEngineAnalyzer(target_engine="New")
        analyzer.parse_pgn(target_file)
        analyzer.generate_report()
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {target_file}。請確認檔案與程式在同一目錄。")