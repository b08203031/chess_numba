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
        
        # 開局統計 (target_engine 持白時)
        self.openings = {}
        
        # 賽果序列 (1.0=勝, 0.5=和, 0.0=負)
        self.score_sequence = []

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
                
                if first_move not in self.openings:
                    self.openings[first_move] = {"W": 0, "D": 0, "L": 0, "Total": 0}
                self.openings[first_move][res_type] += 1
                self.openings[first_move]["Total"] += 1
            else:
                if res_type == "W": self.b_wins += 1
                elif res_type == "D": self.b_draws += 1
                else: self.b_losses += 1

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
            "Margin_of_Error": (elo_max - elo_min) / 2
        }

    def plot_cumulative_wins(self, output_file="tournament_analysis/cumulative_wins.png"):
        """繪製累積勝場圖與理論期望線"""
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
        
        return {"W_Rate": w_rate, "B_Rate": b_rate, "Chi2": chi2, "p_value": p_val}

    def analyze_openings(self):
        """分析主流開局的效率與雙比例 Z-test"""
        sorted_ops = sorted(self.openings.items(), key=lambda x: x[1]["Total"], reverse=True)
        if len(sorted_ops) < 2: return sorted_ops
        
        op1_name, op1_data = sorted_ops[0]
        op2_name, op2_data = sorted_ops[1]
        
        n1, n2 = op1_data["Total"], op2_data["Total"]
        p1 = (op1_data["W"] + 0.5 * op1_data["D"]) / n1
        p2 = (op2_data["W"] + 0.5 * op2_data["D"]) / n2
        
        p_pool = (op1_data["W"] + 0.5 * op1_data["D"] + op2_data["W"] + 0.5 * op2_data["D"]) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z_score = (p1 - p2) / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {"Top_Openings": sorted_ops, "Z_test": {"op1": op1_name, "op2": op2_name, "Z": z_score, "p_value": p_val}}

    def analyze_lengths(self):
        """分析對局長度的動差"""
        return {
            "Overall": (np.mean(self.lengths), np.std(self.lengths)),
            "Wins": (np.mean(self.lengths_w), np.std(self.lengths_w)),
            "Losses": (np.mean(self.lengths_l), np.std(self.lengths_l)),
            "Draws": (np.mean(self.lengths_d), np.std(self.lengths_d))
        }

    def analyze_sequence(self):
        """分析賽果序列的自相關性與馬可夫性質"""
        if len(self.score_sequence) < 2: return None
        seq = np.array(self.score_sequence)
        
        # Lag-1 自相關
        autocorr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
        
        # 游程檢定 (Wald-Wolfowitz Runs Test)
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
        """列印完整統計報告"""
        print(f"========== 引擎對戰深度分析報告 ({self.target_engine}) ==========\n")
        
        elo_data = self.calculate_elo_and_ci()
        print("[1. 總體戰績與 Elo 映射]")
        print(f"總對局數: {elo_data['Total']} (勝: {elo_data['W']}, 和: {elo_data['D']}, 負: {elo_data['L']})")
        print(f"期望得分率 (E): {elo_data['Expected_Score']:.4f}")
        print(f"相對 Elo 差距: {elo_data['Elo_Diff']:.2f}")
        print(f"95% 信賴區間: [{elo_data['CI_95'][0]:.2f}, {elo_data['CI_95'][1]:.2f}] (誤差界限: ±{elo_data['Margin_of_Error']:.2f})\n")
        
        color_data = self.analyze_color_bias()
        print("[2. 對稱性破缺與執色優勢 (Chi-Square Test)]")
        print(f"持白勝率期望: {color_data['W_Rate']:.4f} (勝:{self.w_wins} 和:{self.w_draws} 負:{self.w_losses})")
        print(f"持黑勝率期望: {color_data['B_Rate']:.4f} (勝:{self.b_wins} 和:{self.b_draws} 負:{self.b_losses})")
        print(f"卡方統計量 (Chi^2): {color_data['Chi2']:.4f}, p-value: {color_data['p_value']:.4e}\n")
        
        op_data = self.analyze_openings()
        print("[3. 持白開局策略分析]")
        for op, data in op_data["Top_Openings"][:3]: # 印出前三大開局
            rate = (data['W'] + 0.5 * data['D']) / data['Total']
            print(f"開局 {op}: {data['Total']} 局, 勝率期望: {rate:.4f} (勝:{data['W']} 和:{data['D']} 負:{data['L']})")
        if "Z_test" in op_data:
            zt = op_data["Z_test"]
            print(f"主流開局 {zt['op1']} vs {zt['op2']} 雙比例 Z-test:")
            print(f"Z-score: {zt['Z']:.4f}, p-value: {zt['p_value']:.4f}\n")
            
        len_data = self.analyze_lengths()
        print("[4. 狀態空間收斂深度 (對局步數分佈)]")
        print(f"總體平均: {len_data['Overall'][0]:.2f} ± {len_data['Overall'][1]:.2f} 步")
        print(f"獲勝平均: {len_data['Wins'][0]:.2f} ± {len_data['Wins'][1]:.2f} 步")
        print(f"落敗平均: {len_data['Losses'][0]:.2f} ± {len_data['Losses'][1]:.2f} 步")
        print(f"和局平均: {len_data['Draws'][0]:.2f} ± {len_data['Draws'][1]:.2f} 步\n")
        
        seq_data = self.analyze_sequence()
        print("[5. 系統馬可夫性質與序列獨立性]")
        print(f"Lag-1 自相關係數: {seq_data['Autocorrelation']:.4f}")
        print(f"游程檢定 Z-score: {seq_data['Runs_Z']:.4f}, p-value: {seq_data['Runs_P']:.4f}\n")
        
        self.plot_cumulative_wins()
        print(f"已生成累積勝場圖，儲存為 cumulative_wins.png")

if __name__ == "__main__":
    # 確保當前目錄存在 tournament_results.pgn 檔案
    target_file = "tournament_analysis/tournament_results.pgn"
    try:
        analyzer = ChessEngineAnalyzer(target_engine="chess_engine")
        analyzer.parse_pgn(target_file)
        analyzer.generate_report()
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {target_file}。請確認檔案與程式在同一目錄。")