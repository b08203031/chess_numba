import re
import os
import argparse
import numpy as np
import scipy.stats as sp_stats
import matplotlib.pyplot as plt

def parse_pgn(file_path, name1, name2):
    if not os.path.exists(file_path):
        print(f"Error: PGN file not found: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by games
    game_blocks = content.split('[Event "')
    
    new_stats = {'depth': [], 'nodes': [], 'time': [], 'move_num': []}
    old_stats = {'depth': [], 'nodes': [], 'time': [], 'move_num': []}

    # Regex patterns
    # Matches white moves: 1. Qxe4+ { d=11, eval=+0.07, n=332609, t=528ms }
    # Avoid matching black move indicators like 1...
    white_move_pattern = re.compile(r'\b(?<!\.)\b(\d+)\.\s+(\S+)\s*\{\s*d=(\d+),\s*eval=([^,]+),\s*n=(\d+),\s*t=(\d+)ms\s*\}')
    # Matches black moves: 1... Nxe4 { d=13, eval=+0.05, n=349642, t=562ms }
    black_move_pattern = re.compile(r'\b(\d+)\.\.\.\s+(\S+)\s*\{\s*d=(\d+),\s*eval=([^,]+),\s*n=(\d+),\s*t=(\d+)ms\s*\}')

    games_parsed = 0
    for block in game_blocks:
        if not block.strip():
            continue
        
        # Determine who is White and Black
        white_match = re.search(r'\[White\s+"([^"]+)"\]', block)
        black_match = re.search(r'\[Black\s+"([^"]+)"\]', block)
        
        if not white_match or not black_match:
            continue
            
        white_player = white_match.group(1)
        black_player = black_match.group(1)
        
        # We only care about games involving name1 and name2
        if white_player not in [name1, name2] or black_player not in [name1, name2]:
            continue
            
        games_parsed += 1
        
        # Find all white moves and black moves
        white_moves = white_move_pattern.findall(block)
        black_moves = black_move_pattern.findall(block)
        
        # Parse white moves
        for move in white_moves:
            move_num = int(move[0])
            depth = int(move[2])
            nodes = int(move[4])
            time_ms = int(move[5])
            
            if white_player == name1:
                new_stats['depth'].append(depth)
                new_stats['nodes'].append(nodes)
                new_stats['time'].append(time_ms)
                new_stats['move_num'].append(move_num)
            else:
                old_stats['depth'].append(depth)
                old_stats['nodes'].append(nodes)
                old_stats['time'].append(time_ms)
                old_stats['move_num'].append(move_num)
                
        # Parse black moves
        for move in black_moves:
            move_num = int(move[0])
            depth = int(move[2])
            nodes = int(move[4])
            time_ms = int(move[5])
            
            if black_player == name1:
                new_stats['depth'].append(depth)
                new_stats['nodes'].append(nodes)
                new_stats['time'].append(time_ms)
                new_stats['move_num'].append(move_num)
            else:
                old_stats['depth'].append(depth)
                old_stats['nodes'].append(nodes)
                old_stats['time'].append(time_ms)
                old_stats['move_num'].append(move_num)

    print(f"Successfully parsed {games_parsed} games.")
    return new_stats, old_stats

def plot_line_metric(new_moves, new_means, new_sems, old_moves, old_means, old_sems, 
                     title, xlabel, ylabel, filename, name1, name2, use_log=False):
    plt.figure(figsize=(10, 5.5))
    
    # Plot New Engine
    plt.plot(new_moves, new_means, color='#1f77b4', label=f'{name1} (Avg)', linewidth=2.0, marker='o', markersize=4)
    if len(new_sems) > 0:
        plt.fill_between(new_moves, new_means - new_sems, new_means + new_sems, color='#1f77b4', alpha=0.15, label=f'{name1} SEM')
        
    # Plot Old Engine
    plt.plot(old_moves, old_means, color='#ff7f0e', label=f'{name2} (Avg)', linewidth=2.0, marker='s', markersize=4)
    if len(old_sems) > 0:
        plt.fill_between(old_moves, old_means - old_sems, old_means + old_sems, color='#ff7f0e', alpha=0.15, label=f'{name2} SEM')
        
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    
    if use_log:
        plt.yscale('log')
        
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def analyze_and_plot(new_stats, old_stats, output_dir, name1, name2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    new_depths = np.array(new_stats['depth'])
    new_nodes = np.array(new_stats['nodes'])
    new_times = np.array(new_stats['time'])
    new_move_nums = np.array(new_stats.get('move_num', []))
    
    old_depths = np.array(old_stats['depth'])
    old_nodes = np.array(old_stats['nodes'])
    old_times = np.array(old_stats['time'])
    old_move_nums = np.array(old_stats.get('move_num', []))
    
    if len(new_depths) == 0 or len(old_depths) == 0:
        print(f"Error: Not enough move data to analyze. {name1} moves: {len(new_depths)}, {name2} moves: {len(old_depths)}")
        return
        
    # Calculate NPS (Nodes Per Second)
    # Prevent division by zero
    new_nps = np.where(new_times > 0, new_nodes / (new_times / 1000.0), 0)
    old_nps = np.where(old_times > 0, old_nodes / (old_times / 1000.0), 0)
    
    # Filter out 0 NPS for statistics
    new_nps_clean = new_nps[new_nps > 0]
    old_nps_clean = old_nps[old_nps > 0]

    # Calculate Standard Deviations (ddof=1 for sample standard deviation)
    new_depth_std = np.std(new_depths, ddof=1) if len(new_depths) > 1 else 0.0
    old_depth_std = np.std(old_depths, ddof=1) if len(old_depths) > 1 else 0.0
    new_node_std = np.std(new_nodes, ddof=1) if len(new_nodes) > 1 else 0.0
    old_node_std = np.std(old_nodes, ddof=1) if len(old_nodes) > 1 else 0.0
    new_time_std = np.std(new_times, ddof=1) if len(new_times) > 1 else 0.0
    old_time_std = np.std(old_times, ddof=1) if len(old_times) > 1 else 0.0
    new_nps_std = np.std(new_nps_clean, ddof=1) if len(new_nps_clean) > 1 else 0.0
    old_nps_std = np.std(old_nps_clean, ddof=1) if len(old_nps_clean) > 1 else 0.0

    # Perform Welch's t-test to check if differences are statistically significant
    t_depth, p_depth = sp_stats.ttest_ind(new_depths, old_depths, equal_var=False)
    t_node, p_node = sp_stats.ttest_ind(new_nodes, old_nodes, equal_var=False)
    t_time, p_time = sp_stats.ttest_ind(new_times, old_times, equal_var=False)
    
    if len(new_nps_clean) > 0 and len(old_nps_clean) > 0:
        t_nps, p_nps = sp_stats.ttest_ind(new_nps_clean, old_nps_clean, equal_var=False)
    else:
        t_nps, p_nps = 0.0, 1.0

    report = []
    report.append("========== 搜尋效能與深度對比統計報告 (含標準差與顯著性分析) ==========\n")
    report.append(f"【樣本數據量】")
    report.append(f"  * {name1} 引擎總著步數: {len(new_depths)}")
    report.append(f"  * {name2} 引擎總著步數: {len(old_depths)}\n")
    
    report.append(f"【1. 搜尋深度 (Depth) 分析】")
    report.append(f"  * {name1} 引擎平均深度: {np.mean(new_depths):.2f} 層 (標準差: {new_depth_std:.2f}, 中位數: {np.median(new_depths):.0f}, 最大: {np.max(new_depths)})")
    report.append(f"  * {name2} 引擎平均深度: {np.mean(old_depths):.2f} 層 (標準差: {old_depth_std:.2f}, 中位數: {np.median(old_depths):.0f}, 最大: {np.max(old_depths)})")
    depth_diff = np.mean(new_depths) - np.mean(old_depths)
    report.append(f"  * 深度差異 ({name1} - {name2}): {depth_diff:+.2f} 層")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_depth:+.3f}, p-value = {p_depth:.4e}")
    if p_depth < 0.05:
        report.append("    [+] 深度差異具有統計顯著性 (p < 0.05)")
    else:
        report.append("    [-] 深度差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)")
    
    # Analyze proportion of deep searches (depth >= 14)
    new_deep_prop = np.mean(new_depths >= 14) * 100
    old_deep_prop = np.mean(old_depths >= 14) * 100
    report.append(f"  * 深度 >= 14 比例: {name1} {new_deep_prop:.1f}% vs {name2} {old_deep_prop:.1f}%\n")
    
    report.append(f"【2. 搜尋節點數 (Nodes) 分析】")
    report.append(f"  * {name1} 引擎平均節點數: {np.mean(new_nodes):.0f} (標準差: {new_node_std:.0f}, 中位數: {np.median(new_nodes):.0f})")
    report.append(f"  * {name2} 引擎平均節點數: {np.mean(old_nodes):.0f} (標準差: {old_node_std:.0f}, 中位數: {np.median(old_nodes):.0f})")
    nodes_diff_pct = ((np.mean(new_nodes) - np.mean(old_nodes)) / np.mean(old_nodes) * 100) if np.mean(old_nodes) > 0 else 0
    report.append(f"  * 節點數差異: {nodes_diff_pct:+.1f}%")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_node:+.3f}, p-value = {p_node:.4e}")
    if p_node < 0.05:
        report.append("    [+] 節點數差異具有統計顯著性 (p < 0.05)\n")
    else:
        report.append("    [-] 節點數差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)\n")
    
    report.append(f"【3. 每步耗時 (Time) 分析】")
    report.append(f"  * {name1} 引擎平均每步耗時: {np.mean(new_times):.1f} ms (標準差: {new_time_std:.1f} ms)")
    report.append(f"  * {name2} 引擎平均每步耗時: {np.mean(old_times):.1f} ms (標準差: {old_time_std:.1f} ms)")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_time:+.3f}, p-value = {p_time:.4e}")
    if p_time < 0.05:
        report.append("    [+] 耗時差異具有統計顯著性 (p < 0.05)\n")
    else:
        report.append("    [-] 耗時差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)\n")
    
    report.append(f"【4. 搜尋速度 (NPS) 分析】")
    if len(new_nps_clean) > 0 and len(old_nps_clean) > 0:
        report.append(f"  * {name1} 引擎平均 NPS: {np.mean(new_nps_clean):.0f} (標準差: {new_nps_std:.0f}, 中位數: {np.median(new_nps_clean):.0f})")
        report.append(f"  * {name2} 引擎平均 NPS: {np.mean(old_nps_clean):.0f} (標準差: {old_nps_std:.0f}, 中位數: {np.median(old_nps_clean):.0f})")
        nps_diff_pct = (np.mean(new_nps_clean) - np.mean(old_nps_clean)) / np.mean(old_nps_clean) * 100
        report.append(f"  * NPS 差異: {nps_diff_pct:+.1f}%")
        report.append(f"  * Welch's t-test 顯著性檢定: t = {t_nps:+.3f}, p-value = {p_nps:.4e}")
        if p_nps < 0.05:
            report.append("    [+] NPS 差異具有統計顯著性 (p < 0.05)\n")
        else:
            report.append("    [-] NPS 差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)\n")
    else:
        report.append("  * 無法計算 NPS 統計（耗時數據不全）\n")
    
    report.append("=========================================\n")
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(os.path.join(output_dir, "search_stats_report.txt"), "w", encoding="utf-8") as rf:
        rf.write(report_text)

    # Clean up old distribution plots if they exist
    for old_file in ["depth_distribution.png", "nps_distribution.png"]:
        file_path = os.path.join(output_dir, old_file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: could not remove old file {file_path}: {e}")

    # Calculate statistics per move number
    def get_metric_per_move(move_nums, values, max_move=100, min_samples=5):
        if len(move_nums) == 0:
            return np.array([]), np.array([]), np.array([])
        unique_moves = np.unique(move_nums)
        unique_moves = unique_moves[unique_moves <= max_move]
        
        m_list, mean_list, sem_list = [], [], []
        for m in unique_moves:
            mask = (move_nums == m)
            samples = values[mask]
            if len(samples) >= min_samples:
                m_list.append(m)
                mean_list.append(np.mean(samples))
                sem_list.append(np.std(samples, ddof=1) / np.sqrt(len(samples)) if len(samples) > 1 else 0.0)
        return np.array(m_list), np.array(mean_list), np.array(sem_list)

    # Group metrics by move number
    new_m_depth, new_mean_depth, new_sem_depth = get_metric_per_move(new_move_nums, new_depths)
    old_m_depth, old_mean_depth, old_sem_depth = get_metric_per_move(old_move_nums, old_depths)

    new_m_nodes, new_mean_nodes, new_sem_nodes = get_metric_per_move(new_move_nums, new_nodes)
    old_m_nodes, old_mean_nodes, old_sem_nodes = get_metric_per_move(old_move_nums, old_nodes)

    new_m_time, new_mean_time, new_sem_time = get_metric_per_move(new_move_nums, new_times)
    old_m_time, old_mean_time, old_sem_time = get_metric_per_move(old_move_nums, old_times)

    # Plot Line Charts per Move
    # 1. Depth per Move
    if len(new_m_depth) > 0 and len(old_m_depth) > 0:
        plot_line_metric(
            new_m_depth, new_mean_depth, new_sem_depth,
            old_m_depth, old_mean_depth, old_sem_depth,
            title=f"Average Search Depth per Move ({name1} vs {name2})",
            xlabel="Move Number",
            ylabel="Average Depth (Plies)",
            filename=os.path.join(output_dir, "depth_per_move.png"),
            name1=name1,
            name2=name2
        )

    # 2. Nodes per Move
    if len(new_m_nodes) > 0 and len(old_m_nodes) > 0:
        plot_line_metric(
            new_m_nodes, new_mean_nodes, new_sem_nodes,
            old_m_nodes, old_mean_nodes, old_sem_nodes,
            title=f"Average Nodes Searched per Move ({name1} vs {name2})",
            xlabel="Move Number",
            ylabel="Average Nodes Searched",
            filename=os.path.join(output_dir, "nodes_per_move.png"),
            name1=name1,
            name2=name2,
            use_log=True
        )

    # 3. Time per Move
    if len(new_m_time) > 0 and len(old_m_time) > 0:
        plot_line_metric(
            new_m_time, new_mean_time, new_sem_time,
            old_m_time, old_mean_time, old_sem_time,
            title=f"Average Search Time per Move ({name1} vs {name2})",
            xlabel="Move Number",
            ylabel="Average Search Time (ms)",
            filename=os.path.join(output_dir, "time_per_move.png"),
            name1=name1,
            name2=name2
        )

    # Plot 4: Average Nodes vs Depth
    plt.figure(figsize=(10, 5))
    common_depths = sorted(list(set(new_depths).intersection(set(old_depths))))
    # Filter common depths to reasonable range
    common_depths = [d for d in common_depths if d < 30] # Filter out mates
    
    if common_depths:
        new_avg_nodes = [np.mean(new_nodes[new_depths == d]) for d in common_depths]
        old_avg_nodes = [np.mean(old_nodes[old_depths == d]) for d in common_depths]
        
        plt.plot(common_depths, new_avg_nodes, marker='o', label=f'{name1} Engine', color='#1f77b4', linewidth=2)
        plt.plot(common_depths, old_avg_nodes, marker='s', label=f'{name2} Engine', color='#ff7f0e', linewidth=2)
        plt.title(f"Average Nodes Searched vs Depth ({name1} vs {name2})")
        plt.xlabel("Search Depth (Plies)")
        plt.ylabel("Average Nodes Searched")
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avg_nodes_vs_depth.png"), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgn_path = os.path.join(project_root, "tournament_analysis", "tournament_results.pgn")
    out_dir = os.path.join(project_root, "tournament_analysis")
    
    parser = argparse.ArgumentParser(description="Parse search stats from PGN.")
    parser.add_argument("--name1", default=None, help="Name of the first engine (e.g., New/MyEngine)")
    parser.add_argument("--name2", default=None, help="Name of the second engine (e.g., Old/Stockfish)")
    args = parser.parse_args()

    # Auto-detect names if not specified
    name1 = args.name1
    name2 = args.name2
    
    if name1 is None or name2 is None:
        if os.path.exists(pgn_path):
            with open(pgn_path, 'r', encoding='utf-8') as f:
                content = f.read()
            white_names = re.findall(r'\[White\s+"([^"]+)"\]', content)
            black_names = re.findall(r'\[Black\s+"([^"]+)"\]', content)
            unique_names = list(set(white_names + black_names))
            
            # Prefer 'New' and 'Old'
            if 'New' in unique_names and 'Old' in unique_names:
                name1 = 'New'
                name2 = 'Old'
            # Or 'MyEngine' and 'Stockfish'
            elif 'MyEngine' in unique_names and 'Stockfish' in unique_names:
                name1 = 'MyEngine'
                name2 = 'Stockfish'
            elif len(unique_names) >= 2:
                # Retain consistent order of appearance or alphabetical order
                unique_names.sort()
                name1 = unique_names[0]
                name2 = unique_names[1]
            elif len(unique_names) == 1:
                name1 = unique_names[0]
                name2 = "Unknown"
            else:
                name1 = "New"
                name2 = "Old"
        else:
            name1 = "New"
            name2 = "Old"

    print(f"Parsing stats comparing Player 1: '{name1}' vs Player 2: '{name2}'")
    stats = parse_pgn(pgn_path, name1, name2)
    if stats:
        analyze_and_plot(stats[0], stats[1], out_dir, name1, name2)
