import re
import os
import numpy as np
import scipy.stats as sp_stats
import matplotlib.pyplot as plt

def parse_pgn(file_path):
    if not os.path.exists(file_path):
        print(f"Error: PGN file not found: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by games
    game_blocks = content.split('[Event "')
    
    new_stats = {'depth': [], 'nodes': [], 'time': []}
    old_stats = {'depth': [], 'nodes': [], 'time': []}

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
        
        # We only care about games involving New and Old
        if white_player not in ['New', 'Old'] or black_player not in ['New', 'Old']:
            continue
            
        games_parsed += 1
        
        # Find all white moves and black moves
        white_moves = white_move_pattern.findall(block)
        black_moves = black_move_pattern.findall(block)
        
        # Parse white moves
        for move in white_moves:
            depth = int(move[2])
            nodes = int(move[4])
            time_ms = int(move[5])
            
            if white_player == 'New':
                new_stats['depth'].append(depth)
                new_stats['nodes'].append(nodes)
                new_stats['time'].append(time_ms)
            else:
                old_stats['depth'].append(depth)
                old_stats['nodes'].append(nodes)
                old_stats['time'].append(time_ms)
                
        # Parse black moves
        for move in black_moves:
            depth = int(move[2])
            nodes = int(move[4])
            time_ms = int(move[5])
            
            if black_player == 'New':
                new_stats['depth'].append(depth)
                new_stats['nodes'].append(nodes)
                new_stats['time'].append(time_ms)
            else:
                old_stats['depth'].append(depth)
                old_stats['nodes'].append(nodes)
                old_stats['time'].append(time_ms)

    print(f"Successfully parsed {games_parsed} games.")
    return new_stats, old_stats

def analyze_and_plot(new_stats, old_stats, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    new_depths = np.array(new_stats['depth'])
    new_nodes = np.array(new_stats['nodes'])
    new_times = np.array(new_stats['time'])
    
    old_depths = np.array(old_stats['depth'])
    old_nodes = np.array(old_stats['nodes'])
    old_times = np.array(old_stats['time'])
    
    # Calculate NPS (Nodes Per Second)
    # Prevent division by zero
    new_nps = np.where(new_times > 0, new_nodes / (new_times / 1000.0), 0)
    old_nps = np.where(old_times > 0, old_nodes / (old_times / 1000.0), 0)
    
    # Filter out 0 NPS for statistics
    new_nps_clean = new_nps[new_nps > 0]
    old_nps_clean = old_nps[old_nps > 0]

    # Calculate Standard Deviations (ddof=1 for sample standard deviation)
    new_depth_std = np.std(new_depths, ddof=1)
    old_depth_std = np.std(old_depths, ddof=1)
    new_node_std = np.std(new_nodes, ddof=1)
    old_node_std = np.std(old_nodes, ddof=1)
    new_time_std = np.std(new_times, ddof=1)
    old_time_std = np.std(old_times, ddof=1)
    new_nps_std = np.std(new_nps_clean, ddof=1)
    old_nps_std = np.std(old_nps_clean, ddof=1)

    # Perform Welch's t-test to check if differences are statistically significant
    t_depth, p_depth = sp_stats.ttest_ind(new_depths, old_depths, equal_var=False)
    t_node, p_node = sp_stats.ttest_ind(new_nodes, old_nodes, equal_var=False)
    t_time, p_time = sp_stats.ttest_ind(new_times, old_times, equal_var=False)
    t_nps, p_nps = sp_stats.ttest_ind(new_nps_clean, old_nps_clean, equal_var=False)

    report = []
    report.append("========== 搜尋效能與深度對比統計報告 (含標準差與顯著性分析) ==========\n")
    report.append(f"【樣本數據量】")
    report.append(f"  * New 引擎總著步數: {len(new_depths)}")
    report.append(f"  * Old 引擎總著步數: {len(old_depths)}\n")
    
    report.append(f"【1. 搜尋深度 (Depth) 分析】")
    report.append(f"  * New 引擎平均深度: {np.mean(new_depths):.2f} 層 (標準差: {new_depth_std:.2f}, 中位數: {np.median(new_depths):.0f}, 最大: {np.max(new_depths)})")
    report.append(f"  * Old 引擎平均深度: {np.mean(old_depths):.2f} 層 (標準差: {old_depth_std:.2f}, 中位數: {np.median(old_depths):.0f}, 最大: {np.max(old_depths)})")
    depth_diff = np.mean(new_depths) - np.mean(old_depths)
    report.append(f"  * 深度差異 (New - Old): {depth_diff:+.2f} 層")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_depth:+.3f}, p-value = {p_depth:.4e}")
    if p_depth < 0.05:
        report.append("    ✅ 深度差異具有統計顯著性 (p < 0.05)")
    else:
        report.append("    ⚖️ 深度差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)")
    
    # Analyze proportion of deep searches (depth >= 14)
    new_deep_prop = np.mean(new_depths >= 14) * 100
    old_deep_prop = np.mean(old_depths >= 14) * 100
    report.append(f"  * 深度 >= 14 比例: New {new_deep_prop:.1f}% vs Old {old_deep_prop:.1f}%\n")
    
    report.append(f"【2. 搜尋節點數 (Nodes) 分析】")
    report.append(f"  * New 引擎平均節點數: {np.mean(new_nodes):.0f} (標準差: {new_node_std:.0f}, 中位數: {np.median(new_nodes):.0f})")
    report.append(f"  * Old 引擎平均節點數: {np.mean(old_nodes):.0f} (標準差: {old_node_std:.0f}, 中位數: {np.median(old_nodes):.0f})")
    nodes_diff_pct = (np.mean(new_nodes) - np.mean(old_nodes)) / np.mean(old_nodes) * 100
    report.append(f"  * 節點數差異: {nodes_diff_pct:+.1f}%")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_node:+.3f}, p-value = {p_node:.4e}")
    if p_node < 0.05:
        report.append("    ✅ 節點數差異具有統計顯著性 (p < 0.05)\n")
    else:
        report.append("    ⚖️ 節點數差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)\n")
    
    report.append(f"【3. 每步耗時 (Time) 分析】")
    report.append(f"  * New 引擎平均每步耗時: {np.mean(new_times):.1f} ms (標準差: {new_time_std:.1f} ms)")
    report.append(f"  * Old 引擎平均每步耗時: {np.mean(old_times):.1f} ms (標準差: {old_time_std:.1f} ms)")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_time:+.3f}, p-value = {p_time:.4e}")
    if p_time < 0.05:
        report.append("    ✅ 耗時差異具有統計顯著性 (p < 0.05)\n")
    else:
        report.append("    ⚖️ 耗時差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)\n")
    
    report.append(f"【4. 搜尋速度 (NPS) 分析】")
    report.append(f"  * New 引擎平均 NPS: {np.mean(new_nps_clean):.0f} (標準差: {new_nps_std:.0f}, 中位數: {np.median(new_nps_clean):.0f})")
    report.append(f"  * Old 引擎平均 NPS: {np.mean(old_nps_clean):.0f} (標準差: {old_nps_std:.0f}, 中位數: {np.median(old_nps_clean):.0f})")
    nps_diff_pct = (np.mean(new_nps_clean) - np.mean(old_nps_clean)) / np.mean(old_nps_clean) * 100
    report.append(f"  * NPS 差異: {nps_diff_pct:+.1f}%")
    report.append(f"  * Welch's t-test 顯著性檢定: t = {t_nps:+.3f}, p-value = {p_nps:.4e}")
    if p_nps < 0.05:
        report.append("    ✅ NPS 差異具有統計顯著性 (p < 0.05)\n")
    else:
        report.append("    ⚖️ NPS 差異無統計顯著性，可能為隨機統計誤差 (p >= 0.05)\n")
    
    report.append("=========================================\n")
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(os.path.join(output_dir, "search_stats_report.txt"), "w", encoding="utf-8") as rf:
        rf.write(report_text)


    # Plot 1: Depth Distribution Comparison
    plt.figure(figsize=(10, 5))
    min_d = min(np.min(new_depths), np.min(old_depths))
    max_d = max(np.max(new_depths), np.max(old_depths))
    bins = np.arange(min_d, max_d + 2) - 0.5
    
    plt.hist([new_depths, old_depths], bins=bins, label=['New Engine', 'Old Engine'], 
             alpha=0.8, color=['#1f77b4', '#ff7f0e'], density=True)
    plt.title("Search Depth Distribution Comparison")
    plt.xlabel("Search Depth (Plies)")
    plt.ylabel("Proportion of Moves")
    plt.xticks(np.arange(min_d, max_d + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "depth_distribution.png"), dpi=150)
    plt.close()
    
    # Plot 2: Average Nodes vs Depth
    plt.figure(figsize=(10, 5))
    common_depths = sorted(list(set(new_depths).intersection(set(old_depths))))
    # Filter common depths to reasonable range
    common_depths = [d for d in common_depths if d < 30] # Filter out mates
    
    new_avg_nodes = [np.mean(new_nodes[new_depths == d]) for d in common_depths]
    old_avg_nodes = [np.mean(old_nodes[old_depths == d]) for d in common_depths]
    
    plt.plot(common_depths, new_avg_nodes, marker='o', label='New Engine', color='#1f77b4', linewidth=2)
    plt.plot(common_depths, old_avg_nodes, marker='s', label='Old Engine', color='#ff7f0e', linewidth=2)
    plt.title("Average Nodes Searched vs Depth")
    plt.xlabel("Search Depth (Plies)")
    plt.ylabel("Average Nodes Searched")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_nodes_vs_depth.png"), dpi=150)
    plt.close()

    # Plot 3: Cumulative NPS distribution
    plt.figure(figsize=(10, 5))
    plt.hist([new_nps_clean / 1000.0, old_nps_clean / 1000.0], bins=30, label=['New NPS', 'Old NPS'],
             alpha=0.7, color=['#2ca02c', '#d62728'], density=True)
    plt.title("NPS Distribution Comparison")
    plt.xlabel("NPS (Kilo Nodes Per Second)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nps_distribution.png"), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgn_path = os.path.join(project_root, "tournament_analysis", "tournament_results.pgn")
    out_dir = os.path.join(project_root, "tournament_analysis")
    
    stats = parse_pgn(pgn_path)
    if stats:
        analyze_and_plot(stats[0], stats[1], out_dir)
