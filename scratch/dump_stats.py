import json

with open('scratch/two_runners_stats.txt', 'w', encoding='utf-8') as out_f:
    for nb_path in ['tournament_colab_runner_se_multicut.ipynb', 'tournament_colab_runner_se_multicut copy 7.ipynb']:
        out_f.write(f"\n==================================================\n")
        out_f.write(f"FILE: {nb_path}\n")
        out_f.write(f"==================================================\n")
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get('cells', [])):
            all_text = ""
            for o in cell.get('outputs', []):
                all_text += ''.join(o.get('text', []))
            if '搜尋效能與深度對比統計報告' in all_text or '引擎對戰深度分析報告' in all_text:
                out_f.write(f"--- Cell {i} Output ---\n")
                out_f.write(all_text + "\n")

print("Dumped two_runners_stats.txt")

