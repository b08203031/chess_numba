import json

for path in ['tournament_colab_runner_se_multicut copy 4.ipynb', 'tournament_colab_runner_se_multicut copy 6.ipynb']:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {path} ===")
    for cell in nb.get('cells', []):
        all_text = ""
        for o in cell.get('outputs', []):
            all_text += ''.join(o.get('text', []))
        lines = all_text.splitlines()
        for i, l in enumerate(lines):
            if any(k in l for k in ['After ', 'Stage Finished', 'Elo difference:', 'LLR', 'Score of']):
                # print last matching lines
                print(l)

