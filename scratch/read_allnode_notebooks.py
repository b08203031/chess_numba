import json

notebooks = [
    ('multicut.ipynb', 'tournament_colab_runner_se_multicut.ipynb'),
    ('copy 7.ipynb', 'tournament_colab_runner_se_multicut copy 7.ipynb'),
]

for label, path in notebooks:
    print(f"==================================================")
    print(f"FILE: {label} ({path})")
    print(f"==================================================")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue

    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        all_text = ""
        for o in cell.get('outputs', []):
            all_text += ''.join(o.get('text', []))

        # Check setup or diff
        if any(k in src for k in ['diff-code', 'constants.py', 'git log', 'HEAD is now at']):
            print(f"--- Cell {i} (setup/diff) ---")
            for l in all_text.splitlines():
                if any(k in l for k in ['HEAD is now at', 'Differences found', 'constants.py', '+', '-', '@@']):
                    print("  ", l)

        # Check tournament outputs
        if any(k in all_out for k in ['After ', 'Stage Finished', 'Score of', 'SPRT', 'Elo difference:'] for all_out in [all_text]):
            print(f"--- Cell {i} (tournament) ---")
            lines = all_text.splitlines()
            print(f"  Total lines: {len(lines)}")
            for l in lines[-25:]:
                print("  ", l)

