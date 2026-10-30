import json, glob

for nb_path in sorted(glob.glob("tournament_colab_runner*.ipynb")):
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        continue

    commit = "unknown"
    diff = "none"
    last_game_line = ""
    llr_line = ""

    for cell in nb.get('cells', []):
        all_text = ""
        for o in cell.get('outputs', []):
            all_text += ''.join(o.get('text', []))
        
        for line in all_text.splitlines():
            if 'HEAD is now at' in line:
                commit = line.strip()
            if 'LMP_SCALE_PERCENT' in line or 'LMR_' in line or 'SINGULAR_' in line:
                if '+' in line or '-' in line or 'vs' in line:
                    diff = line.strip()
            if 'After ' in line and 'games):' in line:
                last_game_line = line.strip()
            if 'SPRT LLR=' in line or 'Elo difference:' in line:
                llr_line = line.strip()

    print(f"[{nb_path}]")
    print(f"  Commit: {commit}")
    print(f"  Diff: {diff}")
    print(f"  Progress: {last_game_line}")
    print(f"  SPRT / Status: {llr_line}")
    print()

