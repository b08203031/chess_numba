import json, glob, os

for nb_path in glob.glob("tournament_colab_runner*.ipynb"):
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        continue
    
    commit_info = ""
    diff_info = ""
    result_info = ""
    
    for cell in nb.get('cells', []):
        outputs = cell.get('outputs', [])
        for o in outputs:
            text = ''.join(o.get('text', []))
            for line in text.splitlines():
                if 'HEAD is now at' in line:
                    commit_info = line.strip()
                if 'Differences found in search constants' in line or '-SINGULAR_' in line or '+SINGULAR_' in line or '-LMR_' in line or '+LMR_' in line or '-PROBCUT_' in line or '+PROBCUT_' in line:
                    diff_info += line.strip() + " | "
                if 'After 500 pairs' in line or '=== Stage Finished ===' in line or 'Score of' in line or 'Elo difference:' in line:
                    result_info += line.strip() + " \n "
    
    print(f"=== {nb_path} ===")
    print("  Commit:", commit_info)
    print("  Diff:", diff_info[:150])
    if result_info:
        print("  Result:\n ", result_info[-300:])
    else:
        print("  Result: (In progress or no final summary)")
    print()

