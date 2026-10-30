import json

with open('tournament_colab_runner_se_multicut.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    for o in cell.get('outputs', []):
        text = ''.join(o.get('text', []))
        for line in text.splitlines():
            if 'Game 1:' in line or 'Game 2:' in line or 'Game 50:' in line or 'Game 100:' in line or 'Game 300:' in line or 'Game 370:' in line:
                print(line)

