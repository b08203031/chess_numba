import json

with open('tournament_colab_runner_se_multicut copy 3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    for o in cell.get('outputs', []):
        text = ''.join(o.get('text', []))
        for line in text.splitlines():
            if any(k in line for k in ['HEAD is now at', 'Differences found', 'LMP_SCALE_PERCENT', 'After ', 'SPRT', '錦標賽參數配置']):
                print(line)

