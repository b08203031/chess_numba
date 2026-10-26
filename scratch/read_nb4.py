import json

with open('tournament_colab_runner_se_multicut copy 4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/nb4_dump.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        out_f.write(f'=== Cell {i} ({cell.get("cell_type")}) ===\n')
        for line in src.splitlines():
            out_f.write(f'  src: {line}\n')
        outputs = cell.get('outputs', [])
        for j, out in enumerate(outputs):
            text = ''.join(out.get('text', []))
            if text:
                out_f.write(f'  --- Output {j} (total lines {len(text.splitlines())}) ---\n')
                for line in text.splitlines():
                    out_f.write(f'    out: {line}\n')

print("Dump completed.")

