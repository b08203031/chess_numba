import json

nb_path = 'tournament_colab_runner_se_multicut.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/multicut_runner_full.txt', 'w', encoding='utf-8') as out_f:
    out_f.write(f"=== Number of cells: {len(nb.get('cells', []))} ===\n")
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        src = ''.join(cell.get('source', []))
        out_f.write(f"\n--- Cell {i} ({cell_type}) ---\n")
        out_f.write("Source:\n" + src + "\n")
        outputs = cell.get('outputs', [])
        out_f.write(f"Outputs ({len(outputs)}):\n")
        for j, out in enumerate(outputs):
            text = ''.join(out.get('text', []))
            lines = text.splitlines()
            out_f.write(f"  [Output {j}, total lines: {len(lines)}]\n")
            if len(lines) <= 50:
                for l in lines:
                    out_f.write("    " + l + "\n")
            else:
                out_f.write("    (first 25 lines)\n")
                for l in lines[:25]:
                    out_f.write("    " + l + "\n")
                out_f.write("    ...\n    (last 25 lines)\n")
                for l in lines[-25:]:
                    out_f.write("    " + l + "\n")

print("Dumped multicut_runner_full.txt")

