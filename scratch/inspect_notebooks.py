import json, sys

notebooks = [
    ('copy.ipynb', 'tournament_colab_runner_se_multicut copy.ipynb'),
    ('copy 5.ipynb', 'tournament_colab_runner_se_multicut copy 5.ipynb'),
    ('copy 7.ipynb', 'tournament_colab_runner_se_multicut copy 7.ipynb'),
    ('copy 2.ipynb', 'tournament_colab_runner_se_multicut copy 2.ipynb'),
]

with open('scratch/nb_eval_report.txt', 'w', encoding='utf-8') as out_f:
    for short_name, nb_path in notebooks:
        out_f.write(f'==================================================\n')
        out_f.write(f'FILE: {short_name} ({nb_path})\n')
        out_f.write(f'==================================================\n')
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            out_f.write(f'Error reading notebook: {e}\n')
            continue

        # Look for setup/diff cell
        for i, cell in enumerate(nb.get('cells', [])):
            src = ''.join(cell.get('source', []))
            outputs = cell.get('outputs', [])
            all_out = ''
            for o in outputs:
                all_out += ''.join(o.get('text', []))

            # Check if this cell contains diff or git checkout or setup
            if 'diff' in src or 'checkout' in src or 'constants.py' in src:
                out_f.write(f'--- Setup / Diff Cell {i} ---\n')
                out_f.write(f'Source:\n{src}\n')
                if all_out:
                    out_f.write(f'Output:\n{all_out}\n')

            # Check if this cell contains tournament progress / results
            if 'After ' in all_out or 'Score of' in all_out or 'Stage Finished' in all_out or 'SPRT' in all_out:
                out_f.write(f'--- Tournament Output Cell {i} ---\n')
                # Grab the last 40 lines of output
                lines = all_out.splitlines()
                out_f.write(f'Total output lines: {len(lines)}\n')
                out_f.write('Last 40 lines:\n')
                for line in lines[-40:]:
                    out_f.write(line + '\n')

print("Evaluation report written to scratch/nb_eval_report.txt")

