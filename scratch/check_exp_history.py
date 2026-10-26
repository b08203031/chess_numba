with open('tournament_analysis/SEARCH_EXP_LOG.md', 'r', encoding='utf-8') as f:
    text = f.read()

import re
exps = re.findall(r'(### E\d+.*?)(?=\n### E|\n## 4|\Z)', text, re.DOTALL)
with open('scratch/exp_history_out.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total experiments found: {len(exps)}\n")
    for e in exps:
        first_line = e.splitlines()[0]
        if any(k in e for k in ['SINGULAR_MARGIN_BASE', 'PROBCUT_MARGIN', 'LMR_BASE_OFFSET', 'SINGULAR_DOUBLE_EXT']):
            out.write(first_line + '\n')
            for l in e.splitlines()[1:10]:
                if any(k in l for k in ['Elo', '勝率', '處置', '總戰績', '勝', 'Candidate']):
                    out.write('   ' + l.strip() + '\n')

print("Wrote history.")

