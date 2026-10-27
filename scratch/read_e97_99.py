with open('tournament_analysis/SEARCH_EXP_LOG.md', 'r', encoding='utf-8') as f:
    text = f.read()

import re
matches = re.findall(r'(### E9[7-9].*?)(?=\n### E|\n## 4|\Z)', text, re.DOTALL)
with open('scratch/e97_99.txt', 'w', encoding='utf-8') as out:
    for m in matches:
        out.write(m + '\n---\n')

print(f"Found {len(matches)} sections.")

