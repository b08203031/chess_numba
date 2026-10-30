path = 'chess_engine/classical/constants.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if l.startswith('LMR_ALLNODE_SCALE_DENOM_BASE ='):
        lines[i] = 'LMR_ALLNODE_SCALE_DENOM_BASE = 256\n'
    elif l.startswith('LMR_ALLNODE_SCALE_DENOM_OFFSET ='):
        lines[i] = 'LMR_ALLNODE_SCALE_DENOM_OFFSET = 298  # Candidate 298 vs Baseline 268 (gradient probe 298 vs 238: +9.04 Elo)\n'

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Updated classical/constants.py")

