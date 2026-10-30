path = 'chess_engine/classical_old/constants.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if l.startswith('LMR_ALLNODE_SCALE_DENOM_BASE ='):
        lines[i] = 'LMR_ALLNODE_SCALE_DENOM_BASE = 256\n'
    elif l.startswith('LMR_ALLNODE_SCALE_DENOM_OFFSET ='):
        lines[i] = 'LMR_ALLNODE_SCALE_DENOM_OFFSET = 268  # Original production baseline\n'

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Updated classical_old/constants.py")

