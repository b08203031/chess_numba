with open('chess_engine/classical/constants.py', 'r', encoding='utf-8') as f:
    lines = [f'{i+1}: {l.rstrip()}' for i, l in enumerate(f) if 'LMR_' in l or 'TUNE_LMR' in l]

with open('scratch/lmr_out.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(lines[30:75]))

print('Wrote safely.')

