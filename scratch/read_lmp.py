with open('chess_engine/classical/constants.py', 'r', encoding='utf-8') as f:
    lines = [f'{i+1}: {l.rstrip()}' for i, l in enumerate(f) if 'LMP_' in l or 'TUNE_LMP' in l]

print('\n'.join(lines))

