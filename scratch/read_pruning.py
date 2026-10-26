with open('chess_engine/classical/constants.py', 'r', encoding='utf-8') as f:
    lines = [f'{i+1}: {l.rstrip()}' for i, l in enumerate(f) if any(k in l for k in ['RFP_', 'FP_', 'RAZORING_', 'DELTA_PRUNING_'])]

with open('scratch/rfp_fp_out.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(lines))

print('Done.')

