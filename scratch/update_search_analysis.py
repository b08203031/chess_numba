path = 'chess_engine/classical/SEARCH_ANALYSIS.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target1 = '`SINGULAR_TTPV_BONUS = 62`'
repl1 = '`SINGULAR_TTPV_BONUS = 65`（E96 採納至 65）'
assert target1 in text, 'target1 not found'
text = text.replace(target1, repl1, 1)

target2 = '接納SINGULAR_MARGIN_BASE = 58 正式採納進生產基線。'
repl2 = '''接納SINGULAR_MARGIN_BASE = 58 正式採納進生產基線。
        *   第六階段（TT PV 延展獎勵最佳化，E96，2026-10-26）：探索 `SINGULAR_TTPV_BONUS`（54 vs 62 vs 65 vs 70）。1000 局成對換先實測（`copy 4.ipynb`），`65 vs 62` 取得 Candidate **203W 626D 171L**（勝率 **51.60%**，淨勝 **+32 局**，Elo: **+11.12 ± 13.16**，LLR=+1.55），且在 51~80 步中長盤取得 +35 局優勢，平均深度與同深度節點比完全持平（1.000）。正式採納 `SINGULAR_TTPV_BONUS = 65` 為生產基準。'''
assert target2 in text, 'target2 not found'
text = text.replace(target2, repl2, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SEARCH_ANALYSIS.md updated successfully.')

