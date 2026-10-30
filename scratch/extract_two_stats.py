with open('scratch/two_runners_stats.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('scratch/extracted_summary.txt', 'w', encoding='utf-8') as out:
    parts = text.split('==================================================')
    for p in parts[1:]:
        lines = p.strip().splitlines()
        out.write('*** ' + lines[0] + '\n')
        capture = False
        for l in lines:
            if any(k in l for k in ['【1. 搜尋深度', '【2. 節點數', '【3. 每步耗時', '【4. NPS', 'short(', 'medium(', 'long(', 'very_long(', '勝率(含和) :', '戰績統計 :']):
                out.write('  ' + l.strip() + '\n')
            if '【1. 搜尋深度' in l:
                capture = True
            if '【5. 分執色' in l:
                capture = False
            if capture:
                out.write('    ' + l.strip() + '\n')

print("Summary extracted.")

