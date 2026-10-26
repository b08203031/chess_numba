with open('tournament_analysis/SEARCH_EXP_LOG.md', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('### E91')
idx2 = text.find('### E92', idx)
with open('scratch/e91_out.txt', 'w', encoding='utf-8') as out:
    out.write(text[idx:idx2])
print("Wrote E91.")

