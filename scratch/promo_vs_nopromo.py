import chess, chess.pgn, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

pgn_path = 'tournament_analysis/tournament_results.pgn'

games_by_fen = defaultdict(list)
with open(pgn_path, 'r', encoding='utf-8') as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None: break
        fen = g.headers.get('FEN', 'startpos')
        games_by_fen[fen].append(g)

def get_score(g, eng):
    res = g.headers.get('Result', '*')
    w, b = g.headers.get('White'), g.headers.get('Black')
    if res == '1-0': return 1.0 if eng == w else 0.0
    if res == '0-1': return 1.0 if eng == b else 0.0
    if res == '1/2-1/2': return 0.5
    return 0.0

total_pairs = len(games_by_fen)
print(f"Total pairs: {total_pairs}")

pairs_with_promos = 0
pairs_promo_new_better = 0
pairs_promo_old_better = 0
pairs_promo_equal = 0

pairs_nopromo_new_better = 0
pairs_nopromo_old_better = 0
pairs_nopromo_equal = 0

for fen, glist in games_by_fen.items():
    if len(glist) != 2: continue
    s_new = get_score(glist[0], 'New') + get_score(glist[1], 'New')
    s_old = get_score(glist[0], 'Old') + get_score(glist[1], 'Old')
    diff = s_new - s_old
    has_promo = any(any(m.promotion for m in g.mainline_moves()) for g in glist)
    
    if has_promo:
        pairs_with_promos += 1
        if diff > 0: pairs_promo_new_better += 1
        elif diff < 0: pairs_promo_old_better += 1
        else: pairs_promo_equal += 1
    else:
        if diff > 0: pairs_nopromo_new_better += 1
        elif diff < 0: pairs_nopromo_old_better += 1
        else: pairs_nopromo_equal += 1

print(f"\n--- Pairs WITH promotions (n={pairs_with_promos}) ---")
print(f"New scored higher: {pairs_promo_new_better} ({(pairs_promo_new_better/pairs_with_promos)*100:.1f}%)")
print(f"Old scored higher: {pairs_promo_old_better} ({(pairs_promo_old_better/pairs_with_promos)*100:.1f}%)")
print(f"Equal (both drew or split): {pairs_promo_equal}")
net_promo = pairs_promo_new_better - pairs_promo_old_better
print(f"Net pair diff on promotion games: {net_promo:+d}")

nopromo_total = total_pairs - pairs_with_promos
print(f"\n--- Pairs WITHOUT promotions (n={nopromo_total}) ---")
print(f"New scored higher: {pairs_nopromo_new_better} ({(pairs_nopromo_new_better/nopromo_total)*100:.1f}%)")
print(f"Old scored higher: {pairs_nopromo_old_better} ({(pairs_nopromo_old_better/nopromo_total)*100:.1f}%)")
print(f"Equal (both drew or split): {pairs_nopromo_equal}")
net_nopromo = pairs_nopromo_new_better - pairs_nopromo_old_better
print(f"Net pair diff on NO promotion games: {net_nopromo:+d}")

