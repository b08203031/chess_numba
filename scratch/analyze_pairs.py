import chess.pgn
import sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

pgn_path = 'tournament_analysis/tournament_results.pgn'

games_by_fen = defaultdict(list)
with open(pgn_path, 'r', encoding='utf-8') as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        fen = game.headers.get('FEN', 'startpos')
        games_by_fen[fen].append(game)

print(f'Distinct FENs: {len(games_by_fen)}')

pair_diffs = []
for fen, glist in games_by_fen.items():
    if len(glist) != 2:
        print(f'Warning: FEN has {len(glist)} games!')
        continue
    g1, g2 = glist[0], glist[1]
    def get_score(game, engine):
        res = game.headers.get('Result', '*')
        w = game.headers.get('White', '')
        b = game.headers.get('Black', '')
        if res == '1-0':
            return 1.0 if engine == w else 0.0
        elif res == '0-1':
            return 1.0 if engine == b else 0.0
        elif res == '1/2-1/2':
            return 0.5
        return 0.0
    
    new_score = get_score(g1, 'New') + get_score(g2, 'New')
    old_score = get_score(g1, 'Old') + get_score(g2, 'Old')
    pair_diffs.append((new_score - old_score, fen, glist))

pair_diffs.sort(key=lambda x: x[0])

print('\n=== Pairs where OLD outscored NEW (new_score < old_score) ===')
count_old = 0
for diff, fen, glist in pair_diffs:
    if diff < 0:
        count_old += 1
        rounds = [g.headers.get('Round', '?') for g in glist]
        r0 = f"{glist[0].headers.get('White')} vs {glist[0].headers.get('Black')}: {glist[0].headers.get('Result')}"
        r1 = f"{glist[1].headers.get('White')} vs {glist[1].headers.get('Black')}: {glist[1].headers.get('Result')}"
        print(f"Diff: {diff:+.1f} | Rounds {rounds} | {r0} ; {r1}")
print(f'Total pairs where Old outscored New: {count_old}')

print('\n=== Pairs where NEW outscored OLD (new_score > old_score) ===')
count_new = 0
for diff, fen, glist in pair_diffs:
    if diff > 0:
        count_new += 1
        rounds = [g.headers.get('Round', '?') for g in glist]
        r0 = f"{glist[0].headers.get('White')} vs {glist[0].headers.get('Black')}: {glist[0].headers.get('Result')}"
        r1 = f"{glist[1].headers.get('White')} vs {glist[1].headers.get('Black')}: {glist[1].headers.get('Result')}"
        print(f"Diff: {diff:+.1f} | Rounds {rounds} | {r0} ; {r1}")
print(f'Total pairs where New outscored Old: {count_new}')

