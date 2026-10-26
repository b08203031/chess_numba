import chess.pgn
import sys, math
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

pgn_path = 'tournament_analysis/tournament_results.pgn'

games = []
with open(pgn_path, 'r', encoding='utf-8') as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None: break
        games.append(g)

print(f"Total games loaded: {len(games)}")

def get_game_score(g):
    res = g.headers.get("Result")
    w = g.headers.get("White")
    b = g.headers.get("Black")
    if res == "1-0":
        return 1.0 if w == "New" else 0.0
    elif res == "0-1":
        return 1.0 if b == "New" else 0.0
    elif res == "1/2-1/2":
        return 0.5
    return 0.0

# Group games into chunks of 100
chunk_size = 100
for start in range(0, len(games), chunk_size):
    end = min(start + chunk_size, len(games))
    chunk = games[start:end]
    w_cnt = sum(1 for g in chunk if get_game_score(g) == 1.0)
    d_cnt = sum(1 for g in chunk if get_game_score(g) == 0.5)
    l_cnt = sum(1 for g in chunk if get_game_score(g) == 0.0)
    pts = w_cnt + 0.5 * d_cnt
    total = len(chunk)
    pct = (pts / total) * 100
    elo = -400 * math.log10(1 / (pts / total) - 1) if 0 < pts < total else 0.0
    print(f"Games {start+1:>3d} - {end:>3d} (n={total:>3d}): W={w_cnt:>2d} D={d_cnt:>2d} L={l_cnt:>2d} | Score={pct:5.1f}% | Elo={elo:+6.1f}")

