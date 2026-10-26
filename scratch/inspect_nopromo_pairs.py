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

nopromo_diff_pairs = []
for fen, glist in games_by_fen.items():
    if len(glist) != 2: continue
    has_promo = any(any(m.promotion for m in g.mainline_moves()) for g in glist)
    if has_promo: continue
    s_new = get_score(glist[0], 'New') + get_score(glist[1], 'New')
    s_old = get_score(glist[0], 'Old') + get_score(glist[1], 'Old')
    diff = s_new - s_old
    if diff != 0:
        nopromo_diff_pairs.append((diff, fen, glist))

print(f"Total NO-PROMO pairs with non-zero diff: {len(nopromo_diff_pairs)}")
for diff, fen, glist in nopromo_diff_pairs:
    rounds = [g.headers.get('Round') for g in glist]
    results = [f"{g.headers.get('White')} vs {g.headers.get('Black')}: {g.headers.get('Result')}" for g in glist]
    # Check if king-pawn endgame reached
    has_kp = False
    for g in glist:
        b = g.board()
        for m in g.mainline_moves():
            b.push(m)
            if (len(b.pieces(chess.KNIGHT, chess.WHITE)) + len(b.pieces(chess.BISHOP, chess.WHITE)) + len(b.pieces(chess.ROOK, chess.WHITE)) + len(b.pieces(chess.QUEEN, chess.WHITE)) == 0 and
                len(b.pieces(chess.KNIGHT, chess.BLACK)) + len(b.pieces(chess.BISHOP, chess.BLACK)) + len(b.pieces(chess.ROOK, chess.BLACK)) + len(b.pieces(chess.QUEEN, chess.BLACK)) == 0):
                has_kp = True
                break
        if has_kp: break

    # Find divergence ply
    m1 = list(glist[0].mainline_moves())
    m2 = list(glist[1].mainline_moves())
    div_ply = 0
    while div_ply < min(len(m1), len(m2)) and m1[div_ply] == m2[div_ply]:
        div_ply += 1
    
    print(f"Rounds {rounds}: diff={diff:+.1f} | {results[0]} ; {results[1]} | div_ply={div_ply+1} | kp={has_kp}")

