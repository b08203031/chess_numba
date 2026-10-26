import chess
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

old_favored_pairs = []
new_favored_pairs = []
drawn_pairs = []

for fen, glist in games_by_fen.items():
    if len(glist) != 2:
        continue
    g1, g2 = glist[0], glist[1]
    s_new = get_score(g1, 'New') + get_score(g2, 'New')
    s_old = get_score(g1, 'Old') + get_score(g2, 'Old')
    diff = s_new - s_old
    if diff < 0:
        old_favored_pairs.append((diff, fen, glist))
    elif diff > 0:
        new_favored_pairs.append((diff, fen, glist))
    else:
        drawn_pairs.append((diff, fen, glist))

print(f"Old favored pairs: {len(old_favored_pairs)} (total score diff: {sum(x[0] for x in old_favored_pairs)})")
print(f"New favored pairs: {len(new_favored_pairs)} (total score diff: {sum(x[0] for x in new_favored_pairs)})")
print(f"Drawn pairs: {len(drawn_pairs)}")

# Analyze promotions in pairs
def check_promotions(glist):
    promos = []
    for g in glist:
        node = g
        while node.variations:
            node = node.variation(0)
            if node.move.promotion:
                promos.append((g.headers.get('Round'), node.move))
    return promos

# Check if King and pawn endgame reached (non-pawns == 0)
def check_kp_endgame(glist):
    kp_games = []
    for g in glist:
        board = g.board()
        node = g
        reached_kp = False
        while node.variations:
            node = node.variation(0)
            board.push(node.move)
            # check non-pawns
            w_non_pawns = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.WHITE))
            b_non_pawns = len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + len(board.pieces(chess.ROOK, chess.BLACK)) + len(board.pieces(chess.QUEEN, chess.BLACK))
            if w_non_pawns == 0 and b_non_pawns == 0:
                reached_kp = True
                break
        if reached_kp:
            kp_games.append(g.headers.get('Round'))
    return kp_games

old_promos = 0
for _, _, glist in old_favored_pairs:
    p = check_promotions(glist)
    if p: old_promos += 1

new_promos = 0
for _, _, glist in new_favored_pairs:
    p = check_promotions(glist)
    if p: new_promos += 1

drawn_promos = 0
for _, _, glist in drawn_pairs:
    p = check_promotions(glist)
    if p: drawn_promos += 1

print(f"\nPairs with promotions:")
print(f"  Old favored: {old_promos} / {len(old_favored_pairs)}")
print(f"  New favored: {new_promos} / {len(new_favored_pairs)}")
print(f"  Drawn:       {drawn_promos} / {len(drawn_pairs)}")

old_kp = 0
for _, _, glist in old_favored_pairs:
    kp = check_kp_endgame(glist)
    if kp: old_kp += 1

new_kp = 0
for _, _, glist in new_favored_pairs:
    kp = check_kp_endgame(glist)
    if kp: new_kp += 1

print(f"\nPairs that reached pure King-Pawn endgames:")
print(f"  Old favored: {old_kp} / {len(old_favored_pairs)}")
print(f"  New favored: {new_kp} / {len(new_favored_pairs)}")

