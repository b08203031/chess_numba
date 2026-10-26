import chess.pgn
import sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

pgn_path = 'tournament_analysis/tournament_results.pgn'

losses_as_white = []
losses_as_black = []

with open(pgn_path, 'r', encoding='utf-8') as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        w = game.headers.get('White')
        b = game.headers.get('Black')
        res = game.headers.get('Result')
        rnd = game.headers.get('Round')
        
        if w == 'New' and res == '0-1':
            losses_as_white.append((rnd, game))
        elif b == 'New' and res == '1-0':
            losses_as_black.append((rnd, game))

print(f"Total losses for New: {len(losses_as_white) + len(losses_as_black)}")
print(f"Losses as White (n={len(losses_as_white)}): {[x[0] for x in losses_as_white]}")
print(f"Losses as Black (n={len(losses_as_black)}): {[x[0] for x in losses_as_black]}")

def analyze_losses(loss_list, color):
    print(f"\n--- Analyzing New's losses as {color} ---")
    move_counts = []
    reasons = []
    for rnd, g in loss_list:
        node = g
        moves = []
        comments = []
        while node.variations:
            node = node.variation(0)
            moves.append(node.move)
            comments.append(node.comment)
        move_counts.append(len(moves)//2 + 1)
        last_comment = comments[-1] if comments else ""
        last_few = comments[-5:] if len(comments) >= 5 else comments
        print(f"Round {rnd:>3s} ({len(moves)//2 + 1:>3d} moves): final comment: {last_comment}")

analyze_losses(losses_as_white, "White")
analyze_losses(losses_as_black, "Black")

