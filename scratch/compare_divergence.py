import chess.pgn
import sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

pgn_path = 'tournament_analysis/tournament_results.pgn'

games_by_round = {}
with open(pgn_path, 'r', encoding='utf-8') as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        rnd = game.headers.get('Round', '?')
        games_by_round[rnd] = game

def compare_pair(r_w_new, r_w_old):
    g_new_w = games_by_round[r_w_new] # New is White, Old is Black
    g_old_w = games_by_round[r_w_old] # Old is White, New is Black
    
    fen = g_new_w.headers.get('FEN')
    print(f"==================================================")
    print(f"Pair: Round {r_w_new} (New as W: {g_new_w.headers.get('Result')}) vs Round {r_w_old} (Old as W: {g_old_w.headers.get('Result')})")
    print(f"FEN: {fen}")
    
    # Extract move lists and evals
    def get_moves_and_evals(g):
        moves = []
        comments = []
        node = g
        while node.variations:
            node = node.variation(0)
            moves.append(node.move)
            comments.append(node.comment)
        return moves, comments

    m_new_w, c_new_w = get_moves_and_evals(g_new_w)
    m_old_w, c_old_w = get_moves_and_evals(g_old_w)

    # Move 1 of White:
    print(f"  First move as White: New played {m_new_w[0].uci()} ({c_new_w[0]}), Old played {m_old_w[0].uci()} ({c_old_w[0]})")
    
    # Find where the game trees diverged:
    # Notice that New as White was playing against Old as Black in g_new_w.
    # While Old as White was playing against New as Black in g_old_w.
    # So if they diverge on move 1 White, they played different first moves!
    # Let's check how long they matched:
    min_len = min(len(m_new_w), len(m_old_w))
    div_idx = 0
    while div_idx < min_len and m_new_w[div_idx] == m_old_w[div_idx]:
        div_idx += 1
    
    ply = div_idx + 1
    move_num = (div_idx // 2) + 1
    side = "White" if div_idx % 2 == 0 else "Black"
    print(f"  Divergence at ply {ply} ({move_num}.{side}):")
    if div_idx < len(m_new_w):
        engine_new_w = "New" if side == "White" else "Old"
        print(f"    In Round {r_w_new}: {engine_new_w} played {m_new_w[div_idx].uci()} [{c_new_w[div_idx]}]")
    if div_idx < len(m_old_w):
        engine_old_w = "Old" if side == "White" else "New"
        print(f"    In Round {r_w_old}: {engine_old_w} played {m_old_w[div_idx].uci()} [{c_old_w[div_idx]}]")

for r_new, r_old in [('7', '8'), ('45', '46'), ('65', '66'), ('1', '2'), ('15', '16'), ('23', '24'), ('35', '36'), ('37', '38'), ('39', '40')]:
    compare_pair(r_new, r_old)

