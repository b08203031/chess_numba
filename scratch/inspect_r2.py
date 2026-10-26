import chess.pgn
import sys
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

g1 = games_by_round['1']
g2 = games_by_round['2']

def print_game(g):
    print(f"=== Round {g.headers.get('Round')} : {g.headers.get('White')} vs {g.headers.get('Black')} ({g.headers.get('Result')}) ===")
    node = g
    m_idx = 0
    while node.variations:
        node = node.variation(0)
        m = node.move
        m_str = f"{(m_idx//2)+1}.{'..' if m_idx%2 else ' '} {m.uci()}"
        print(f"{m_str:<10s} {node.comment}")
        m_idx += 1

print_game(g2)

