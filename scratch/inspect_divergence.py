import chess.pgn
import sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

pgn_path = 'tournament_analysis/tournament_results.pgn'

games_by_round = {}
games_by_fen = defaultdict(list)
with open(pgn_path, 'r', encoding='utf-8') as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        rnd = game.headers.get('Round', '?')
        fen = game.headers.get('FEN', 'startpos')
        games_by_round[rnd] = game
        games_by_fen[fen].append(game)

# Let's inspect Rounds 7 and 8 (Double loss for New: Round 7 New (W) vs Old (B) 0-1; Round 8 Old (W) vs New (B) 1-0)
def inspect_game(game):
    w = game.headers.get('White')
    b = game.headers.get('Black')
    res = game.headers.get('Result')
    rnd = game.headers.get('Round')
    print(f"--- Round {rnd}: {w} (White) vs {b} (Black) -> {res} ---")
    node = game
    moves = []
    evals = []
    promotions = []
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        moves.append(move)
        comment = next_node.comment
        if move.promotion:
            promotions.append((len(moves), move, comment))
        evals.append(comment)
        node = next_node
    print(f"Total moves: {len(moves)}")
    if promotions:
        print(f"Promotions in game: {promotions}")
    # Print the last 10 moves and comments
    start_m = max(0, len(moves) - 10)
    for i in range(start_m, len(moves)):
        ply_num = i + 1
        m_str = f"{(i//2)+1}.{'..' if i%2 else ''} {moves[i].uci()}"
        print(f"  {m_str:10s} {evals[i]}")

print("=== INSPECTING ROUND 7 & 8 ===")
inspect_game(games_by_round['7'])
inspect_game(games_by_round['8'])

print("\n=== INSPECTING ROUND 45 & 46 ===")
inspect_game(games_by_round['45'])
inspect_game(games_by_round['46'])

print("\n=== INSPECTING ROUND 65 & 66 ===")
inspect_game(games_by_round['65'])
inspect_game(games_by_round['66'])

