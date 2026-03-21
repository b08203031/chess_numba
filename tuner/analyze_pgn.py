import chess.pgn
import os
import sys

def analyze_pgn(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Analyzing PGN: {file_path}")
    print("-" * 40)

    total_games = 0
    elo_2400_plus = 0
    elo_2300_plus = 0
    elo_2200_plus = 0
    no_elo = 0
    
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}

    with open(file_path, encoding='utf-8') as pgn_file:
        while True:
            headers = chess.pgn.read_headers(pgn_file)
            if headers is None:
                break
            
            total_games += 1
            
            white_elo = headers.get("WhiteElo")
            black_elo = headers.get("BlackElo")
            result = headers.get("Result", "*")
            
            if result in results:
                results[result] += 1
            else:
                results["*"] += 1

            if white_elo and black_elo:
                try:
                    w = int(white_elo)
                    b = int(black_elo)
                    if w >= 2400 and b >= 2400:
                        elo_2400_plus += 1
                    if w >= 2300 and b >= 2300:
                        elo_2300_plus += 1
                    if w >= 2200 and b >= 2200:
                        elo_2200_plus += 1
                except ValueError:
                    no_elo += 1
            else:
                no_elo += 1

            if total_games % 1000 == 0:
                print(f"Read {total_games} games...")

    print("-" * 40)
    print(f"總對局數 (Total Games): {total_games}")
    print(f"無 ELO 資訊 (No ELO): {no_elo}")
    print("-" * 40)
    print("雙方 ELO 門檻統計 (Both players above threshold):")
    print(f"  >= 2400 (GM+): {elo_2400_plus} ({elo_2400_plus/total_games*100:.1f}%)")
    print(f"  >= 2300 (IM+): {elo_2300_plus} ({elo_2300_plus/total_games*100:.1f}%)")
    print(f"  >= 2200 (NM+): {elo_2200_plus} ({elo_2200_plus/total_games*100:.1f}%)")
    print("-" * 40)
    print("結果統計 (Results):")
    for r, count in results.items():
        print(f"  {r:8}: {count}")

if __name__ == "__main__":
    path = "tuner/twic1613.pgn"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    if not os.path.exists(path) and os.path.exists("twic1613.pgn"):
        path = "twic1613.pgn"
        
    analyze_pgn(path)
