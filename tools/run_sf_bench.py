import os
import subprocess
import time

def run_sf_bench():
    # Use absolute path and double quotes for potential spaces
    _tools_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_tools_dir)
    sf_path = os.path.join(_root_dir, "external", "stockfish", "stockfish-windows-x86-64-avx2.exe" if os.name == "nt" else "src/stockfish")
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    
    commands = [
        "uci\n",
        "isready\n",
        f"position fen {fen}\n",
        "go depth 25\n",
        "quit\n"
    ]
    
    print(f"Starting Stockfish from {sf_path}...")
    
    process = subprocess.Popen(
        sf_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        bufsize=0
    )
    
    # Send commands
    for cmd in commands:
        process.stdin.write(cmd)
        process.stdin.flush()
        time.sleep(0.5)
    
    # Read output
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output_lines.append(line)
        if "bestmove" in line:
            break
            
    with open("sf_full_output_v2.txt", "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    
    print("Full Stockfish output captured in sf_full_output_v2.txt")

if __name__ == "__main__":
    run_sf_bench()
