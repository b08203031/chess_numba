
import sys
import numpy as np

from chess_engine.board_operations import make_move
from chess_engine.core import generate_legal_moves
from chess_engine.debug_utils import log_info
from chess_engine.fen_parser import parse_fen
from chess_engine.move import move_to_uci
from chess_engine.search import iterative_deepening_search
from chess_engine.transposition_table import (
    create_transposition_table,
    clear_transposition_table,
    TT_SIZE_MB,
)

# Maximum search depth (Ply) for arrays like killer moves
MAX_PLY = 64


def uci_loop():
    """
    The main UCI loop of the engine.
    Listens for and responds to UCI commands from a GUI.
    """
    # Initialize engine components before the loop starts
    transposition_table = create_transposition_table(TT_SIZE_MB)
    killer_moves = np.zeros((MAX_PLY, 2), dtype=np.uint16)

    board_state = None

    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line:
            continue

        tokens = line.strip().split()
        command = tokens[0]

        if command == "uci":
            print("id name MyChessEngine")
            print("id author YourName")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            clear_transposition_table(transposition_table)
            killer_moves.fill(0)
        elif command == "position":
            # --- Parse position command ---
            if "startpos" in tokens:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                board_state = piece_bbs + occupancy_bbs + game_state
            elif "fen" in tokens:
                fen_start_index = tokens.index("fen") + 1
                fen = " ".join(tokens[fen_start_index:])
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                board_state = piece_bbs + occupancy_bbs + game_state

            if "moves" in tokens:
                moves_start_index = tokens.index("moves") + 1
                for move_uci in tokens[moves_start_index:]:
                    legal_moves = generate_legal_moves(
                        board_state[:12], board_state[12:15], board_state[15:]
                    )
                    found_move = False
                    for legal_move in legal_moves:
                        if move_to_uci(legal_move) == move_uci:
                            board_state, _ = make_move(
                                board_state[:12],
                                board_state[12:15],
                                board_state[15:],
                                legal_move,
                            )
                            found_move = True
                            break
                    if not found_move:
                        log_info(f"Illegal move {move_uci} received. Ignoring.")
                        break
        elif command == "go":
            # --- Parse go command ---
            max_depth = 64  # Default max depth
            move_time_ms = float('inf')  # Default to almost infinite time

            if "movetime" in tokens:
                move_time_ms = int(tokens[tokens.index("movetime") + 1])
            elif "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])
            else:
                side_to_move = board_state[15]
                wtime = int(tokens[tokens.index("wtime") + 1]) if "wtime" in tokens else 0
                btime = int(tokens[tokens.index("btime") + 1]) if "btime" in tokens else 0
                movestogo = int(tokens[tokens.index("movestogo") + 1]) if "movestogo" in tokens else 40

                time_for_move = (wtime if side_to_move == 0 else btime) / movestogo
                move_time_ms = time_for_move - 100 # Safety margin

            if board_state:
                best_move, _, _, _, _ = iterative_deepening_search(
                    board_state, max_depth, move_time_ms, transposition_table, killer_moves
                )
                print(f"bestmove {move_to_uci(best_move)}")
        elif command == "stop":
            # (Advanced) Handle stop command
            pass
        elif command == "quit":
            break


if __name__ == "__main__":
    uci_loop()
