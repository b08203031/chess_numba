import argparse
import glob
import json
import os

import chess
import numpy as np


def board_to_arrays(board: chess.Board):
    """Convert a ``chess.Board`` into HCE tuner ``(piece_bbs, occupancy_bbs, game_state)``.

    Matches ``chess_engine.classical.fen_parser.parse_fen`` bit encodings:
    - ``game_state[0]``: side_to_move (0=White, 1=Black)
    - ``game_state[1]``: castling_rights (K=1 | Q=2 | k=4 | q=8)
    - ``game_state[2]``: legal en_passant_sq (0..63, or 64 if none)
    - ``game_state[3]``: halfmove_clock (rule50 decay)
    """
    piece_bbs = np.zeros(12, dtype=np.uint64)
    mapping = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for piece_type in range(1, 7):
        piece_bbs[mapping[piece_type]] = np.uint64(
            board.pieces_mask(piece_type, chess.WHITE)
        )
        piece_bbs[mapping[piece_type] + 6] = np.uint64(
            board.pieces_mask(piece_type, chess.BLACK)
        )

    occupancy_bbs = np.zeros(3, dtype=np.uint64)
    occupancy_bbs[0] = np.uint64(board.occupied_co[chess.WHITE])
    occupancy_bbs[1] = np.uint64(board.occupied_co[chess.BLACK])
    occupancy_bbs[2] = np.uint64(board.occupied)

    game_state = np.zeros(6, dtype=np.uint64)
    game_state[0] = np.uint64(0 if board.turn == chess.WHITE else 1)

    # Preserve castling rights (used by pawns.py virtualized shelter & TRAPPED_ROOK)
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    game_state[1] = np.uint64(castling)

    ep_sq = board.ep_square if board.has_legal_en_passant() else None
    game_state[2] = np.uint64(ep_sq if ep_sq is not None else 64)
    game_state[3] = np.uint64(board.halfmove_clock)
    game_state[4] = np.uint64(0)
    game_state[5] = np.uint64(0)

    return piece_bbs, occupancy_bbs, game_state


def parse_fen_to_arrays(fen: str):
    return board_to_arrays(chess.Board(fen))


def preprocess(input_files, output_file):
    if not input_files:
        print("No input files provided.")
        return

    all_piece_bbs = []
    all_occupancy_bbs = []
    all_game_states = []
    all_results = []

    count = 0

    for filename in input_files:
        print(f"Processing {filename}...")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fen = data["fen"]
                        result = float(data["result"])

                        p_bbs, o_bbs, g_st = parse_fen_to_arrays(fen)

                        all_piece_bbs.append(p_bbs)
                        all_occupancy_bbs.append(o_bbs)
                        all_game_states.append(g_st)
                        all_results.append(result)

                        count += 1
                        if count % 10000 == 0:
                            print(f"Parsed {count} positions...")

                    except Exception as e:
                        print(f"Error parsing line: {line.strip()} -> {e}")
                        continue
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping.")

    if count == 0:
        print("No valid data found. Aborting save.")
        return

    print(f"Saving {count} positions to {output_file}...")

    np.savez_compressed(
        output_file,
        piece_bbs=np.array(all_piece_bbs, dtype=np.uint64),
        occupancy_bbs=np.array(all_occupancy_bbs, dtype=np.uint64),
        game_states=np.array(all_game_states, dtype=np.uint64),
        results=np.array(all_results, dtype=np.float32),
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL training data into NPZ format for tuner."
    )
    parser.add_argument("--files", nargs="+", default=None, help="List of input JSONL files.")
    parser.add_argument(
        "--output", type=str, default="tuner/data/hce/dataset.npz", help="Output NPZ file."
    )

    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob("tuner/data/jsonl/training_data_*_cleaned.jsonl"))
        tactical_file = "tuner/data/jsonl/tactical_data.jsonl"
        if os.path.exists(tactical_file):
            files.append(tactical_file)

    if not files:
        print(
            "No training data found under tuner/data/jsonl/ "
            "(expected training_data_*_cleaned.jsonl)."
        )
        print("Run: python -m tuner.data_pipeline.generate_training_data")
    else:
        print(f"Found input files: {files}")
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        preprocess(files, args.output)
