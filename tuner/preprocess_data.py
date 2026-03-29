import json
import numpy as np
import chess
import glob
import os
import argparse

def parse_fen_to_arrays(fen):
    board = chess.Board(fen)
    
    # Piece Bitboards
    # Index 0-5: White P, N, B, R, Q, K
    # Index 6-11: Black P, N, B, R, Q, K
    piece_bbs = np.zeros(12, dtype=np.uint64)
    
    mapping = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for piece_type in range(1, 7): # 1 to 6
        piece_bbs[mapping[piece_type]] = np.uint64(board.pieces_mask(piece_type, chess.WHITE))
        piece_bbs[mapping[piece_type] + 6] = np.uint64(board.pieces_mask(piece_type, chess.BLACK))
        
    # Occupancy Bitboards
    # 0: White, 1: Black, 2: All
    occupancy_bbs = np.zeros(3, dtype=np.uint64)
    occupancy_bbs[0] = np.uint64(board.occupied_co[chess.WHITE])
    occupancy_bbs[1] = np.uint64(board.occupied_co[chess.BLACK])
    occupancy_bbs[2] = np.uint64(board.occupied)
    
    # Game State
    # [side_to_move, castling_rights, en_passant_sq, halfmove_clock, zobrist_key, pawn_key]
    game_state = np.zeros(6, dtype=np.uint64)
    
    # Side to move: 0 for White, 1 for Black
    game_state[0] = np.uint64(0 if board.turn == chess.WHITE else 1)
    
    # Castling rights (not strictly needed for pure static eval but part of signature)
    # This engine likely uses specific bit encoding. For tuning static eval, 0 is safe.
    game_state[1] = np.uint64(0) 
    
    # En passant square (64 if none)
    ep_sq = board.ep_square
    game_state[2] = np.uint64(ep_sq if ep_sq is not None else 64)
    
    # Halfmove clock
    game_state[3] = np.uint64(board.halfmove_clock)
    
    # Zobrist key (dummy)
    game_state[4] = np.uint64(0)
    
    return piece_bbs, occupancy_bbs, game_state

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
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fen = data['fen']
                        result = float(data['result'])
                        
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
        results=np.array(all_results, dtype=np.float32)
    )
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JSONL training data into NPZ format for tuner.")
    parser.add_argument("--files", nargs='+', default=None, help="List of input JSONL files.")
    parser.add_argument("--output", type=str, default="tuner/dataset.npz", help="Output NPZ file.")
    
    args = parser.parse_args()
    
    # If explicit files not provided, default to glob logic
    if args.files:
        files = args.files
    else:
        # 預設行為：抓取所有生成的 cleaned 資料以及 tactical 數據
        files = glob.glob("tuner/training_data_*_cleaned.jsonl")
        tactical_file = "tuner/tactical_data.jsonl"
        if os.path.exists(tactical_file):
            files.append(tactical_file)
    
    if not files:
        print("No training data found in tuner/ directory (matching 'tuner/training_data_*_cleaned.jsonl').")
        print("Please run generate_training_data.py first.")
    else:
        print(f"Found input files: {files}")
        preprocess(files, args.output)
