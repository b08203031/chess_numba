import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.engine_types import SearchContext, MAX_PLY
from chess_engine.classical.transposition_table import create_transposition_table, clear_transposition_table
from chess_engine.classical.search import iterative_deepening_search, warmup_search_jit

POSITIONS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("Position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("Position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("Position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("Position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
]

def make_context(tt):
    return SearchContext(
        tt,
        np.zeros(MAX_PLY * 2, dtype=np.uint16),
        np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16),
        np.zeros((12, 64), dtype=np.int32),
        np.zeros((64, 64), dtype=np.int32),
        np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
        np.zeros((12, 64, 6), dtype=np.int32),
        np.full((8192, 12, 64), -1238, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros((12, 64, 12, 64), dtype=np.int16),
    )

def reset_context(ctx, tt):
    clear_transposition_table(tt)
    ctx.nodes_searched = np.uint64(0)
    ctx.killer_moves.fill(0)
    ctx.history_table.fill(0)
    ctx.butterfly_history.fill(0)
    ctx.capture_history.fill(0)
    ctx.continuation_history.fill(0)
    ctx.pawn_history.fill(-1238)
    ctx.pawn_correction_history.fill(0)
    ctx.minor_correction_history.fill(0)
    ctx.non_pawn_correction_history_white.fill(0)
    ctx.non_pawn_correction_history_black.fill(0)
    ctx.continuation_correction_history.fill(0)
    ctx.low_ply_history.fill(0)
    ctx.counter_moves.fill(0)
    ctx.pv_table.fill(0)

def run_bench(depth=15, runs_per_pos=5):
    print("Initializing TT (64MB) and SearchContext...")
    tt = create_transposition_table(64)
    search_context = make_context(tt)
    
    print("Warming up JIT...")
    warmup_search_jit(search_context)
    print(f"Warmup complete. Starting benchmark (Depth: {depth}, Runs per pos: {runs_per_pos})...\n")
    
    time_config = {'optimum_time': 0, 'maximum_time': 0}
    all_results = []
    
    for name, fen in POSITIONS:
        print(f"[{name}] FEN: {fen}")
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        
        pos_times = []
        pos_nodes = []
        pos_nps = []
        
        for r in range(runs_per_pos):
            reset_context(search_context, tt)
            
            p_bbs = piece_bbs.copy()
            o_bbs = occupancy_bbs.copy()
            g_state = game_state.copy()
            
            t0 = time.perf_counter()
            best_move, score, search_n, q_n, tt_h, last_d = iterative_deepening_search(
                p_bbs, o_bbs, g_state, depth, time_config, search_context, verbose=False
            )
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            tot_n = search_n + q_n
            nps = tot_n / elapsed if elapsed > 0 else 0
            
            pos_times.append(elapsed)
            pos_nodes.append(tot_n)
            pos_nps.append(nps)
            print(f"  Run {r+1}: time={elapsed:.4f}s, nodes={tot_n}, depth={last_d}, NPS={int(nps)}")
        
        avg_time = sum(pos_times) / runs_per_pos
        avg_nodes = sum(pos_nodes) / runs_per_pos
        avg_nps = avg_nodes / avg_time if avg_time > 0 else 0
        
        print(f"  --> Average [{name}]: Time={avg_time:.4f}s, Nodes={int(avg_nodes)}, NPS={int(avg_nps)}\n")
        all_results.append({
            'name': name,
            'avg_time': avg_time,
            'avg_nodes': avg_nodes,
            'avg_nps': avg_nps,
            'pos_times': pos_times,
            'pos_nodes': pos_nodes,
            'pos_nps': pos_nps,
        })
        
    total_time = sum(r['avg_time'] for r in all_results)
    total_nodes = sum(r['avg_nodes'] for r in all_results)
    overall_nps = total_nodes / total_time if total_time > 0 else 0
    
    print("=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    for r in all_results:
        print(f"{r['name']:<12}: Time={r['avg_time']:.4f}s | Nodes={int(r['avg_nodes']):<9} | NPS={int(r['avg_nps'])}")
    print("-" * 65)
    print(f"Total Time: {total_time:.4f}s | Total Nodes: {int(total_nodes)} | Overall NPS: {int(overall_nps)}")
    print("=" * 65)

if __name__ == "__main__":
    run_bench()

