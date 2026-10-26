import sys
sys.stdout.reconfigure(encoding='utf-8')
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.transposition_table import create_transposition_table
from chess_engine.classical.engine_types import SearchContext
from chess_engine.classical.search import iterative_deepening_search as ids_new
from chess_engine.classical.move import move_to_uci

from chess_engine.classical_old.search import iterative_deepening_search as ids_old
from chess_engine.classical_old.engine_types import SearchContext as SearchContextOld
from chess_engine.classical_old.transposition_table import create_transposition_table as create_tt_old

fen = "rn2kb1r/pb3pp1/1ppppq1p/1B6/3PP3/P1N2N2/1PP2PPP/R2QK2R w KQkq - 0 1"

p_bbs1, o_bbs1, g_state1 = parse_fen(fen)
tt1 = create_transposition_table(64)
ctx1 = SearchContext(tt1)
ctx1.max_nodes = 200000

p_bbs2, o_bbs2, g_state2 = parse_fen(fen)
tt2 = create_tt_old(64)
ctx2 = SearchContextOld(tt2)
ctx2.max_nodes = 200000

print("Running New search...")
val1, m1, n1, qn1, th1 = ids_new(p_bbs1, o_bbs1, g_state1, 64, ctx1)
print(f"New: val={val1}, move={move_to_uci(m1)}, nodes={n1}")

print("Running Old search...")
val2, m2, n2, qn2, th2 = ids_old(p_bbs2, o_bbs2, g_state2, 64, ctx2)
print(f"Old: val={val2}, move={move_to_uci(m2)}, nodes={n2}")

