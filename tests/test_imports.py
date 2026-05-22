import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 50)
print("正在測試經典引擎模組匯入...")
try:
    import chess_engine.classical.search as classical_search
    import chess_engine.classical.evaluation as classical_eval
    print("經典引擎核心模組匯入成功！")
except Exception as e:
    print(f"經典引擎匯入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 50)
print("正在測試 NNUE 引擎模組匯入...")
try:
    import chess_engine.nnue.search as nnue_search
    import chess_engine.nnue.evaluation as nnue_eval
    print("NNUE 引擎核心模組匯入成功！")
except Exception as e:
    print(f"NNUE 引擎匯入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("-" * 50)
print("正在測試主要入口腳本匯入...")
try:
    import main
    import main_nn
    print("主要入口腳本匯入成功！")
except Exception as e:
    print(f"主要入口腳本匯入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 50)
print("恭喜！所有核心模組與入口匯入驗證完全成功！")
sys.exit(0)
