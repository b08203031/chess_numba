import json
import io
import sys
import contextlib
from debug_static import get_eval_breakdown
from debug_deep_search import debug_search

def capture_output(func, *args, **kwargs):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func(*args, **kwargs)
    return f.getvalue()

def run_debug_tools():
    # Load failed puzzles
    try:
        with open("failed_puzzles.json", "r") as f:
            failed_puzzles = json.load(f)
    except FileNotFoundError:
        print("Error: failed_puzzles.json not found. Run test_puzzle.py first.")
        return

    report_lines = []
    report_lines.append("# Puzzle Analysis Report")
    report_lines.append(f"Total Failed Puzzles: {len(failed_puzzles)}\n")

    for i, puzzle in enumerate(failed_puzzles):
        name = puzzle['name']
        fen = puzzle['fen']
        correct_move = puzzle['solution']
        engine_move = puzzle['engine_move']

        print(f"[{i+1}/{len(failed_puzzles)}] Analyzing {name}...")
        report_lines.append(f"## {name}")
        report_lines.append(f"**FEN:** `{fen}`")
        report_lines.append(f"**Correct Move:** `{correct_move}`")
        report_lines.append(f"**Engine Move:** `{engine_move}`\n")

        # 1. Static Evaluation of the Root Position
        report_lines.append("### Static Evaluation (Root)")
        report_lines.append("```")
        try:
            output = capture_output(get_eval_breakdown, fen)
            report_lines.append(output)
        except Exception as e:
            report_lines.append(f"Error running debug_static: {e}")
        report_lines.append("```\n")

        # 2. Deep Search Comparison
        report_lines.append("### Search Comparison")
        report_lines.append("Comparing Engine Move vs Correct Move:")
        report_lines.append("```")
        try:
            # Analyze both moves at depth 12
            output = capture_output(debug_deep_search_wrapper, fen, [engine_move, correct_move], 12, 2000)
            report_lines.append(output)
        except Exception as e:
            report_lines.append(f"Error running debug_deep_search: {e}")
            import traceback
            traceback.print_exc()
        report_lines.append("```\n")

        report_lines.append("---\n")

    with open("PUZZLE_ANALYSIS.md", "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print("Analysis complete. Report saved to PUZZLE_ANALYSIS.md")

def debug_deep_search_wrapper(fen, moves, depth, time_limit):
    # Wrapper to call debug_search from debug_deep_search.py
    debug_search(fen, moves, depth, time_limit)

if __name__ == "__main__":
    run_debug_tools()
