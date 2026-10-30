import sys
import os
import ctypes

# Enable Windows Per-Monitor DPI awareness before initializing GUI/screen capture
# Ensures SetCursorPos coordinates align 1:1 with physical pixels captured by mss
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import collections
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))
import time
import chess
import chess.pgn
import traceback
import re

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# --- Configuration & Constants ---
UNICODE_PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    None: ''
}

BOARD_COLORS = ["#B58863", "#F0D9B5"]  # Dark, Light squares (Wood theme)
HIGHLIGHT_COLOR = "#FFFF00"            # Yellow for best move
LAST_MOVE_COLOR = "#BBCB2B"            # Greenish for last move


def sanitize_fen_input(text, field_type):
    """
    Sanitizes user input for FEN fields to prevent UCI command injection.
    """
    stripped = text.strip()

    if field_type == 'castling':
        if stripped == "-":
            return "-"
        if re.match(r"^[KQkq]+$", stripped) and len(stripped) <= 4:
            return stripped
        return "-"

    elif field_type == 'ep':
        if stripped == "-":
            return "-"
        if re.match(r"^[a-h][36]$", stripped):
            return stripped
        return "-"

    return "-"


class ToolTip(object):
    """
    Creates a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)
        self.tw = None

    def enter(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20

        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            self.tw, text=self.text, justify='left',
            background='#ffffe0', relief='solid', borderwidth=1,
            font=("Segoe UI", "9", "normal")
        )
        label.pack(ipadx=5, ipady=2)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


class EngineProcess:
    """
    Manages the subprocess for the chess engine (main.py).
    Handles UCI communication via stdin/stdout with strict search_id tracking
    so stale bestmove/info outputs from interrupted searches never leak into new searches.
    """
    def __init__(self, callback_queue, engine_script="main.py"):
        self.callback_queue = callback_queue
        self.process = None
        self.running = False
        self.threads = []
        self._lock = threading.Lock()
        self._inflight_search_ids = collections.deque()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.engine_path = os.path.join(script_dir, engine_script)

    def start(self):
        if self.running:
            return

        try:
            cmd = [sys.executable, "-u", self.engine_path]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
                bufsize=1
            )
            self.running = True

            t_out = threading.Thread(target=self._read_stdout, daemon=True)
            t_err = threading.Thread(target=self._read_stderr, daemon=True)
            self.threads = [t_out, t_err]
            for t in self.threads:
                t.start()

            # Initialize UCI; main.py runs warmup_search_jit() on first `isready` and replies `readyok`
            self.send_command("uci")
            self.send_command("isready")

        except Exception as e:
            self.callback_queue.put({"type": "error", "message": f"Failed to start engine: {e}"})

    def start_search(self, position_cmd: str, go_cmd: str, search_id: int, own_book: bool = True):
        """Atomically configures OwnBook, sets position (with full move history), and starts `go`."""
        if not (self.running and self.process):
            return
        with self._lock:
            self._inflight_search_ids.append(search_id)
            try:
                self.process.stdin.write(f"setoption name OwnBook value {'true' if own_book else 'false'}\n")
                self.process.stdin.write(position_cmd + "\n")
                self.process.stdin.write(go_cmd + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                self.running = False

    def stop_calculation(self):
        """Sends 'stop' command to engine to gracefully end current search."""
        if self.running and self.process:
            self.send_command("stop")

    def quit_engine(self):
        """Terminates the engine process."""
        if self.running and self.process:
            self.send_command("quit")
            time.sleep(0.1)
            if self.process.poll() is None:
                self.process.terminate()
            self.running = False

    def send_command(self, cmd):
        if self.running and self.process:
            with self._lock:
                try:
                    self.process.stdin.write(cmd + "\n")
                    self.process.stdin.flush()
                except BrokenPipeError:
                    self.running = False

    def _read_stdout(self):
        while self.running and self.process:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                print(line, flush=True)
                self.callback_queue.put({"type": "raw_log", "message": line})
                self._parse_uci_output(line)
            except Exception:
                break

    def _read_stderr(self):
        while self.running and self.process:
            try:
                line = self.process.stderr.readline()
                if not line:
                    break
                line = line.strip()
                self.callback_queue.put({"type": "raw_log", "message": f"[STDERR] {line}"})
            except Exception:
                break

    def _parse_uci_output(self, line):
        parts = line.split()
        if not parts:
            return

        cmd = parts[0]

        if cmd == "readyok":
            self.callback_queue.put({"type": "readyok"})

        elif cmd == "info":
            with self._lock:
                cur_sid = self._inflight_search_ids[0] if self._inflight_search_ids else None
            info_data = {"type": "info", "search_id": cur_sid}

            if "string" in parts:
                idx = parts.index("string")
                info_data["string"] = " ".join(parts[idx + 1:])

            if "score" in parts:
                idx = parts.index("score")
                if idx + 2 < len(parts):
                    info_data["score_type"] = parts[idx + 1]
                    info_data["score_val"] = parts[idx + 2]

            for key in ("depth", "seldepth", "nodes", "nps", "time"):
                if key in parts:
                    idx = parts.index(key)
                    if idx + 1 < len(parts):
                        info_data[key] = parts[idx + 1]

            castling_map = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
            if "pv" in parts:
                idx = parts.index("pv")
                pv_moves = parts[idx + 1:]
                mapped_pv = [castling_map.get(m, m) for m in pv_moves]
                info_data["pv"] = " ".join(mapped_pv)

            self.callback_queue.put(info_data)

        elif cmd == "bestmove":
            move = parts[1] if len(parts) > 1 else "0000"
            with self._lock:
                sid = self._inflight_search_ids.popleft() if self._inflight_search_ids else None
            self.callback_queue.put({"type": "bestmove", "move": move, "search_id": sid})


class ChessVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("西洋棋視覺助理 (Chess Vision App)")
        self.geometry("500x800")

        # Defer topmost to avoid locking the window on startup
        self.after(500, lambda: self.attributes('-topmost', True))

        # State
        self.board = chess.Board()
        self.game_pgn = chess.pgn.Game.from_board(self.board)
        self.game_node = self.game_pgn
        self.message_queue = queue.Queue()

        self._recognizer_lock = threading.Lock()
        self.recognizer = None

        self.analyzing = False
        self.warming_up = True
        self.current_search_id = 0
        self.is_first_analysis = True
        self.auto_detecting_opponent = False
        self.last_detected_fen = ""
        self.autoplay_mode_var = tk.StringVar(value="fixed")  # "fixed" or "self"
        self.use_book_var = tk.BooleanVar(value=True)

        # Manual Mode & Arrow States
        self.redo_stack = []
        self.selected_square = None
        self.best_move_arrow = None
        self.dragging = False

        # Logs Initialization
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.join(script_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.logs_dir, f"game_eval_{timestamp}.log")

        # UI Components
        self._ui_state = {}
        self._cached_hwnd = None
        self._create_ui()
        self._refresh_ui_snapshot()

        # Start engine after UI is ready so `readyok` updates status label cleanly
        self.engine = EngineProcess(self.message_queue)
        self.engine.start()

        # Initialize recognizer asynchronously in background so buttons never hit NoneType
        threading.Thread(target=self._ensure_recognizer, daemon=True).start()

        # Start message polling
        self.after(20, self._process_queue)

    def _post_ui(self, fn, *args):
        """
        Thread-safe UI callback dispatch:
        - Executes synchronously if already on the Tkinter main thread.
        - Otherwise enqueues onto `self.message_queue` so worker threads never touch Tcl locks.
        """
        if threading.current_thread() is threading.main_thread():
            fn(*args)
        else:
            self.message_queue.put({"type": "ui_call", "fn": fn, "args": args})

    def _refresh_ui_snapshot(self):
        """Captures Tkinter variable values & HWND onto plain Python attributes on the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        d = self.__dict__
        state = d.setdefault("_ui_state", {})
        for key, attr in (
            ("my_color", "my_color_var"),
            ("side", "side_var"),
            ("auto_detect", "auto_detect_var"),
            ("auto_detect_opponent", "auto_detect_opponent_var"),
            ("auto_play", "auto_play_var"),
            ("autoplay_mode", "autoplay_mode_var"),
            ("use_book", "use_book_var"),
        ):
            var = d.get(attr)
            if var is not None:
                try:
                    state[key] = var.get()
                except Exception:
                    pass
        try:
            state["go_cmd"] = self._build_go_cmd()
        except Exception:
            pass
        try:
            if d.get("tk") is not None:
                d["_cached_hwnd"] = self.winfo_id()
        except Exception:
            pass

    def _ui_get(self, key: str, var_attr: str, default=None):
        """Reads UI state safely from either main thread or background worker threads."""
        d = self.__dict__
        if threading.current_thread() is threading.main_thread():
            var = d.get(var_attr)
            if var is not None:
                try:
                    val = var.get()
                    state = d.setdefault("_ui_state", {})
                    state[key] = val
                    return val
                except Exception:
                    pass
        state = d.get("_ui_state")
        if state and key in state:
            return state[key]
        var = d.get(var_attr)
        if var is not None:
            try:
                return var.get()
            except Exception:
                pass
        return default

    def _ensure_recognizer(self):
        """Thread-safe lazy initialization of ScreenRecognizer."""
        if self.recognizer is not None:
            return self.recognizer
        with self._recognizer_lock:
            if self.recognizer is None:
                from tools import screen_recognizer
                rec = screen_recognizer.ScreenRecognizer(mode="universal")
                self.recognizer = rec
                if rec.mode != "universal":
                    self._post_ui(lambda: self.piece_style_var.set("Default"))
        return self.recognizer

    @staticmethod
    def build_uci_position_cmd(board: chess.Board) -> str:
        """
        Builds a full-history UCI `position` command (`position startpos moves ...` or
        `position fen <root> moves ...`) so the engine accurately tracks 3-fold repetition
        and the 50-move rule throughout the game.
        """
        root_board = board.root()
        moves_uci = [m.uci() for m in board.move_stack]
        root_fen = root_board.fen()
        base = "startpos" if root_fen == chess.STARTING_FEN else f"fen {root_fen}"
        if moves_uci:
            return f"position {base} moves {' '.join(moves_uci)}"
        return f"position {base}"

    def _build_go_cmd(self) -> str:
        d = self.__dict__
        limit_nodes_var = d.get("limit_by_nodes_var")
        if limit_nodes_var is not None and limit_nodes_var.get():
            try:
                nodes_limit = max(100, int(d["nodes_limit_var"].get()))
            except Exception:
                nodes_limit = 100000
            return f"go nodes {nodes_limit}"
        try:
            time_var = d.get("time_var")
            time_limit = max(50, int(float(time_var.get()) * 1000)) if time_var is not None else 10000
        except Exception:
            time_limit = 10000
        return f"go movetime {time_limit}"

    def get_pgn_string(self):
        """Returns the current game in standard PGN format."""
        if self.game_pgn is None:
            return ""
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return self.game_pgn.accept(exporter)

    def copy_pgn_to_clipboard(self):
        """Copies the current game PGN to system clipboard."""
        pgn_str = self.get_pgn_string()
        if not pgn_str:
            messagebox.showinfo("PGN", "目前尚無對局紀錄 (No PGN history yet)")
            return
        self.clipboard_clear()
        self.clipboard_append(pgn_str)
        messagebox.showinfo("PGN", "PGN 已複製到剪貼簿 (PGN copied to clipboard)")

    def _sync_ui_vars_from_board(self):
        """Must be called on the Tkinter main thread."""
        ep = chess.square_name(self.board.ep_square) if self.board.ep_square is not None else "-"
        self.ep_var.set(ep)
        self.castling_var.set(self.board.fen().split()[2])
        self.side_var.set("w" if self.board.turn == chess.WHITE else "b")

    def init_game_from_board(self, board):
        """Initializes a new game / PGN rooted at the given board position (Main thread)."""
        self.board = chess.Board(board.fen())
        self.game_pgn = chess.pgn.Game.from_board(self.board)
        self.game_node = self.game_pgn
        self.best_move_arrow = None
        self._sync_ui_vars_from_board()

    def find_matching_legal_move(self, target_board):
        """1-ply convenience wrapper."""
        seq = self.find_matching_move_sequence(target_board, max_plies=1)
        return seq[0] if seq and len(seq) == 1 else None

    def find_matching_move_sequence(self, target_board, max_plies=2):
        """
        Finds a 1-ply or 2-ply legal move sequence from `self.board` that produces
        `target_board`'s exact piece placement.
          - Returns `[]` if `target_board` already matches `self.board`.
          - Returns `[m1]` if 1 legal move matches.
          - Returns `[m1, m2]` if 2 legal moves (e.g. our move + opponent reply) match.
          - Returns `None` if no legal <=max_plies sequence matches.
        """
        if target_board is None:
            return None
        if isinstance(target_board, str):
            target_board = chess.Board(target_board)

        target_pieces = target_board.board_fen()
        if self.board.board_fen() == target_pieces:
            return []

        b_work = self.board.copy(stack=False)

        # Changed squares filter: any move m1's from_square MUST differ between b_work and target_board
        diff_squares = {
            sq for sq in chess.SQUARES
            if b_work.piece_at(sq) != target_board.piece_at(sq)
        }

        # 1. Check 1-ply legal moves first
        for m1 in b_work.legal_moves:
            if m1.from_square not in diff_squares:
                continue
            b_work.push(m1)
            if b_work.board_fen() == target_pieces:
                b_work.pop()
                return [m1]
            b_work.pop()

        if max_plies < 2:
            return None

        # 2. Check 2-ply legal move sequences (our move + fast opponent reply / manual pair)
        for m1 in b_work.legal_moves:
            if m1.from_square not in diff_squares:
                continue
            b_work.push(m1)
            diff_after_m1 = {
                sq for sq in chess.SQUARES
                if b_work.piece_at(sq) != target_board.piece_at(sq)
            }
            for m2 in b_work.legal_moves:
                if m2.from_square not in diff_after_m1:
                    continue
                b_work.push(m2)
                if b_work.board_fen() == target_pieces:
                    b_work.pop()
                    b_work.pop()
                    return [m1, m2]
                b_work.pop()
            b_work.pop()

        return None

    def apply_move(self, move: chess.Move):
        """
        Applies a legal move to `self.board` and records it into the PGN game tree (Main thread).
        """
        if move not in self.board.legal_moves:
            print(f"[WARN] Move {move} is not legal in position {self.board.fen()}")
            return False

        self.board.push(move)
        if self.game_node is not None:
            self.game_node = self.game_node.add_variation(move)

        self._sync_ui_vars_from_board()
        return True

    def _is_text_input_focused(self) -> bool:
        focused = self.focus_get()
        if focused is None:
            return False
        return isinstance(focused, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text))

    def _create_ui(self):
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Arial", 11, "bold"), padding=[12, 6])

        # 1. ttk.Notebook (Tabs Container)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="x", padx=10, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # 2. Main Control Buttons Frame
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.btn_analyze = ttk.Button(btn_frame, text="▶ 開始分析 (Start)", command=self.start_analysis_thread)
        self.btn_analyze.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ToolTip(self.btn_analyze, "快速鍵 (Shortcut): Enter")

        self.btn_stop = ttk.Button(btn_frame, text="⏹ 停止 (Stop)", command=self.stop_analysis, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ToolTip(self.btn_stop, "快速鍵 (Shortcut): Esc")

        self.btn_new_game = ttk.Button(btn_frame, text="🔄 新遊戲 (New Game)", command=self.new_game)
        self.btn_new_game.pack(side="left", fill="x", expand=True, padx=(5, 0))
        ToolTip(self.btn_new_game, "快速鍵 (Shortcut): Ctrl+N")

        # Tab 1: Auto Capture Settings
        self.tab_auto = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_auto, text="自動偵測 (Auto)")

        row_idx = 0

        # Side to Move
        ttk.Label(self.tab_auto, text="輪到誰走 (Side to Move):").grid(row=row_idx, column=0, sticky="w")
        self.side_var = tk.StringVar(value="w")
        ttk.Radiobutton(self.tab_auto, text="白方 (White)", variable=self.side_var, value="w",
                        command=self._on_side_radio_changed).grid(row=row_idx, column=1, sticky="w")
        ttk.Radiobutton(self.tab_auto, text="黑方 (Black)", variable=self.side_var, value="b",
                        command=self._on_side_radio_changed).grid(row=row_idx, column=2, sticky="w")
        row_idx += 1

        # My Color (for board orientation)
        ttk.Label(self.tab_auto, text="我的顏色 (My Color):").grid(row=row_idx, column=0, sticky="w")
        self.my_color_var = tk.StringVar(value="w")
        ttk.Radiobutton(self.tab_auto, text="白方 (White)", variable=self.my_color_var, value="w",
                        command=self.draw_board).grid(row=row_idx, column=1, sticky="w")
        ttk.Radiobutton(self.tab_auto, text="黑方 (Black)", variable=self.my_color_var, value="b",
                        command=self.draw_board).grid(row=row_idx, column=2, sticky="w")
        row_idx += 1

        # Auto Detect Rights Checkbox
        self.auto_detect_var = tk.BooleanVar(value=True)
        chk_auto = ttk.Checkbutton(self.tab_auto, text="自動偵測權利與視角 (Auto-detect Rights & Side)", variable=self.auto_detect_var)
        chk_auto.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)
        ToolTip(chk_auto, "自動推斷王車易位權、吃過路兵與開局黑白視角")
        row_idx += 1

        # Castling Rights
        ttk.Label(self.tab_auto, text="王車易位 (Castling Rights):").grid(row=row_idx, column=0, sticky="w")
        self.castling_var = tk.StringVar(value="KQkq")
        castling_choices = ["KQkq", "KQ", "kq", "K", "Q", "k", "q", "Kk", "Kq", "Qk", "Qq", "KQk", "KQq", "Kkq", "Qkq", "-"]
        cb_castling = ttk.Combobox(self.tab_auto, textvariable=self.castling_var, values=castling_choices, width=8)
        cb_castling.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(cb_castling, "王車易位權 (例如: KQkq, -)\nCastling rights (e.g. KQkq, -)")
        row_idx += 1

        # En Passant
        ttk.Label(self.tab_auto, text="吃過路兵 (En Passant):").grid(row=row_idx, column=0, sticky="w")
        self.ep_var = tk.StringVar(value="-")
        ep_choices = ["-", "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6"]
        cb_ep = ttk.Combobox(self.tab_auto, textvariable=self.ep_var, values=ep_choices, width=6, state="readonly")
        cb_ep.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(cb_ep, "過路兵目標格 (例如: e3, -)\nTarget square (e.g. e3, -)")
        row_idx += 1

        # Time Limit
        ttk.Label(self.tab_auto, text="思考時間 (Time Limit s):").grid(row=row_idx, column=0, sticky="w")
        self.time_var = tk.StringVar(value="10")
        time_choices = ["0.5", "1", "2", "5", "10", "30", "60"]
        cb_time = ttk.Combobox(self.tab_auto, textvariable=self.time_var, values=time_choices, width=6)
        cb_time.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(cb_time, "秒 (1 = 1秒)\nSeconds (1 = 1s)")
        row_idx += 1

        # Limit by Nodes Option
        self.limit_by_nodes_var = tk.BooleanVar(value=False)
        chk_limit_nodes = ttk.Checkbutton(self.tab_auto, text="限定節點數 (Limit Nodes)", variable=self.limit_by_nodes_var)
        chk_limit_nodes.grid(row=row_idx, column=0, sticky="w", pady=2)
        ToolTip(chk_limit_nodes, "開啟時以設定的節點數為搜尋上限\nWhen enabled, limit search by node count")

        self.nodes_limit_var = tk.StringVar(value="100000")
        self.entry_nodes_limit = ttk.Entry(self.tab_auto, textvariable=self.nodes_limit_var, width=8)
        self.entry_nodes_limit.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(self.entry_nodes_limit, "節點數上限 (預設 100000)\nMaximum node count (default 100000)")
        row_idx += 1

        # Opening Book Option
        chk_book = ttk.Checkbutton(self.tab_auto, text="使用開局書 (Use Opening Book)", variable=self.use_book_var, command=self.on_toggle_book)
        chk_book.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)
        ToolTip(chk_book, "開啟時在開局階段直接查閱 Polyglot 開局書快速出子")
        row_idx += 1

        # Always on top Checkbox
        self.always_on_top_var = tk.BooleanVar(value=True)
        chk_top = ttk.Checkbutton(self.tab_auto, text="視窗置頂 (Always on Top)", variable=self.always_on_top_var, command=self.toggle_topmost)
        chk_top.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)
        row_idx += 1

        # Auto-Play Checkbox and Mode
        auto_play_frame = ttk.Frame(self.tab_auto)
        auto_play_frame.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)

        self.auto_play_var = tk.BooleanVar(value=False)
        chk_autoplay = ttk.Checkbutton(
            auto_play_frame, text="自動下棋 (Auto-Play)",
            variable=self.auto_play_var, command=self.on_toggle_auto_play
        )
        chk_autoplay.pack(side="left")
        ToolTip(chk_autoplay, "自動在螢幕上點擊滑鼠下棋 (Auto-click best move)")

        ttk.Radiobutton(
            auto_play_frame, text="固定方 (Fixed)", variable=self.autoplay_mode_var,
            value="fixed", command=self.on_autoplay_mode_changed
        ).pack(side="left", padx=(10, 0))
        ttk.Radiobutton(
            auto_play_frame, text="自己對戰 (Self-Play)", variable=self.autoplay_mode_var,
            value="self", command=self.on_autoplay_mode_changed
        ).pack(side="left")
        row_idx += 1

        # Auto-Detect Opponent Checkbox
        self.auto_detect_opponent_var = tk.BooleanVar(value=False)
        chk_autodetect_opp = ttk.Checkbutton(
            self.tab_auto, text="自動偵測對手 (Auto-Detect Opponent)",
            variable=self.auto_detect_opponent_var, command=self.on_toggle_auto_detect_opponent
        )
        chk_autodetect_opp.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)
        ToolTip(chk_autodetect_opp, "推薦步顯示後自動偵測我方與對手落子，輪到我方時自動啟動分析")
        row_idx += 1

        # Piece Style Selector & Learn Button
        ttk.Label(self.tab_auto, text="辨識模式 (Piece Mode):").grid(row=row_idx, column=0, sticky="w", pady=2)
        self.piece_style_var = tk.StringVar(value="Universal (AI)")
        cb_style = ttk.Combobox(
            self.tab_auto, textvariable=self.piece_style_var,
            values=["Universal (AI)", "Default", "Custom"], width=13, state="readonly"
        )
        cb_style.grid(row=row_idx, column=1, sticky="w", pady=2)
        cb_style.bind("<<ComboboxSelected>>", self.on_piece_style_selected)
        ToolTip(cb_style, "Universal (AI): 跨網站通用 CNN 免學習辨識\nDefault/Custom: 傳統圖示模板匹配")

        self.btn_learn = ttk.Button(self.tab_auto, text="學習 (Learn)", command=self.learn_piece_style, width=10)
        self.btn_learn.grid(row=row_idx, column=2, sticky="w", padx=2, pady=2)
        ToolTip(self.btn_learn, "遇到特殊造型棋子時，設為起始盤面點此一鍵擷取自訂模板")
        row_idx += 1

        # Calibration & Quick Re-detect Frame
        calib_frame = ttk.Frame(self.tab_auto)
        calib_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=4)
        self.btn_redetect = ttk.Button(calib_frame, text="🎯 重新定位棋盤", command=self.redetect_board_now)
        self.btn_redetect.pack(side="left", fill="x", expand=True, padx=(0, 3))
        ToolTip(self.btn_redetect, "移動或捲動網頁視窗後，一鍵重新自動搜尋 8x8 棋盤格網")

        self.btn_calibrate = ttk.Button(calib_frame, text="🔍 預覽與手動校正", command=self.open_calibration_window)
        self.btn_calibrate.pack(side="left", fill="x", expand=True, padx=(3, 0))
        ToolTip(self.btn_calibrate, "開啟螢幕截圖預覽並手動微調四個棋盤角點")
        row_idx += 1

        # Tab 2: Manual Play / Analysis Mode
        self.tab_manual = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_manual, text="手動分析 (Manual)")

        ttk.Label(self.tab_manual, text="輸入 FEN (FEN Input):").grid(row=0, column=0, sticky="w", pady=5)
        self.manual_fen_var = tk.StringVar(value=chess.STARTING_FEN)
        self.entry_manual_fen = ttk.Entry(self.tab_manual, textvariable=self.manual_fen_var, width=32)
        self.entry_manual_fen.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)
        self.entry_manual_fen.bind("<Return>", lambda e: self.load_manual_fen())

        ttk.Button(self.tab_manual, text="載入 FEN (Load)", command=self.load_manual_fen).grid(row=1, column=0, pady=5, sticky="ew", padx=2)
        ttk.Button(self.tab_manual, text="清空棋盤 (Clear)", command=self.clear_manual_board).grid(row=1, column=1, pady=5, sticky="ew", padx=2)
        ttk.Button(self.tab_manual, text="重置起始位置 (Reset)", command=self.reset_manual_board).grid(row=1, column=2, pady=5, sticky="ew", padx=2)

        self.live_eval_var = tk.BooleanVar(value=True)
        chk_live = ttk.Checkbutton(self.tab_manual, text="落子自動即時分析 (Live Eval)", variable=self.live_eval_var)
        chk_live.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)
        ToolTip(chk_live, "在手動分析模式下，每次移動棋子後是否立即自動開始分析")

        chk_book_manual = ttk.Checkbutton(self.tab_manual, text="使用開局書 (Use Opening Book)", variable=self.use_book_var, command=self.on_toggle_book)
        chk_book_manual.grid(row=3, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Button(self.tab_manual, text="翻轉棋盤 (Flip Board)", command=self.flip_board).grid(row=4, column=0, columnspan=3, pady=5, sticky="ew")

        # Tab 3: Diagnostic Log Console
        self.tab_console = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_console, text="日誌主控台 (Console)")

        console_frame = ttk.Frame(self.tab_console)
        console_frame.pack(fill="both", expand=True)
        self.console_text = tk.Text(console_frame, height=9, width=50, state="disabled", wrap="word", font=("Consolas", 9))
        self.console_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(console_frame, orient="vertical", command=self.console_text.yview)
        sb.pack(side="right", fill="y")
        self.console_text.configure(yscrollcommand=sb.set)

        self.btn_open_logs = ttk.Button(self.tab_console, text="📂 開啟日誌資料夾 (Open Log Folder)", command=self.open_log_folder)
        self.btn_open_logs.pack(pady=5, fill="x")

        self.btn_copy_pgn = ttk.Button(self.tab_console, text="📋 複製當前對局 PGN (Copy PGN)", command=self.copy_pgn_to_clipboard)
        self.btn_copy_pgn.pack(pady=2, fill="x")

        # 3. Info Frame
        info_frame = ttk.LabelFrame(self, text="分析結果 (Analysis)", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_status = ttk.Label(info_frame, text="狀態 (Status): 引擎熱機中... (Warming Up JIT...)", font=("Arial", 10, "italic"), foreground="red")
        self.lbl_status.pack(anchor="w", pady=(0, 5))

        self.lbl_score = ttk.Label(info_frame, text="評分 (Score): --", font=("Arial", 12, "bold"))
        self.lbl_score.pack(anchor="w")

        self.lbl_bestmove = ttk.Label(info_frame, text="最佳著法 (Best Move): --", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_bestmove.pack(anchor="w")

        self.lbl_pv = ttk.Label(info_frame, text="變例 (PV): --", wraplength=460)
        self.lbl_pv.pack(anchor="w", fill="x")

        # 4. Board Canvas
        self.canvas_size = 400
        self.square_size = self.canvas_size // 8

        self.load_piece_images()

        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Global Shortcuts (guarded against hijacking text Entry / Combobox widgets)
        self.bind('<Return>', self._on_return_shortcut)
        self.bind('<Escape>', self._on_escape_shortcut)
        self.bind('<Control-n>', lambda e: self.new_game())

        self.draw_board()

    def _on_return_shortcut(self, event=None):
        if self._is_text_input_focused():
            return
        self.start_analysis_thread()

    def _on_escape_shortcut(self, event=None):
        self.canvas.focus_set()
        self.stop_analysis()

    def _on_side_radio_changed(self):
        self.board.turn = (chess.WHITE if self.side_var.get() == 'w' else chess.BLACK)

    def on_toggle_auto_play(self):
        if self.auto_play_var.get() and self.autoplay_mode_var.get() == "fixed":
            # In Fixed Auto-Play, users almost always want continuous play across opponent replies
            if not self.auto_detect_opponent_var.get():
                self.auto_detect_opponent_var.set(True)

    def on_autoplay_mode_changed(self):
        if self.autoplay_mode_var.get() == "self":
            # Selecting Self-Play radio button automatically enables Auto-Play
            if not self.auto_play_var.get():
                self.auto_play_var.set(True)

    def on_toggle_auto_detect_opponent(self):
        """Starts or stops background turn monitoring immediately when toggled."""
        if not self.auto_detect_opponent_var.get():
            self.auto_detecting_opponent = False
            if not self.analyzing and not self.warming_up:
                self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")
            return

        if self.warming_up or self.analyzing:
            return
        try:
            if self.notebook.index("current") != 0:
                return
        except Exception:
            pass

        if self.autoplay_mode_var.get() == "fixed":
            if self.side_var.get() != self.my_color_var.get():
                if not self.auto_detecting_opponent:
                    threading.Thread(target=self._auto_detect_loop, daemon=True).start()
            elif self.best_move_arrow is not None and not self.auto_play_var.get():
                # Recommendation already shown for our turn -> wait for our move then opponent's
                if not self.auto_detecting_opponent:
                    threading.Thread(target=self._auto_detect_manual_play_loop, daemon=True).start()

    def toggle_topmost(self):
        self.attributes('-topmost', self.always_on_top_var.get())

    def on_toggle_book(self):
        val = self.use_book_var.get()
        if hasattr(self, "engine"):
            self.engine.send_command(f"setoption name OwnBook value {'true' if val else 'false'}")
        print(f"[INFO] 開局書設定為 (OwnBook set to): {val}")

    def on_canvas_resize(self, event):
        new_size = min(event.width, event.height)
        if new_size > 50 and abs(self.canvas_size - new_size) > 2:
            self.canvas_size = new_size
            self.square_size = self.canvas_size // 8
            self.load_piece_images()
            self.draw_board()

    def load_piece_images(self):
        """Loads piece images for the assistant's preview canvas."""
        self.piece_images = {}
        if not HAS_PIL:
            return

        try:
            self.original_pil_images = {}
            script_dir = os.path.dirname(os.path.abspath(__file__))
            style_name = self.piece_style_var.get() if hasattr(self, 'piece_style_var') else "Universal (AI)"

            custom_dir = os.path.abspath(os.path.join(script_dir, "data", "templates", "custom"))
            default_dir = os.path.abspath(os.path.join(script_dir, "data", "templates"))
            if style_name == "Custom" and os.path.isdir(custom_dir):
                template_dir = custom_dir
            else:
                template_dir = default_dir

            pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
            for p in pieces:
                path = os.path.join(template_dir, f"{p}.png")
                if os.path.exists(path):
                    self.original_pil_images[p] = Image.open(path)

            for p, img in self.original_pil_images.items():
                img_resized = img.resize((self.square_size, self.square_size), Image.Resampling.LANCZOS)
                self.piece_images[p] = ImageTk.PhotoImage(img_resized)
        except Exception as e:
            print(f"[WARN] Failed to load piece images: {e}")
            self.piece_images = {}

    def redetect_board_now(self):
        """Force re-detecting the 8x8 chessboard on screen."""
        if self.warming_up:
            messagebox.showwarning("Warning", "引擎正在熱機中，請稍候... (Engine Warming Up)")
            return
        try:
            rec = self._ensure_recognizer()
            rec.invalidate_board_cache()
            screenshot = rec.capture_screen()
            board_img = rec.find_board(screenshot, exclude_rect=self.get_exclude_rect(), verbose=True)
            if board_img is None:
                self.lbl_status.config(text="狀態 (Status): 找不到棋盤，請確認棋盤完整顯示或使用手動校正", foreground="red")
                messagebox.showwarning("Warning", "未能在螢幕上自動找到 8x8 棋盤！\n請確認棋盤完整顯示且未被遮擋，或點擊「預覽與手動校正」。")
                return

            # Also auto-check starting orientation if applicable
            vis_pieces = rec.identify_board_pieces(board_img)
            orient = rec.detect_starting_orientation(vis_pieces)
            if orient in ('w', 'b') and self.auto_detect_var.get():
                self.my_color_var.set(orient)
                self.draw_board()

            self.lbl_status.config(text="狀態 (Status): 已成功重新定位螢幕 8x8 棋盤！(Board Locked)", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"重新定位棋盤失敗: {e}")

    def new_game(self):
        """Resets engine TT & starts a fresh PGN game rooted at the current screen/manual board."""
        if self.analyzing or self.auto_detecting_opponent:
            self.stop_analysis()

        self.engine.send_command("ucinewgame")
        self.engine.send_command(f"setoption name OwnBook value {'true' if self.use_book_var.get() else 'false'}")

        new_board = None
        try:
            current_tab = self.notebook.index("current")
        except Exception:
            current_tab = 0

        if current_tab == 0:
            try:
                rec = self._ensure_recognizer()
                if not rec.manual_calibrated:
                    rec.invalidate_board_cache()
                screenshot = rec.capture_screen()
                board_img = rec.get_warped_board(screenshot) if rec.M is not None else None
                if board_img is None:
                    board_img = rec.find_board(screenshot, exclude_rect=self.get_exclude_rect(), verbose=False)
                if board_img is not None:
                    vis_pieces = rec.identify_board_pieces(board_img)
                    if self.auto_detect_var.get():
                        orient = rec.detect_starting_orientation(vis_pieces)
                        if orient in ('w', 'b'):
                            self.my_color_var.set(orient)
                    fen_raw = rec.get_fen_from_board_image(
                        board_img,
                        player_color=self.my_color_var.get(),
                        active_player=self.side_var.get(),
                        castling='-',
                        en_passant='-',
                    )
                    b_screen = chess.Board(fen_raw)
                    # At standard starting position, White always moves first
                    active = 'w' if b_screen.board_fen() == chess.STARTING_BOARD_FEN else self.side_var.get()
                    rights = self.infer_castling_rights(b_screen, "KQkq")
                    parts = fen_raw.split()
                    parts[1] = active
                    parts[2] = rights
                    parts[3] = "-"
                    new_board = chess.Board(" ".join(parts))
            except Exception as e:
                print(f"[INFO] 讀取螢幕盤面作為新局面起點失敗: {e}")

        elif current_tab == 1:
            try:
                new_board = chess.Board(self.manual_fen_var.get().strip())
            except Exception:
                new_board = None

        if new_board is None:
            new_board = chess.Board()

        self.init_game_from_board(new_board)
        self.manual_fen_var.set(self.board.fen())
        self.last_detected_fen = self.board.fen()
        self.is_first_analysis = True
        self.redo_stack = []
        self.draw_board()
        self.lbl_pv.config(text="變例 (PV): --")
        self.lbl_score.config(text="評分 (Score): --")
        self.lbl_bestmove.config(text="最佳著法 (Best Move): --")
        self.lbl_status.config(text="狀態 (Status): 新局面已開啟，等待下一步... (New Game Ready)", foreground="green")
        messagebox.showinfo("Info", "新局面已開啟 (New Game Started) - PGN已清空並以當前盤面作為起點")

    def start_analysis_thread(self, current_fen=None):
        if self.warming_up:
            messagebox.showwarning("Warning", "引擎正在熱機中，請稍候... (Engine Warming Up)")
            return

        try:
            current_tab = self.notebook.index("current")
        except Exception:
            current_tab = 0

        if current_tab == 1:
            self.start_manual_analysis()
            return

        # Stop any active background opponent/player watcher before starting fresh analysis
        self.auto_detecting_opponent = False
        self._refresh_ui_snapshot()

        is_self_play = (self.auto_play_var.get() and self.autoplay_mode_var.get() == "self")

        # In Fixed mode, if UI currently indicates opponent's turn and no pre-synced FEN was given:
        # start opponent watcher (which also immediately checks if opponent has ALREADY moved on screen!)
        if not is_self_play and current_fen is None and self.side_var.get() != self.my_color_var.get():
            if self.auto_detect_opponent_var.get():
                print("[INFO] 目前為固定方模式且輪到對手，啟動對手偵測...")
                threading.Thread(target=self._auto_detect_loop, daemon=True).start()
                return
            # Even if auto_detect_opponent is off, let _run_analysis capture the screen once:
            # if the opponent already moved on screen, _run_analysis will advance turn to us and analyze!

        if self.analyzing:
            return

        self.analyzing = True
        self.current_search_id += 1
        search_id = self.current_search_id

        self.btn_analyze.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="狀態 (Status): 思考計算中... (Analyzing...)", foreground="orange")
        self.lbl_score.config(text="評分 (Score): 計算中...")
        self.lbl_bestmove.config(text="最佳著法 (Best Move): ...")
        self.lbl_pv.config(text="變例 (PV): ...")
        self.best_move_arrow = None

        threading.Thread(target=self._run_analysis, args=(current_fen, search_id), daemon=True).start()

    def stop_analysis(self):
        self.auto_detecting_opponent = False
        if self.analyzing:
            self.current_search_id += 1  # Invalidate any in-flight search bestmove
            self.engine.stop_calculation()
            self.analyzing = False

        self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready) [停止]", foreground="green")
        self.btn_analyze.config(state="normal")
        self.btn_stop.config(state="disabled")

    def toggle_side_to_move(self):
        current = self.side_var.get()
        new_side = 'b' if current == 'w' else 'w'
        self.side_var.set(new_side)
        self.board.turn = (chess.WHITE if new_side == 'w' else chess.BLACK)

    def execute_auto_play(self, move_uci, player_color=None, is_self_play=None):
        """Simulate mouse clicks to play `move_uci` on screen using Windows API."""
        rec = self._ensure_recognizer()
        if player_color is None:
            player_color = self._ui_get("my_color", "my_color_var", "w")
        if is_self_play is None:
            is_self_play = (
                self._ui_get("auto_play", "auto_play_var", False)
                and self._ui_get("autoplay_mode", "autoplay_mode_var", "fixed") == "self"
            )

        coords = rec.get_move_screen_coords(move_uci, player_color=player_color)
        if not coords:
            print("[WARN] 無法計算自動下棋的螢幕座標")
            return

        start_pt, end_pt = coords
        print(f"[INFO] Auto-playing {move_uci}: clicking {start_pt} then {end_pt}")

        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004

        def click(x, y):
            ctypes.windll.user32.SetCursorPos(x, y)
            time.sleep(0.015)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.015)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(0.02)

        click(start_pt[0], start_pt[1])
        time.sleep(0.04)
        click(end_pt[0], end_pt[1])

        # Handle Pawn Promotion (supports both White-at-bottom and Black-at-bottom orientations!)
        if len(move_uci) == 5 and move_uci[-1] in ['q', 'r', 'b', 'n']:
            promo_piece = move_uci[-1]
            print(f"[INFO] 偵測到升變: {promo_piece}，準備點擊選擇選單...")
            time.sleep(0.15)

            # Compute screen-space square height accurately via inv_M
            pt0 = rec.get_move_screen_coords("a1a2", player_color=player_color)
            sq_screen_h = abs(pt0[1][1] - pt0[0][1]) if pt0 else (rec.board_side_length // 8)

            target_rank = int(move_uci[3]) - 1  # 7 for rank 8 (White), 0 for rank 1 (Black)
            promo_at_visual_top = (target_rank == 7) if (player_color == 'w') else (target_rank == 0)
            offset_multiplier = {'q': 0, 'n': 1, 'r': 2, 'b': 3}

            if promo_piece == 'q':
                click(end_pt[0], end_pt[1])
            else:
                direction = 1 if promo_at_visual_top else -1
                y_offset = direction * offset_multiplier[promo_piece] * sq_screen_h
                click(end_pt[0], end_pt[1] + int(y_offset))

        # Wait briefly for move animation to settle before next turn / opponent detection
        time.sleep(0.22 if is_self_play else 0.10)
        self._post_ui(self.on_our_move_completed)

    def get_exclude_rect(self):
        try:
            from ctypes import wintypes
            d = self.__dict__
            if threading.current_thread() is threading.main_thread() and d.get("tk") is not None:
                d["_cached_hwnd"] = self.winfo_id()
            hwnd = d.get("_cached_hwnd")
            if hwnd:
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w > 0 and h > 0:
                        return (rect.left, rect.top, w, h)

            if threading.current_thread() is threading.main_thread() and d.get("tk") is not None:
                x, y = self.winfo_x(), self.winfo_y()
                w, h = self.winfo_width(), self.winfo_height()
                if w > 0 and h > 0:
                    return (x, y, w, h)
            return None
        except Exception:
            return None

    def _check_and_handle_game_over(self, board_obj: chess.Board) -> bool:
        if board_obj.is_game_over(claim_draw=True):
            outcome = board_obj.outcome(claim_draw=True)
            res_str = outcome.result() if outcome else "*"
            reason = "將殺 (Checkmate)" if board_obj.is_checkmate() else "平局/和棋 (Draw)"
            self.lbl_status.config(text=f"狀態 (Status): 遊戲結束 ({reason}) - {res_str}", foreground="purple")
            self.auto_play_var.set(False)
            self.auto_detect_opponent_var.set(False)
            self.auto_detecting_opponent = False
            self.analyzing = False
            self.btn_analyze.config(state="normal")
            self.btn_stop.config(state="disabled")
            return True
        return False

    def on_our_move_completed(self):
        """Called on main thread right after `execute_auto_play` finishes clicking."""
        if self._check_and_handle_game_over(self.board):
            return

        # 1. Self-Play Mode: does NOT depend on auto_detect_opponent_var!
        if self.auto_play_var.get() and self.autoplay_mode_var.get() == "self":
            print("[INFO] 自己對戰模式：自動觸發下一回合分析...")
            self.start_analysis_thread(current_fen=self.board.fen())
            return

        # 2. Fixed Mode with Auto-Detect Opponent
        if self.auto_detect_opponent_var.get():
            print("[INFO] 已完成我方著法，啟動對手極速偵測...")
            threading.Thread(target=self._auto_detect_loop, daemon=True).start()
        else:
            self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")

    def _capture_warped_board_safe(self, rec):
        screenshot = rec.capture_screen()
        board_img = None
        if rec.M is not None:
            board_img = rec.get_warped_board(screenshot, validate=True)
        if board_img is None:
            board_img = rec.find_board(screenshot, exclude_rect=self.get_exclude_rect(), verbose=False)
        return board_img

    def _auto_detect_manual_play_loop(self):
        """
        Background watcher when Auto-Play is OFF and Auto-Detect Opponent is ON:
          Phase 1: Wait for the user to make their move on screen (recommendation arrow stays visible).
          Phase 2: Wait for the opponent to reply on screen, then automatically trigger engine analysis.
        Handles instant bot/premove replies (2-ply jump) seamlessly!
        """
        if self.auto_detecting_opponent:
            return
        self.auto_detecting_opponent = True
        print("[INFO] Started Auto-Detect Manual Play loop (Phase 1: user move -> Phase 2: opponent move)...")

        def set_status(text, color):
            self.lbl_status.config(text=text, foreground=color)

        self._post_ui(lambda: self.btn_stop.config(state="normal"))
        self._post_ui(set_status, "狀態 (Status): 推薦步已標示，等待我方下棋... (Waiting for your move)", "blue")

        try:
            import cv2
            import numpy as np
            rec = self._ensure_recognizer()

            baseline_board = self._capture_warped_board_safe(rec)
            if baseline_board is None:
                print("[WARN] Manual-detect failed to locate chessboard.")
                self._post_ui(set_status, "狀態 (Status): 自動偵測失敗 - 找不到棋盤 (Board not found)", "red")
                return

            # Check if the user ALREADY moved on screen while the engine was finishing its calculation!
            init_fen = rec.get_fen_from_board_image(
                baseline_board,
                player_color=self._ui_get("my_color", "my_color_var", "w"),
                active_player=self._ui_get("side", "side_var", "w"),
                castling='-',
                en_passant='-',
            )
            init_screen_board = chess.Board(init_fen)
            init_seq = self.find_matching_move_sequence(init_screen_board, max_plies=2)

            phase = 1
            if init_seq is not None and len(init_seq) == 2:
                # Both user and instant-bot already moved!
                def apply_two_and_analyze(m_pair):
                    self.apply_move(m_pair[0])
                    self.apply_move(m_pair[1])
                    self.best_move_arrow = None
                    self.draw_board()
                    if not self._check_and_handle_game_over(self.board):
                        self.start_analysis_thread(current_fen=self.board.fen())
                self.auto_detecting_opponent = False
                self._post_ui(apply_two_and_analyze, init_seq)
                return
            elif init_seq is not None and len(init_seq) == 1:
                # User already made their move; jump straight to Phase 2 (waiting for opponent)!
                done_evt = threading.Event()
                def apply_user_early(mv):
                    self.apply_move(mv)
                    self.best_move_arrow = None
                    self.draw_board()
                    self.lbl_status.config(text="狀態 (Status): 等待對手下棋... (Waiting for opponent)", foreground="blue")
                    self._refresh_ui_snapshot()
                    done_evt.set()
                self._post_ui(apply_user_early, init_seq[0])
                done_evt.wait(timeout=1.0)
                phase = 2

            baseline_gray = cv2.cvtColor(baseline_board, cv2.COLOR_BGR2GRAY)
            previous_gray = baseline_gray
            stable_count = 0

            while self.auto_detecting_opponent:
                time.sleep(0.04)
                if not self.auto_detecting_opponent:
                    break

                current_board = self._capture_warped_board_safe(rec)
                if current_board is None:
                    continue

                current_gray = cv2.cvtColor(current_board, cv2.COLOR_BGR2GRAY)
                diff_baseline = cv2.absdiff(current_gray, baseline_gray)
                changed_pixels = np.count_nonzero(diff_baseline > 26)

                if changed_pixels > 900:
                    diff_prev = cv2.absdiff(current_gray, previous_gray)
                    settling_pixels = np.count_nonzero(diff_prev > 24)
                    if settling_pixels < 350:
                        stable_count += 1
                    else:
                        stable_count = 0

                    if stable_count >= 2:
                        new_fen_raw = rec.get_fen_from_board_image(
                            current_board,
                            player_color=self._ui_get("my_color", "my_color_var", "w"),
                            active_player=self._ui_get("side", "side_var", "w"),
                            castling='-',
                            en_passant='-',
                        )
                        board_screen = chess.Board(new_fen_raw)
                        seq = self.find_matching_move_sequence(board_screen, max_plies=2)

                        if phase == 1:
                            if seq is not None and len(seq) == 2:
                                print(f"[INFO] 偵測到我方與對手連續落子: {seq[0].uci()} -> {seq[1].uci()}")
                                self.auto_detecting_opponent = False
                                def on_both_moves(m_pair):
                                    self.apply_move(m_pair[0])
                                    self.apply_move(m_pair[1])
                                    self.best_move_arrow = None
                                    self.draw_board()
                                    if not self._check_and_handle_game_over(self.board):
                                        self.start_analysis_thread(current_fen=self.board.fen())
                                self._post_ui(on_both_moves, seq)
                                return

                            elif seq is not None and len(seq) == 1:
                                print(f"[INFO] 玩家走子識別成功: {seq[0].uci()}，切換為等待對手...")
                                sync_evt = threading.Event()
                                is_over_box = [False]
                                def on_player_moved(mv):
                                    self.apply_move(mv)
                                    self.best_move_arrow = None
                                    self.draw_board()
                                    if self._check_and_handle_game_over(self.board):
                                        is_over_box[0] = True
                                    else:
                                        self.lbl_status.config(
                                            text="狀態 (Status): 等待對手下棋... (Waiting for opponent)",
                                            foreground="blue",
                                        )
                                    self._refresh_ui_snapshot()
                                    sync_evt.set()
                                self._post_ui(on_player_moved, seq[0])
                                sync_evt.wait(timeout=1.0)
                                if is_over_box[0]:
                                    return

                                baseline_gray = current_gray
                                previous_gray = current_gray
                                stable_count = 0
                                phase = 2
                                continue

                            elif board_screen.board_fen() != self.board.board_fen():
                                # Non-standard jump -> resync as opponent's turn and move to Phase 2
                                print("[WARN] 玩家盤面跳躍，重新同步並等待對手...")
                                sync_evt = threading.Event()
                                def on_player_resync(fen_val):
                                    b_sync = chess.Board(fen_val)
                                    opp_color = 'b' if self.my_color_var.get() == 'w' else 'w'
                                    rights = self.infer_castling_rights(b_sync, self.castling_var.get())
                                    parts = fen_val.split()
                                    parts[1] = opp_color
                                    parts[2] = rights
                                    parts[3] = "-"
                                    self.init_game_from_board(chess.Board(" ".join(parts)))
                                    self.draw_board()
                                    self.lbl_status.config(
                                        text="狀態 (Status): 等待對手下棋... (Waiting for opponent)",
                                        foreground="blue",
                                    )
                                    self._refresh_ui_snapshot()
                                    sync_evt.set()
                                self._post_ui(on_player_resync, new_fen_raw)
                                sync_evt.wait(timeout=1.0)
                                baseline_gray = current_gray
                                previous_gray = current_gray
                                stable_count = 0
                                phase = 2
                                continue
                            else:
                                # Cursor / highlight change without piece move
                                baseline_gray = current_gray
                                stable_count = 0

                        else:  # phase == 2 (Waiting for opponent)
                            if seq is not None and len(seq) >= 1:
                                print(f"[INFO] 對手走子識別成功: {[m.uci() for m in seq]}")
                                self.auto_detecting_opponent = False
                                def on_opp_moved(moves_to_apply):
                                    for mv in moves_to_apply:
                                        self.apply_move(mv)
                                    self.draw_board()
                                    if not self._check_and_handle_game_over(self.board):
                                        self.start_analysis_thread(current_fen=self.board.fen())
                                self._post_ui(on_opp_moved, seq)
                                return
                            elif board_screen.board_fen() != self.board.board_fen():
                                print("[INFO] 對手盤面變動非單步推進，同步新局面並立即分析")
                                self.auto_detecting_opponent = False
                                def on_opp_resync(fen_val):
                                    b_sync = chess.Board(fen_val)
                                    rights = self.infer_castling_rights(b_sync, self.castling_var.get())
                                    parts = fen_val.split()
                                    parts[1] = self.my_color_var.get()
                                    parts[2] = rights
                                    parts[3] = "-"
                                    new_fen = " ".join(parts)
                                    self.init_game_from_board(chess.Board(new_fen))
                                    self.draw_board()
                                    if not self._check_and_handle_game_over(self.board):
                                        self.start_analysis_thread(current_fen=self.board.fen())
                                self._post_ui(on_opp_resync, new_fen_raw)
                                return
                            else:
                                baseline_gray = current_gray
                                stable_count = 0
                else:
                    stable_count = 0

                previous_gray = current_gray

        except Exception as e:
            print(f"[WARN] Error in manual auto-detect loop: {e}")
            traceback.print_exc()
        finally:
            self.auto_detecting_opponent = False

    def _auto_detect_loop(self):
        """
        High-speed background thread to detect when the opponent has moved.
        Also immediately detects if the opponent ALREADY moved before baseline was captured
        (e.g., instant bot reply or human premove, or starting as Black after White moved).
        """
        if self.auto_detecting_opponent:
            return
        self.auto_detecting_opponent = True
        print("[INFO] Started Ultra-Fast Auto-Detect Opponent loop...")

        def update_status(text, color):
            self.lbl_status.config(text=text, foreground=color)

        self._post_ui(lambda: self.btn_stop.config(state="normal"))
        self._post_ui(update_status, "狀態 (Status): 等待對手下棋... (Waiting for opponent)", "blue")

        try:
            import cv2
            import numpy as np
            rec = self._ensure_recognizer()

            time.sleep(0.06)

            baseline_board = self._capture_warped_board_safe(rec)
            if baseline_board is None:
                print("[WARN] Auto-detect failed to get baseline board.")
                self._post_ui(update_status, "狀態 (Status): 自動偵測失敗 - 找不到棋盤 (Board not found)", "red")
                return

            player_color = self._ui_get("my_color", "my_color_var", "w")
            # Orientation check on starting board (e.g. user playing Black at move 1)
            if self._ui_get("auto_detect", "auto_detect_var", True) and len(self.board.move_stack) == 0:
                vis_pieces = rec.identify_board_pieces(baseline_board)
                orient = rec.detect_starting_orientation(vis_pieces)
                if orient in ('w', 'b') and orient != player_color:
                    player_color = orient
                    self._post_ui(lambda o=orient: (self.my_color_var.set(o), self._refresh_ui_snapshot(), self.draw_board()))
                    time.sleep(0.02)

            # Immediate check: did a fast bot / premove opponent ALREADY move on `baseline_board`?
            init_fen_raw = rec.get_fen_from_board_image(
                baseline_board,
                player_color=player_color,
                active_player=self._ui_get("side", "side_var", "w"),
                castling='-',
                en_passant='-',
            )
            init_board_screen = chess.Board(init_fen_raw)
            init_seq = self.find_matching_move_sequence(init_board_screen, max_plies=1)
            if init_seq and len(init_seq) == 1:
                mv = init_seq[0]
                print(f"[INFO] 對手已提前完成走子 (即時偵測命中): {mv.uci()}")
                self.auto_detecting_opponent = False
                def on_early_opp(move_obj):
                    self.apply_move(move_obj)
                    self.draw_board()
                    if not self._check_and_handle_game_over(self.board):
                        self.start_analysis_thread(current_fen=self.board.fen())
                self._post_ui(on_early_opp, mv)
                return

            baseline_gray = cv2.cvtColor(baseline_board, cv2.COLOR_BGR2GRAY)
            previous_gray = baseline_gray
            stable_count = 0

            while self.auto_detecting_opponent:
                time.sleep(0.035)
                if not self.auto_detecting_opponent:
                    break

                current_board = self._capture_warped_board_safe(rec)
                if current_board is None:
                    continue

                current_gray = cv2.cvtColor(current_board, cv2.COLOR_BGR2GRAY)
                diff_baseline = cv2.absdiff(current_gray, baseline_gray)
                changed_pixels = np.count_nonzero(diff_baseline > 26)

                if changed_pixels > 900:
                    diff_prev = cv2.absdiff(current_gray, previous_gray)
                    settling_pixels = np.count_nonzero(diff_prev > 24)
                    if settling_pixels < 350:
                        stable_count += 1
                    else:
                        stable_count = 0

                    if stable_count >= 2:
                        new_fen_raw = rec.get_fen_from_board_image(
                            current_board,
                            player_color=self._ui_get("my_color", "my_color_var", "w"),
                            active_player=self._ui_get("side", "side_var", "w"),
                            castling='-',
                            en_passant='-',
                        )
                        board_screen = chess.Board(new_fen_raw)
                        seq = self.find_matching_move_sequence(board_screen, max_plies=2)

                        if seq and len(seq) >= 1:
                            print(f"[INFO] 對手走子極速識別成功: {[m.uci() for m in seq]}")
                            self.auto_detecting_opponent = False
                            def on_opponent_matched(moves_list):
                                for m_obj in moves_list:
                                    self.apply_move(m_obj)
                                self.draw_board()
                                if not self._check_and_handle_game_over(self.board):
                                    self.start_analysis_thread(current_fen=self.board.fen())
                            self._post_ui(on_opponent_matched, seq)
                            return
                        elif self.board.board_fen() != board_screen.board_fen():
                            print("[INFO] 對手盤面變動但非單步推進，同步新局面並即刻分析")
                            self.auto_detecting_opponent = False
                            def on_resync_matched(fen_val):
                                b_sync = chess.Board(fen_val)
                                rights = self.infer_castling_rights(b_sync, self.castling_var.get())
                                parts = fen_val.split()
                                parts[1] = self.my_color_var.get()
                                parts[2] = rights
                                parts[3] = "-"
                                new_fen = " ".join(parts)
                                self.init_game_from_board(chess.Board(new_fen))
                                self.draw_board()
                                if not self._check_and_handle_game_over(self.board):
                                    self.start_analysis_thread(current_fen=self.board.fen())
                            self._post_ui(on_resync_matched, new_fen_raw)
                            return
                        else:
                            baseline_gray = current_gray
                            stable_count = 0
                else:
                    stable_count = 0

                previous_gray = current_gray

        except Exception as e:
            print(f"[WARN] Error in opponent auto-detect loop: {e}")
        finally:
            self.auto_detecting_opponent = False

    def infer_castling_rights(self, board, current_rights=None):
        """
        Infer likely castling rights based on piece positions and existing rights.
        Castling rights can ONLY be lost, NEVER regained.
        """
        if current_rights is None:
            current_rights = self.castling_var.get()

        current_rights = sanitize_fen_input(current_rights, 'castling')
        if current_rights == "-":
            return "-"

        rights = ""
        if "K" in current_rights:
            if (board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and
                    board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE)):
                rights += "K"
        if "Q" in current_rights:
            if (board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and
                    board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE)):
                rights += "Q"
        if "k" in current_rights:
            if (board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and
                    board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK)):
                rights += "k"
        if "q" in current_rights:
            if (board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and
                    board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK)):
                rights += "q"

        return rights if rights else "-"

    def infer_en_passant_square(self, prev_board, current_board, side_just_moved=None):
        if prev_board is None or current_board is None:
            return "-"
        try:
            if isinstance(current_board, str):
                current_board = chess.Board(current_board)
            if isinstance(prev_board, str):
                prev_board = chess.Board(prev_board)

            curr_pieces = current_board.board_fen()
            b_test = prev_board.copy()
            if side_just_moved is not None:
                b_test.turn = side_just_moved
            for move in b_test.legal_moves:
                b_test.push(move)
                if b_test.board_fen() == curr_pieces:
                    ep_sq = b_test.ep_square
                    b_test.pop()
                    return chess.square_name(ep_sq) if ep_sq is not None else "-"
                b_test.pop()
            return "-"
        except Exception:
            return "-"

    def _run_analysis(self, precomputed_fen=None, search_id=0):
        try:
            if precomputed_fen:
                # Board was already updated on the main thread
                position_cmd = self.build_uci_position_cmd(self.board)
                fen_final = self.board.fen()
            else:
                try:
                    rec = self._ensure_recognizer()
                except Exception as e:
                    self.message_queue.put({"type": "error", "message": f"Recognizer Init Failed: {e}"})
                    self.message_queue.put({"type": "analysis_aborted", "search_id": search_id})
                    return

                board_img = self._capture_warped_board_safe(rec)
                if board_img is None:
                    self.message_queue.put({"type": "error", "message": "無法從螢幕辨識到 8x8 棋盤 (Board Not Found)\n請確認棋盤完整顯示或使用「預覽與手動校正」。"})
                    self.message_queue.put({"type": "analysis_aborted", "search_id": search_id})
                    return

                # Auto-detect board orientation at game start so playing Black never gets inverted
                player_color = self._ui_get("my_color", "my_color_var", "w")
                if self._ui_get("auto_detect", "auto_detect_var", True) and (self.is_first_analysis or len(self.board.move_stack) == 0):
                    vis_pieces = rec.identify_board_pieces(board_img)
                    orient = rec.detect_starting_orientation(vis_pieces)
                    if orient in ('w', 'b') and orient != player_color:
                        player_color = orient
                        self._post_ui(lambda o=orient: (self.my_color_var.set(o), self._refresh_ui_snapshot(), self.draw_board()))

                self.is_first_analysis = False

                fen_raw = rec.get_fen_from_board_image(
                    board_img,
                    player_color=player_color,
                    active_player=self._ui_get("side", "side_var", "w"),
                    castling='-',
                    en_passant='-',
                )
                if not fen_raw:
                    self.message_queue.put({"type": "error", "message": "無法辨識棋盤 (Recognition Failed)"})
                    self.message_queue.put({"type": "analysis_aborted", "search_id": search_id})
                    return

                # Synchronize `self.board` safely on the main thread
                sync_done = threading.Event()
                sync_result = {"wait_for_opponent": False}

                def sync_board_main_thread():
                    try:
                        is_self_play = (self.auto_play_var.get() and self.autoplay_mode_var.get() == "self")
                        if self.auto_detect_var.get():
                            board_screen = chess.Board(fen_raw)

                            # Special guard: at fresh starting board, White MUST move first!
                            if (len(self.board.move_stack) == 0 and
                                    board_screen.board_fen() == chess.STARTING_BOARD_FEN):
                                self.init_game_from_board(chess.Board())
                                if (not is_self_play and player_color == 'b' and
                                        self.auto_detect_opponent_var.get()):
                                    sync_result["wait_for_opponent"] = True
                            else:
                                seq = self.find_matching_move_sequence(board_screen, max_plies=2)
                                if seq is not None:
                                    for mv in seq:
                                        self.apply_move(mv)
                                else:
                                    new_rights = self.infer_castling_rights(board_screen, self.castling_var.get())
                                    parts = fen_raw.split()
                                    parts[2] = new_rights
                                    parts[3] = "-"
                                    self.init_game_from_board(chess.Board(" ".join(parts)))

                                # After syncing screen moves, if in Fixed mode + Auto-Detect Opponent
                                # and it's actually opponent's turn (e.g. user already moved on screen!),
                                # switch directly into opponent detection instead of analyzing opponent's turn!
                                if (not is_self_play and self.auto_detect_opponent_var.get() and
                                        self.side_var.get() != player_color):
                                    sync_result["wait_for_opponent"] = True
                        else:
                            user_rights = sanitize_fen_input(self.castling_var.get(), 'castling')
                            user_ep = sanitize_fen_input(self.ep_var.get(), 'ep')
                            parts = fen_raw.split()
                            parts[2] = user_rights
                            parts[3] = user_ep
                            self.init_game_from_board(chess.Board(" ".join(parts)))

                        self._refresh_ui_snapshot()
                        self.draw_board()
                    except Exception as ex:
                        print(f"[WARN] sync_board_main_thread exception: {ex}")
                    finally:
                        sync_done.set()

                self._post_ui(sync_board_main_thread)
                sync_done.wait(timeout=2.0)

                if sync_result["wait_for_opponent"]:
                    def switch_to_opp_wait():
                        if search_id == self.current_search_id:
                            self.analyzing = False
                            self.btn_analyze.config(state="normal")
                        threading.Thread(target=self._auto_detect_loop, daemon=True).start()
                    self._post_ui(switch_to_opp_wait)
                    return

                position_cmd = self.build_uci_position_cmd(self.board)
                fen_final = self.board.fen()

            if search_id != self.current_search_id:
                return

            self.message_queue.put({"type": "fen_update", "fen": fen_final, "search_id": search_id})

            # Check for Game Over before sending to engine
            temp_board = chess.Board(fen_final)
            if temp_board.is_game_over(claim_draw=True):
                self._post_ui(self._check_and_handle_game_over, temp_board)
                return

            go_cmd = self.__dict__.get("_ui_state", {}).get("go_cmd") or "go movetime 10000"
            if threading.current_thread() is threading.main_thread():
                go_cmd = self._build_go_cmd()
            self.engine.start_search(
                position_cmd=position_cmd,
                go_cmd=go_cmd,
                search_id=search_id,
                own_book=bool(self._ui_get("use_book", "use_book_var", True)),
            )

        except Exception as e:
            traceback.print_exc()
            self.message_queue.put({"type": "error", "message": f"Analysis Error: {e}"})
            self.message_queue.put({"type": "analysis_aborted", "search_id": search_id})

    def _process_queue(self):
        self._refresh_ui_snapshot()
        try:
            while True:
                msg = self.message_queue.get_nowait()
                mtype = msg.get("type")

                if mtype == "ui_call":
                    fn = msg.get("fn")
                    args = msg.get("args", ())
                    if fn is not None:
                        fn(*args)

                elif mtype == "raw_log":
                    self.write_to_console(msg["message"])

                elif mtype == "readyok":
                    if self.warming_up:
                        self.warming_up = False
                        self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")
                        self.engine.send_command("ucinewgame")
                        self.engine.send_command(
                            f"setoption name OwnBook value {'true' if self.use_book_var.get() else 'false'}"
                        )
                        print("[INFO] Engine JIT readyok complete.")

                elif mtype == "info":
                    if self.warming_up:
                        continue
                    if msg.get("search_id") != self.current_search_id:
                        continue

                    if "string" in msg and "Playing from book" in msg["string"]:
                        self.lbl_score.config(text="評分 (Score): 開局庫 (Book)")
                        self.lbl_pv.config(text="變例 (PV): [Book Move]")

                    if "score_type" in msg:
                        s_type = msg["score_type"]
                        s_val = msg["score_val"]
                        display_score = f"{s_val} cp" if s_type == "cp" else f"Mate in {s_val}"
                        self.lbl_score.config(text=f"評分 (Score): {display_score}")

                    if "pv" in msg:
                        self.lbl_pv.config(text=f"變例 (PV): {msg['pv']}")

                    if "depth" in msg and "score_type" in msg:
                        depth = msg["depth"]
                        seldepth = msg.get("seldepth", "--")
                        s_type = msg["score_type"]
                        s_val = msg["score_val"]
                        display_score = f"{s_val} cp" if s_type == "cp" else f"Mate {s_val}"
                        time_ms = msg.get("time", "--")
                        nodes = msg.get("nodes", "--")
                        nps = msg.get("nps", "--")
                        pv = msg.get("pv", "")
                        first_move = pv.split()[0] if pv else "--"

                        log_line = (
                            f"Depth: {depth:<3} | SelDepth: {seldepth:<3} | Score: {display_score:<10} | "
                            f"Time: {time_ms:<6} ms | Nodes: {nodes:<10} | NPS: {nps:<10} | Best Move: {first_move}"
                        )
                        self.log_search_info(log_line)

                elif mtype == "bestmove":
                    # Ignore stale bestmoves from earlier interrupted searches
                    if msg.get("search_id") != self.current_search_id or not self.analyzing:
                        continue

                    # Mark search completed IMMEDIATELY (prevents any queue race when chaining next turn!)
                    self.analyzing = False
                    self.btn_analyze.config(state="normal")
                    self.btn_stop.config(state="disabled")

                    move_uci = msg["move"]
                    castling_map = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
                    if (move_uci in castling_map and
                            self.board.piece_at(chess.parse_square(move_uci[:2])) == chess.Piece(chess.KING, self.board.turn)):
                        display_move = castling_map[move_uci]
                    else:
                        display_move = move_uci

                    self.lbl_bestmove.config(text=f"最佳著法 (Best Move): {display_move}")
                    self.log_search_info(f"[Best Move Chosen]: {display_move}")
                    self.log_search_info("=" * 110)

                    try:
                        current_tab = self.notebook.index("current")
                    except Exception:
                        current_tab = 0

                    was_our_turn = (self.side_var.get() == self.my_color_var.get())
                    is_auto_play = self.auto_play_var.get()
                    is_self_play = (is_auto_play and self.autoplay_mode_var.get() == "self")

                    try:
                        move = chess.Move.from_uci(display_move)
                        self.best_move_arrow = move
                        # ONLY push `move` onto `self.board` immediately if Auto-Play is going to click it right now!
                        # When Auto-Play is OFF, keep `self.board` at the current screen state so the arrow points
                        # cleanly on the un-moved board and `_auto_detect_manual_play_loop` can detect whatever move
                        # the human actually plays!
                        if current_tab == 0 and (is_self_play or (is_auto_play and was_our_turn)):
                            if move in self.board.legal_moves:
                                self.apply_move(move)
                        self.draw_board()
                    except Exception:
                        pass

                    if current_tab == 1:
                        self.lbl_status.config(text="狀態 (Status): 手動分析完成 (Ready)", foreground="green")
                        continue

                    # --- Tab 0 Automation State Machine ---
                    p_color = self.my_color_var.get()
                    self._refresh_ui_snapshot()
                    if is_self_play:
                        # Self-Play plays BOTH White and Black turns continuously!
                        self.lbl_status.config(text=f"狀態 (Status): 自我對弈落子 {display_move}...", foreground="blue")
                        threading.Thread(
                            target=self.execute_auto_play,
                            args=(display_move, p_color, True),
                            daemon=True,
                        ).start()
                    elif was_our_turn:
                        if is_auto_play:
                            self.lbl_status.config(text=f"狀態 (Status): 自動落子 {display_move}...", foreground="blue")
                            threading.Thread(
                                target=self.execute_auto_play,
                                args=(display_move, p_color, False),
                                daemon=True,
                            ).start()
                        elif self.auto_detect_opponent_var.get():
                            threading.Thread(target=self._auto_detect_manual_play_loop, daemon=True).start()
                        else:
                            self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")
                    else:
                        # Analyzed opponent's turn (e.g. Auto-Detect was off)
                        if self.auto_detect_opponent_var.get():
                            threading.Thread(target=self._auto_detect_loop, daemon=True).start()
                        else:
                            self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")

                elif mtype == "fen_update":
                    if msg.get("search_id") != self.current_search_id:
                        continue
                    fen = msg["fen"]
                    self.draw_board()
                    self.log_search_info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === 分析盤面 ===")
                    self.log_search_info(f"FEN: {fen}")
                    self.log_search_info("-" * 110)

                elif mtype == "analysis_aborted":
                    if msg.get("search_id") == self.current_search_id:
                        self.analyzing = False
                        self.btn_analyze.config(state="normal")
                        self.btn_stop.config(state="disabled")

                elif mtype == "error":
                    messagebox.showerror("Error", msg["message"])
                    self.lbl_status.config(text="狀態 (Status): 發生錯誤 (Error)", foreground="red")

                elif mtype == "log":
                    print(f"[LOG] {msg['message']}")

        except queue.Empty:
            pass

        self.after(20, self._process_queue)

    def draw_board(self):
        self.canvas.delete("all")

        is_flipped = (self.my_color_var.get() == 'b')

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w = self.canvas_size
            canvas_h = self.canvas_size

        offset_x = max(0, (canvas_w - (self.square_size * 8)) // 2)
        offset_y = max(0, (canvas_h - (self.square_size * 8)) // 2)

        last_move = self.board.peek() if self.board.move_stack else None

        for rank in range(8):
            for file in range(8):
                display_rank = rank if is_flipped else (7 - rank)
                display_file = (7 - file) if is_flipped else file

                x1 = offset_x + display_file * self.square_size
                y1 = offset_y + display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                square_idx = chess.square(file, rank)
                color = BOARD_COLORS[(rank + file) % 2]

                if last_move and square_idx in (last_move.from_square, last_move.to_square):
                    color = LAST_MOVE_COLOR

                if hasattr(self, 'selected_square') and self.selected_square == square_idx:
                    color = "#E0E68C"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                label_color = "#B58863" if (rank + file) % 2 == 1 else "#F0D9B5"
                if display_file == 0:
                    self.canvas.create_text(x1 + 2, y1 + 2, text=str(rank + 1), anchor="nw", font=("Arial", 8), fill=label_color)
                if display_rank == 7:
                    self.canvas.create_text(x2 - 2, y2 - 2, text=chr(ord('a') + file), anchor="se", font=("Arial", 8), fill=label_color)

                piece = self.board.piece_at(square_idx)
                if piece:
                    if hasattr(self, 'dragging') and self.dragging and self.selected_square == square_idx:
                        continue

                    drawn_image = False
                    if hasattr(self, 'piece_images') and self.piece_images:
                        piece_key = f"{'w' if piece.color else 'b'}{piece.symbol().upper()}"
                        if piece_key in self.piece_images:
                            self.canvas.create_image(
                                x1 + self.square_size // 2,
                                y1 + self.square_size // 2,
                                image=self.piece_images[piece_key],
                                anchor="center"
                            )
                            drawn_image = True

                    if not drawn_image:
                        symbol = UNICODE_PIECES.get(piece.symbol())
                        self.canvas.create_text(
                            x1 + self.square_size // 2,
                            y1 + self.square_size // 2,
                            text=symbol,
                            font=("Segoe UI Symbol", 32)
                        )

        # Draw dragging piece at cursor
        if hasattr(self, 'dragging') and self.dragging and self.selected_square is not None:
            piece = self.board.piece_at(self.selected_square)
            if piece:
                drawn_image = False
                if hasattr(self, 'piece_images') and self.piece_images:
                    piece_key = f"{'w' if piece.color else 'b'}{piece.symbol().upper()}"
                    if piece_key in self.piece_images:
                        self.canvas.create_image(self.drag_x, self.drag_y, image=self.piece_images[piece_key], anchor="center")
                        drawn_image = True
                if not drawn_image:
                    symbol = UNICODE_PIECES.get(piece.symbol())
                    self.canvas.create_text(self.drag_x, self.drag_y, text=symbol, font=("Segoe UI Symbol", 32))

        # Draw recommended move blue arrow
        if hasattr(self, 'best_move_arrow') and self.best_move_arrow:
            move = self.best_move_arrow

            def get_square_center(square):
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                display_file = 7 - file_idx if is_flipped else file_idx
                display_rank = rank_idx if is_flipped else 7 - rank_idx
                x = offset_x + display_file * self.square_size + self.square_size // 2
                y = offset_y + display_rank * self.square_size + self.square_size // 2
                return x, y

            try:
                x1, y1 = get_square_center(move.from_square)
                x2, y2 = get_square_center(move.to_square)
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="#1E90FF",
                    width=5,
                    arrow=tk.LAST,
                    arrowshape=(16, 20, 6)
                )
            except Exception as e:
                print(f"Error drawing arrow: {e}")

    def on_tab_changed(self, event):
        current_tab = self.notebook.index("current")
        if self.analyzing or self.auto_detecting_opponent:
            self.stop_analysis()

        if current_tab == 1:  # Manual Tab
            self.auto_play_var.set(False)
            self.auto_detect_opponent_var.set(False)
            self.auto_detecting_opponent = False

            self.canvas.bind("<Button-1>", self.on_board_click)
            self.canvas.bind("<B1-Motion>", self.on_board_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_board_release)

            self.bind("<Left>", self._on_manual_left_key)
            self.bind("<Right>", self._on_manual_right_key)
            self.bind("<space>", self._on_manual_space_key)

            self.manual_fen_var.set(self.board.fen())
            self.redo_stack = []
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
        else:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

            self.unbind("<Left>")
            self.unbind("<Right>")
            self.unbind("<space>")

            try:
                self._sync_ui_vars_from_board()
            except Exception:
                pass

            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()

    def _on_manual_left_key(self, event=None):
        if not self._is_text_input_focused():
            self.manual_undo()

    def _on_manual_right_key(self, event=None):
        if not self._is_text_input_focused():
            self.manual_redo()

    def _on_manual_space_key(self, event=None):
        if not self._is_text_input_focused():
            self.toggle_manual_eval()

    def get_square_from_coords(self, x, y):
        is_flipped = (self.my_color_var.get() == 'b')
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        offset_x = max(0, (canvas_w - (self.square_size * 8)) // 2)
        offset_y = max(0, (canvas_h - (self.square_size * 8)) // 2)

        file_x = (x - offset_x) // self.square_size
        rank_y = (y - offset_y) // self.square_size

        if file_x < 0 or file_x >= 8 or rank_y < 0 or rank_y >= 8:
            return None

        display_file = 7 - file_x if is_flipped else file_x
        display_rank = rank_y if is_flipped else 7 - rank_y

        return chess.square(display_file, display_rank)

    def on_board_click(self, event):
        self.canvas.focus_set()
        sq = self.get_square_from_coords(event.x, event.y)
        if sq is not None:
            piece = self.board.piece_at(sq)
            if self.selected_square is not None:
                if sq != self.selected_square:
                    promotion = None
                    p_piece = self.board.piece_at(self.selected_square)
                    if p_piece and p_piece.piece_type == chess.PAWN:
                        target_rank = chess.square_rank(sq)
                        if (p_piece.color == chess.WHITE and target_rank == 7) or (p_piece.color == chess.BLACK and target_rank == 0):
                            promotion = chess.QUEEN

                    move = chess.Move(self.selected_square, sq, promotion=promotion)
                    if move in self.board.legal_moves:
                        self.apply_move(move)
                        self.redo_stack = []
                        self.manual_fen_var.set(self.board.fen())
                        self.selected_square = None
                        self.dragging = False
                        self.best_move_arrow = None
                        self.draw_board()
                        if self.live_eval_var.get():
                            self.start_manual_analysis()
                        return
                    else:
                        if piece and piece.color == self.board.turn:
                            self.selected_square = sq
                            self.dragging = True
                            self.drag_x = event.x
                            self.drag_y = event.y
                            self.draw_board()
                            return
                        else:
                            self.selected_square = None
                            self.dragging = False
                            self.draw_board()
                            return
                else:
                    self.dragging = True
                    self.drag_x = event.x
                    self.drag_y = event.y
                    self.draw_board()
                    return

            if piece and piece.color == self.board.turn:
                self.selected_square = sq
                self.dragging = True
                self.drag_x = event.x
                self.drag_y = event.y
                self.draw_board()

    def on_board_drag(self, event):
        if self.dragging and self.selected_square is not None:
            self.drag_x = event.x
            self.drag_y = event.y
            self.draw_board()

    def on_board_release(self, event):
        if self.dragging:
            self.dragging = False
            target_sq = self.get_square_from_coords(event.x, event.y)
            if target_sq is not None and target_sq != self.selected_square:
                promotion = None
                piece = self.board.piece_at(self.selected_square)
                if piece and piece.piece_type == chess.PAWN:
                    target_rank = chess.square_rank(target_sq)
                    if (piece.color == chess.WHITE and target_rank == 7) or (piece.color == chess.BLACK and target_rank == 0):
                        promotion = chess.QUEEN

                move = chess.Move(self.selected_square, target_sq, promotion=promotion)
                if move in self.board.legal_moves:
                    self.apply_move(move)
                    self.redo_stack = []
                    self.manual_fen_var.set(self.board.fen())
                    self.selected_square = None
                    self.best_move_arrow = None
                    self.draw_board()
                    if self.live_eval_var.get():
                        self.start_manual_analysis()
                    return
            self.draw_board()

    def load_manual_fen(self):
        fen = self.manual_fen_var.get().strip()
        try:
            board = chess.Board(fen)
            self.init_game_from_board(board)
            self.redo_stack = []
            self.selected_square = None
            self.best_move_arrow = None
            self.canvas.focus_set()
            self.draw_board()
            if self.live_eval_var.get():
                self.start_manual_analysis()
        except Exception as e:
            messagebox.showerror("Error", f"無效的 FEN (Invalid FEN): {e}")

    def clear_manual_board(self):
        if self.analyzing:
            self.stop_analysis()
        self.board.clear()
        self.game_pgn = chess.pgn.Game.from_board(self.board)
        self.game_node = self.game_pgn
        self.manual_fen_var.set(self.board.fen())
        self.redo_stack = []
        self.selected_square = None
        self.best_move_arrow = None
        self.draw_board()

    def reset_manual_board(self):
        self.init_game_from_board(chess.Board())
        self.manual_fen_var.set(self.board.fen())
        self.redo_stack = []
        self.selected_square = None
        self.best_move_arrow = None
        self.draw_board()
        if self.live_eval_var.get():
            self.start_manual_analysis()

    def flip_board(self):
        current = self.my_color_var.get()
        self.my_color_var.set('b' if current == 'w' else 'w')
        self.draw_board()

    def start_manual_analysis(self):
        if self.warming_up:
            return
        if self.analyzing:
            self.engine.stop_calculation()

        if self._check_and_handle_game_over(self.board):
            return

        self.analyzing = True
        self.current_search_id += 1
        search_id = self.current_search_id

        self.btn_analyze.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="狀態 (Status): 手動分析計算中... (Analyzing...)", foreground="orange")
        self.lbl_score.config(text="評分 (Score): 計算中...")
        self.lbl_bestmove.config(text="最佳著法 (Best Move): ...")
        self.lbl_pv.config(text="變例 (PV): ...")
        self.best_move_arrow = None

        self.log_search_info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === 手動分析盤面 ===")
        self.log_search_info(f"FEN: {self.board.fen()}")
        self.log_search_info("-" * 110)

        position_cmd = self.build_uci_position_cmd(self.board)
        go_cmd = self._build_go_cmd()
        self.engine.start_search(
            position_cmd=position_cmd,
            go_cmd=go_cmd,
            search_id=search_id,
            own_book=self.use_book_var.get(),
        )

    def manual_undo(self):
        if self.notebook.index("current") != 1:
            return
        if self.board.move_stack:
            move = self.board.pop()
            self.redo_stack.append(move)
            if self.game_node is not None and self.game_node.parent is not None:
                self.game_node = self.game_node.parent
            self.manual_fen_var.set(self.board.fen())
            self._sync_ui_vars_from_board()
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
            if self.live_eval_var.get():
                self.start_manual_analysis()

    def manual_redo(self):
        if self.notebook.index("current") != 1:
            return
        if self.redo_stack:
            move = self.redo_stack.pop()
            self.apply_move(move)
            self.manual_fen_var.set(self.board.fen())
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
            if self.live_eval_var.get():
                self.start_manual_analysis()

    def toggle_manual_eval(self):
        if self.notebook.index("current") != 1:
            return
        if self.analyzing:
            self.stop_analysis()
        else:
            self.start_manual_analysis()

    def write_to_console(self, text):
        if hasattr(self, 'console_text'):
            self.console_text.configure(state="normal")
            self.console_text.insert(tk.END, text + "\n")
            self.console_text.see(tk.END)
            self.console_text.configure(state="disabled")

    def log_search_info(self, text):
        try:
            with open(self.log_filename, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def open_log_folder(self):
        try:
            os.startfile(self.logs_dir)
        except Exception as e:
            messagebox.showerror("Error", f"無法開啟日誌資料夾 (Cannot open log folder): {e}")

    def on_piece_style_selected(self, event=None):
        style = self.piece_style_var.get()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            rec = self._ensure_recognizer()
            if style == "Custom":
                custom_dir = os.path.abspath(os.path.join(script_dir, "data", "templates", "custom"))
                if not os.path.exists(custom_dir) or not os.listdir(custom_dir):
                    messagebox.showwarning("Warning", "尚未學習自訂風格！請先點擊「學習」按鈕學習風格。")
                    self.piece_style_var.set("Universal (AI)" if rec.dnn_net is not None else "Default")
                    return
                rec.set_recognition_mode("custom")
            elif style == "Default":
                rec.set_recognition_mode("default")
            else:
                rec.set_recognition_mode("universal")
        except Exception as e:
            messagebox.showerror("Error", f"切換辨識模式失敗: {e}")
            self.piece_style_var.set("Default")

        self.load_piece_images()
        self.draw_board()

    def learn_piece_style(self):
        """
        一鍵從當前螢幕起始盤面學習 12 種自訂棋子圖示風格。
        """
        if self.warming_up:
            messagebox.showwarning("Warning", "引擎正在熱機中，請稍候... (Engine Warming Up)")
            return

        color_desc = "白方在下方、黑方在上方" if self.my_color_var.get() == 'w' else "黑方在下方、白方在上方"
        confirm = messagebox.askyesno(
            "學習自訂棋子風格",
            f"請確認：\n1. 網頁棋盤已重置為「新遊戲 (Starting Position)」位置。\n2. 視角為：{color_desc}。\n\n確認已就緒並開始學習？"
        )
        if not confirm:
            return

        try:
            rec = self._ensure_recognizer()
            screenshot = rec.capture_screen()
            board_img = rec.get_warped_board(screenshot) if rec.manual_calibrated else None
            if board_img is None:
                board_img = rec.find_board(screenshot, exclude_rect=self.get_exclude_rect())

            if board_img is None:
                messagebox.showerror("Error", "找不到棋盤！請確認網頁棋盤完整顯示於螢幕上，或先使用「預覽與手動校正」對齊。")
                return

            rec.learn_pieces_from_board(board_img, player_color=self.my_color_var.get())
            self.piece_style_var.set("Custom")
            self.load_piece_images()
            self.draw_board()

            messagebox.showinfo("Success", "自訂棋子風格學習成功！\n模板已自動儲存至 data/templates/custom/ 並啟用。")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"學習棋子風格失敗: {e}")

    def open_calibration_window(self):
        """
        開啟手動校正與螢幕預覽視窗。
        """
        if self.warming_up:
            messagebox.showwarning("Warning", "引擎正在熱機中，請稍候... (Engine Warming Up)")
            return

        try:
            rec = self._ensure_recognizer()
            screenshot = rec.capture_screen()
            # If not yet detected, run auto-detection once so the 4 handles snap onto the board automatically!
            if rec.inv_M is None:
                rec.find_board(screenshot, exclude_rect=self.get_exclude_rect(), verbose=False)
        except Exception as e:
            messagebox.showerror("Error", f"無法擷取螢幕 (Capture Failed): {e}")
            return

        calib_win = tk.Toplevel(self)
        calib_win.title("螢幕辨識預覽與手動校正 (Screen Preview & Board Calibration)")
        calib_win.geometry("980x640")
        calib_win.attributes('-topmost', True)

        lbl_desc = ttk.Label(
            calib_win,
            text="說明：紅色外框為目前偵測到的棋盤範圍。可拖曳四個紅點微調對齊網頁棋盤四角，完成後點選「套用校正」。",
            foreground="#1E90FF", font=("Segoe UI", 10, "bold")
        )
        lbl_desc.pack(fill="x", padx=10, pady=5)

        import cv2
        import numpy as np
        screen_h, screen_w, _ = screenshot.shape
        scale = min(960 / screen_w, 500 / screen_h)
        preview_w = int(screen_w * scale)
        preview_h = int(screen_h * scale)

        preview_img = cv2.resize(screenshot, (preview_w, preview_h))
        preview_pil = Image.fromarray(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))
        preview_tk = ImageTk.PhotoImage(preview_pil)

        canvas = tk.Canvas(calib_win, width=preview_w, height=preview_h, bg="#1E1E1E")
        canvas.pack(padx=10, pady=5)
        canvas.image = preview_tk
        canvas.create_image(0, 0, image=preview_tk, anchor="nw")

        points = []
        if rec.inv_M is not None:
            w_val = rec.board_side_length
            corners_warped = np.array([[[0, 0]], [[w_val, 0]], [[w_val, w_val]], [[0, w_val]]], dtype="float32")
            corners_screen = cv2.perspectiveTransform(corners_warped, rec.inv_M)
            for i in range(4):
                px = max(0, min(preview_w, int(corners_screen[i][0][0] * scale)))
                py = max(0, min(preview_h, int(corners_screen[i][0][1] * scale)))
                points.append([px, py])
        else:
            cx, cy = preview_w // 2, preview_h // 2
            size = min(preview_w, preview_h) // 4
            points = [
                [cx - size, cy - size],
                [cx + size, cy - size],
                [cx + size, cy + size],
                [cx - size, cy + size],
            ]

        active_handle = None
        handle_radius = 8

        def draw_calibration():
            canvas.delete("overlay")
            coords = []
            for p in points:
                coords.extend(p)
            canvas.create_polygon(coords, fill="", outline="red", width=2, tags="overlay")
            for idx, p in enumerate(points):
                canvas.create_oval(
                    p[0] - handle_radius, p[1] - handle_radius,
                    p[0] + handle_radius, p[1] + handle_radius,
                    fill="red", outline="white", width=1, tags="overlay"
                )
                canvas.create_text(p[0], p[1], text=str(idx + 1), fill="white", font=("Segoe UI", 9, "bold"), tags="overlay")

        def get_nearest_handle(event):
            nonlocal active_handle
            for idx, p in enumerate(points):
                if np.hypot(p[0] - event.x, p[1] - event.y) < 15:
                    active_handle = idx
                    return
            active_handle = None

        def drag_handle(event):
            if active_handle is not None:
                x = max(0, min(preview_w, event.x))
                y = max(0, min(preview_h, event.y))
                points[active_handle] = [x, y]
                draw_calibration()

        def release_handle(event):
            nonlocal active_handle
            active_handle = None

        canvas.bind("<Button-1>", get_nearest_handle)
        canvas.bind("<B1-Motion>", drag_handle)
        canvas.bind("<ButtonRelease-1>", release_handle)

        draw_calibration()

        btn_frame = ttk.Frame(calib_win)
        btn_frame.pack(fill="x", padx=10, pady=10)

        def apply_calibration():
            orig_points = np.array([[p[0] / scale, p[1] / scale] for p in points], dtype="float32")
            sorted_rect = rec._sort_quad_points(orig_points)

            side_length = rec.board_side_length
            dst = np.array([[0, 0], [side_length - 1, 0], [side_length - 1, side_length - 1], [0, side_length - 1]], dtype="float32")
            rec.M = cv2.getPerspectiveTransform(sorted_rect, dst)
            rec.inv_M = cv2.getPerspectiveTransform(dst, sorted_rect)
            rec.manual_calibrated = True

            self.lbl_status.config(text="狀態 (Status): 手動校正棋盤完成 (Calibration complete)", foreground="green")
            self.draw_board()
            calib_win.destroy()
            messagebox.showinfo("Success", "棋盤校正成功！\n棋盤邊界已鎖定完成。")

        ttk.Button(btn_frame, text="套用校正 (Apply)", command=apply_calibration).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="取消 (Cancel)", command=calib_win.destroy).pack(side="right", padx=5)

    def on_closing(self):
        self.auto_detecting_opponent = False
        self.engine.quit_engine()
        self.destroy()


if __name__ == "__main__":
    app = ChessVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
