import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import subprocess
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))
import time
import chess
import traceback
import re
import ctypes

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

BOARD_COLORS = [ "#B58863", "#F0D9B5"]  # Light, Dark squares (Wood theme)
HIGHLIGHT_COLOR = "#FFFF00" # Yellow for best move
LAST_MOVE_COLOR = "#BBCB2B" # Greenish for last move

def sanitize_fen_input(text, field_type):
    """
    Sanitizes user input for FEN fields to prevent UCI command injection.
    """
    # Security: Remove any potential control characters and validate format
    stripped = text.strip()
    
    if field_type == 'castling':
        # Castling rights: only K, Q, k, q, or -
        if stripped == "-":
            return "-"
            
        if re.match(r"^[KQkq]+$", stripped):
            if len(stripped) <= 4:
                return stripped
        return "-"

    elif field_type == 'ep':
        # En Passant: [a-h][36] or -
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
        x = y = 0
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20
        
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background='#ffffe0', relief='solid', borderwidth=1,
                       font=("Segoe UI", "9", "normal"))
        label.pack(ipadx=5, ipady=2)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None

class EngineProcess:
    """
    Manages the subprocess for the chess engine (main.py).
    Handles UCI communication via stdin/stdout.
    """
    def __init__(self, callback_queue):
        self.callback_queue = callback_queue
        self.process = None
        self.running = False
        self.threads = []
        
        # Determine the path to main.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.engine_path = os.path.join(script_dir, "main.py")

    def start(self):
        if self.running:
            return

        try:
            # Use 'python' or sys.executable to run main.py
            # Use -u for unbuffered output to ensure we get prints immediately
            cmd = [sys.executable, "-u", self.engine_path]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,  # Text mode for strings
                bufsize=1   # Line buffered
            )
            self.running = True
            
            # Start threads to read output
            t_out = threading.Thread(target=self._read_stdout, daemon=True)
            t_err = threading.Thread(target=self._read_stderr, daemon=True)
            self.threads = [t_out, t_err]
            for t in self.threads:
                t.start()
            
            # Initialize UCI
            self.send_command("uci")
            self.send_command("isready")
            
        except Exception as e:
            self.callback_queue.put({"type": "error", "message": f"Failed to start engine: {e}"})

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
                
                # Print raw output to terminal for user visibility
                print(line, flush=True)

                # Forward raw line to the GUI console
                self.callback_queue.put({"type": "raw_log", "message": line})

                # Parse UCI output
                self._parse_uci_output(line)
            except Exception as e:
                # self.callback_queue.put({"type": "log", "message": f"Stdout Error: {e}"})
                break

    def _read_stderr(self):
        """Read stderr to prevent deadlock, can be logged if needed."""
        while self.running and self.process:
            try:
                line = self.process.stderr.readline()
                if not line:
                    break
                line = line.strip()
                # Forward stderr to main thread logs
                self.callback_queue.put({"type": "raw_log", "message": f"[STDERR] {line}"})
            except:
                break

    def _parse_uci_output(self, line):
        parts = line.split()
        if not parts:
            return

        cmd = parts[0]
        
        if cmd == "info":
            # Example: info depth 5 score cp 20 pv e2e4 e7e5
            info_data = {"type": "info"}
            
            # Parse Score
            if "score" in parts:
                idx = parts.index("score")
                if idx + 2 < len(parts):
                    score_type = parts[idx+1] # cp or mate
                    score_val = parts[idx+2]
                    info_data["score_type"] = score_type
                    info_data["score_val"] = score_val

            # Parse Depth
            if "depth" in parts:
                idx = parts.index("depth")
                if idx + 1 < len(parts):
                    info_data["depth"] = parts[idx+1]

            # Parse SelDepth
            if "seldepth" in parts:
                idx = parts.index("seldepth")
                if idx + 1 < len(parts):
                    info_data["seldepth"] = parts[idx+1]

            # Parse Nodes
            if "nodes" in parts:
                idx = parts.index("nodes")
                if idx + 1 < len(parts):
                    info_data["nodes"] = parts[idx+1]

            # Parse NPS
            if "nps" in parts:
                idx = parts.index("nps")
                if idx + 1 < len(parts):
                    info_data["nps"] = parts[idx+1]

            # Parse Time
            if "time" in parts:
                idx = parts.index("time")
                if idx + 1 < len(parts):
                    info_data["time"] = parts[idx+1]

            # Parse PV and map FRC castling to Standard Castling for display
            castling_map = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
            if "pv" in parts:
                idx = parts.index("pv")
                pv_moves = parts[idx+1:]
                
                # We do a simple string replacement for the first move if it's a King move
                # This could be more robust using chess.Board, but for display purposes this is usually sufficient
                mapped_pv = []
                for move in pv_moves:
                    # Very simple heuristic: if the move is in castling_map and starts with e1/e8
                    # we just replace it. A true robust way requires board state tracking through the PV.
                    if move in castling_map:
                        mapped_pv.append(castling_map[move])
                    else:
                        mapped_pv.append(move)
                        
                info_data["pv"] = " ".join(mapped_pv)

            self.callback_queue.put(info_data)

        elif cmd == "bestmove":
            # Example: bestmove e2e4 ponder ...
            move = parts[1]
            self.callback_queue.put({"type": "bestmove", "move": move})

class ChessVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("西洋棋視覺助理 (Chess Vision App)")
        self.geometry("500x780")
        
        # Defer topmost to avoid locking the window on startup
        self.after(500, lambda: self.attributes('-topmost', True))
        
        # State
        self.board = chess.Board()
        self.message_queue = queue.Queue()
        self.recognizer = None # Lazy load
        self.engine = EngineProcess(self.message_queue)
        self.engine.start()
        self.analyzing = False
        self.warming_up = False
        self.is_first_analysis = True
        self.auto_detecting_opponent = False
        self.last_detected_fen = ""
        self.autoplay_mode_var = tk.StringVar(value="fixed") # "fixed" or "self"
        
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
        self._create_ui()
        
        # Start message polling
        self.after(100, self._process_queue)
        
        # Warm-up (Silent)
        self.after(1000, self._run_warmup)

    def _create_ui(self):
        # Configure Notebook Tab style to make tabs larger
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Arial", 11, "bold"), padding=[12, 6])

        # 1. ttk.Notebook (Tabs Container)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="x", padx=10, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # 2. Main Control Buttons Frame (Put back to original place - below tabs, above analysis results)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=5)

        # Start Button
        self.btn_analyze = ttk.Button(btn_frame, text="▶ 開始分析 (Start)", command=self.start_analysis_thread)
        self.btn_analyze.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ToolTip(self.btn_analyze, "快速鍵 (Shortcut): Enter")

        # Stop Button
        self.btn_stop = ttk.Button(btn_frame, text="⏹ 停止 (Stop)", command=self.stop_analysis, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ToolTip(self.btn_stop, "快速鍵 (Shortcut): Esc")

        # New Game Button
        self.btn_new_game = ttk.Button(btn_frame, text="🔄 新遊戲 (New Game)", command=self.new_game)
        self.btn_new_game.pack(side="left", fill="x", expand=True, padx=(5, 0))
        ToolTip(self.btn_new_game, "快速鍵 (Shortcut): Ctrl+N")

        # Tab 1: Auto Capture Settings
        self.tab_auto = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_auto, text="自動偵測 (Auto)")

        # Grid settings for Tab 1
        row_idx = 0
        
        # Side to Move
        ttk.Label(self.tab_auto, text="輪到誰走 (Side to Move):").grid(row=row_idx, column=0, sticky="w")
        self.side_var = tk.StringVar(value="w")
        ttk.Radiobutton(self.tab_auto, text="白方 (White)", variable=self.side_var, value="w").grid(row=row_idx, column=1, sticky="w")
        ttk.Radiobutton(self.tab_auto, text="黑方 (Black)", variable=self.side_var, value="b").grid(row=row_idx, column=2, sticky="w")
        row_idx += 1

        # My Color (for board orientation)
        ttk.Label(self.tab_auto, text="我的顏色 (My Color):").grid(row=row_idx, column=0, sticky="w")
        self.my_color_var = tk.StringVar(value="w")
        ttk.Radiobutton(self.tab_auto, text="白方 (White)", variable=self.my_color_var, value="w", command=self.draw_board).grid(row=row_idx, column=1, sticky="w")
        ttk.Radiobutton(self.tab_auto, text="黑方 (Black)", variable=self.my_color_var, value="b", command=self.draw_board).grid(row=row_idx, column=2, sticky="w")
        row_idx += 1

        # Auto Detect Checkbox
        self.auto_detect_var = tk.BooleanVar(value=True)
        chk_auto = ttk.Checkbutton(self.tab_auto, text="自動偵測權利 (Auto-detect Rights)", variable=self.auto_detect_var)
        chk_auto.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)
        ToolTip(chk_auto, "嘗試自動判斷王車易位權 (Try to infer castling rights)")
        row_idx += 1

        # Castling Rights (Combobox instead of checkboxes)
        ttk.Label(self.tab_auto, text="王車易位 (Castling Rights):").grid(row=row_idx, column=0, sticky="w")
        self.castling_var = tk.StringVar(value="KQkq")
        castling_choices = ["KQkq", "KQ", "kq", "K", "Q", "k", "q", "Kk", "Kq", "Qk", "Qq", "KQk", "KQq", "Kkq", "Qkq", "-"]
        cb_castling = ttk.Combobox(self.tab_auto, textvariable=self.castling_var, values=castling_choices, width=8)
        cb_castling.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(cb_castling, "王車易位權 (例如: KQkq, -)\nCastling rights (e.g. KQkq, -)")
        row_idx += 1

        # En Passant (Combobox instead of entry)
        ttk.Label(self.tab_auto, text="吃過路兵 (En Passant):").grid(row=row_idx, column=0, sticky="w")
        self.ep_var = tk.StringVar(value="-")
        ep_choices = ["-", "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6"]
        cb_ep = ttk.Combobox(self.tab_auto, textvariable=self.ep_var, values=ep_choices, width=6, state="readonly")
        cb_ep.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(cb_ep, "過路兵目標格 (例如: e3, -)\nTarget square (e.g. e3, -)")
        row_idx += 1

        # Time Limit (Combobox in seconds instead of milliseconds entry)
        ttk.Label(self.tab_auto, text="思考時間 (Time Limit s):").grid(row=row_idx, column=0, sticky="w")
        self.time_var = tk.StringVar(value="10")
        time_choices = ["1", "2", "5", "10", "30", "60"]
        cb_time = ttk.Combobox(self.tab_auto, textvariable=self.time_var, values=time_choices, width=6)
        cb_time.grid(row=row_idx, column=1, columnspan=2, sticky="w")
        ToolTip(cb_time, "秒 (1 = 1秒)\nSeconds (1 = 1s)")
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
        chk_autoplay = ttk.Checkbutton(auto_play_frame, text="自動下棋 (Auto-Play)", variable=self.auto_play_var)
        chk_autoplay.pack(side="left")
        ToolTip(chk_autoplay, "自動在螢幕上點擊滑鼠下棋 (Auto-click best move)")
        
        ttk.Radiobutton(auto_play_frame, text="固定方 (Fixed)", variable=self.autoplay_mode_var, value="fixed").pack(side="left", padx=(10, 0))
        ttk.Radiobutton(auto_play_frame, text="自己對戰 (Self-Play)", variable=self.autoplay_mode_var, value="self").pack(side="left")
        row_idx += 1

        # Auto-Detect Checkbox
        self.auto_detect_opponent_var = tk.BooleanVar(value=False)
        chk_autodetect_opp = ttk.Checkbutton(self.tab_auto, text="自動偵測對手 (Auto-Detect Opponent)", variable=self.auto_detect_opponent_var)
        chk_autodetect_opp.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=2)
        ToolTip(chk_autodetect_opp, "對手下棋後自動開始分析 (Auto-start on opponent move)")
        row_idx += 1

        # Piece Style Selector & Calibration Buttons
        ttk.Label(self.tab_auto, text="棋子風格 (Piece Style):").grid(row=row_idx, column=0, sticky="w", pady=2)
        self.piece_style_var = tk.StringVar(value="Default")
        cb_style = ttk.Combobox(self.tab_auto, textvariable=self.piece_style_var, values=["Default", "Custom"], width=8, state="readonly")
        cb_style.grid(row=row_idx, column=1, sticky="w", pady=2)
        cb_style.bind("<<ComboboxSelected>>", self.on_piece_style_selected)
        
        self.btn_learn = ttk.Button(self.tab_auto, text="學習 (Learn)", command=self.learn_piece_style, width=8)
        self.btn_learn.grid(row=row_idx, column=2, sticky="w", padx=2, pady=2)
        ToolTip(self.btn_learn, "將網頁棋盤設為起始位置後，點此一鍵學習自訂風格")
        row_idx += 1

        self.btn_calibrate = ttk.Button(self.tab_auto, text="🔍 偵測預覽與手動校正 (Preview & Calibrate)", command=self.open_calibration_window)
        self.btn_calibrate.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=5)
        row_idx += 1

        # Tab 2: Manual Play / Analysis Mode
        self.tab_manual = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_manual, text="手動分析 (Manual)")

        # Grid settings for Tab 2
        ttk.Label(self.tab_manual, text="輸入 FEN (FEN Input):").grid(row=0, column=0, sticky="w", pady=5)
        self.manual_fen_var = tk.StringVar(value=chess.STARTING_FEN)
        self.entry_manual_fen = ttk.Entry(self.tab_manual, textvariable=self.manual_fen_var, width=32)
        self.entry_manual_fen.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        ttk.Button(self.tab_manual, text="載入 FEN (Load)", command=self.load_manual_fen).grid(row=1, column=0, pady=5, sticky="ew", padx=2)
        ttk.Button(self.tab_manual, text="清空棋盤 (Clear)", command=self.clear_manual_board).grid(row=1, column=1, pady=5, sticky="ew", padx=2)
        ttk.Button(self.tab_manual, text="重置起始位置 (Reset)", command=self.reset_manual_board).grid(row=1, column=2, pady=5, sticky="ew", padx=2)

        self.live_eval_var = tk.BooleanVar(value=True)
        chk_live = ttk.Checkbutton(self.tab_manual, text="落子自動即時分析 (Live Eval)", variable=self.live_eval_var)
        chk_live.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)
        ToolTip(chk_live, "在手動分析模式下，每次移動棋子後是否立即自動開始分析")

        ttk.Button(self.tab_manual, text="翻轉棋盤 (Flip Board)", command=self.flip_board).grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

        # Tab 3: Diagnostic Log Console
        self.tab_console = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_console, text="日誌主控台 (Console)")

        # Scrolled Text Box for Console in Tab 3
        console_frame = ttk.Frame(self.tab_console)
        console_frame.pack(fill="both", expand=True)
        self.console_text = tk.Text(console_frame, height=9, width=50, state="disabled", wrap="word", 
                                    font=("Consolas", 9))
        self.console_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(console_frame, orient="vertical", command=self.console_text.yview)
        sb.pack(side="right", fill="y")
        self.console_text.configure(yscrollcommand=sb.set)

        self.btn_open_logs = ttk.Button(self.tab_console, text="📂 開啟日誌資料夾 (Open Log Folder)", command=self.open_log_folder)
        self.btn_open_logs.pack(pady=5, fill="x")

        # 3. Info Frame (Always visible below tabs and buttons)
        info_frame = ttk.LabelFrame(self, text="分析結果 (Analysis)", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Status Label (Warming Up / Ready)
        self.lbl_status = ttk.Label(info_frame, text="狀態 (Status): 等待熱機... (Warming Up...)", font=("Arial", 10, "italic"), foreground="red")
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
        
        # Try to load images
        self.load_piece_images()
        
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Shortcuts (Global Default)
        self.bind('<Return>', lambda e: self.start_analysis_thread())
        self.bind('<Escape>', lambda e: self.stop_analysis())
        self.bind('<Control-n>', lambda e: self.new_game())

        # Initial Draw
        self.draw_board()

    def toggle_topmost(self):
        self.attributes('-topmost', self.always_on_top_var.get())

    def on_canvas_resize(self, event):
        new_size = min(event.width, event.height)
        if new_size > 50 and abs(self.canvas_size - new_size) > 2:
            self.canvas_size = new_size
            self.square_size = self.canvas_size // 8
            self.load_piece_images()
            self.draw_board()

    def load_piece_images(self):
        """Loads piece images from templates folder."""
        self.piece_images = {}
        if not HAS_PIL:
            print("[WARN] Pillow library not found. Falling back to unicode pieces.")
            return

        try:
            # Clear original cache to force reload if switching styles
            self.original_pil_images = {}
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            style_name = "Default"
            if hasattr(self, 'piece_style_var'):
                style_name = self.piece_style_var.get()
                
            if style_name == "Custom":
                template_dir = os.path.abspath(os.path.join(script_dir, "data", "templates", "custom"))
            else:
                template_dir = os.path.abspath(os.path.join(script_dir, "data", "templates"))
                
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

    def _run_warmup(self):
        """Runs a silent short search to warm up the engine JIT."""
        if not self.analyzing:
            print("[INFO] Warming up engine...")
            self.warming_up = True
            # Use Kiwipete position to avoid book hits and force search
            kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
            self.engine.send_command(f"position fen {kiwipete}")
            self.engine.send_command("go depth 12")

    def new_game(self):
        """Resets the engine's transposition table."""
        if self.analyzing:
            messagebox.showwarning("Warning", "請先停止當前分析 (Please stop analysis first)")
            return
        
        self.engine.send_command("ucinewgame")
        self.lbl_pv.config(text="變例 (PV): --")
        self.lbl_score.config(text="評分 (Score): --")
        self.lbl_bestmove.config(text="最佳著法 (Best Move): --")
        messagebox.showinfo("Info", "新遊戲已開始 (New Game Started) - 置換表已清除 (Hash Cleared)")

    def start_analysis_thread(self):
        if self.warming_up:
             messagebox.showwarning("Warning", "引擎正在熱機中，請稍候... (Engine Warming Up)")
             return
             
        # Check current tab to see if we are in manual analysis mode
        try:
            current_tab = self.notebook.index("current")
        except:
            current_tab = 0

        if current_tab == 1:
            self.start_manual_analysis()
            return

        # Prevent starting analysis if in fixed mode and it's not our turn
        if self.autoplay_mode_var.get() == "fixed" and self.side_var.get() != self.my_color_var.get() and self.auto_detect_opponent_var.get():
             print("[INFO] 目前為固定方對戰模式且輪到對手，不主動進行分析。")
             return
             
        if self.analyzing:
            return
        
        self.analyzing = True
        self.btn_analyze.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_score.config(text="評分 (Score): 計算中...")
        self.lbl_bestmove.config(text="最佳著法 (Best Move): ...")
        self.lbl_pv.config(text="變例 (PV): ...")
        self.best_move_arrow = None
        
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def stop_analysis(self):
        if self.analyzing:
            self.engine.stop_calculation()
            self.analyzing = False # Stop immediately internally
            
        self.auto_detecting_opponent = False
        self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready) [停止]", foreground="green")
        self.btn_analyze.config(state="normal")
        self.btn_stop.config(state="disabled")

    def toggle_side_to_move(self):
        current = self.side_var.get()
        self.side_var.set('b' if current == 'w' else 'w')

    def execute_auto_play(self, move_uci):
        """Simulate mouse clicks to play the move on screen using Windows API."""
        if not self.recognizer:
            return

        coords = self.recognizer.get_move_screen_coords(move_uci, player_color=self.my_color_var.get())
        if not coords:
            print("[WARN] 無法計算自動下棋的螢幕座標")
            return

        start_pt, end_pt = coords
        print(f"[INFO] Auto-playing {move_uci}: clicking {start_pt} then {end_pt}")

        # Mouse event constants
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004

        def click(x, y):
            ctypes.windll.user32.SetCursorPos(x, y)
            time.sleep(0.05)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(0.1)

        # Execute start and end clicks
        click(start_pt[0], start_pt[1])
        time.sleep(0.2)
        click(end_pt[0], end_pt[1])
        
        # Handle Pawn Promotion
        # Usually UCI for promotion is 5 chars, e.g. "e7e8q"
        # When a pawn reaches the 8th/1st rank, chess.com (and most sites) shows a popup.
        # The popup usually appears right at the destination square, extending downwards or upwards.
        # Typically the Queen is the first option (exactly on the destination square or slightly offset).
        # We will add a small delay and then click slightly around the destination square
        # depending on the piece requested.
        if len(move_uci) == 5 and move_uci[-1] in ['q', 'r', 'b', 'n']:
            promo_piece = move_uci[-1]
            print(f"[INFO] 偵測到升變: {promo_piece}，準備點擊選擇選單...")
            # Wait for the promotion menu to appear
            time.sleep(0.4) 
            
            # Estimate screen coordinates for the popup menu.
            # On chess.com, for white's promotion on e8 (top), the menu opens downwards [Q, N, R, B]
            # For black's promotion on e1 (bottom), the menu opens upwards.
            # It's highly site-specific. We'll implement a basic standard offset.
            # Assuming Q is always exactly at the target square (which covers 95% of cases as we usually promote to Q).
            sq_size = self.recognizer.board_side_length // 8
            
            target_file = ord(move_uci[2]) - ord('a')
            target_rank = int(move_uci[3]) - 1 # 0 for rank 1, 7 for rank 8
            
            # Is promotion UI going down or up?
            is_white_promo = (target_rank == 7)
            
            # Simple offset logic (Assuming standard chess.com layout: Q, N, R, B)
            # We add vertical offset based on the piece.
            offset_multiplier = {'q': 0, 'n': 1, 'r': 2, 'b': 3}
            # Many sites auto-promote to Queen if you just click the square again, 
            # or the Queen button replaces the square.
            
            # To be safe, try to click the exact spot if it's 'q'
            if promo_piece == 'q':
                # Just click the destination square again
                click(end_pt[0], end_pt[1])
            else:
                 # If not Q, we try to guess the offset. This might need tweaking per website.
                 # Let's assume the pieces are stacked vertically on top of each other.
                 # For white, pieces go down. For black, pieces might go up or down depending on the site.
                 # Let's assume downward for simplicity, each piece taking 1 square size.
                 y_offset = offset_multiplier[promo_piece] * sq_size
                 # If we are Black promoting at the bottom (rank 1), popup usually goes UP.
                 if not is_white_promo:
                     y_offset = -y_offset 
                 click(end_pt[0], end_pt[1] + y_offset)
        
        # Wait a moment for move animation to finish and then transition to opponent turn detection
        time.sleep(0.8)
        self.after(0, self.on_our_move_completed)

    def get_exclude_rect(self):
        try:
            import ctypes
            from ctypes import wintypes
            
            hwnd = self.winfo_id()
            rect = wintypes.RECT()
            # GetWindowRect returns coordinates in physical pixels
            if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                x = rect.left
                y = rect.top
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 0 and h > 0:
                    return (x, y, w, h)
            
            # Fallback to tkinter's winfo if GetWindowRect fails
            x = self.winfo_x()
            y = self.winfo_y()
            w = self.winfo_width()
            h = self.winfo_height()
            if w > 0 and h > 0:
                return (x, y, w, h)
            return None
        except:
            return None

    def on_our_move_completed(self):
        # Detect Game Over
        if self.board.is_game_over():
            outcome = self.board.outcome()
            reason = "將殺 (Checkmate)" if self.board.is_checkmate() else "平局/逼和 (Draw/Stalemate)"
            self.lbl_status.config(text=f"狀態 (Status): 遊戲結束 ({reason}) - {outcome.result()}", foreground="purple")
            self.auto_play_var.set(False)
            self.auto_detect_opponent_var.set(False)
            self.analyzing = False
            return

        if self.auto_detect_opponent_var.get():
            if self.autoplay_mode_var.get() == "self":
                # In self-play mode, just toggle side and analyze again
                self.toggle_side_to_move()
                print("[INFO] 自己對戰模式：自動觸發下一回合分析...")
                self.start_analysis_thread()
            else:
                # Standard play: toggle side to move to opponent, then start opponent detect loop
                self.toggle_side_to_move()
                print("[INFO] 已完成我方著法，啟動對手偵測...")
                threading.Thread(target=self._auto_detect_loop, daemon=True).start()
        else:
            self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")

    def _auto_detect_manual_play_loop(self):
        """Background thread to detect manual play: first wait for player, then wait for opponent."""
        print("[INFO] Started Auto-Detect Manual Play loop...")
        self.auto_detecting_opponent = True
        
        def update_status(text, color):
            self.lbl_status.config(text=text, foreground=color)
            
        self.after(0, update_status, "狀態 (Status): 等待我方下棋... (Waiting for your move)", "blue")
        
        # Phase 1: Wait for player's move
        try:
            baseline_fen_raw = self.recognizer.get_fen_from_screen(
                player_color=self.my_color_var.get(),
                active_player=self.side_var.get(),
                castling='-', en_passant='-',
                exclude_rect=self.get_exclude_rect(),
                verbose=False
            )
            self.last_detected_fen = baseline_fen_raw
            if not baseline_fen_raw:
                print("[WARN] Manual-detect failed to get initial baseline FEN.")
                self.auto_detecting_opponent = False
                self.after(0, update_status, "狀態 (Status): 自動偵測失敗 (Auto-detect failed)", "red")
                return
            
            print(f"[INFO] Initial baseline FEN for manual play: {baseline_fen_raw}")
        except Exception as e:
            print(f"Error getting initial FEN: {e}")
            self.auto_detecting_opponent = False
            return

        consecutive_stable_frames = 0
        current_detected_fen = baseline_fen_raw

        # 1. Loop waiting for player to move
        while self.auto_detecting_opponent and self.analyzing:
            time.sleep(0.5)
            if not self.auto_detecting_opponent or not self.analyzing:
                break
                
            try:
                new_fen_raw = self.recognizer.get_fen_from_screen(
                    player_color=self.my_color_var.get(),
                    active_player=self.side_var.get(),
                    castling='-', en_passant='-',
                    exclude_rect=self.get_exclude_rect(),
                    verbose=False
                )
                
                if new_fen_raw and new_fen_raw != baseline_fen_raw:
                    if new_fen_raw == current_detected_fen:
                        consecutive_stable_frames += 1
                    else:
                        consecutive_stable_frames = 1
                        current_detected_fen = new_fen_raw
                        
                    if consecutive_stable_frames >= 2:
                        print(f"[INFO] Player manual move detected! New FEN: {current_detected_fen}")
                        
                        # Update virtual board and side to move
                        def on_player_moved(new_fen):
                            try:
                                self.board.set_fen(new_fen)
                                self.draw_board()
                            except:
                                pass
                            self.toggle_side_to_move()
                            self.lbl_status.config(text="狀態 (Status): 等待對手下棋... (Waiting for opponent)", foreground="blue")
                            
                        self.after(0, on_player_moved, current_detected_fen)
                        
                        # Wait 0.8s for animations to settle
                        time.sleep(0.8)
                        
                        # Capture new baseline FEN (now after player's move, turn is opponent)
                        baseline_fen_after_us = self.recognizer.get_fen_from_screen(
                            player_color=self.my_color_var.get(),
                            active_player=self.side_var.get(),
                            castling='-', en_passant='-',
                            exclude_rect=self.get_exclude_rect(),
                            verbose=False
                        )
                        if not baseline_fen_after_us:
                            baseline_fen_after_us = current_detected_fen
                        
                        self.last_detected_fen = baseline_fen_after_us
                        print(f"[INFO] New baseline FEN for opponent: {baseline_fen_after_us}")
                        
                        # Reset for Phase 2
                        baseline_fen_raw = baseline_fen_after_us
                        consecutive_stable_frames = 0
                        current_detected_fen = baseline_fen_after_us
                        break
            except Exception as e:
                print(f"Error during manual player detect: {e}")
                
        # 2. Loop waiting for opponent to move
        while self.auto_detecting_opponent and self.analyzing:
            time.sleep(0.5)
            if not self.auto_detecting_opponent or not self.analyzing:
                break
                
            try:
                new_fen_raw = self.recognizer.get_fen_from_screen(
                    player_color=self.my_color_var.get(),
                    active_player=self.side_var.get(),
                    castling='-', en_passant='-',
                    exclude_rect=self.get_exclude_rect(),
                    verbose=False
                )
                
                if new_fen_raw and new_fen_raw != baseline_fen_raw:
                    if new_fen_raw == current_detected_fen:
                        consecutive_stable_frames += 1
                    else:
                        consecutive_stable_frames = 1
                        current_detected_fen = new_fen_raw
                        
                    if consecutive_stable_frames >= 2:
                        print(f"[INFO] Opponent move detected in manual loop! New FEN: {current_detected_fen}")
                        self.auto_detecting_opponent = False
                        
                        def resume_turn():
                            self.toggle_side_to_move()
                            self.start_analysis_thread()
                            
                        self.after(0, resume_turn)
                        break
            except Exception as e:
                print(f"Error during manual opponent detect: {e}")

    def _auto_detect_loop(self):
        """Background thread to detect when the opponent has moved."""
        print("[INFO] Started Auto-Detect Opponent loop...")
        self.auto_detecting_opponent = True
        
        def update_status(text, color):
            self.lbl_status.config(text=text, foreground=color)
            
        self.after(0, update_status, "狀態 (Status): 等待對手下棋... (Waiting for opponent)", "blue")

        # Wait a moment before taking the baseline FEN so animations can finish
        time.sleep(0.1)
        
        try:
            # We assume recognizer exists because we just finished our own analysis
            baseline_fen_raw = self.recognizer.get_fen_from_screen(
                player_color=self.my_color_var.get(),
                active_player=self.side_var.get(), # Now opponent's side
                castling='-', en_passant='-',
                exclude_rect=self.get_exclude_rect(),
                verbose=False
            )
            self.last_detected_fen = baseline_fen_raw
            if not baseline_fen_raw:
                print("[WARN] Auto-detect failed to get baseline FEN.")
                self.auto_detecting_opponent = False
                self.after(0, update_status, "狀態 (Status): 自動偵測失敗 (Auto-detect failed)", "red")
                return

            print(f"[INFO] Baseline FEN for auto-detect: {self.last_detected_fen}")
            
        except Exception as e:
            print(f"Error getting baseline FEN: {e}")
            self.auto_detecting_opponent = False
            return

        consecutive_stable_frames = 0
        current_detected_fen = baseline_fen_raw

        while self.auto_detecting_opponent:
            time.sleep(0.5) # Check every 0.5 second
            if not self.auto_detecting_opponent:
                 break
                 
            try:
                # Run full recognition
                new_fen_raw = self.recognizer.get_fen_from_screen(
                    player_color=self.my_color_var.get(),
                    active_player=self.side_var.get(),
                    castling='-', en_passant='-',
                    exclude_rect=self.get_exclude_rect(),
                    verbose=False
                )
                
                if new_fen_raw and new_fen_raw != baseline_fen_raw:
                    # FEN changed! Let's ensure it's stable for 2 frames to avoid animation mid-points
                    if new_fen_raw == current_detected_fen:
                        consecutive_stable_frames += 1
                    else:
                        consecutive_stable_frames = 1
                        current_detected_fen = new_fen_raw
                        
                    if consecutive_stable_frames >= 2:
                        print(f"[INFO] Opponent move detected! New FEN: {current_detected_fen}")
                        self.auto_detecting_opponent = False
                        
                        def resume_turn():
                            self.toggle_side_to_move()
                            self.start_analysis_thread()
                            
                        # Trigger analysis automatically via UI thread
                        self.after(0, resume_turn)
                        break
            except Exception as e:
                 print(f"Error during auto-detect: {e}")
                 pass

    def infer_castling_rights(self, board):
        """
        Infer likely castling rights based on piece positions.
        Heuristic: If King and Rook are on starting squares, assume castling is possible.
        """
        rights = ""
        # White
        if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE): rights += "K"
            if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE): rights += "Q"
        # Black
        if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK): rights += "k"
            if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK): rights += "q"
        
        return rights if rights else "-"

    def _run_analysis(self):
        try:
            # First time analysis delay (requested by user)
            if self.is_first_analysis:
                print("[INFO] First analysis: Waiting 2 seconds before capture...")
                time.sleep(2)
                self.is_first_analysis = False

            try:
                from tools import screen_recognizer
            except ImportError as e:
                self.message_queue.put({"type": "error", "message": f"無法載入 screen_recognizer: {e}"})
                self.message_queue.put({"type": "analysis_finished"})
                return
            # Lazy init recognizer to avoid slow startup if not used
            if not self.recognizer:
                try:
                    self.recognizer = screen_recognizer.ScreenRecognizer()
                except Exception as e:
                    self.message_queue.put({"type": "error", "message": f"Recognizer Init Failed: {e}"})
                    traceback.print_exc()
                    self.message_queue.put({"type": "analysis_finished"}) # Fail safe
                    return

            # Capture Raw Pieces (using defaults for rights to get just pieces first)
            fen_raw = self.recognizer.get_fen_from_screen(
                player_color=self.my_color_var.get(),
                active_player=self.side_var.get(),
                castling='-', # Placeholder
                en_passant='-', # Placeholder
                exclude_rect=self.get_exclude_rect()
            )
            
            if not fen_raw:
                self.message_queue.put({"type": "error", "message": "無法辨識棋盤 (Recognition Failed)"})
                self.message_queue.put({"type": "analysis_finished"})
                return

            fen_final = fen_raw

            if self.auto_detect_var.get():
                try:
                    # Parse board to infer rights
                    board = chess.Board(fen_raw)
                    new_rights = self.infer_castling_rights(board)
                    
                    # Update UI in main thread
                    def update_ui_fields(r):
                        self.castling_var.set(r)
                        # Keep existing EP square if it was set manually by the user
                        # self.ep_var.set("-") 

                    self.after(0, update_ui_fields, new_rights)

                    # Reconstruct FEN
                    parts = fen_raw.split()
                    parts[2] = new_rights
                    parts[3] = sanitize_fen_input(self.ep_var.get(), 'ep')
                    fen_final = " ".join(parts)
                except Exception as e:
                    print(f"Auto-detect rights failed: {e}")
            else:
                # Use User Input
                user_rights_raw = self.castling_var.get()
                user_ep_raw = self.ep_var.get()

                # Security: Sanitize input to prevent UCI Command Injection
                user_rights = sanitize_fen_input(user_rights_raw, 'castling')
                user_ep = sanitize_fen_input(user_ep_raw, 'ep')

                # Update UI to reflect sanitized values if they changed significantly (optional, but good for feedback)
                # Note: We are in a thread, so use self.after if we wanted to update UI. 
                # For now, we just use the sanitized values for the engine.

                parts = fen_raw.split()
                parts[2] = user_rights
                parts[3] = user_ep
                fen_final = " ".join(parts)

            self.message_queue.put({"type": "fen_update", "fen": fen_final})
            
            # Check for Game Over before sending to engine
            temp_board = chess.Board(fen_final)
            if temp_board.is_game_over():
                outcome = temp_board.outcome()
                reason = "將殺 (Checkmate)" if temp_board.is_checkmate() else "平局/逼和 (Draw/Stalemate)"
                msg = f"偵測到遊戲已結束 ({reason}): {outcome.result()}"
                self.message_queue.put({"type": "log", "message": msg})
                
                def stop_on_over():
                    self.lbl_status.config(text=f"狀態 (Status): 遊戲結束 - {outcome.result()}", foreground="purple")
                    self.auto_play_var.set(False)
                    self.auto_detect_opponent_var.set(False)
                    self.analyzing = False
                    self.btn_analyze.config(state="normal")
                    self.btn_stop.config(state="disabled")
                
                self.after(0, stop_on_over)
                return

            # Send to Engine
            try:
                time_limit = int(float(self.time_var.get()) * 1000)
            except:
                time_limit = 10000

            # Removed automatic ucinewgame to preserve TT across moves
            # self.engine.send_command("ucinewgame")
            
            self.engine.send_command(f"position fen {fen_final}")
            self.engine.send_command(f"go movetime {time_limit}")

            # Note: We do NOT send "analysis_finished" here. 
            # We wait for "bestmove" from the engine to signal completion.

        except Exception as e:
            traceback.print_exc() # Print full stack trace for debugging
            self.message_queue.put({"type": "error", "message": f"Analysis Error: {e}"})
            self.message_queue.put({"type": "analysis_finished"})

    def _process_queue(self):
        try:
            while True:
                msg = self.message_queue.get_nowait()
                
                if msg["type"] == "raw_log":
                    self.write_to_console(msg["message"])

                elif msg["type"] == "info":
                    if self.warming_up: continue

                    # Update Info
                    if "score_type" in msg:
                        s_type = msg["score_type"]
                        s_val = msg["score_val"]
                        display_score = f"{s_val} cp" if s_type == "cp" else f"Mate in {s_val}"
                        self.lbl_score.config(text=f"評分 (Score): {display_score}")
                    
                    if "pv" in msg:
                        self.lbl_pv.config(text=f"變例 (PV): {msg['pv']}")

                    # Performance Logging: Log info depth diagnostics
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
                        
                        log_line = f"Depth: {depth:<3} | SelDepth: {seldepth:<3} | Score: {display_score:<10} | Time: {time_ms:<6} ms | Nodes: {nodes:<10} | NPS: {nps:<10} | Best Move: {first_move}"
                        self.log_search_info(log_line)

                elif msg["type"] == "bestmove":
                    if self.warming_up:
                        self.warming_up = False
                        self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")
                        print("[INFO] Warmup complete.")
                        continue
                        
                    # If user pressed stop, analyzing is False, so we should discard this bestmove
                    if not self.analyzing:
                         print("[INFO] Muted bestmove because analysis was stopped.")
                         self.btn_analyze.config(state="normal")
                         self.btn_stop.config(state="disabled")
                         continue

                    move_uci = msg["move"]
                    
                    # Display proper castling move
                    castling_map = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
                    if move_uci in castling_map and self.board.piece_at(chess.parse_square(move_uci[:2])) == chess.Piece(chess.KING, self.board.turn):
                        display_move = castling_map[move_uci]
                    else:
                        display_move = move_uci
                        
                    self.lbl_bestmove.config(text=f"最佳著法 (Best Move): {display_move}")
                    
                    # Log best move chosen
                    self.log_search_info(f"[Best Move Chosen]: {display_move}")
                    self.log_search_info("=" * 110)

                    # Highlight move on board
                    try:
                        move = chess.Move.from_uci(display_move)
                        self.best_move_arrow = move
                        
                        # Only push move automatically in screen-capturing auto mode (not in manual tab)
                        current_tab = 0
                        try:
                            current_tab = self.notebook.index("current")
                        except:
                            pass
                            
                        if current_tab != 1: # If not in manual mode
                            if move in self.board.legal_moves:
                                self.board.push(move)
                        
                        self.draw_board()
                    except:
                        pass
                    
                    # Signal that analysis is done
                    self.message_queue.put({"type": "analysis_finished"})
                    
                    # --- Automation Logic ---
                    is_our_turn = (self.side_var.get() == self.my_color_var.get())
                    
                    if is_our_turn:
                        if self.auto_play_var.get():
                            # Auto-play Scenario
                            threading.Thread(target=self.execute_auto_play, args=(display_move,), daemon=True).start()
                        elif self.auto_detect_opponent_var.get():
                            # Manual play with auto-detect Scenario: wait for our move, then opponent's move
                            threading.Thread(target=self._auto_detect_manual_play_loop, daemon=True).start()
                        else:
                            self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")
                    else:
                        # It's opponent's turn
                        if self.auto_detect_opponent_var.get():
                            if self.autoplay_mode_var.get() == "self":
                                # Self play mode: just toggle side and analyze
                                self.toggle_side_to_move()
                                time.sleep(0.8)
                                self.start_analysis_thread()
                            else:
                                # Standard play: wait for opponent
                                threading.Thread(target=self._auto_detect_loop, daemon=True).start()
                        else:
                            self.lbl_status.config(text="狀態 (Status): 準備就緒 (Ready)", foreground="green")

                elif msg["type"] == "fen_update":
                    fen = msg["fen"]
                    try:
                        self.board.set_fen(fen)
                        self.draw_board()
                    except:
                        pass
                    
                    # Log the FEN header
                    self.log_search_info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === 分析盤面 ===")
                    self.log_search_info(f"FEN: {fen}")
                    self.log_search_info("-" * 110)

                elif msg["type"] == "error":
                    messagebox.showerror("Error", msg["message"])
                    self.lbl_status.config(text="狀態 (Status): 發生錯誤 (Error)", foreground="red")

                elif msg["type"] == "analysis_finished":
                    self.analyzing = False
                    self.btn_analyze.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    # Note: we update label inside bestmove now if auto-detecting
                
                elif msg["type"] == "log":
                    print(f"[LOG] {msg['message']}")

        except queue.Empty:
            pass
        
        self.after(100, self._process_queue)

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
        
        for rank in range(8):
            for file in range(8):
                # Calculate coordinates
                display_rank = rank if is_flipped else (7 - rank)
                display_file = (7 - file) if is_flipped else file
                
                x1 = offset_x + display_file * self.square_size
                y1 = offset_y + display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Color
                square_idx = chess.square(file, rank)
                color = BOARD_COLORS[(rank + file) % 2]
                
                # Highlight selected square in manual mode
                if hasattr(self, 'selected_square') and self.selected_square == square_idx:
                    color = "#E0E68C" # light khaki for selected square
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Labels (Coordinates)
                label_color = "#B58863" if (rank + file) % 2 == 1 else "#F0D9B5"
                if display_file == 0: # Left edge
                    self.canvas.create_text(x1 + 2, y1 + 2, text=str(rank + 1), anchor="nw", font=("Arial", 8), fill=label_color)
                if display_rank == 7: # Bottom edge
                    self.canvas.create_text(x2 - 2, y2 - 2, text=chr(ord('a') + file), anchor="se", font=("Arial", 8), fill=label_color)

                # Piece
                piece = self.board.piece_at(square_idx)
                
                if piece:
                    # If dragging, skip drawing the piece on its original square
                    if hasattr(self, 'dragging') and self.dragging and hasattr(self, 'selected_square') and self.selected_square == square_idx:
                        continue
                        
                    # Try to draw image first
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
                    
                    # Fallback to Unicode text
                    if not drawn_image:
                        symbol = UNICODE_PIECES.get(piece.symbol())
                        self.canvas.create_text(
                            x1 + self.square_size // 2,
                            y1 + self.square_size // 2,
                            text=symbol,
                            font=("Segoe UI Symbol", 32)
                        )

        # Draw dragging piece at cursor
        if hasattr(self, 'dragging') and self.dragging and hasattr(self, 'selected_square') and self.selected_square is not None:
            piece = self.board.piece_at(self.selected_square)
            if piece:
                drawn_image = False
                if hasattr(self, 'piece_images') and self.piece_images:
                    piece_key = f"{'w' if piece.color else 'b'}{piece.symbol().upper()}"
                    if piece_key in self.piece_images:
                        self.canvas.create_image(
                            self.drag_x,
                            self.drag_y,
                            image=self.piece_images[piece_key],
                            anchor="center"
                        )
                        drawn_image = True
                if not drawn_image:
                    symbol = UNICODE_PIECES.get(piece.symbol())
                    self.canvas.create_text(
                        self.drag_x,
                        self.drag_y,
                        text=symbol,
                        font=("Segoe UI Symbol", 32)
                    )

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
                    fill="#1E90FF", # Dodger Blue
                    width=5,
                    arrow=tk.LAST,
                    arrowshape=(16, 20, 6)
                )
            except Exception as e:
                print(f"Error drawing arrow: {e}")

    def on_tab_changed(self, event):
        current_tab = self.notebook.index("current")
        if self.analyzing:
            self.stop_analysis()

        if current_tab == 1: # Manual Tab
            # Disable auto play and auto opponent detect to avoid conflicts
            self.auto_play_var.set(False)
            self.auto_detect_opponent_var.set(False)
            self.auto_detecting_opponent = False
            
            # Bind manual interactions
            self.canvas.bind("<Button-1>", self.on_board_click)
            self.canvas.bind("<B1-Motion>", self.on_board_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_board_release)
            
            # Bind shortcuts
            self.bind("<Left>", lambda e: self.manual_undo())
            self.bind("<Right>", lambda e: self.manual_redo())
            self.bind("<space>", lambda e: self.toggle_manual_eval())
            
            # Synchronize manual_fen_var with current board FEN
            self.manual_fen_var.set(self.board.fen())
            self.redo_stack = []
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
        else:
            # Unbind manual interactions
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            
            # Unbind shortcuts
            self.unbind("<Left>")
            self.unbind("<Right>")
            self.unbind("<space>")
            
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()

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
        sq = self.get_square_from_coords(event.x, event.y)
        if sq is not None:
            piece = self.board.piece_at(sq)
            # Click-to-Move logic
            if hasattr(self, 'selected_square') and self.selected_square is not None:
                # If click target is different, check move
                if sq != self.selected_square:
                    promotion = None
                    p_piece = self.board.piece_at(self.selected_square)
                    if p_piece and p_piece.piece_type == chess.PAWN:
                        target_rank = chess.square_rank(sq)
                        if (p_piece.color == chess.WHITE and target_rank == 7) or (p_piece.color == chess.BLACK and target_rank == 0):
                            promotion = chess.QUEEN
                    
                    move = chess.Move(self.selected_square, sq, promotion=promotion)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.redo_stack = []
                        self.manual_fen_var.set(self.board.fen())
                        self.selected_square = None
                        self.dragging = False
                        self.draw_board()
                        if self.live_eval_var.get():
                            self.start_manual_analysis()
                        return
                    else:
                        # Move is illegal. If target has our turn piece, switch selection to it
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
                    # Clicked same square, prepare for potential drag
                    self.dragging = True
                    self.drag_x = event.x
                    self.drag_y = event.y
                    self.draw_board()
                    return
            
            # Fresh click
            if piece and piece.color == self.board.turn:
                self.selected_square = sq
                self.dragging = True
                self.drag_x = event.x
                self.drag_y = event.y
                self.draw_board()

    def on_board_drag(self, event):
        if hasattr(self, 'dragging') and self.dragging and hasattr(self, 'selected_square') and self.selected_square is not None:
            self.drag_x = event.x
            self.drag_y = event.y
            self.draw_board()

    def on_board_release(self, event):
        if hasattr(self, 'dragging') and self.dragging:
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
                    self.board.push(move)
                    self.redo_stack = []
                    self.manual_fen_var.set(self.board.fen())
                    self.selected_square = None
                    self.draw_board()
                    if self.live_eval_var.get():
                        self.start_manual_analysis()
                    return
            self.draw_board()

    def load_manual_fen(self):
        fen = self.manual_fen_var.get().strip()
        try:
            board = chess.Board(fen)
            self.board = board
            self.redo_stack = []
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
            if self.live_eval_var.get():
                self.start_manual_analysis()
        except Exception as e:
            messagebox.showerror("Error", f"無效的 FEN (Invalid FEN): {e}")

    def clear_manual_board(self):
        self.board.clear()
        self.manual_fen_var.set(self.board.fen())
        self.redo_stack = []
        self.selected_square = None
        self.best_move_arrow = None
        self.draw_board()
        if self.analyzing:
            self.stop_analysis()

    def reset_manual_board(self):
        self.board.reset()
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
             self.analyzing = False
             self.after(50, self.start_manual_analysis)
             return
             
        self.analyzing = True
        self.btn_analyze.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_score.config(text="評分 (Score): 計算中...")
        self.lbl_bestmove.config(text="最佳著法 (Best Move): ...")
        self.lbl_pv.config(text="變例 (PV): ...")
        self.best_move_arrow = None
        
        # Log FEN header to Performance Log
        self.log_search_info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === 手動分析盤面 ===")
        self.log_search_info(f"FEN: {self.board.fen()}")
        self.log_search_info("-" * 110)

        # Send FEN to engine
        fen = self.board.fen()
        self.engine.send_command(f"position fen {fen}")
        try:
            time_limit = int(float(self.time_var.get()) * 1000)
        except:
            time_limit = 10000
        self.engine.send_command(f"go movetime {time_limit}")

    def manual_undo(self):
        current_tab = self.notebook.index("current")
        if current_tab != 1:
            return
        if self.board.move_stack:
            move = self.board.pop()
            if not hasattr(self, 'redo_stack'):
                self.redo_stack = []
            self.redo_stack.append(move)
            self.manual_fen_var.set(self.board.fen())
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
            if self.live_eval_var.get():
                self.start_manual_analysis()

    def manual_redo(self):
        current_tab = self.notebook.index("current")
        if current_tab != 1:
            return
        if hasattr(self, 'redo_stack') and self.redo_stack:
            move = self.redo_stack.pop()
            self.board.push(move)
            self.manual_fen_var.set(self.board.fen())
            self.selected_square = None
            self.best_move_arrow = None
            self.draw_board()
            if self.live_eval_var.get():
                self.start_manual_analysis()

    def toggle_manual_eval(self):
        current_tab = self.notebook.index("current")
        if current_tab != 1:
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



    def on_piece_style_selected(self, event):
        style = self.piece_style_var.get()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if style == "Custom":
            custom_dir = os.path.abspath(os.path.join(script_dir, "data", "templates", "custom"))
            if not os.path.exists(custom_dir) or not os.listdir(custom_dir):
                messagebox.showwarning("Warning", "尚未學習自訂風格！請先點擊「學習」按鈕學習風格。")
                self.piece_style_var.set("Default")
                return
            self.recognizer.piece_templates = self.recognizer._load_templates(custom_dir)
        else:
            default_dir = os.path.abspath(os.path.join(script_dir, "data", "templates"))
            self.recognizer.piece_templates = self.recognizer._load_templates(default_dir)
            
        self.load_piece_images()
        self.draw_board()

    def learn_piece_style(self):
        """
        引導使用者一鍵學習自訂棋格內的棋子圖示風格。
        """
        if self.warming_up:
            messagebox.showwarning("Warning", "引擎正在熱機中，請稍候... (Engine Warming Up)")
            return
            
        confirm = messagebox.askyesno("學習自訂棋子風格", 
            "請確認：\n1. 網頁棋盤已重置為「新遊戲 (Starting Position)」位置。\n2. 白方棋子在下方，黑方在上方。\n\n確認已就緒並開始學習？")
        if not confirm:
            return
            
        try:
            # 抓取目前畫面
            screenshot = self.recognizer.capture_screen()
            # 偵測棋盤（這會自動更新 M 與 inv_M，支援無色彩形狀偵測）
            board_img = self.recognizer.find_board(screenshot, exclude_rect=self.get_exclude_rect())
            
            if board_img is None:
                messagebox.showerror("Error", "找不到棋盤！請先確認網頁棋盤是否完整顯示於螢幕上，或者先使用「手動校正」對齊。")
                return
                
            # 一鍵學習
            self.recognizer.learn_pieces_from_board(board_img, player_color=self.my_color_var.get())
            
            # 更新為 Custom 風格
            self.piece_style_var.set("Custom")
            self.load_piece_images()
            self.draw_board()
            
            messagebox.showinfo("Success", "自訂棋子風格學習成功！\n模板已自動儲存至 data/templates/custom/ 並成功啟用。")
            
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
            
        # 1. 抓取目前螢幕截圖
        try:
            screenshot = self.recognizer.capture_screen()
        except Exception as e:
            messagebox.showerror("Error", f"無法擷取螢幕 (Capture Failed): {e}")
            return
            
        # 建立子視窗
        calib_win = tk.Toplevel(self)
        calib_win.title("螢幕辨識預覽與手動校正 (Screen Preview & Board Calibration)")
        calib_win.geometry("980x640")
        calib_win.attributes('-topmost', True)
        
        # 顯示說明 Label
        lbl_desc = ttk.Label(calib_win, 
                             text="說明：拖曳四個紅色控制點，使其精準對齊網頁棋盤的四個外框角點。對齊完成後點選「套用校正」。", 
                             foreground="#1E90FF", font=("Segoe UI", 10, "bold"))
        lbl_desc.pack(fill="x", padx=10, pady=5)
        
        # 縮放截圖以適應 Canvas (最大 960x500)
        import cv2
        import numpy as np
        screen_h, screen_w, _ = screenshot.shape
        scale = min(960 / screen_w, 500 / screen_h)
        preview_w = int(screen_w * scale)
        preview_h = int(screen_h * scale)
        
        preview_img = cv2.resize(screenshot, (preview_w, preview_h))
        # 轉成 PIL 格式
        preview_pil = Image.fromarray(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))
        preview_tk = ImageTk.PhotoImage(preview_pil)
        
        # Canvas
        canvas = tk.Canvas(calib_win, width=preview_w, height=preview_h, bg="#1E1E1E")
        canvas.pack(padx=10, pady=5)
        
        # Keep reference to avoid garbage collection
        canvas.image = preview_tk
        canvas.create_image(0, 0, image=preview_tk, anchor="nw")
        
        # 初始化 4 個控制點的位置 (頂點)
        # 如果 recognizer.inv_M 已經存在，我們將棋盤的 4 個角點 (0,0), (W,0), (W,H), (0,H) 映射回螢幕，再乘以 scale
        points = []
        if self.recognizer.inv_M is not None:
            # warp corners back to screen space
            w_val = self.recognizer.board_side_length
            corners_warped = np.array([[[0, 0]], [[w_val, 0]], [[w_val, w_val]], [[0, w_val]]], dtype="float32")
            corners_screen = cv2.perspectiveTransform(corners_warped, self.recognizer.inv_M)
            for i in range(4):
                px = int(corners_screen[i][0][0] * scale)
                py = int(corners_screen[i][0][1] * scale)
                # Keep within bounds
                px = max(0, min(preview_w, px))
                py = max(0, min(preview_h, py))
                points.append([px, py])
        else:
            # 預設居中正方形
            cx, cy = preview_w // 2, preview_h // 2
            size = min(preview_w, preview_h) // 4
            points = [
                [cx - size, cy - size], # TL
                [cx + size, cy - size], # TR
                [cx + size, cy + size], # BR
                [cx - size, cy + size]  # BL
            ]
            
        active_handle = None
        handle_radius = 8
        
        def draw_calibration():
            canvas.delete("overlay")
            # 畫多邊形連線
            coords = []
            for p in points:
                coords.extend(p)
            canvas.create_polygon(coords, fill="", outline="red", width=2, tags="overlay")
            # 畫四個紅色圓形控制點與序號
            for idx, p in enumerate(points):
                canvas.create_oval(p[0] - handle_radius, p[1] - handle_radius, 
                                  p[0] + handle_radius, p[1] + handle_radius, 
                                  fill="red", outline="white", width=1, tags="overlay")
                canvas.create_text(p[0], p[1], text=str(idx+1), fill="white", font=("Segoe UI", 9, "bold"), tags="overlay")
                
        # 綁定拖曳事件
        def get_nearest_handle(event):
            nonlocal active_handle
            for idx, p in enumerate(points):
                dist = np.sqrt((p[0] - event.x)**2 + (p[1] - event.y)**2)
                if dist < 15:
                    active_handle = idx
                    return
            active_handle = None
            
        def drag_handle(event):
            if active_handle is not None:
                # Keep within canvas limits
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
        
        # 按鈕 Frame
        btn_frame = ttk.Frame(calib_win)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        def apply_calibration():
            # 將 4 個角點縮放回原始解析度
            orig_points = []
            for p in points:
                orig_points.append([p[0] / scale, p[1] / scale])
            orig_points = np.array(orig_points, dtype="float32")
            
            # 極角排序確保順序為: tl, tr, br, bl
            center_pt = orig_points.mean(axis=0)
            angles = np.arctan2(orig_points[:, 1] - center_pt[1], orig_points[:, 0] - center_pt[0])
            sorted_indices = np.argsort(angles)
            sorted_points = orig_points[sorted_indices]
            
            tl_idx = np.argmin(np.sum(sorted_points, axis=1))
            sorted_rect = np.roll(sorted_points, -tl_idx, axis=0)
            
            # 計算新的 M 與 inv_M
            side_length = self.recognizer.board_side_length
            dst = np.array([[0,0], [side_length-1,0], [side_length-1,side_length-1], [0,side_length-1]], dtype="float32")
            self.recognizer.M = cv2.getPerspectiveTransform(sorted_rect, dst)
            self.recognizer.inv_M = cv2.getPerspectiveTransform(dst, sorted_rect)
            
            self.lbl_status.config(text="狀態 (Status): 手動校正棋盤完成 (Calibration complete)", foreground="green")
            self.draw_board()
            calib_win.destroy()
            messagebox.showinfo("Success", "棋盤校正成功！\n棋盤邊界已重新校正完成。")
            
        ttk.Button(btn_frame, text="套用校正 (Apply)", command=apply_calibration).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="取消 (Cancel)", command=calib_win.destroy).pack(side="right", padx=5)

    def on_closing(self):
        self.engine.quit_engine()
        self.destroy()

if __name__ == "__main__":
    app = ChessVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
