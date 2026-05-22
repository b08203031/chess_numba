import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import subprocess
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))
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
    Manages the subprocess for the chess engine (main_nn.py).
    Handles UCI communication via stdin/stdout.
    """
    def __init__(self, callback_queue):
        self.callback_queue = callback_queue
        self.process = None
        self.running = False
        self.threads = []
        
        # Determine the path to main_nn.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.engine_path = os.path.join(script_dir, "main_nn.py")

    def start(self):
        if self.running:
            return

        try:
            # Use 'python' or sys.executable to run main_nn.py
            # Use -u for unbuffered output to ensure we get prints immediately
            cmd = [sys.executable, "-u", self.engine_path]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
                # Forward stderr to main thread logs if needed
                # self.callback_queue.put({"type": "log", "message": f"Engine Log: {line.strip()}"})
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
        self.geometry("600x850") # Increased height for new controls
        
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
        
        # UI Components
        self._create_ui()
        
        # Start message polling
        self.after(100, self._process_queue)
        
        # Warm-up (Silent)
        self.after(1000, self._run_warmup)

    def _create_ui(self):
        # 1. Controls Frame
        control_frame = ttk.LabelFrame(self, text="設定 (Settings)", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        row_idx = 0
        
        # Side to Move
        ttk.Label(control_frame, text="輪到誰走 (Side to Move):").grid(row=row_idx, column=0, sticky="w")
        self.side_var = tk.StringVar(value="w")
        ttk.Radiobutton(control_frame, text="白方 (White)", variable=self.side_var, value="w").grid(row=row_idx, column=1)
        ttk.Radiobutton(control_frame, text="黑方 (Black)", variable=self.side_var, value="b").grid(row=row_idx, column=2)
        row_idx += 1

        # My Color (for board orientation)
        ttk.Label(control_frame, text="我的顏色 (My Color):").grid(row=row_idx, column=0, sticky="w")
        self.my_color_var = tk.StringVar(value="w")
        ttk.Radiobutton(control_frame, text="白方 (White)", variable=self.my_color_var, value="w", command=self.draw_board).grid(row=row_idx, column=1)
        ttk.Radiobutton(control_frame, text="黑方 (Black)", variable=self.my_color_var, value="b", command=self.draw_board).grid(row=row_idx, column=2)
        row_idx += 1

        # Auto Detect Checkbox
        self.auto_detect_var = tk.BooleanVar(value=True)
        chk_auto = ttk.Checkbutton(control_frame, text="自動偵測權利 (Auto-detect Rights)", variable=self.auto_detect_var)
        chk_auto.grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=5)
        ToolTip(chk_auto, "嘗試自動判斷王車易位權 (Try to infer castling rights)")
        row_idx += 1

        # Castling Rights (New Feature)
        ttk.Label(control_frame, text="王車易位 (Castling Rights):").grid(row=row_idx, column=0, sticky="w")
        self.castling_var = tk.StringVar(value="KQkq")
        entry_castling = ttk.Entry(control_frame, textvariable=self.castling_var, width=10)
        entry_castling.grid(row=row_idx, column=1, sticky="w")
        ToolTip(entry_castling, "FEN 格式 (例如: KQkq, -, K, q)\nFEN format (e.g. KQkq, -, K, q)")
        row_idx += 1

        # En Passant (New Feature)
        ttk.Label(control_frame, text="吃過路兵 (En Passant):").grid(row=row_idx, column=0, sticky="w")
        self.ep_var = tk.StringVar(value="-")
        entry_ep = ttk.Entry(control_frame, textvariable=self.ep_var, width=5)
        entry_ep.grid(row=row_idx, column=1, sticky="w")
        ToolTip(entry_ep, "過路兵目標格 (例如: e3, -)\nTarget square (e.g. e3, -)")
        row_idx += 1

        # Time Limit
        ttk.Label(control_frame, text="思考時間 (Time Limit ms):").grid(row=row_idx, column=0, sticky="w")
        self.time_var = tk.StringVar(value="10000")
        entry_time = ttk.Entry(control_frame, textvariable=self.time_var, width=10)
        entry_time.grid(row=row_idx, column=1, sticky="w")
        ToolTip(entry_time, "毫秒 (1000 = 1秒)\nMilliseconds (1000 = 1s)")
        row_idx += 1

        # Always on top Checkbox
        self.always_on_top_var = tk.BooleanVar(value=True)
        chk_top = ttk.Checkbutton(control_frame, text="視窗置頂 (Always on Top)", variable=self.always_on_top_var, command=self.toggle_topmost)
        chk_top.grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=5)
        row_idx += 1

        # Auto-Play Checkbox and Mode
        auto_play_frame = ttk.Frame(control_frame)
        auto_play_frame.grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=5)
        
        self.auto_play_var = tk.BooleanVar(value=False)
        chk_autoplay = ttk.Checkbutton(auto_play_frame, text="自動下棋 (Auto-Play)", variable=self.auto_play_var)
        chk_autoplay.pack(side="left")
        ToolTip(chk_autoplay, "自動在螢幕上點擊滑鼠下棋 (Auto-click best move)")
        
        ttk.Radiobutton(auto_play_frame, text="固定方 (Fixed)", variable=self.autoplay_mode_var, value="fixed").pack(side="left", padx=(10, 0))
        ttk.Radiobutton(auto_play_frame, text="自己對戰 (Self-Play)", variable=self.autoplay_mode_var, value="self").pack(side="left")
        row_idx += 1

        # Auto-Detect Checkbox
        self.auto_detect_opponent_var = tk.BooleanVar(value=False)
        chk_autodetect_opp = ttk.Checkbutton(control_frame, text="自動偵測對手 (Auto-Detect Opponent)", variable=self.auto_detect_opponent_var)
        chk_autodetect_opp.grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=5)
        ToolTip(chk_autodetect_opp, "對手下棋後自動開始分析 (Auto-start on opponent move)")
        row_idx += 1

        # Buttons Frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row_idx, column=0, columnspan=3, pady=10, sticky="ew")

        # Start Button
        self.btn_analyze = ttk.Button(btn_frame, text="開始分析 (Start)", command=self.start_analysis_thread)
        self.btn_analyze.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ToolTip(self.btn_analyze, "快速鍵 (Shortcut): Enter")

        # Stop Button
        self.btn_stop = ttk.Button(btn_frame, text="停止 (Stop)", command=self.stop_analysis, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ToolTip(self.btn_stop, "快速鍵 (Shortcut): Esc")

        # New Game Button
        self.btn_new_game = ttk.Button(btn_frame, text="新遊戲 (New Game)", command=self.new_game)
        self.btn_new_game.pack(side="left", fill="x", expand=True, padx=(5, 0))
        ToolTip(self.btn_new_game, "快速鍵 (Shortcut): Ctrl+N")

        # Shortcuts
        self.bind('<Return>', lambda e: self.start_analysis_thread())
        self.bind('<Escape>', lambda e: self.stop_analysis())
        self.bind('<Control-n>', lambda e: self.new_game())

        # 2. Info Frame
        info_frame = ttk.LabelFrame(self, text="分析結果 (Analysis)", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # Status Label (Warming Up / Ready)
        self.lbl_status = ttk.Label(info_frame, text="狀態 (Status): 等待熱機... (Warming Up...)", font=("Arial", 10, "italic"), foreground="red")
        self.lbl_status.pack(anchor="w", pady=(0, 5))

        self.lbl_score = ttk.Label(info_frame, text="評分 (Score): --", font=("Arial", 12, "bold"))
        self.lbl_score.pack(anchor="w")
        
        self.lbl_bestmove = ttk.Label(info_frame, text="最佳著法 (Best Move): --", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_bestmove.pack(anchor="w")
        
        self.lbl_pv = ttk.Label(info_frame, text="變例 (PV): --", wraplength=550)
        self.lbl_pv.pack(anchor="w", fill="x")

        # 3. Board Canvas
        self.canvas_size = 400
        self.square_size = self.canvas_size // 8
        
        # Try to load images
        self.load_piece_images()
        
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
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
            if not hasattr(self, 'original_pil_images'):
                self.original_pil_images = {}
                script_dir = os.path.dirname(os.path.abspath(__file__))
                template_dir = os.path.join(script_dir, "templates")
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
            self.engine.send_command("go depth 2")

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

    def _auto_detect_loop(self):
        """Background thread to detect when the opponent has moved."""
        print("[INFO] Started Auto-Detect Opponent loop...")
        self.auto_detecting_opponent = True
        
        def update_status(text, color):
            self.lbl_status.config(text=text, foreground=color)
            
        self.after(0, update_status, "狀態 (Status): 等待對手下棋... (Waiting for opponent)", "blue")

        # Wait a moment before taking the baseline FEN so animations can finish
        # Reduced from 0.5 to 0.1 to catch fast responses
        time.sleep(0.1)
        
        try:
            # We assume recognizer exists because we just finished our own analysis
            baseline_fen_raw = self.recognizer.get_fen_from_screen(
                player_color=self.my_color_var.get(),
                active_player=self.side_var.get(), # Now opponent's side
                castling='-', en_passant='-'
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
                new_fen_raw = self.recognizer.get_fen_from_screen(
                    player_color=self.my_color_var.get(),
                    active_player=self.side_var.get(),
                    castling='-', en_passant='-'
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

            import screen_recognizer
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
                en_passant='-' # Placeholder
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
                time_limit = int(self.time_var.get())
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
                
                if msg["type"] == "info":
                    if self.warming_up: continue

                    # Update Info
                    if "score_type" in msg:
                        s_type = msg["score_type"]
                        s_val = msg["score_val"]
                        display_score = f"{s_val} cp" if s_type == "cp" else f"Mate in {s_val}"
                        self.lbl_score.config(text=f"評分 (Score): {display_score}")
                    
                    if "pv" in msg:
                        self.lbl_pv.config(text=f"變例 (PV): {msg['pv']}")

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
                    
                    # Highlight move on board
                    try:
                        move = chess.Move.from_uci(display_move)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            self.draw_board()
                    except:
                        pass
                    
                    # Signal that analysis is done
                    self.message_queue.put({"type": "analysis_finished"})
                    
                    # --- Automation Logic ---
                    # Execute auto play if enabled
                    if self.auto_play_var.get():
                        # If in fixed mode, only auto-play if the current side to move matches "My Color"
                        if self.autoplay_mode_var.get() == "self" or self.side_var.get() == self.my_color_var.get():
                            self.execute_auto_play(display_move)
                        else:
                            print(f"[INFO] 略過自動下棋：目前為固定方對戰模式，且輪到對手 ({self.side_var.get()})。")
                        
                    # Detect Game Over AFTER executing the move (ensures checkmate move is played)
                    if self.board.is_game_over():
                        outcome = self.board.outcome()
                        print(f"[INFO] 遊戲結束: {outcome.result()} - {outcome.termination.name}")
                        reason = ""
                        if self.board.is_checkmate():
                            reason = "將殺 (Checkmate)"
                        elif self.board.is_stalemate():
                            reason = "逼和 (Stalemate)"
                        else:
                            reason = "平局 (Draw)"
                            
                        self.lbl_status.config(text=f"狀態 (Status): 遊戲結束 ({reason}) - {outcome.result()}", foreground="purple")
                        self.auto_play_var.set(False)
                        self.auto_detect_opponent_var.set(False)
                        self.analyzing = False # Stop further analysis
                        continue

                    # If auto detect is enabled, wait for opponent
                    if self.auto_detect_opponent_var.get():
                        if self.autoplay_mode_var.get() == "self":
                           # In self-play mode, we don't wait for the opponent. We just flip the board side and trigger our own analysis again.
                           self.toggle_side_to_move()
                           print("[INFO] 自己對戰模式：正在自動觸發下一回合分析...")
                           # Tiny delay to allow pieces to visually move
                           time.sleep(0.8)
                           self.after(0, self.start_analysis_thread)
                        else:
                           self.toggle_side_to_move()
                           # Start background thread to detect move
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
                color = BOARD_COLORS[(rank + file) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Labels (Coordinates)
                label_color = "#B58863" if (rank + file) % 2 == 1 else "#F0D9B5"
                if display_file == 0: # Left edge
                    self.canvas.create_text(x1 + 2, y1 + 2, text=str(rank + 1), anchor="nw", font=("Arial", 8), fill=label_color)
                if display_rank == 7: # Bottom edge
                    self.canvas.create_text(x2 - 2, y2 - 2, text=chr(ord('a') + file), anchor="se", font=("Arial", 8), fill=label_color)

                # Piece
                square_idx = chess.square(file, rank)
                piece = self.board.piece_at(square_idx)
                
                if piece:
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

    def on_closing(self):
        self.engine.quit_engine()
        self.destroy()

if __name__ == "__main__":
    app = ChessVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
