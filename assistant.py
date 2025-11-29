import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import subprocess
import sys
import os
import time
import chess
import traceback

# --- Configuration & Constants ---
UNICODE_PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    None: ''
}

BOARD_COLORS = [ "#B58863", "#F0D9B5"]  # Light, Dark squares (Wood theme)
HIGHLIGHT_COLOR = "#FFFF00" # Yellow for best move
LAST_MOVE_COLOR = "#BBCB2B" # Greenish for last move

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

            # Parse PV
            if "pv" in parts:
                idx = parts.index("pv")
                info_data["pv"] = " ".join(parts[idx+1:])

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
        
        # State
        self.board = chess.Board()
        self.message_queue = queue.Queue()
        self.recognizer = None # Lazy load
        self.engine = EngineProcess(self.message_queue)
        self.engine.start()
        self.analyzing = False
        self.warming_up = False
        self.is_first_analysis = True
        
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
        ttk.Checkbutton(control_frame, text="自動偵測權利 (Auto-detect Rights)", variable=self.auto_detect_var).grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=5)
        row_idx += 1

        # Castling Rights (New Feature)
        ttk.Label(control_frame, text="王車易位 (Castling Rights):").grid(row=row_idx, column=0, sticky="w")
        self.castling_var = tk.StringVar(value="KQkq")
        ttk.Entry(control_frame, textvariable=self.castling_var, width=10).grid(row=row_idx, column=1, sticky="w")
        row_idx += 1

        # En Passant (New Feature)
        ttk.Label(control_frame, text="吃過路兵 (En Passant):").grid(row=row_idx, column=0, sticky="w")
        self.ep_var = tk.StringVar(value="-")
        ttk.Entry(control_frame, textvariable=self.ep_var, width=5).grid(row=row_idx, column=1, sticky="w")
        row_idx += 1

        # Time Limit
        ttk.Label(control_frame, text="思考時間 (Time Limit ms):").grid(row=row_idx, column=0, sticky="w")
        self.time_var = tk.StringVar(value="10000")
        ttk.Entry(control_frame, textvariable=self.time_var, width=10).grid(row=row_idx, column=1, sticky="w")
        row_idx += 1

        # Buttons Frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row_idx, column=0, columnspan=3, pady=10, sticky="ew")

        # Start Button
        self.btn_analyze = ttk.Button(btn_frame, text="開始分析 (Start)", command=self.start_analysis_thread)
        self.btn_analyze.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Stop Button
        self.btn_stop = ttk.Button(btn_frame, text="停止 (Stop)", command=self.stop_analysis, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # 2. Info Frame
        info_frame = ttk.LabelFrame(self, text="分析結果 (Analysis)", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_score = ttk.Label(info_frame, text="評分 (Score): --", font=("Arial", 12, "bold"))
        self.lbl_score.pack(anchor="w")
        
        self.lbl_bestmove = ttk.Label(info_frame, text="最佳著法 (Best Move): --", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_bestmove.pack(anchor="w")
        
        self.lbl_pv = ttk.Label(info_frame, text="變例 (PV): --", wraplength=550)
        self.lbl_pv.pack(anchor="w", fill="x")

        # 3. Board Canvas
        self.canvas_size = 400
        self.square_size = self.canvas_size // 8
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(padx=10, pady=10)
        
        # Initial Draw
        self.draw_board()

    def _run_warmup(self):
        """Runs a silent short search to warm up the engine JIT."""
        if not self.analyzing:
            print("[INFO] Warming up engine...")
            self.warming_up = True
            # Search startpos for depth 2
            self.engine.send_command("position startpos")
            self.engine.send_command("go depth 2")

    def start_analysis_thread(self):
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
            # We don't enable buttons here; we wait for 'bestmove' from engine

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
                        self.ep_var.set("-") # Default EP to empty as it's hard to guess

                    self.after(0, update_ui_fields, new_rights)

                    # Reconstruct FEN
                    parts = fen_raw.split()
                    parts[2] = new_rights
                    parts[3] = "-"
                    fen_final = " ".join(parts)
                except Exception as e:
                    print(f"Auto-detect rights failed: {e}")
            else:
                # Use User Input
                user_rights = self.castling_var.get().strip()
                if not user_rights: user_rights = "-"
                user_ep = self.ep_var.get().strip()
                if not user_ep: user_ep = "-"

                parts = fen_raw.split()
                parts[2] = user_rights
                parts[3] = user_ep
                fen_final = " ".join(parts)

            self.message_queue.put({"type": "fen_update", "fen": fen_final})

            # Send to Engine
            try:
                time_limit = int(self.time_var.get())
            except:
                time_limit = 10000

            # FORCE CLEAR STATE to prevent hallucinations from previous analyses
            self.engine.send_command("ucinewgame")
            time.sleep(0.05) # Brief pause to ensure processing
            
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
                        print("[INFO] Warmup complete.")
                        continue

                    move_uci = msg["move"]
                    self.lbl_bestmove.config(text=f"最佳著法 (Best Move): {move_uci}")
                    
                    # Highlight move on board
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            self.draw_board()
                    except:
                        pass
                    
                    # Signal that analysis is done
                    self.message_queue.put({"type": "analysis_finished"})

                elif msg["type"] == "fen_update":
                    fen = msg["fen"]
                    try:
                        self.board.set_fen(fen)
                        self.draw_board()
                    except:
                        pass

                elif msg["type"] == "error":
                    messagebox.showerror("Error", msg["message"])

                elif msg["type"] == "analysis_finished":
                    self.analyzing = False
                    self.btn_analyze.config(state="normal")
                    self.btn_stop.config(state="disabled")
                
                elif msg["type"] == "log":
                    print(f"[LOG] {msg['message']}")

        except queue.Empty:
            pass
        
        self.after(100, self._process_queue)

    def draw_board(self):
        self.canvas.delete("all")
        
        is_flipped = (self.my_color_var.get() == 'b')
        
        for rank in range(8):
            for file in range(8):
                # Calculate coordinates
                display_rank = rank if is_flipped else (7 - rank)
                display_file = (7 - file) if is_flipped else file
                
                x1 = display_file * self.square_size
                y1 = display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Color
                color = BOARD_COLORS[(rank + file) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Piece
                square_idx = chess.square(file, rank)
                piece = self.board.piece_at(square_idx)
                
                if piece:
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
