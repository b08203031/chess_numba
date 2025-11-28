import tkinter as tk
from tkinter import ttk, font
import threading
import queue
import time
import chess
import subprocess
import sys
import os
import json
from screen_recognizer import ScreenRecognizer

class EngineProcess:
    """
    管理後端西洋棋引擎子進程的類。

    這個類負責啟動、關閉、重啟 `chess_engine.py`（預期由 PyPy 解釋器運行）
    子進程，並通過標準輸入（stdin）和標準輸出（stdout）與其進行通信。
    它還會創建一個背景線程來非同步讀取引擎的標準錯誤輸出（stderr），
    以避免因管道緩衝區滿而導致的死鎖，並能即時顯示引擎的調試信息。
    """
    def __init__(self, analysis_queue, pypy_executable_path=".pypy-venv/Scripts/pypy.exe"):
        """
        初始化並啟動引擎進程。

        Args:
            analysis_queue (queue.Queue): 用於將分析結果從工作線程傳遞回 GUI 主線程的隊列。
            pypy_executable_path (str, optional): PyPy 解釋器的路徑。
        """
        self.pypy_path = pypy_executable_path
        self.process = None
        self.analysis_queue = analysis_queue
        self.stderr_thread = None
        self.start_engine()

    def _read_stderr(self):
        """
        在背景線程中持續讀取引擎的 stderr 輸出。

        這是一個內部方法，旨在防止 stderr 管道被填滿而阻塞引擎。
        讀取到的任何內容都會被直接打印到主程序的主控台。
        """
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stderr.readline()
                if line:
                    # 直接將引擎的日誌輸出到終端機
                    print(f"[引擎日誌] {line.strip()}", file=sys.stderr)
                else:
                    # 如果管道關閉 (例如程序終止)，則跳出迴圈
                    break
            except (IOError, ValueError):
                # 當程序關閉時，readline 可能會引發錯誤
                break
            except Exception as e:
                print(f"讀取引擎 stderr 時發生意外錯誤: {e}", file=sys.stderr)
                break

    def start_engine(self):
        """
        啟動引擎子進程，並設置好與其通信的管道和 stderr 讀取線程。

        Raises:
            RuntimeError: 如果找不到 PyPy 執行檔或啟動過程中發生其他錯誤。
        """
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            self.process = subprocess.Popen(
                [self.pypy_path, "chess_engine.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                env=env
            )
            
            # 啟動 stderr 讀取執行緒
            self.stderr_thread = threading.Thread(target=self._read_stderr)
            self.stderr_thread.daemon = True
            self.stderr_thread.start()

        except FileNotFoundError:
            raise RuntimeError(f"找不到 PyPy 執行檔: {self.pypy_path}。請檢查路徑。")
        except Exception as e:
            raise RuntimeError(f"啟動引擎時發生錯誤: {e}")

    def analyze_fen(self, fen, time_limit=10.0):
        """
        向引擎發送一個 FEN 局面進行分析。

        此方法會將 `fen <fen_string> time <time_limit>` 命令寫入引擎的 stdin，
        然後阻塞並等待從引擎的 stdout 讀取一行 JSON 格式的分析結果。

        Args:
            fen (str): 要分析的 FEN 字符串。
            time_limit (float, optional): 引擎的思考時間上限（秒）。

        Raises:
            RuntimeError: 如果與引擎通信或解析 JSON 時發生錯誤。

        Returns:
            dict: 從引擎返回的、已解析為字典的 JSON 分析結果。
        """
        if not self.process or self.process.poll() is not None:
            self.start_engine()

        try:
            # 命令現在包含思考時間
            command = f"fen {fen} time {time_limit}\n"
            self.process.stdin.write(command)
            self.process.stdin.flush()
            
            # 讀取引擎的 JSON 輸出 (stdout)
            output_line = self.process.stdout.readline().strip()
            
            if not output_line:
                # 如果 stdout 沒有輸出，等待一小段時間讓 stderr 執行緒可能有機會報告錯誤
                time.sleep(0.5) 
                raise RuntimeError("引擎沒有任何輸出 (stdout)。")
            
            # 解析 JSON
            return json.loads(output_line)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"無法解析來自引擎的 JSON: {e}\n引擎輸出: {output_line}")
        except Exception as e:
            # stderr 的錯誤現在由背景執行緒處理，這裡只處理直接的通訊錯誤
            raise RuntimeError(f"與引擎通訊時發生錯誤: {e}")

    def close(self):
        """
        優雅地關閉引擎子進程。

        它會先嘗試發送 `quit` 命令讓引擎自行退出，如果超時則強制終止進程。
        """
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                # 等待子程序自然結束
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
                # 如果無法正常關閉，則強制終止
                self.process.kill()
            finally:
                # 確保子程序被終止
                if self.process.poll() is None:
                    self.process.kill()
        
        # 等待 stderr 執行緒結束
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=1)

    def restart(self):
        """
        強制並立即重啟引擎子進程。

        此方法會先終止當前正在運行的進程（如果有的話），然後再調用 `start_engine` 重新啟動一個。
        """
        try:
            if self.process and self.process.poll() is None:
                self.process.kill()
        except Exception:
            pass
        finally:
            try:
                if self.stderr_thread and self.stderr_thread.is_alive():
                    self.stderr_thread.join(timeout=0.5)
            except Exception:
                pass
        self.start_engine()

    def set_print_search_info(self, enabled: bool):
        """
        向引擎發送命令以啟用或禁用 `PRINT_SEARCH_INFO` 選項。

        Args:
            enabled (bool): `True` 為啟用，`False` 為禁用。
        """
        if not self.process or self.process.poll() is not None:
            self.start_engine()
        try:
            cmd = f"setoption name PRINT_SEARCH_INFO value {'true' if enabled else 'false'}\n"
            self.process.stdin.write(cmd)
            self.process.stdin.flush()
        except Exception as e:
            print(f"設定 PRINT_SEARCH_INFO 失敗: {e}", file=sys.stderr)

class ChessVisionApp:
    """
    一個整合了螢幕辨識和西洋棋引擎分析功能的 Tkinter GUI 應用程式。

    這個應用程式允許用戶：
    1.  從螢幕上實時捕捉西洋棋棋盤的圖像。
    2.  將圖像轉換為 FEN (Forsyth-Edwards Notation) 字符串。
    3.  將 FEN 發送給後端西洋棋引擎（`EngineProcess`）進行分析。
    4.  在 GUI 上顯示分析結果，包括最佳走法、評估分數和主要變例。
    5.  提供一個可視化的棋盤來預覽當前局面。

    它使用一個工作線程來執行耗時的操作（螢幕辨識和引擎分析），
    並通過一個隊列（`queue.Queue`）將結果安全地傳遞回主 GUI 線程，
    從而避免界面凍結。
    """
    def __init__(self, root):
        """
        初始化應用程式的 GUI 組件和後端連接。

        Args:
            root (tk.Tk): Tkinter 的根窗口對象。
        """
        self.root = root
        self.root.title("西洋棋視覺助理 v3.3 (Book Miss Stop & Controls)")
        self.root.geometry("750x450")

        style = ttk.Style(self.root)
        default_font = ('Arial', 11)
        style.configure('.', font=default_font)
        style.configure('TLabelFrame.Label', font=('Arial', 12, 'bold'))

        self.recognizer = ScreenRecognizer(templates_path="templates/")
        self.analysis_queue = queue.Queue()
        self.engine = EngineProcess(self.analysis_queue)
        self.board = chess.Board()
        self.is_first_analysis = True

        self.player_color_var = tk.StringVar(value='w')
        # 當顏色切換時，重新依照視角重畫棋盤
        self.player_color_var.trace_add('write', self._on_player_color_change)
        self.active_turn_var = tk.StringVar(value='w')
        self.castling_var = tk.StringVar(value='KQkq')
        self.fen_var = tk.StringVar(value="點擊「從螢幕分析」開始")
        self.status_var = tk.StringVar(value="歡迎使用！請設定您的顏色，然後點擊分析按鈕。")
        self.print_info_var = tk.BooleanVar(value=True)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        board_frame = ttk.LabelFrame(main_frame, text="棋盤預覽 (Board Preview)", padding="10")
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.board_canvas = tk.Canvas(board_frame, width=300, height=300, highlightthickness=0)
        self.board_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
        self.root.after(50, self.draw_empty_board)

        controls_frame = ttk.Frame(main_frame, padding="10")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, ipadx=5)

        options_frame = ttk.LabelFrame(controls_frame, text="初始設定 (Initial Setup)", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(options_frame, text="你的顏色:").grid(row=0, column=0, sticky="w", pady=3)
        color_frame = ttk.Frame(options_frame)
        ttk.Radiobutton(color_frame, text="白 (W)", variable=self.player_color_var, value='w').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(color_frame, text="黑 (B)", variable=self.player_color_var, value='b').pack(side=tk.LEFT)
        color_frame.grid(row=0, column=1, sticky="w")

        ttk.Label(options_frame, text="輪到誰走:").grid(row=1, column=0, sticky="w", pady=3)
        turn_frame = ttk.Frame(options_frame)
        ttk.Radiobutton(turn_frame, text="白 (W)", variable=self.active_turn_var, value='w').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(turn_frame, text="黑 (B)", variable=self.active_turn_var, value='b').pack(side=tk.LEFT)
        turn_frame.grid(row=1, column=1, sticky="w")

        ttk.Label(options_frame, text="王車易位:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(options_frame, textvariable=self.castling_var, width=10).grid(row=2, column=1, sticky="w")

        ttk.Label(options_frame, text="思考時長(秒):").grid(row=3, column=0, sticky="w", pady=3)
        self.time_limit_var = tk.StringVar(value='10')
        ttk.Entry(options_frame, textvariable=self.time_limit_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="顯示搜尋資訊 (PRINT_SEARCH_INFO)", variable=self.print_info_var, command=self._on_print_info_toggle).grid(row=4, column=0, columnspan=2, sticky="w", pady=3)

        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, pady=5)
        self.analyze_button = ttk.Button(button_frame, text="從螢幕分析 (Analyze Screen)", command=self.start_analysis_thread)
        self.analyze_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.reset_button = ttk.Button(button_frame, text="重設 (Reset)", command=self.reset_state)
        self.reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        fen_frame = ttk.LabelFrame(controls_frame, text="偵測到的 FEN (Detected FEN)", padding="10")
        fen_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(fen_frame, textvariable=self.fen_var, state='readonly', font=('Courier', 10)).pack(fill=tk.X, expand=True)

        analysis_frame = ttk.LabelFrame(controls_frame, text="引擎分析 (Engine Analysis)", padding="10")
        analysis_frame.pack(fill=tk.BOTH, expand=True)
        
        self.analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, height=10, width=45, font=('Arial', 11))
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        self.analysis_text.insert(tk.END, "點擊「從螢幕分析」按鈕開始...")
        self.analysis_text.config(state=tk.DISABLED)

        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.after(100, self.check_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """在關閉 GUI 窗口時被調用，確保後端引擎進程被優雅地關閉。"""
        self.engine.close()
        self.root.destroy()

    def reset_state(self):
        """
        將應用程式和後端引擎重置到初始狀態。

        這包括重啟引擎進程、清空棋盤、重置 GUI 上的所有顯示和狀態變量。
        """
        # 立即停止當次搜尋：重啟引擎並清空狀態
        try:
            self.engine.restart()
        except Exception:
            pass
        self.board.reset()
        self.player_color_var.set('w')
        self.update_gui_from_board()
        self.fen_var.set("點擊「從螢幕分析」開始")
        self.status_var.set("狀態已重設。請設定您的顏色，然後開始分析。")
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "點擊「從螢幕分析」按鈕開始...")
        self.analysis_text.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.NORMAL)
        self.draw_empty_board()
        self.is_first_analysis = True
        print("State has been reset.")

    def update_gui_from_board(self):
        """根據內部 `self.board` 的狀態更新 GUI 上的 FEN、王車易位權和棋盤畫布。"""
        # self.active_turn_var.set('w' if self.board.turn == chess.WHITE else 'b') # 根據使用者要求，註解此行以保持輪走方固定
        self.castling_var.set(self.board.castling_xfen())
        self.fen_var.set(self.board.fen())
        self.draw_board_from_fen(self.board.fen().split(' ')[0])

    def _on_player_color_change(self, *args):
        """當用戶在 GUI 中切換玩家顏色選項時觸發的回調函數。"""
        # 使用者切換視角（白/黑）時，依照新的視角重畫棋盤
        try:
            self.draw_board_from_fen(self.board.fen().split(' ')[0])
        except Exception:
            # 若目前棋盤尚未就緒或 FEN 無效，忽略
            pass

    def draw_empty_board(self):
        """在 GUI 的畫布上繪製一個空的棋盤格。"""
        self.board_canvas.delete("all")
        square_size = min(self.board_canvas.winfo_width(), self.board_canvas.winfo_height()) / 8
        if square_size < 10: return
        colors = ["#F0D9B5", "#B58863"]
        for r in range(8):
            for c in range(8):
                color = colors[(r + c) % 2]
                x1, y1 = c * square_size, r * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def draw_board_from_fen(self, fen_pieces):
        """
        根據 FEN 字符串的棋子佈局部分，在 GUI 畫布上繪製棋子。

        此函數會考慮用戶選擇的視角（`player_color_var`），
        如果選擇黑方，棋盤會被翻轉 180 度。

        Args:
            fen_pieces (str): FEN 字符串的第一部分，描述了棋子的位置。
        """
        self.draw_empty_board()
        square_size = min(self.board_canvas.winfo_width(), self.board_canvas.winfo_height()) / 8
        if square_size < 10: return

        piece_map = {'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚',
                     'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔'}
        piece_font = font.Font(family='Segoe UI Symbol', size=int(square_size * 0.7))

        rows = fen_pieces.split('/')
        for r, row_str in enumerate(rows):
            c = 0
            for char in row_str:
                if char.isdigit():
                    c += int(char)
                else:
                    # 依據玩家視角（白/黑）決定顯示座標是否要旋轉 180 度
                    if self.player_color_var.get() == 'b':
                        display_r = 7 - r
                        display_c = 7 - c
                    else:
                        display_r = r
                        display_c = c

                    x = display_c * square_size + square_size / 2
                    y = display_r * square_size + square_size / 2
                    piece_color = "black" if char.islower() else "white"
                    self.board_canvas.create_text(x, y, text=piece_map.get(char, ''), 
                                                  font=piece_font, fill=piece_color)
                    c += 1

    def start_analysis_thread(self):
        """
        啟動一個新的背景線程來執行分析流程。

        這是 GUI 中「從螢幕分析」按鈕的回調函數。它會禁用按鈕以防止
        重複點擊，並啟動一個新的守護線程來運行 `run_analysis` 方法。
        """
        self.analyze_button.config(state=tk.DISABLED)
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)
        self.status_var.set("正在啟動分析...")
        self.analysis_text.insert(tk.END, "正在啟動分析...\n")
        self.root.update_idletasks()

        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()

    def run_analysis(self):
        """
        在背景線程中執行的主分析流程。

        此方法按順序執行以下操作：
        1.  （首次运行时）等待用戶切換窗口。
        2.  調用 `ScreenRecognizer` 從螢幕獲取 FEN。
        3.  將 FEN 和狀態更新發送到 GUI 隊列。
        4.  調用 `EngineProcess` 對 FEN 進行分析。
        5.  將最終分析結果或任何發生的錯誤發送到 GUI 隊列。
        """
        try:
            if self.is_first_analysis:
                self.analysis_queue.put(("status", "首次分析：請在 2 秒內切換到您的棋盤視窗！"))
                time.sleep(2)
                self.is_first_analysis = False

            self.analysis_queue.put(("status", "正在辨識棋盤..."))
            
            current_active_turn = self.active_turn_var.get()
            current_castling = self.castling_var.get()
            full_fen = self.recognizer.get_fen_from_screen(
                player_color=self.player_color_var.get(),
                active_player=current_active_turn,
                castling=current_castling
            )

            if full_fen is None:
                raise Exception("無法從螢幕辨識棋盤。")
            
            self.analysis_queue.put(("board_update", full_fen))
            
            self.analysis_queue.put(("status", "FEN 辨識成功，正在呼叫 PyPy 引擎分析..."))
            
            # 從 GUI 獲取思考時間，並進行驗證
            try:
                time_limit = float(self.time_limit_var.get())
                if time_limit <= 0:
                    time_limit = 10.0 # 若輸入不合法，使用預設值
                    self.analysis_queue.put(("status", "無效的思考時長，使用預設值 10 秒。"))
            except ValueError:
                time_limit = 10.0 # 若輸入不是數字，使用預設值
                self.analysis_queue.put(("status", "無效的思考時長，使用預設值 10 秒。"))

            analysis_result = self.engine.analyze_fen(full_fen, time_limit)
            
            self.analysis_queue.put(("done", analysis_result))

        except Exception as e:
            self.analysis_queue.put(("error", str(e)))

    def _display_analysis_results(self, data):
        """
        將從引擎接收到的分析結果格式化並顯示在 GUI 上。

        此方法在 GUI 主線程中被 `check_queue` 調用。它負責解析引擎返回的字典，
        將其轉換為人類可讀的格式，並更新文本框和狀態欄。

        Args:
            data (dict): 包含分析結果的字典。
        """
        # --- 1. 獲取分析結果 (安全地處理 None) ---
        best_move_uci = data.get("best_move")
        score_raw = data.get("score", "N/A")
        stopped_after_book_misses = data.get("stopped_after_book_misses", False)
        
        # 將數值分數四捨五入到小數兩位；將殺字串維持原樣
        def _format_score_for_display(score_value):
            if isinstance(score_value, (int, float)):
                return f"{score_value:.2f}"
            if isinstance(score_value, str):
                if "mate" in score_value:
                    return score_value
                try:
                    return f"{float(score_value):.2f}"
                except ValueError:
                    return score_value
            return str(score_value)

        score_display = _format_score_for_display(score_raw)
        time_elapsed = data.get("time_elapsed", 0)
        nodes_searched = data.get("nodes_searched", 0)
        pv = data.get("pv", [])
        nps = int(nodes_searched / max(time_elapsed, 0.001))
        
        # --- 2. 備份當前棋盤狀態，用於後續分析顯示 ---
        board_before_move = self.board.copy()

        # --- 3. 確定要應用的走法 (若已停用搜尋，則不應用任何走法) ---
        move_to_apply_uci = None if stopped_after_book_misses else (pv[0] if pv else best_move_uci)

        if move_to_apply_uci:
            try:
                move = chess.Move.from_uci(move_to_apply_uci)
                if self.board.is_legal(move):
                    self.board.push(move)
                    self.update_gui_from_board() # 更新棋盤 UI
                    self.analysis_text.insert(tk.END, f"已應用最佳走法: {move_to_apply_uci}\n")
                else:
                    self.analysis_text.insert(tk.END, f"錯誤：引擎回傳非法走法 {move_to_apply_uci}。\n")
            except (ValueError, TypeError) as e:
                self.analysis_text.insert(tk.END, f"應用最佳走法 {move_to_apply_uci} 時發生錯誤: {e}\n")

        if stopped_after_book_misses:
            self.analysis_text.insert(tk.END, "--- 已連續兩步未命中開局書；本局後續不再搜尋 ---\n")
        self.analysis_text.insert(tk.END, "--- 分析完成 ---")

        # --- 4. 將 PV (UCI格式) 轉換為人類可讀的 SAN 格式 ---
        pv_san_str = self._format_pv_to_san(pv, board_before_move)

        # --- 5. 格式化並顯示最終結果 ---
        is_mate = isinstance(score_raw, str) and "mate" in score_raw
        if is_mate:
            try:
                mate_in_moves = int(score_raw.split(' ')[1])
                mating_player = "白方" if (board_before_move.turn == chess.WHITE and mate_in_moves > 0) or \
                                        (board_before_move.turn == chess.BLACK and mate_in_moves < 0) else "黑方"
                result_line = f"結果: {mating_player} {abs(mate_in_moves)} 步殺"
                
                # 當引擎回傳將殺時，best_move 可能為空，此時從 PV (主要變例) 的第一步棋推斷出最佳著法
                best_move_display = "無 (終局)"
                if pv:
                    try:
                        first_move_uci = pv[0]
                        move = chess.Move.from_uci(first_move_uci)
                        # 轉換為 SAN 格式 (例如: e4, Nf3) 以方便閱讀
                        if board_before_move.is_legal(move):
                            best_move_display = board_before_move.san(move)
                        else:
                            best_move_display = first_move_uci  # 若轉換失敗，直接顯示 UCI
                    except Exception:
                        best_move_display = pv[0] # 若發生其他錯誤，直接顯示 UCI

                self.analysis_text.insert(tk.END, f"{result_line}\n")
                self.analysis_text.insert(tk.END, f"最佳著法: {best_move_display}\n")
                self.analysis_text.insert(tk.END, f"將殺路徑: {pv_san_str}\n")
            except (ValueError, IndexError) as mate_e:
                 self.analysis_text.insert(tk.END, f"處理將殺資訊時發生錯誤: {mate_e}\n")
        else:
            best_move_display = move_to_apply_uci or 'N/A'
            self.analysis_text.insert(tk.END, f"最佳著法: {best_move_display}\n")
            self.analysis_text.insert(tk.END, f"主要變例 (PV): {pv_san_str} (分數: {score_display})\n")
        
        self.analysis_text.insert(tk.END, f"搜尋時間: {time_elapsed:.2f} 秒, 節點數: {nodes_searched}, NPS: {nps}\n")
        self.status_var.set("分析完成！")
        self.analyze_button.config(state=tk.NORMAL)

    def _on_print_info_toggle(self):
        """當用戶點擊「顯示搜尋資訊」複選框時的回調函數。"""
        enabled = bool(self.print_info_var.get())
        self.engine.set_print_search_info(enabled)
        self.status_var.set(f"PRINT_SEARCH_INFO 設為 {enabled}")

    def _format_pv_to_san(self, pv, board):
        """
        將 UCI 格式的主要變例（PV）列表轉換為人類可讀的標準代數記譜法（SAN）字符串。

        Args:
            pv (list[str]): UCI 格式的走法字符串列表。
            board (chess.Board): PV 開始前的棋盤狀態。

        Returns:
            str: 格式化後的 SAN 字符串，或在轉換失敗時返回原始的 UCI 字符串。
        """
        if not pv:
            return "N/A"
        try:
            temp_board = board.copy()
            san_parts = []
            for i, uci_move in enumerate(pv):
                move = chess.Move.from_uci(uci_move)
                if temp_board.is_legal(move):
                    if temp_board.turn == chess.WHITE:
                        san_parts.append(f"{temp_board.fullmove_number}. {temp_board.san(move)}")
                    else:
                        if i == 0:
                            san_parts.append(f"{temp_board.fullmove_number}. ... {temp_board.san(move)}")
                        else:
                            san_parts.append(temp_board.san(move))
                    temp_board.push(move)
                else:
                    san_parts.append(f"(非法: {uci_move})")
                    break 
            return " ".join(san_parts)
        except Exception as e:
            print(f"產生 SAN PV 時發生錯誤: {e}", file=sys.stderr)
            return ' '.join(pv) # 轉換失敗時，退回顯示原始 UCI

    def check_queue(self):
        """
        定期檢查從工作線程傳來的消息隊列，並在 GUI 主線程中處理它們。

        這是一個輪詢函數，使用 `root.after` 來安排下一次檢查，
        是 Tkinter 中進行線程間通信的標準模式。
        """
        try:
            while True:
                msg_type, data = self.analysis_queue.get_nowait()
                
                if msg_type == "status":
                    self.analysis_text.insert(tk.END, data + "\n")
                    self.status_var.set(data)
                
                elif msg_type == "board_update":
                    try:
                        self.board.set_fen(data)
                        self.update_gui_from_board()
                        self.status_var.set("棋盤狀態已同步！")
                    except ValueError as e:
                        self.analysis_queue.put(("error", f"無效的 FEN: {data}\n{e}"))

                elif msg_type == "done":
                    self._display_analysis_results(data)
                
                elif msg_type == "error":
                    error_msg = f"\n--- 錯誤---\n{data}\n"
                    self.analysis_text.insert(tk.END, error_msg)
                    self.status_var.set("發生錯誤，請檢查日誌。")
                    self.analyze_button.config(state=tk.NORMAL)

                self.analysis_text.see(tk.END)
                self.root.update_idletasks()

        except queue.Empty:
            pass
        
        finally:
            self.root.after(100, self.check_queue)

if __name__ == '__main__':
    root = tk.Tk()
    app = ChessVisionApp(root)
    root.mainloop()
