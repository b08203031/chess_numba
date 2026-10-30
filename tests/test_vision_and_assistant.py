import os
import unittest
import queue
import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.screen_recognizer import ScreenRecognizer
from tools.train_piece_cnn import load_all_piece_styles, COORD_FONT
import assistant


def render_synthetic_screen(
    fen: str,
    piece_style: dict,
    light_rgb=(238, 238, 210),
    dark_rgb=(118, 150, 86),
    page_bg_rgb=(38, 36, 33),
    screen_size=(1440, 900),
    board_top_left=(220, 130),
    board_px=560,
    player_color="w",
    rounded_radius=12,
    add_player_bars=True,
    highlights=((4, 3), (6, 3)),
    legal_dots=((5, 2), (4, 4)),
    draw_coords=True,
) -> np.ndarray:
    """Renders a realistic full-screen browser screenshot containing a 2D chessboard."""
    W, H = screen_size
    screen = Image.new("RGB", (W, H), page_bg_rgb)
    draw_scr = ImageDraw.Draw(screen)

    bx, by = board_top_left
    if add_player_bars:
        # Attached top and bottom player info bars right against the board edges
        bar_col = (55, 53, 50) if sum(page_bg_rgb) < 380 else (230, 230, 230)
        draw_scr.rectangle((bx, by - 44, bx + board_px, by), fill=bar_col)
        draw_scr.rectangle((bx, by + board_px, bx + board_px, by + board_px + 44), fill=bar_col)
        # Right-side move list panel
        draw_scr.rectangle((bx + board_px + 24, by, bx + board_px + 300, by + board_px), fill=bar_col)

    sq_s = board_px // 8
    board_pil = Image.new("RGBA", (board_px, board_px), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(board_pil, "RGBA")

    b_obj = chess.Board(fen)

    for disp_r in range(8):
        for disp_c in range(8):
            is_light = ((disp_r + disp_c) % 2 == 0)
            col = light_rgb if is_light else dark_rgb
            if (disp_r, disp_c) in highlights:
                # Blend yellow highlight
                col = tuple(int(0.45 * c + 0.55 * h) for c, h in zip(col, (247, 247, 105)))

            x0, y0 = disp_c * sq_s, disp_r * sq_s
            b_draw.rectangle((x0, y0, x0 + sq_s, y0 + sq_s), fill=col + (255,))

            if draw_coords:
                opp_col = (dark_rgb if is_light else light_rgb) + (210,)
                chess_rank = (disp_r + 1) if player_color == "b" else (8 - disp_r)
                chess_file = chr(ord("h") - disp_c) if player_color == "b" else chr(ord("a") + disp_c)
                if disp_c == 0:
                    b_draw.text((x0 + 3, y0 + 2), str(chess_rank), fill=opp_col, font=COORD_FONT)
                if disp_r == 7:
                    b_draw.text((x0 + sq_s - 12, y0 + sq_s - 18), chess_file, fill=opp_col, font=COORD_FONT)

            chess_f = (7 - disp_c) if player_color == "b" else disp_c
            chess_r = disp_r if player_color == "b" else (7 - disp_r)
            piece = b_obj.piece_at(chess.square(chess_f, chess_r))

            if piece is None:
                if (disp_r, disp_c) in legal_dots:
                    rad = int(sq_s * 0.14)
                    cx, cy = x0 + sq_s // 2, y0 + sq_s // 2
                    b_draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(25, 25, 25, 65))
            else:
                code = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()
                p_rgba = piece_style[code]
                ps = int(sq_s * 0.90)
                p_res = cv2.resize(p_rgba, (ps, ps), interpolation=cv2.INTER_AREA)
                p_pil = Image.fromarray(p_res, mode="RGBA")
                pad = (sq_s - ps) // 2
                board_pil.paste(p_pil, (x0 + pad, y0 + pad), p_pil)

    if rounded_radius > 0:
        mask = Image.new("L", (board_px, board_px), 0)
        m_draw = ImageDraw.Draw(mask)
        m_draw.rounded_rectangle((0, 0, board_px, board_px), radius=rounded_radius, fill=255)
        screen.paste(board_pil.convert("RGB"), (bx, by), mask)
    else:
        screen.paste(board_pil.convert("RGB"), (bx, by))

    return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)


class TestUniversalRecognizerAndAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.styles = load_all_piece_styles(repo_root)
        cls.rec = ScreenRecognizer(mode="universal")

    def test_01_onnx_model_loaded_on_unicode_path(self):
        self.assertIsNotNone(self.rec.dnn_net, "ONNX piece classifier must load cleanly even on Unicode paths")
        self.assertEqual(self.rec.mode, "universal")

    def test_02_universal_board_and_piece_recognition_across_themes(self):
        kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        expected_pieces = kiwipete.split()[0]

        theme_cases = [
            # (style_key, light_rgb, dark_rgb, page_bg, pos, size, player_color, rounded)
            ("local_custom_neo", (238, 238, 210), (118, 150, 86), (48, 46, 43), (180, 110), 560, "w", 14),
            ("python_chess_svg", (222, 227, 230), (140, 162, 173), (255, 255, 255), (300, 150), 520, "w", 0),
            ("wikipedia", (115, 115, 115), (62, 60, 58), (24, 24, 24), (120, 80), 640, "b", 10),
            ("local_default_wood", (240, 217, 181), (181, 136, 99), (245, 245, 245), (400, 180), 480, "w", 0),
        ]
        # Add Lichess Merida & California if cached
        if "lichess_merida" in self.styles:
            theme_cases.append(("lichess_merida", (255, 255, 255), (208, 214, 220), (255, 255, 255), (250, 140), 520, "b", 8))

        for style_key, l_rgb, d_rgb, bg_rgb, pos, b_px, p_col, rad in theme_cases:
            with self.subTest(style=style_key, player_color=p_col):
                scr = render_synthetic_screen(
                    fen=kiwipete,
                    piece_style=self.styles[style_key],
                    light_rgb=l_rgb,
                    dark_rgb=d_rgb,
                    page_bg_rgb=bg_rgb,
                    board_top_left=pos,
                    board_px=b_px,
                    player_color=p_col,
                    rounded_radius=rad,
                )
                self.rec.invalidate_board_cache()
                warped = self.rec.find_board(scr, verbose=False)
                self.assertIsNotNone(warped, f"Failed to detect board for {style_key}")

                fen_out = self.rec.get_fen_from_board_image(warped, player_color=p_col)
                self.assertEqual(
                    fen_out.split()[0],
                    expected_pieces,
                    f"FEN mismatch on {style_key} ({p_col}): got {fen_out.split()[0]}",
                )

                # Verify click coordinates land accurately inside e2 and e4 squares
                coords = self.rec.get_move_screen_coords("e2e4", player_color=p_col)
                self.assertIsNotNone(coords)
                (sx, sy), (ex, ey) = coords
                sq_s = b_px / 8.0
                if p_col == "w":
                    exp_sx = pos[0] + 4.5 * sq_s
                    exp_sy = pos[1] + 6.5 * sq_s
                    exp_ex = pos[0] + 4.5 * sq_s
                    exp_ey = pos[1] + 4.5 * sq_s
                else:
                    exp_sx = pos[0] + 3.5 * sq_s
                    exp_sy = pos[1] + 1.5 * sq_s
                    exp_ex = pos[0] + 3.5 * sq_s
                    exp_ey = pos[1] + 3.5 * sq_s
                self.assertLess(abs(sx - exp_sx), 5.0)
                self.assertLess(abs(sy - exp_sy), 5.0)
                self.assertLess(abs(ex - exp_ex), 5.0)
                self.assertLess(abs(ey - exp_ey), 5.0)

    def test_03_auto_invalidation_on_window_scroll_and_multi_monitor_offset(self):
        start_fen = chess.STARTING_FEN
        scr1 = render_synthetic_screen(
            fen=start_fen,
            piece_style=self.styles["local_custom_neo"],
            board_top_left=(140, 100),
            board_px=520,
            player_color="b",
        )
        self.rec.invalidate_board_cache()
        self.rec.capture_offset = (-1920, 0)  # Simulate secondary left monitor
        try:
            w1 = self.rec.find_board(scr1, verbose=False)
            self.assertIsNotNone(w1)

            # Verify orientation auto-detection detects Black at bottom ('b')
            vis = self.rec.identify_board_pieces(w1)
            self.assertEqual(self.rec.detect_starting_orientation(vis), "b")

            # Verify multi-monitor offset is applied to SetCursorPos coordinates
            (sx, _), _ = self.rec.get_move_screen_coords("e7e5", player_color="b")
            self.assertLess(sx, 0, "Secondary monitor negative X offset must be preserved")

            # Now simulate user scrolling/moving browser window to (520, 240)
            b_after = chess.Board()
            b_after.push_uci("e2e4")
            scr2 = render_synthetic_screen(
                fen=b_after.fen(),
                piece_style=self.styles["local_custom_neo"],
                board_top_left=(520, 240),
                board_px=520,
                player_color="b",
            )
            # Cached M points to old (140, 100) -> get_warped_board(validate=True) must invalidate & return None!
            stale_warp = self.rec.get_warped_board(scr2, validate=True)
            self.assertIsNone(stale_warp, "Stale perspective matrix M must auto-invalidate when board moves")

            # Re-detecting finds the moved board and reads 1. e4 accurately
            w2 = self.rec.find_board(scr2, verbose=False)
            self.assertIsNotNone(w2)
            fen2 = self.rec.get_fen_from_board_image(w2, player_color="b")
            self.assertEqual(fen2.split()[0], b_after.board_fen())
        finally:
            self.rec.capture_offset = (0, 0)

    def test_04_exclude_rect_ignores_assistant_preview_board(self):
        # Put a board inside assistant window (50, 50, 420, 420) AND real game board at (620, 150)
        scr = render_synthetic_screen(
            fen=chess.STARTING_FEN,
            piece_style=self.styles["python_chess_svg"],
            board_top_left=(620, 150),
            board_px=560,
        )
        mini = render_synthetic_screen(
            fen="8/8/8/3k4/4K3/8/8/8 w - - 0 1",
            piece_style=self.styles["python_chess_svg"],
            screen_size=(400, 400),
            board_top_left=(0, 0),
            board_px=400,
            add_player_bars=False,
            rounded_radius=0,
        )
        scr[60:460, 50:450] = mini
        self.rec.invalidate_board_cache()
        warped = self.rec.find_board(scr, exclude_rect=(40, 50, 430, 430), verbose=False)
        self.assertIsNotNone(warped)
        fen_out = self.rec.get_fen_from_board_image(warped, player_color="w")
        self.assertEqual(fen_out.split()[0], chess.STARTING_BOARD_FEN)

    def test_05_uci_position_history_and_2ply_move_matching(self):
        # Verify full repetition history command generation
        b = chess.Board()
        self.assertEqual(assistant.ChessVisionApp.build_uci_position_cmd(b), "position startpos")
        b.push_uci("e2e4")
        b.push_uci("c7c5")
        b.push_uci("g1f3")
        self.assertEqual(
            assistant.ChessVisionApp.build_uci_position_cmd(b),
            "position startpos moves e2e4 c7c5 g1f3",
        )

        # Custom FEN root + moves
        custom_root = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        b2 = chess.Board(custom_root)
        b2.push_uci("e1g1")
        self.assertEqual(
            assistant.ChessVisionApp.build_uci_position_cmd(b2),
            f"position fen {custom_root} moves e1g1",
        )

        # Test 0-ply, 1-ply, and 2-ply legal move sequence matching on a dummy instance
        dummy = object.__new__(assistant.ChessVisionApp)
        dummy.board = chess.Board()

        self.assertEqual(dummy.find_matching_move_sequence(chess.Board()), [])

        t1 = chess.Board()
        t1.push_uci("e2e4")
        seq1 = dummy.find_matching_move_sequence(t1, max_plies=2)
        self.assertEqual([m.uci() for m in seq1], ["e2e4"])

        # 2-ply jump (user move + instant bot reply, preserving en passant square e6!)
        t2 = chess.Board()
        t2.push_uci("d2d4")
        t2.push_uci("e7e5")
        seq2 = dummy.find_matching_move_sequence(t2, max_plies=2)
        self.assertEqual([m.uci() for m in seq2], ["d2d4", "e7e5"])

    def test_06_engine_process_search_id_filters_stale_bestmove(self):
        q = queue.Queue()
        ep = assistant.EngineProcess(q)
        # Simulate two overlapping `go` commands (#1 interrupted by #2)
        ep._inflight_search_ids.append(1)
        ep._inflight_search_ids.append(2)

        ep._parse_uci_output("info depth 8 score cp 34 pv e2e4 e7e5")
        ep._parse_uci_output("bestmove e2e4")  # Stale bestmove from search #1
        ep._parse_uci_output("info depth 10 score cp -18 pv c7c5")
        ep._parse_uci_output("bestmove c7c5")  # Active bestmove from search #2

        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())

        self.assertEqual(msgs[0]["type"], "info")
        self.assertEqual(msgs[0]["search_id"], 1)
        self.assertEqual(msgs[1], {"type": "bestmove", "move": "e2e4", "search_id": 1})
        self.assertEqual(msgs[2]["type"], "info")
        self.assertEqual(msgs[2]["search_id"], 2)
        self.assertEqual(msgs[3], {"type": "bestmove", "move": "c7c5", "search_id": 2})

    def test_07_bug1_auto_detect_on_our_turn_waits_for_player_then_opponent(self):
        """
        Regression test for Bug #1:
        When Auto-Play is OFF and Auto-Detect Opponent is ON on our turn:
          1. Engine bestmove must NOT prematurely mutate self.board.
          2. _auto_detect_manual_play_loop must NOT immediately exit due to self.analyzing==False.
          3. When player plays a move (even different from bestmove) and opponent replies,
             both moves are tracked into PGN and the next turn's analysis triggers automatically.
        """
        import tkinter as tk
        from unittest.mock import MagicMock

        root = tk.Tk()
        root.withdraw()
        try:
            app = object.__new__(assistant.ChessVisionApp)
            app.board = chess.Board()
            app.game_pgn = chess.pgn.Game.from_board(app.board)
            app.game_node = app.game_pgn
            app.message_queue = queue.Queue()
            app._recognizer_lock = assistant.threading.Lock()
            app.analyzing = True
            app.warming_up = False
            app.current_search_id = 1
            app.auto_detecting_opponent = False
            app.best_move_arrow = None

            app.side_var = tk.StringVar(master=root, value="w")
            app.my_color_var = tk.StringVar(master=root, value="w")
            app.auto_play_var = tk.BooleanVar(master=root, value=False)
            app.autoplay_mode_var = tk.StringVar(master=root, value="fixed")
            app.auto_detect_opponent_var = tk.BooleanVar(master=root, value=True)
            app.auto_detect_var = tk.BooleanVar(master=root, value=True)
            app.castling_var = tk.StringVar(master=root, value="KQkq")
            app.ep_var = tk.StringVar(master=root, value="-")

            app.lbl_bestmove = MagicMock()
            app.lbl_status = MagicMock()
            app.btn_analyze = MagicMock()
            app.btn_stop = MagicMock()
            app.notebook = MagicMock()
            app.notebook.index.return_value = 0
            app.draw_board = MagicMock()
            app.log_search_info = MagicMock()
            app.after = lambda delay, fn=None, *a: (fn(*a) if (fn and delay == 0) else None)

            # Intercept _auto_detect_manual_play_loop launch on bestmove
            manual_loop_started = []
            orig_loop = app._auto_detect_manual_play_loop
            app._auto_detect_manual_play_loop = lambda: manual_loop_started.append(True)

            app.message_queue.put({"type": "bestmove", "move": "e2e4", "search_id": 1})
            app._process_queue()

            # 1. Board must STILL be at starting position (player hasn't moved yet!)
            self.assertEqual(app.board.fen(), chess.STARTING_FEN)
            self.assertEqual(app.side_var.get(), "w")
            self.assertEqual(app.best_move_arrow, chess.Move.from_uci("e2e4"))
            self.assertFalse(app.analyzing)
            self.assertEqual(len(manual_loop_started), 1)

            # 2. Now run the real _auto_detect_manual_play_loop against simulated screen frames:
            #    Frame 0 (baseline): startpos
            #    Frames 1-3: player plays d2d4 (different from e2e4!)
            #    Frames 4-6: opponent plays d7d5
            style = self.styles["local_custom_neo"]
            b0 = chess.Board()
            b1 = chess.Board()
            b1.push_uci("d2d4")
            b2 = b1.copy()
            b2.push_uci("d7d5")

            f0 = self.rec.find_board(render_synthetic_screen(b0.fen(), style, add_player_bars=False), verbose=False)
            f1 = self.rec.find_board(render_synthetic_screen(b1.fen(), style, add_player_bars=False), verbose=False)
            f2 = self.rec.find_board(render_synthetic_screen(b2.fen(), style, add_player_bars=False), verbose=False)

            frames = [f0, f1, f1, f1, f2, f2, f2]
            idx_box = [0]

            def fake_capture(_rec):
                i = min(idx_box[0], len(frames) - 1)
                idx_box[0] += 1
                return frames[i]

            app.recognizer = self.rec
            app._capture_warped_board_safe = fake_capture

            triggered_next_fens = []
            app.start_analysis_thread = lambda current_fen=None: triggered_next_fens.append(current_fen)

            orig_loop()

            self.assertEqual([m.uci() for m in app.board.move_stack], ["d2d4", "d7d5"])
            self.assertEqual(app.side_var.get(), "w")
            self.assertEqual(len(triggered_next_fens), 1)
            self.assertFalse(app.auto_detecting_opponent, "auto_detecting_opponent flag must not leak")
        finally:
            root.destroy()

    def test_08_bug2_self_play_chains_both_white_and_black_moves(self):
        """
        Regression test for Bug #2:
        Self-Play (Auto-Play=ON, Mode=self, Auto-Detect Opponent=OFF) must click AND chain
        Move 1 (White) -> Move 2 (Black) -> Move 3 (White) without stopping after Move 1.
        """
        import tkinter as tk
        from unittest.mock import MagicMock

        root = tk.Tk()
        root.withdraw()
        try:
            app = object.__new__(assistant.ChessVisionApp)
            app.board = chess.Board()
            app.game_pgn = chess.pgn.Game.from_board(app.board)
            app.game_node = app.game_pgn
            app.message_queue = queue.Queue()
            app._recognizer_lock = assistant.threading.Lock()
            app.analyzing = True
            app.warming_up = False
            app.current_search_id = 1
            app.auto_detecting_opponent = False
            app.best_move_arrow = None

            app.side_var = tk.StringVar(master=root, value="w")
            app.my_color_var = tk.StringVar(master=root, value="w")
            app.auto_play_var = tk.BooleanVar(master=root, value=True)
            app.autoplay_mode_var = tk.StringVar(master=root, value="self")
            app.auto_detect_opponent_var = tk.BooleanVar(master=root, value=False)  # Default unchecked!
            app.castling_var = tk.StringVar(master=root, value="KQkq")
            app.ep_var = tk.StringVar(master=root, value="-")

            app.lbl_bestmove = MagicMock()
            app.lbl_status = MagicMock()
            app.btn_analyze = MagicMock()
            app.btn_stop = MagicMock()
            app.notebook = MagicMock()
            app.notebook.index.return_value = 0
            app.draw_board = MagicMock()
            app.log_search_info = MagicMock()
            app.after = lambda delay, fn=None, *a: (fn(*a) if (fn and delay == 0) else None)

            clicked_moves = []
            chained_fens = []

            def fake_execute_auto_play(mv_uci, player_color=None, is_self_play=None):
                clicked_moves.append(mv_uci)
                app._post_ui(app.on_our_move_completed)

            def fake_start_analysis(current_fen=None):
                chained_fens.append(current_fen)

            app.execute_auto_play = fake_execute_auto_play
            app.start_analysis_thread = fake_start_analysis

            # Move 1 (White: e2e4)
            app.message_queue.put({"type": "bestmove", "move": "e2e4", "search_id": 1})
            app._process_queue()
            # Wait briefly for daemon click thread to post `ui_call` and drain queue on main thread
            assistant.time.sleep(0.05)
            app._process_queue()

            self.assertEqual(clicked_moves, ["e2e4"])
            self.assertEqual(app.side_var.get(), "b")
            self.assertEqual(len(chained_fens), 1)

            # Move 2 (Black: e7e5, even though my_color_var is still 'w'!)
            app.analyzing = True
            app.current_search_id = 2
            app.message_queue.put({"type": "bestmove", "move": "e7e5", "search_id": 2})
            app._process_queue()
            assistant.time.sleep(0.05)
            app._process_queue()

            self.assertEqual(clicked_moves, ["e2e4", "e7e5"])
            self.assertEqual(app.side_var.get(), "w")
            self.assertEqual(len(chained_fens), 2)
            self.assertEqual([m.uci() for m in app.board.move_stack], ["e2e4", "e7e5"])
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

