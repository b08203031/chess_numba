import os
import math
import cv2
import numpy as np
import mss
from PIL import Image

PIECE_CLASSES = [
    None,
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]


class ScreenRecognizer:
    """
    負責從螢幕截圖中自動定位任意配色/邊框的 8x8 棋盤，並解析出棋局狀態 (FEN)。
    支援：
      1. Universal (AI) 模式：8x8 鞍點格網定位 + 輕量 ONNX 13 類 CNN (cv2.dnn 批次推理)
      2. Default / Custom 模板模式：改進型四角背景色自適應去背 + 模板匹配
    """

    def __init__(self, templates_path=None, mode="universal"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if templates_path is None:
            templates_path = os.path.abspath(os.path.join(script_dir, "..", "data", "templates"))

        self.default_templates_path = templates_path
        self.custom_templates_path = os.path.abspath(os.path.join(script_dir, "..", "data", "templates", "custom"))
        self.onnx_model_path = os.path.abspath(os.path.join(script_dir, "..", "data", "models", "piece_classifier.onnx"))

        self.M = None                 # Perspective transform: screenshot -> warped 800x800
        self.inv_M = None             # Inverse transform: warped 800x800 -> screenshot
        self.manual_calibrated = False
        self.board_side_length = 800
        self.capture_offset = (0, 0)  # Virtual desktop (left, top) offset for multi-monitor setups
        self._sct = None

        self.dnn_net = None
        self._load_dnn_model()

        self.piece_templates = {}
        try:
            self.piece_templates = self._load_templates(self.default_templates_path)
        except Exception as e:
            print(f"[WARN] 載入預設模板略過: {e}")

        if mode == "universal" and self.dnn_net is None:
            self.mode = "default"
        else:
            self.mode = mode

    def _load_dnn_model(self):
        if os.path.isfile(self.onnx_model_path):
            try:
                # 使用記憶體緩衝區載入 ONNX，避免 Windows 中文路徑 (例如「桌面」) 導致 OpenCV C++ ifstream 失敗
                with open(self.onnx_model_path, "rb") as f:
                    onnx_buf = np.frombuffer(f.read(), dtype=np.uint8)
                net = cv2.dnn.readNetFromONNX(onnx_buf)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.dnn_net = net
            except Exception as e:
                print(f"[WARN] 無法載入通用 ONNX 棋子分類器 ({self.onnx_model_path}): {e}")
                self.dnn_net = None

    def set_recognition_mode(self, mode: str):
        """
        切換辨識模式：'universal' | 'default' | 'custom'
        """
        mode_lower = mode.lower()
        if "universal" in mode_lower or "ai" in mode_lower or "通用" in mode_lower:
            if self.dnn_net is None:
                self._load_dnn_model()
            if self.dnn_net is None:
                raise FileNotFoundError(f"找不到通用模型檔案: {self.onnx_model_path}")
            self.mode = "universal"
        elif "custom" in mode_lower or "自訂" in mode_lower:
            self.piece_templates = self._load_templates(self.custom_templates_path)
            self.mode = "custom"
        else:
            self.piece_templates = self._load_templates(self.default_templates_path)
            self.mode = "default"

    def invalidate_board_cache(self):
        """清除快取的棋盤投影矩陣，強制下一次重新搜尋螢幕上的棋盤。"""
        self.M = None
        self.inv_M = None
        self.manual_calibrated = False

    def _load_templates(self, path):
        """
        從指定路徑載入 12 種棋子的模板圖片 (wP.png .. bK.png)。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = path if os.path.isabs(path) else os.path.join(script_dir, path)

        templates = {}
        pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
        for piece in pieces:
            file_path = os.path.join(base_path, f"{piece}.png")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"找不到模板檔案: {file_path}")
            with open(file_path, "rb") as img_stream:
                img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
                template = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            if template is None:
                raise FileNotFoundError(f"無法解碼模板檔案: {file_path}")

            if template.ndim == 3 and template.shape[2] == 4:
                templates[piece] = {
                    'template': cv2.cvtColor(template, cv2.COLOR_BGRA2BGR),
                    'mask': template[:, :, 3],
                }
            else:
                templates[piece] = {
                    'template': template,
                    'mask': None,
                }
        return templates

    def capture_screen(self, region=None):
        """
        擷取螢幕畫面（預設涵蓋完整虛擬桌面 monitors[0]，支援多螢幕與任意螢幕位置）。
        同時紀錄 capture_offset = (left, top) 以供點擊座標與排除視窗精確對齊。
        """
        for attempt in range(2):
            try:
                if self._sct is None:
                    self._sct = mss.mss()
                sct = self._sct
                if region is None:
                    # monitors[0] 為涵蓋所有螢幕的虛擬桌面；單螢幕下與 monitors[1] 相同
                    monitor = sct.monitors[0]
                else:
                    monitor = region

                self.capture_offset = (int(monitor.get("left", 0)), int(monitor.get("top", 0)))
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception:
                try:
                    if self._sct is not None:
                        self._sct.close()
                except Exception:
                    pass
                self._sct = None
                if attempt == 1:
                    raise

    # =========================================================================
    # Universal 8x8 Checkerboard Validation & Lattice Detection
    # =========================================================================

    def _extract_square_corner_colors(self, board_img: np.ndarray) -> np.ndarray:
        """
        從已投影為正方形的棋盤影像中，取樣 64 格各自的四個對角角落背景色中位數。
        取 [5%..16%] 角落區避開中央棋子底座 (即使棋子佔 90% 格寬也不會覆蓋四角) 與最外緣 4% 邊線。
        回傳 shape=(8, 8, 3) float32 Lab 色彩陣列。
        """
        h, w, _ = board_img.shape
        lab = cv2.cvtColor(board_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        sh, sw = h // 8, w // 8
        c0 = max(2, int(sh * 0.05))
        c1 = max(c0 + 2, int(sh * 0.16))

        colors = np.empty((8, 8, 3), dtype=np.float32)
        for r in range(8):
            y0 = r * sh
            for c in range(8):
                x0 = c * sw
                sq = lab[y0:y0 + sh, x0:x0 + sw]
                p1 = sq[c0:c1, c0:c1].reshape(-1, 3)
                p2 = sq[c0:c1, sw - c1:sw - c0].reshape(-1, 3)
                p3 = sq[sh - c1:sh - c0, c0:c1].reshape(-1, 3)
                p4 = sq[sh - c1:sh - c0, sw - c1:sw - c0].reshape(-1, 3)
                patches = np.concatenate([p1, p2, p3, p4], axis=0)
                colors[r, c] = np.median(patches, axis=0)
        return colors

    def score_checkerboard_validity(self, board_img: np.ndarray) -> float:
        """
        計算 board_img 符合 8x8 雙色交替棋盤格的信心分數 (0.0 ~ 1.0)。
        同時驗證全部 8 列與 8 行的雙色交替對比，徹底排除上下/左右偏移 1~2 格切入背景的偽對齊。
        """
        if board_img is None or board_img.size == 0:
            return 0.0
        colors = self._extract_square_corner_colors(board_img)
        parity = (np.indices((8, 8)).sum(axis=0) % 2)
        even_cols = colors[parity == 0]
        odd_cols = colors[parity == 1]

        med_even = np.median(even_cols, axis=0)
        med_odd = np.median(odd_cols, axis=0)
        contrast = float(np.linalg.norm(med_even - med_odd))
        if contrast < 5.0:
            return 0.0

        d_to_even = np.linalg.norm(colors - med_even, axis=2)
        d_to_odd = np.linalg.norm(colors - med_odd, axis=2)

        correct_even = (d_to_even[parity == 0] < d_to_odd[parity == 0]).sum()
        correct_odd = (d_to_odd[parity == 1] < d_to_even[parity == 1]).sum()
        match_ratio = float(correct_even + correct_odd) / 64.0

        # Per-row and per-column alternation check: if a candidate box overhangs onto solid webpage
        # background or player bars, those overhanging rows/cols will have near-zero alternation!
        active_rows = 0
        for r in range(8):
            r_even = np.median(colors[r, parity[r] == 0], axis=0)
            r_odd = np.median(colors[r, parity[r] == 1], axis=0)
            if float(np.linalg.norm(r_even - r_odd)) >= 0.35 * contrast:
                active_rows += 1

        active_cols = 0
        for c in range(8):
            c_even = np.median(colors[parity[:, c] == 0, c], axis=0)
            c_odd = np.median(colors[parity[:, c] == 1, c], axis=0)
            if float(np.linalg.norm(c_even - c_odd)) >= 0.35 * contrast:
                active_cols += 1

        line_coverage = (active_rows + active_cols) / 16.0
        return match_ratio * (0.45 + 0.55 * (line_coverage ** 2))

    def is_valid_warped_board(self, board_img: np.ndarray, min_ratio: float = 0.72) -> bool:
        return self.score_checkerboard_validity(board_img) >= min_ratio

    def _mask_exclude_rect(self, img: np.ndarray, exclude_rect):
        """將輔助視窗自身區域轉為中性灰，避免偵測到輔助視窗內的預覽小棋盤。"""
        out = img.copy()
        if exclude_rect is None:
            return out, None
        ex_x, ex_y, ex_w, ex_h = exclude_rect
        off_x, off_y = self.capture_offset
        x1 = max(0, int(ex_x - off_x))
        y1 = max(0, int(ex_y - off_y))
        x2 = min(out.shape[1], int(ex_x + ex_w - off_x))
        y2 = min(out.shape[0], int(ex_y + ex_h - off_y))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 127
            return out, (x1, y1, x2, y2)
        return out, None

    def _find_board_by_saddle_lattice(self, detect_img: np.ndarray, local_excl=None):
        """
        通用 2D 棋盤定位演算法（免顏色設定、免外框、支援圓角與無邊界棋盤）：
        1. 在 CIE Lab 色彩空間計算多尺度棋盤鞍點響應 S(y,x) 與相位向量 v(y,x)
        2. 萃取非極大值抑制峰值，尋找相位交替 (-1)^(u+v) 的 7x7 內部交點正交格網
        3. 以最小平方法擬合 x_k = x0 + k*sx, y_m = y0 + m*sy (k,m=1..7) 外推精確 8x8 外框四角
        """
        H, W, _ = detect_img.shape
        max_dim = max(H, W)
        ds = 1100.0 / max_dim if max_dim > 1100 else 1.0
        if ds < 1.0:
            Ws, Hs = int(round(W * ds)), int(round(H * ds))
            work = cv2.resize(detect_img, (Ws, Hs), interpolation=cv2.INTER_AREA)
        else:
            Ws, Hs = W, H
            work = detect_img

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2Lab).astype(np.float32)

        best_saddle = np.zeros((Hs, Ws), dtype=np.float32)
        best_vec = np.zeros((Hs, Ws, 3), dtype=np.float32)

        for r_box in (3, 6):
            ksize = 2 * r_box + 1
            sh = r_box + 2
            if Hs <= 2 * sh + 4 or Ws <= 2 * sh + 4:
                continue
            box = cv2.boxFilter(lab, ddepth=-1, ksize=(ksize, ksize), normalize=True)
            A = box[0:Hs - 2 * sh, 0:Ws - 2 * sh]
            B = box[0:Hs - 2 * sh, 2 * sh:Ws]
            C = box[2 * sh:Hs, 0:Ws - 2 * sh]
            D = box[2 * sh:Hs, 2 * sh:Ws]

            diag1 = np.linalg.norm(A - D, axis=2)
            diag2 = np.linalg.norm(B - C, axis=2)
            cvec = 0.5 * ((A + D) - (B + C))
            cmag = np.linalg.norm(cvec, axis=2)
            score = np.maximum(0.0, cmag - 1.10 * (diag1 + diag2))

            cur_s = np.zeros((Hs, Ws), dtype=np.float32)
            cur_s[sh:Hs - sh, sh:Ws - sh] = score
            mask_better = cur_s > best_saddle
            best_saddle = np.where(mask_better, cur_s, best_saddle)
            cur_v = np.zeros((Hs, Ws, 3), dtype=np.float32)
            cur_v[sh:Hs - sh, sh:Ws - sh] = cvec
            best_vec[mask_better] = cur_v[mask_better]

        if local_excl is not None:
            ex1, ey1, ex2, ey2 = [int(round(v * ds)) for v in local_excl]
            ex1, ey1 = max(0, ex1 - 4), max(0, ey1 - 4)
            ex2, ey2 = min(Ws, ex2 + 4), min(Hs, ey2 + 4)
            best_saddle[ey1:ey2, ex1:ex2] = 0.0

        # Non-maximum suppression
        dil = cv2.dilate(best_saddle, np.ones((9, 9), dtype=np.uint8))
        nms_mask = (best_saddle == dil) & (best_saddle > 4.0)
        ys, xs = np.nonzero(nms_mask)
        if len(xs) < 14:
            return None

        scores = best_saddle[ys, xs]
        if len(scores) > 280:
            top_idx = np.argpartition(scores, -280)[-280:]
            xs, ys, scores = xs[top_idx], ys[top_idx], scores[top_idx]
        order = np.argsort(-scores)
        xs = xs[order].astype(np.float32)
        ys = ys[order].astype(np.float32)
        vecs = best_vec[ys.astype(np.int32), xs.astype(np.int32)]

        # Estimate candidate step sizes s from axis-aligned neighbor pairs with opposite phase
        min_step = max(14.0, 20.0 * ds)
        max_step = min(float(min(Ws, Hs)) / 7.2, 220.0 * ds)
        if max_step <= min_step:
            return None

        step_votes = []
        N = len(xs)
        for i in range(N):
            # Horizontal opposite-phase neighbors
            dy = np.abs(ys - ys[i])
            dx = xs - xs[i]
            dots = vecs @ vecs[i]
            h_mask = (dy <= 3.5) & (dx >= min_step) & (dx <= max_step) & (dots < 0)
            for dval in dx[h_mask]:
                step_votes.append(float(dval))
            # Vertical opposite-phase neighbors
            dx_v = np.abs(xs - xs[i])
            dy_v = ys - ys[i]
            v_mask = (dx_v <= 3.5) & (dy_v >= min_step) & (dy_v <= max_step) & (dots < 0)
            for dval in dy_v[v_mask]:
                step_votes.append(float(dval))

        if len(step_votes) < 6:
            return None

        # Cluster step_votes into candidate steps
        votes_arr = np.array(step_votes, dtype=np.float32)
        bins = np.arange(min_step - 1.0, max_step + 3.0, 2.0)
        hist, edges = np.histogram(votes_arr, bins=bins)
        # Smooth 3-bin window
        smooth = np.convolve(hist, [1, 1, 1], mode="same")
        cand_bins = np.argsort(-smooth)[:6]
        cand_steps = []
        for b_idx in cand_bins:
            if smooth[b_idx] < 3:
                continue
            lo, hi = edges[max(0, b_idx - 1)], edges[min(len(edges) - 1, b_idx + 2)]
            in_bin = votes_arr[(votes_arr >= lo) & (votes_arr <= hi)]
            if len(in_bin) > 0:
                s_est = float(np.median(in_bin))
                if all(abs(s_est - prev_s) > 2.5 for prev_s in cand_steps):
                    cand_steps.append(s_est)

        lattice_hypotheses = []

        for s_est in cand_steps:
            tol = max(2.5, 0.14 * s_est)
            for a_idx in range(min(N, 60)):
                xa, ya = xs[a_idx], ys[a_idx]
                va = vecs[a_idx]
                u_float = (xs - xa) / s_est
                v_float = (ys - ya) / s_est
                u_int = np.round(u_float).astype(np.int32)
                v_int = np.round(v_float).astype(np.int32)

                res_x = np.abs(xs - (xa + u_int * s_est))
                res_y = np.abs(ys - (ya + v_int * s_est))
                dots = vecs @ va
                expected_sign = np.where(((u_int + v_int) % 2) == 0, 1.0, -1.0)
                valid = (
                    (res_x <= tol)
                    & (res_y <= tol)
                    & (np.abs(u_int) <= 7)
                    & (np.abs(v_int) <= 7)
                    & (dots * expected_sign > 0)
                )
                if np.count_nonzero(valid) < 12:
                    continue

                u_val = u_int[valid]
                v_val = v_int[valid]
                x_val = xs[valid]
                y_val = ys[valid]
                sc_val = scores[valid]

                u_min, u_max = int(u_val.min()), int(u_val.max())
                v_min, v_max = int(v_val.min()), int(v_val.max())
                # Enumerate ALL valid 7x7 lattice placements (including when 1-2 outer piece ranks
                # occlude edge saddle rows, e.g. 5 interior rows fitting at offset 1..5, 2..6, or 3..7!)
                u0_lo = max(-6, u_max - 6)
                u0_hi = min(0, u_min)
                v0_lo = max(-6, v_max - 6)
                v0_hi = min(0, v_min)
                if u0_lo > u0_hi:
                    u0_lo, u0_hi = max(-6, u_min), min(0, u_max - 3)
                if v0_lo > v0_hi:
                    v0_lo, v0_hi = max(-6, v_min), min(0, v_max - 3)

                for u0 in range(u0_lo, u0_hi + 1):
                    for v0 in range(v0_lo, v0_hi + 1):
                        in_win = (u_val >= u0) & (u_val <= u0 + 6) & (v_val >= v0) & (v_val <= v0 + 6)
                        if np.count_nonzero(in_win) < 12:
                            continue
                        uw = u_val[in_win] - u0 + 1  # 1..7
                        vw = v_val[in_win] - v0 + 1  # 1..7
                        xw = x_val[in_win]
                        yw = y_val[in_win]
                        sw = sc_val[in_win]

                        cell_map = {}
                        for k_i in range(len(uw)):
                            key = (int(uw[k_i]), int(vw[k_i]))
                            if key not in cell_map or sw[k_i] > cell_map[key][2]:
                                cell_map[key] = (float(xw[k_i]), float(yw[k_i]), float(sw[k_i]))

                        count = len(cell_map)
                        if count < 12:
                            continue
                        cols_seen = len({k[0] for k in cell_map})
                        rows_seen = len({k[1] for k in cell_map})
                        if cols_seen < 3 or rows_seen < 3:
                            continue

                        ks = np.array([k[0] for k in cell_map], dtype=np.float64)
                        ms = np.array([k[1] for k in cell_map], dtype=np.float64)
                        x_pts = np.array([v[0] for v in cell_map.values()], dtype=np.float64)
                        y_pts = np.array([v[1] for v in cell_map.values()], dtype=np.float64)

                        sx, x0 = np.polyfit(ks, x_pts, 1)
                        sy, y0 = np.polyfit(ms, y_pts, 1)
                        if sx <= 0 or sy <= 0 or not (0.93 <= sx / sy <= 1.07):
                            continue

                        bx1, by1 = x0 / ds, y0 / ds
                        bx2, by2 = (x0 + 8.0 * sx) / ds, (y0 + 8.0 * sy) / ds
                        margin_tol = 6.0
                        if bx1 < -margin_tol or by1 < -margin_tol or bx2 > W + margin_tol or by2 > H + margin_tol:
                            continue
                        bx1, by1 = max(0.0, bx1), max(0.0, by1)
                        bx2, by2 = min(float(W), bx2), min(float(H), by2)
                        area = (bx2 - bx1) * (by2 - by1)
                        if area < 32000:
                            continue

                        # Reject if center falls inside excluded assistant window
                        if local_excl is not None:
                            cx_b, cy_b = 0.5 * (bx1 + bx2) * ds, 0.5 * (by1 + by2) * ds
                            ex1_s, ey1_s, ex2_s, ey2_s = [v * ds for v in local_excl]
                            if ex1_s <= cx_b <= ex2_s and ey1_s <= cy_b <= ey2_s:
                                continue

                        pre_score = (count / 49.0) * math.sqrt(area)
                        lattice_hypotheses.append((pre_score, bx1, by1, bx2, by2))

        if not lattice_hypotheses:
            return None

        # Deduplicate near-identical bounding boxes (within 6px) and evaluate exact 8x8 checkerboard score
        lattice_hypotheses.sort(key=lambda item: -item[0])
        unique_boxes = []
        for pre_sc, bx1, by1, bx2, by2 in lattice_hypotheses:
            if any(
                abs(bx1 - ux1) < 6 and abs(by1 - uy1) < 6 and abs(bx2 - ux2) < 6 and abs(by2 - uy2) < 6
                for _, ux1, uy1, ux2, uy2 in unique_boxes
            ):
                continue
            unique_boxes.append((pre_sc, bx1, by1, bx2, by2))
            if len(unique_boxes) >= 24:
                break

        best_candidate = None
        best_cand_metric = -1.0
        for pre_sc, bx1, by1, bx2, by2 in unique_boxes:
            pts = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
            # Fast 256x256 warp for candidate disambiguation
            dst_small = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32)
            M_small = cv2.getPerspectiveTransform(pts, dst_small)
            warped_small = cv2.warpPerspective(detect_img, M_small, (256, 256))
            cb_score = self.score_checkerboard_validity(warped_small)
            if cb_score < 0.65:
                continue
            area = (bx2 - bx1) * (by2 - by1)
            # Strongly prioritize full 8x8 row/col alternation (cb_score^3) so shifted windows never win!
            metric = (cb_score ** 3) * (area ** 0.25) * (0.5 + 0.5 * min(1.0, pre_sc / max(1.0, math.sqrt(area) * 0.5)))
            if metric > best_cand_metric:
                best_cand_metric = metric
                best_candidate = pts

        return best_candidate

    def _warp_from_sorted_rect(self, screenshot: np.ndarray, sorted_rect: np.ndarray):
        side_length = self.board_side_length
        dst = np.array(
            [[0, 0], [side_length - 1, 0], [side_length - 1, side_length - 1], [0, side_length - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(sorted_rect, dst)
        inv_M = cv2.getPerspectiveTransform(dst, sorted_rect)
        warped = cv2.warpPerspective(screenshot, M, (side_length, side_length))
        return warped, M, inv_M

    @staticmethod
    def _sort_quad_points(points: np.ndarray) -> np.ndarray:
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_points = points[np.argsort(angles)]
        tl_index = np.argmin(np.sum(sorted_points, axis=1))
        return np.roll(sorted_points, -tl_index, axis=0).astype("float32")

    def find_board(self, screenshot, exclude_rect=None, verbose=True):
        """
        通用多階段棋盤偵測：
          Stage 1: 無色彩/無邊框限制的 8x8 鞍點正交格網擬合 (支援圓角、無外框、任意配色)
          Stage 2: Canny 四邊形輪廓搜尋 + 8x8 雙色交替驗證
          Stage 3: 傳統木頭色 HSV 遮罩備援
        """
        detect_img, local_excl = self._mask_exclude_rect(screenshot, exclude_rect)

        # Stage 1: Universal 8x8 Saddle Lattice Detector
        lattice_pts = self._find_board_by_saddle_lattice(detect_img, local_excl=local_excl)
        if lattice_pts is not None:
            sorted_rect = self._sort_quad_points(lattice_pts)
            warped, M, inv_M = self._warp_from_sorted_rect(screenshot, sorted_rect)
            if self.score_checkerboard_validity(warped) >= 0.65:
                self.M = M
                self.inv_M = inv_M
                self.manual_calibrated = False
                return warped

        # Stage 2: Canny Quadrilateral Contour + Checkerboard Validation
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        quad_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 32000:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if h > 0 and 0.93 <= float(w) / h <= 1.07:
                    if local_excl is not None:
                        cx, cy = x + w * 0.5, y + h * 0.5
                        if local_excl[0] <= cx <= local_excl[2] and local_excl[1] <= cy <= local_excl[3]:
                            continue
                    quad_candidates.append((area, approx.reshape(4, 2).astype("float32")))

        quad_candidates.sort(key=lambda item: -item[0])
        for _, pts in quad_candidates[:10]:
            sorted_rect = self._sort_quad_points(pts)
            warped, M, inv_M = self._warp_from_sorted_rect(screenshot, sorted_rect)
            if self.score_checkerboard_validity(warped) >= 0.65:
                self.M = M
                self.inv_M = inv_M
                self.manual_calibrated = False
                return warped

        # If a clear single outer square frame exists even with unusual low-contrast theme
        if quad_candidates:
            sorted_rect = self._sort_quad_points(quad_candidates[0][1])
            warped, M, inv_M = self._warp_from_sorted_rect(screenshot, sorted_rect)
            self.M = M
            self.inv_M = inv_M
            self.manual_calibrated = False
            return warped

        # Stage 3: Legacy Wood-Color HSV Fallback
        if verbose:
            print("[INFO] 格網與幾何輪廓偵測未命中，嘗試傳統木紋色彩遮罩備援...")
        hsv = cv2.cvtColor(detect_img, cv2.COLOR_BGR2HSV)
        lower_wood = np.array([10, 80, 50])
        upper_wood = np.array([25, 255, 255])
        board_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if verbose:
                print("錯誤：未能從螢幕找到 8x8 棋盤。")
            return None

        board_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(board_contour) < 32000:
            return None
        x, y, w, h = cv2.boundingRect(board_contour)
        padding = 5
        x, y = max(0, x - padding), max(0, y - padding)
        w = min(screenshot.shape[1] - x, w + 2 * padding)
        h = min(screenshot.shape[0] - y, h + 2 * padding)
        board_roi = screenshot[y:y + h, x:x + w]

        hsv_roi = cv2.cvtColor(board_roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 180]), np.array([180, 50, 255]))
        black_mask = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 255, 80]))
        pieces_mask = cv2.morphologyEx(
            cv2.bitwise_or(white_mask, black_mask),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        board_only_mask = cv2.bitwise_not(pieces_mask)
        contours, _ = cv2.findContours(board_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        points = np.intp(cv2.boxPoints(rect)).reshape(4, 2).astype("float32") + (x, y)
        sorted_rect = self._sort_quad_points(points)
        warped, M, inv_M = self._warp_from_sorted_rect(screenshot, sorted_rect)
        self.M = M
        self.inv_M = inv_M
        self.manual_calibrated = False
        return warped

    def get_warped_board(self, screenshot, validate=True):
        """
        使用快取的投影矩陣 M 快速擷取棋盤。
        若 validate=True 且非手動強制校正，會自動驗證 8x8 棋盤格結構；
        一旦瀏覽器捲動、縮放或視窗移動導致失準，自動回傳 None 觸發無感重定位！
        """
        if self.M is None:
            if self.inv_M is not None:
                ok, M = cv2.invert(self.inv_M)
                if ok:
                    self.M = M
                else:
                    return None
            else:
                return None

        side_length = self.board_side_length
        warped = cv2.warpPerspective(screenshot, self.M, (side_length, side_length))
        if validate and not self.manual_calibrated:
            if not self.is_valid_warped_board(warped, min_ratio=0.66):
                self.invalidate_board_cache()
                return None
        return warped

    # =========================================================================
    # Custom Template Learning & Upgraded Piece Classification
    # =========================================================================

    def learn_pieces_from_board(self, board_img, player_color='w'):
        """
        在起始位置狀態下，從 board_img 自動擷取 12 種棋子的圖示模板並去背儲存至 custom/，
        並自動切換至 'custom' 模式。
        """
        if not os.path.exists(self.custom_templates_path):
            os.makedirs(self.custom_templates_path)

        height, width, _ = board_img.shape
        sq_h, sq_w = height // 8, width // 8

        piece_squares = {
            'wR': (7, 0), 'wN': (7, 1), 'wB': (7, 2), 'wQ': (7, 3), 'wK': (7, 4), 'wP': (6, 0),
            'bR': (0, 0), 'bN': (0, 1), 'bB': (0, 2), 'bQ': (0, 3), 'bK': (0, 4), 'bP': (1, 0),
        }
        if player_color == 'b':
            piece_squares = {
                'wR': (0, 7), 'wN': (0, 6), 'wB': (0, 5), 'wQ': (0, 4), 'wK': (0, 3), 'wP': (1, 7),
                'bR': (7, 7), 'bN': (7, 6), 'bB': (7, 5), 'bQ': (7, 4), 'bK': (7, 3), 'bP': (6, 7),
            }

        for piece, (rank, file) in piece_squares.items():
            square = board_img[rank * sq_h:(rank + 1) * sq_h, file * sq_w:(file + 1) * sq_w]
            h, w, _ = square.shape

            # 取內縮角落像素中位數做精準背景色估計
            c0, c1 = max(1, int(h * 0.05)), max(3, int(h * 0.16))
            corner_patches = np.concatenate([
                square[c0:c1, c0:c1].reshape(-1, 3),
                square[c0:c1, w - c1:w - c0].reshape(-1, 3),
                square[h - c1:h - c0, c0:c1].reshape(-1, 3),
                square[h - c1:h - c0, w - c1:w - c0].reshape(-1, 3),
            ], axis=0)
            bg_color = np.median(corner_patches, axis=0)

            diff = np.sqrt(np.sum((square.astype(np.float32) - bg_color) ** 2, axis=2))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[diff > 24.0] = 255

            kernel_small = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask, kernel_small, iterations=1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded)
            if num_labels > 1:
                best_label = 1
                best_score = -1.0
                center_x, center_y = w * 0.5, h * 0.5
                for idx in range(1, num_labels):
                    area = stats[idx, cv2.CC_STAT_AREA]
                    cx, cy = centroids[idx]
                    dist = math.hypot(cx - center_x, cy - center_y)
                    score = area / (1.0 + dist)
                    if score > best_score:
                        best_score = score
                        best_label = idx

                body_mask = np.zeros_like(mask)
                body_mask[labels == best_label] = 255
                body_mask_dilated = cv2.dilate(body_mask, kernel_small, iterations=2)
                mask = cv2.bitwise_and(mask, body_mask_dilated)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cv2.drawContours(mask, [c], -1, 255, -1)

            b, g, r = cv2.split(square)
            bgra = cv2.merge([b, g, r, mask])
            file_path = os.path.join(self.custom_templates_path, f"{piece}.png")
            _, img_encoded = cv2.imencode('.png', bgra)
            with open(file_path, "wb") as f:
                f.write(img_encoded.tobytes())

        self.piece_templates = self._load_templates(self.custom_templates_path)
        self.mode = "custom"

    def split_into_squares(self, board_img):
        height, width, _ = board_img.shape
        square_h, square_w = height // 8, width // 8
        squares = []
        for i in range(8):
            for j in range(8):
                squares.append(board_img[i * square_h:(i + 1) * square_h, j * square_w:(j + 1) * square_w])
        return squares

    @staticmethod
    def _is_empty_or_center_dot(sq: np.ndarray) -> bool:
        """
        幾何空位與合法步圓點過濾器：
        任何真實西洋棋子 (即使縮小至 0.78 倍) 必有頭部或底座延伸至 [14%..86%] \\ [33%..67%] 環形區；
        而空格或 Chess.com / Lichess 的合法步中心半透明圓點在該環形區完全等於棋格背景色 (< 1.8% 前景)。
        """
        h, w, _ = sq.shape
        if h < 16 or w < 16:
            return False
        c0, c1 = max(1, int(h * 0.05)), max(3, int(h * 0.16))
        corners = np.concatenate([
            sq[c0:c1, c0:c1].reshape(-1, 3),
            sq[c0:c1, w - c1:w - c0].reshape(-1, 3),
            sq[h - c1:h - c0, c0:c1].reshape(-1, 3),
            sq[h - c1:h - c0, w - c1:w - c0].reshape(-1, 3),
        ], axis=0)
        bg = np.median(corners, axis=0)

        r0, r1 = int(h * 0.14), int(h * 0.86)
        k0, k1 = int(w * 0.14), int(w * 0.86)
        sub_diff = np.sqrt(np.sum((sq[r0:r1, k0:k1].astype(np.float32) - bg) ** 2, axis=2))

        # Mask out the inner [33%..67%] center where legal-move indicator dots live
        ir0, ir1 = int(h * 0.33) - r0, int(h * 0.67) - r0
        ik0, ik1 = int(w * 0.33) - k0, int(w * 0.67) - k0
        ring_mask = np.ones(sub_diff.shape, dtype=bool)
        ring_mask[max(0, ir0):max(0, ir1), max(0, ik0):max(0, ik1)] = False

        ring_vals = sub_diff[ring_mask]
        if ring_vals.size == 0:
            return False
        return float((ring_vals > 18.0).mean()) < 0.018

    def _identify_squares_dnn(self, squares) -> list:
        """
        使用幾何前景預篩選 + ONNX 13 類通用分類器，單次批次推理全部非空棋格。
        """
        results = [None] * len(squares)
        non_empty_indices = []
        non_empty_squares = []
        for idx, sq in enumerate(squares):
            if not self._is_empty_or_center_dot(sq):
                non_empty_indices.append(idx)
                non_empty_squares.append(sq)

        if not non_empty_squares:
            return results

        blob = cv2.dnn.blobFromImages(
            non_empty_squares,
            scalefactor=1.0 / 255.0,
            size=(48, 48),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self.dnn_net.setInput(blob)
        logits = self.dnn_net.forward()  # (M, 13)
        if logits.ndim == 1:
            logits = logits[None, :]
        cls_indices = np.argmax(logits, axis=1)
        for out_pos, sq_idx in enumerate(non_empty_indices):
            results[sq_idx] = PIECE_CLASSES[int(cls_indices[out_pos])]
        return results

    def _identify_square_template(self, square_img, threshold=0.48):
        """
        改進版模板匹配（淘汰舊版錯誤的 center_region > 128 COLOR_BIAS）：
        1. 以四角邊緣中位數精準估計該格實際背景色（支援高亮色塊與深淺任意主題）
        2. 若中央區域對背景色差過低（空格或僅有淡色合法步小圓點）直接判空
        3. 以去背前景內部亮度自動區分黑白子，避免淺格黑兵或深格白兵誤判
        """
        if not self.piece_templates:
            return None

        h, w, _ = square_img.shape
        c0, c1 = max(1, int(h * 0.06)), max(3, int(h * 0.18))
        corners = np.concatenate([
            square_img[c0:c1, c0:c1].reshape(-1, 3),
            square_img[c0:c1, w - c1:w - c0].reshape(-1, 3),
            square_img[h - c1:h - c0, c0:c1].reshape(-1, 3),
            square_img[h - c1:h - c0, w - c1:w - c0].reshape(-1, 3),
        ], axis=0)
        bg_bgr = np.median(corners, axis=0)

        # 中央 56% 區域的前景像素比例檢查
        y0, y1 = int(h * 0.22), int(h * 0.78)
        x0, x1 = int(w * 0.22), int(w * 0.78)
        center_bgr = square_img[y0:y1, x0:x1].astype(np.float32)
        fg_diff = np.sqrt(np.sum((center_bgr - bg_bgr) ** 2, axis=2))
        fg_mask = fg_diff > 26.0
        # 合法步中心小圓點通常半徑 < 0.16*S 且對比柔和，或佔中心區域面積小、無垂直延伸
        if np.count_nonzero(fg_mask) < int(center_bgr.shape[0] * center_bgr.shape[1] * 0.10):
            return None

        gray_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        center_gray = gray_square[y0:y1, x0:x1]
        fg_pixels_gray = center_gray[fg_mask]
        # 高分位亮度區分白子內部填充 vs 全黑子填充
        fg_p75 = float(np.percentile(fg_pixels_gray, 75)) if len(fg_pixels_gray) > 0 else 128.0

        best_match = (None, -1.0)
        for piece_code, template_data in self.piece_templates.items():
            template = template_data['template']
            mask = template_data['mask']

            # 縮放模板以對齊當前棋格尺寸 (舊版只縮小不放大，不同解析度會失準)
            if abs(template.shape[0] - h) > 2 or abs(template.shape[1] - w) > 2:
                scale = min(h / template.shape[0], w / template.shape[1])
                if template.shape[0] > h or template.shape[1] > w:
                    scale *= 0.96
                nh = max(8, int(round(template.shape[0] * scale)))
                nw = max(8, int(round(template.shape[1] * scale)))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                template = cv2.resize(template, (nw, nh), interpolation=interp)
                if mask is not None:
                    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

            if template.ndim == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template

            res = cv2.matchTemplate(gray_square, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if not np.isfinite(max_val):
                continue

            # 溫和的前景填充亮度一致性加權 (不依賴棋盤格深淺)
            score = float(max_val)
            is_white_piece = piece_code.startswith('w')
            if fg_p75 > 165.0:
                score += 0.08 if is_white_piece else -0.08
            elif fg_p75 < 95.0:
                score += 0.08 if not is_white_piece else -0.08

            if score > best_match[1]:
                best_match = (piece_code, score)

        if best_match[1] >= threshold:
            return best_match[0]
        return None

    def identify_piece_in_square(self, square_img, threshold=0.48):
        if self.mode == "universal" and self.dnn_net is not None:
            return self._identify_squares_dnn([square_img])[0]
        return self._identify_square_template(square_img, threshold=threshold)

    def identify_board_pieces(self, board_img) -> list:
        """回傳視覺由上而下、由左至右共 64 格的棋子代碼清單 (尚未依黑方視角翻轉)。"""
        squares = self.split_into_squares(board_img)
        if self.mode == "universal" and self.dnn_net is not None:
            return self._identify_squares_dnn(squares)
        return [self._identify_square_template(sq) for sq in squares]

    @staticmethod
    def detect_starting_orientation(visual_pieces_64: list):
        """
        根據畫面上下兩側各兩排的黑白子分佈，在開局/前中期安全推斷視角 ('w' 或 'b')。
        若為殘局或混戰（無法百分百確定），回傳 None 以保留使用者設定。
        """
        if not visual_pieces_64 or len(visual_pieces_64) != 64:
            return None
        top_16 = [p for p in visual_pieces_64[:16] if p is not None]
        bot_16 = [p for p in visual_pieces_64[48:] if p is not None]
        top_w = sum(1 for p in top_16 if p.startswith('w'))
        top_b = sum(1 for p in top_16 if p.startswith('b'))
        bot_w = sum(1 for p in bot_16 if p.startswith('w'))
        bot_b = sum(1 for p in bot_16 if p.startswith('b'))

        if top_b >= 6 and top_w == 0 and bot_w >= 6 and bot_b == 0:
            return 'w'
        if top_w >= 6 and top_b == 0 and bot_b >= 6 and bot_w == 0:
            return 'b'
        return None

    def board_to_fen(self, board_representation):
        fen = ""
        for i in range(8):
            empty_count = 0
            for j in range(8):
                piece = board_representation[i * 8 + j]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen_char = piece[1]
                    if piece[0] == 'b':
                        fen_char = fen_char.lower()
                    fen += fen_char
            if empty_count > 0:
                fen += str(empty_count)
            if i < 7:
                fen += '/'
        return fen

    def get_fen_from_screen(
        self,
        player_color='w',
        active_player='w',
        castling='KQkq',
        en_passant='-',
        halfmove=0,
        fullmove=1,
        exclude_rect=None,
        verbose=True,
        reuse_transform=False,
    ):
        if verbose:
            print("正在擷取螢幕...")
        screenshot = self.capture_screen()

        board_img = None
        if reuse_transform and self.M is not None:
            try:
                board_img = self.get_warped_board(screenshot, validate=True)
            except Exception:
                board_img = None

        if board_img is None:
            if verbose:
                print("正在自動偵測 8x8 棋盤格網...")
            try:
                board_img = self.find_board(screenshot, exclude_rect=exclude_rect, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"棋盤偵測發生錯誤: {e}")
                return None

        if board_img is None:
            if verbose:
                print("無法從螢幕上找到棋盤。")
            return None

        return self.get_fen_from_board_image(
            board_img,
            player_color=player_color,
            active_player=active_player,
            castling=castling,
            en_passant=en_passant,
            halfmove=halfmove,
            fullmove=fullmove,
        )

    def get_fen_from_board_image(
        self,
        board_img,
        player_color='w',
        active_player='w',
        castling='KQkq',
        en_passant='-',
        halfmove=0,
        fullmove=1,
    ):
        board_representation = self.identify_board_pieces(board_img)
        if player_color == 'b':
            board_representation = board_representation[::-1]

        piece_fen = self.board_to_fen(board_representation)
        return f"{piece_fen} {active_player} {castling} {en_passant} {halfmove} {fullmove}"

    def get_move_screen_coords(self, uci_move, player_color='w'):
        """
        將 UCI 步法 (例如 "e2e4") 轉換為 Windows 虛擬桌面上的絕對實體像素座標。
        """
        if self.inv_M is None:
            print("找不到棋盤座標映射矩陣。請確保已成功辨識過一次棋盤。")
            return None

        if len(uci_move) < 4:
            return None

        start_sq = uci_move[0:2]
        end_sq = uci_move[2:4]
        off_x, off_y = self.capture_offset

        def sq_to_center_pt(sq_str):
            file_idx = ord(sq_str[0]) - ord('a')
            rank_idx = int(sq_str[1]) - 1

            if player_color == 'b':
                display_file = 7 - file_idx
                display_rank = rank_idx
            else:
                display_file = file_idx
                display_rank = 7 - rank_idx

            square_size = self.board_side_length / 8.0
            x_warped = display_file * square_size + square_size * 0.5
            y_warped = display_rank * square_size + square_size * 0.5

            pt = np.array([[[x_warped, y_warped]]], dtype="float32")
            pt_img = cv2.perspectiveTransform(pt, self.inv_M)
            return int(round(pt_img[0][0][0])) + off_x, int(round(pt_img[0][0][1])) + off_y

        try:
            return (sq_to_center_pt(start_sq), sq_to_center_pt(end_sq))
        except Exception as e:
            print(f"轉換座標失敗: {e}")
            return None


if __name__ == '__main__':
    import time
    recognizer = ScreenRecognizer()
    print(f"準備在 2 秒後自動偵測棋盤 (模式: {recognizer.mode})...")
    time.sleep(2)
    fen_string = recognizer.get_fen_from_screen()
    if fen_string:
        print(f"\n辨識完成！\nFEN: {fen_string}")
    else:
        print("\n辨識失敗。")
