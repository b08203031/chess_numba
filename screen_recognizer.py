import cv2
import numpy as np
import mss
from PIL import Image
import os
import time

class ScreenRecognizer:
    """
    負責從螢幕截圖中解析出棋局狀態 (FEN)。
    Supports dynamic calibration for different board styles.
    """

    def __init__(self, templates_path="templates/"):
        """
        初始化辨識器。
        
        Args:
            templates_path (str): 存放預設棋子模板圖片的資料夾路徑 (可選)。
        """
        # Load default templates if available, but primarily rely on calibration
        self.piece_templates = {}
        try:
            self.piece_templates = self._load_templates(templates_path)
        except Exception as e:
            # print(f"Warning: Could not load default templates: {e}")
            pass

        # Dictionary to store dynamically learned templates
        # Format: {'wP': template_img, 'bP': template_img, ...}
        self.learned_templates = {}
        self.is_calibrated = False
        self.board_orientation = 'w' # 'w' (White at bottom) or 'b' (Black at bottom)

    def _load_templates(self, path):
        """
        從指定路徑載入12種棋子的模板圖片 (Legacy/Fallback).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(script_dir, path)

        templates = {}
        pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
        
        if not os.path.exists(base_path):
            return templates

        for piece in pieces:
            file_path = os.path.join(base_path, f"{piece}.png")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as img_stream:
                        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
                        template = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    
                    if template is not None:
                        if len(template.shape) == 3 and template.shape[2] == 4:
                            templates[piece] = {
                                'template': cv2.cvtColor(template, cv2.COLOR_BGRA2BGR),
                                'mask': template[:,:,3]
                            }
                        else:
                            templates[piece] = {
                                'template': template,
                                'mask': None
                            }
                except Exception:
                    pass 
        return templates

    def capture_screen(self, region=None):
        """
        擷取螢幕畫面。
        """
        with mss.mss() as sct:
            if region is None:
                if len(sct.monitors) > 1:
                    monitor = sct.monitors[1]
                else:
                    monitor = sct.monitors[0]
            else:
                monitor = region
            
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def find_board(self, screenshot):
        """
        使用通用的邊緣檢測尋找棋盤輪廓 (The largest square-like grid).
        不再依賴特定顏色 (No more wood color dependency).
        """
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Detection
        # Use Canny with automatic parameter tuning or safe defaults
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # 2. Dilate to connect broken edges
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("錯誤：未能找到任何輪廓。")
            return None
        
        # Filter and find the best board candidate
        best_cnt = None
        max_area = 0
        min_area = (screenshot.shape[0] * screenshot.shape[1]) / 64 # At least 1/64 of screen size
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Stricter polygon
            
            # Check if it has 4 corners and looks like a square
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.85 <= aspect_ratio <= 1.15: # Strict Square-ish
                    if area > max_area:
                        max_area = area
                        best_cnt = approx

        if best_cnt is None:
            print("錯誤：未能找到符合條件的棋盤輪廓 (Square-like, large enough).")
            return None

        # 4. Perspective Transform
        # Re-order points: top-left, top-right, bottom-right, bottom-left
        points = best_cnt.reshape(4, 2).astype("float32")
        
        # Use simple sum/diff method to order points
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = points[np.argmin(s)]      # Top-left
        rect[2] = points[np.argmax(s)]      # Bottom-right
        rect[1] = points[np.argmin(diff)]   # Top-right
        rect[3] = points[np.argmax(diff)]   # Bottom-left

        side_length = 800
        dst = np.array([
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(screenshot, M, (side_length, side_length))
        
        return warped

    def split_into_squares(self, board_img):
        height, width, _ = board_img.shape
        square_h, square_w = height // 8, width // 8
        squares = []
        
        # Crop margin to avoid border artifacts (10%)
        margin_h = int(square_h * 0.1)
        margin_w = int(square_w * 0.1)
        
        for i in range(8):
            for j in range(8):
                y1 = i * square_h + margin_h
                y2 = (i + 1) * square_h - margin_h
                x1 = j * square_w + margin_w
                x2 = (j + 1) * square_w - margin_w
                
                # Ensure valid slice
                if y2 > y1 and x2 > x1:
                    square = board_img[y1:y2, x1:x2]
                else:
                    square = board_img[i*square_h:(i+1)*square_h, j*square_w:(j+1)*square_w]
                    
                squares.append(square)
        return squares

    def calibrate(self):
        """
        執行校準程序：
        1. 擷取螢幕
        2. 尋找棋盤
        3. 判斷方向 (白方/黑方在下)
        4. 從標準初始局面中學習棋子樣式
        """
        print("正在執行校準 (Calibrating)...")
        screenshot = self.capture_screen()
        
        try:
            board_img = self.find_board(screenshot)
        except Exception as e:
            print(f"校準錯誤: {e}")
            return False
            
        if board_img is None:
            print("校準失敗：無法找到棋盤。")
            return False

        # Split into 64 squares
        squares = self.split_into_squares(board_img)
        if len(squares) != 64:
            return False

        # --- 3. Determine Orientation ---
        # Compare brightness of pieces on Rank 1-2 vs Rank 7-8 (0-indexed: rows 0-1 vs 6-7)
        # Note: In image coordinates, Row 0 is Top, Row 7 is Bottom.
        
        # Top 2 rows (Screen Top)
        top_pieces_brightness = []
        for r in range(0, 2):
            for c in range(8):
                idx = r * 8 + c
                gray = cv2.cvtColor(squares[idx], cv2.COLOR_BGR2GRAY)
                top_pieces_brightness.append(np.mean(gray))
        
        # Bottom 2 rows (Screen Bottom)
        bottom_pieces_brightness = []
        for r in range(6, 8):
            for c in range(8):
                idx = r * 8 + c
                gray = cv2.cvtColor(squares[idx], cv2.COLOR_BGR2GRAY)
                bottom_pieces_brightness.append(np.mean(gray))

        avg_top = np.mean(top_pieces_brightness)
        avg_bottom = np.mean(bottom_pieces_brightness)
        
        print(f"Debug: Avg Top Brightness: {avg_top}, Avg Bottom Brightness: {avg_bottom}")

        # Assumption: White pieces are lighter (higher brightness) than Black pieces.
        # If Bottom (Rows 6-7) is brighter -> White at Bottom.
        # If Top (Rows 0-1) is brighter -> White at Top (Black at Bottom).
        
        if avg_bottom > avg_top:
            self.board_orientation = 'w' # White at bottom
            print("校準結果：白方在下 (White at Bottom)")
        else:
            self.board_orientation = 'b' # Black at bottom
            print("校準結果：黑方在下 (Black at Bottom)")

        # --- 4. Learn Templates ---
        temp_map = {} # piece_code -> list of images

        if self.board_orientation == 'w':
            # White at Bottom (Standard)
            # Row 0: bR, bN, bB, bQ, bK, bB, bN, bR
            layout_row0 = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR']
            layout_row1 = ['bP'] * 8
            layout_row6 = ['wP'] * 8
            layout_row7 = ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
            
        else:
            # Black at Bottom (Rotated)
            # Row 0 (Top): wR, wN, wB, wK, wQ, wB, wN, wR
            # Note the King/Queen swap relative to files because we scan left-to-right (h1 -> a1)
            layout_row0 = ['wR', 'wN', 'wB', 'wK', 'wQ', 'wB', 'wN', 'wR']
            layout_row1 = ['wP'] * 8
            layout_row6 = ['bP'] * 8
            layout_row7 = ['bR', 'bN', 'bB', 'bK', 'bQ', 'bB', 'bN', 'bR']

        def extract_row(row_idx, layout):
            for col_idx, piece in enumerate(layout):
                square_idx = row_idx * 8 + col_idx
                if piece not in temp_map:
                    temp_map[piece] = []
                temp_map[piece].append(squares[square_idx])

        extract_row(0, layout_row0)
        extract_row(1, layout_row1)
        extract_row(6, layout_row6)
        extract_row(7, layout_row7)

        # Store ALL samples as templates (to handle Light/Dark square backgrounds)
        self.learned_templates = {}
        for p, imgs in temp_map.items():
            self.learned_templates[p] = imgs

        # Learn Empty Squares (Light and Dark)
        self.learned_templates['empty_light'] = []
        self.learned_templates['empty_dark'] = []
        
        empty_light = squares[24]
        if np.std(cv2.cvtColor(empty_light, cv2.COLOR_BGR2GRAY)) > 5.0:
            self.learned_templates['empty_light'].append(empty_light)
            
        empty_dark = squares[25]
        if np.std(cv2.cvtColor(empty_dark, cv2.COLOR_BGR2GRAY)) > 5.0:
            self.learned_templates['empty_dark'].append(empty_dark)

        self.is_calibrated = True
        print("校準完成！已學習棋子樣式。")
        return True

    def identify_piece_in_square(self, square_img, threshold=0.7):
        """
        辨識格子中的棋子，使用校準後的模板。
        """
        if not self.is_calibrated:
             return self._identify_piece_legacy(square_img)

        # Optimization: Check for low variance (likely empty square on digital boards)
        gray_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_square)
        if std_dev < 10.0: # Threshold for "flat color"
            return None

        best_match = (None, -1)
        
        # Check against all learned templates (Pieces + Empty)
        # learned_templates is now Dict[label, List[image]]
        for label, templates_list in self.learned_templates.items():
            if not templates_list:
                continue
                
            for template in templates_list:
                if template.shape != square_img.shape:
                    template = cv2.resize(template, (square_img.shape[1], square_img.shape[0]))

                # Color Match
                res = cv2.matchTemplate(square_img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_match[1]:
                    best_match = (label, max_val)

        label, score = best_match
        
        # If the best match is "empty", return None
        if label and label.startswith('empty'):
            return None
            
        if score > threshold:
            return label
            
        return None

    def _identify_piece_legacy(self, square_img):
        """
        Original logic for backward compatibility.
        """
        gray_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        best_match = (None, -1)
        
        std_dev = np.std(gray_square)
        if std_dev < 15.0: return None

        # Determine if light square (heuristic)
        avg_brightness = np.mean(gray_square)
        is_light_square = avg_brightness > 128
        COLOR_BIAS = 0.25

        for piece_code, template_data in self.piece_templates.items():
            template = template_data['template']
            mask = template_data['mask']
            
            if template.shape[0] > gray_square.shape[0]:
                 scale = min(gray_square.shape[0]/template.shape[0], gray_square.shape[1]/template.shape[1]) * 0.9
                 new_h, new_w = int(template.shape[0]*scale), int(template.shape[1]*scale)
                 if new_h==0 or new_w==0: continue
                 template = cv2.resize(template, (new_w, new_h))
                 if mask is not None: mask = cv2.resize(mask, (new_w, new_h))

            if len(template.shape)==3: template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            res = cv2.matchTemplate(gray_square, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            # Legacy Color Bias logic
            adjusted_max_val = max_val
            is_white_piece = piece_code.startswith('w')
            if is_light_square:
                if is_white_piece: adjusted_max_val += COLOR_BIAS
                else: adjusted_max_val -= COLOR_BIAS
            else:
                if not is_white_piece: adjusted_max_val += COLOR_BIAS
                else: adjusted_max_val -= COLOR_BIAS

            if adjusted_max_val > best_match[1]:
                best_match = (piece_code, adjusted_max_val)
            
        if best_match[1] > 0.45: return best_match[0]
        return None

    def board_to_fen(self, board_representation):
        fen = ""
        for i in range(8):
            empty_count = 0
            for j in range(8):
                piece = board_representation[i*8 + j]
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

    def get_fen_from_screen(self, player_color='w', active_player='w', castling='KQkq', en_passant='-', halfmove=0, fullmove=1):
        print("正在擷取螢幕...")
        screenshot = self.capture_screen()
        
        try:
            board_img = self.find_board(screenshot)
        except Exception as e:
            print(f"辨識錯誤: {e}")
            return None
            
        if board_img is None:
            return None

        squares = self.split_into_squares(board_img)
        board_representation = []
        
        for square in squares:
            piece = self.identify_piece_in_square(square)
            board_representation.append(piece)

        # FEN Ordering Logic
        if self.is_calibrated:
            if self.board_orientation == 'b':
                # Black at Bottom -> Screen shows reversed board.
                # Board Representation needs to be reversed to match FEN (Rank 8..1)
                board_representation = board_representation[::-1]
            else:
                # White at Bottom -> Screen matches FEN order (Rank 8..1)
                # Wait. White at Bottom means:
                # Top Row of Image is Rank 8 (Black Pieces).
                # Bottom Row is Rank 1 (White Pieces).
                # FEN string starts with Rank 8.
                # So standard order is correct.
                pass
        else:
            # Legacy logic fallback
            if player_color == 'b':
                board_representation = board_representation[::-1]

        piece_fen = self.board_to_fen(board_representation)
        
        full_fen = f"{piece_fen} {active_player} {castling} {en_passant} {halfmove} {fullmove}"
        return full_fen

if __name__ == '__main__':
    rec = ScreenRecognizer()
    print("ScreenRecognizer initialized.")
