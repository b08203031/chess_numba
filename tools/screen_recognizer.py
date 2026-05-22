import cv2
import numpy as np
import mss
from PIL import Image
import os

class ScreenRecognizer:
    """
    負責從螢幕截圖中解析出棋局狀態 (FEN)。
    """

    def __init__(self, templates_path=None):
        if templates_path is None:
            templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "templates")
        
        """
        初始化辨識器，載入所有棋子模板。

        Args:
            templates_path (str): 存放棋子模板圖片的資料夾路徑。
        """
        self.piece_templates = self._load_templates(templates_path)
        self.inv_M = None # To store inverse perspective matrix for coordinate mapping
        self.board_side_length = 800

    def _load_templates(self, path):
        """
        從指定路徑載入12種棋子的模板圖片。
        模板檔名應為 "wP.png", "bK.png" 等格式。
        
        重要：模板圖片必須是帶有透明背景的 PNG 檔案，且棋子樣式需與螢幕上完全一致。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(script_dir, path)

        templates = {}
        pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
        for piece in pieces:
            file_path = os.path.join(base_path, f"{piece}.png")
            try:
                # 修正：使用能處理 Unicode 路徑的方式讀取圖片
                with open(file_path, "rb") as img_stream:
                    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
                    template = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                
                if template is None:
                    raise FileNotFoundError(f"無法解碼模板檔案: {file_path}")
                
                if template.shape[2] == 4:
                    templates[piece] = {
                        'template': cv2.cvtColor(template, cv2.COLOR_BGRA2BGR),
                        'mask': template[:,:,3]
                    }
                else:
                    templates[piece] = {
                        'template': template,
                        'mask': None
                    }
            except FileNotFoundError:
                 raise FileNotFoundError(f"找不到模板檔案: {file_path}")
            except Exception as e:
                raise Exception(f"載入模板失敗 {file_path}: {e}")

        return templates

    def capture_screen(self, region=None):
        """
        擷取螢幕畫面。

        Args:
            region (dict): 指定要擷取的區域。如果為 None，則擷取整個螢幕。

        Returns:
            numpy.ndarray: 返回 OpenCV 格式的圖像。
        """
        with mss.mss() as sct:
            # 安全地選擇顯示器：如果 monitors[1] 不存在（例如無顯示器環境），則退回到 monitors[0]
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
        最終方案：利用反轉的棋子遮罩，直接找到棋盤輪廓進行校正。
        """
        # 1. 尋找並裁切出棋盤的大致區域 (ROI)
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        lower_wood = np.array([10, 80, 50])
        upper_wood = np.array([25, 255, 255])
        board_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("錯誤：顏色篩選未能找到棋盤的初始輪廓。")
            return None
        board_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(board_contour)
        padding = 5
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(screenshot.shape[1] - x, w + 2*padding), min(screenshot.shape[0] - y, h + 2*padding)
        board_roi = screenshot[y:y+h, x:x+w]
        
        # 2. 在ROI內，建立精確的棋子遮罩
        hsv_roi = cv2.cvtColor(board_roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
        black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
        pieces_mask = cv2.bitwise_or(white_mask, black_mask)
        kernel = np.ones((3,3), np.uint8)
        pieces_mask = cv2.morphologyEx(pieces_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. 反轉棋子遮罩，得到棋盤本身的遮罩
        board_only_mask = cv2.bitwise_not(pieces_mask)

        # 4. 在棋盤遮罩上找到最外層的輪廓
        contours, _ = cv2.findContours(board_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("錯誤：在反轉遮罩後未能找到棋盤輪廓。")
            return None
        
        final_board_contour = max(contours, key=cv2.contourArea)

        # 5. 獲取輪廓的四個角點
        rect = cv2.minAreaRect(final_board_contour)
        box = cv2.boxPoints(rect)
        approx = np.intp(box)

        # 6. 執行透視校正
        # 角點座標是相對於ROI的，需要轉換回原始截圖座標
        points = approx.reshape(4, 2).astype("float32")
        points += (x, y) # 加上ROI的左上角偏移量

        # 使用穩健的排序方法
        sorted_rect = np.zeros((4, 2), dtype="float32")
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        tl_index = np.argmin(np.sum(sorted_points, axis=1))
        sorted_rect = np.roll(sorted_points, -tl_index, axis=0)

        # Debug outputs removed or commented out to avoid I/O errors
        # debug_screenshot = screenshot.copy()
        # cv2.drawContours(debug_screenshot, [sorted_rect.astype(np.int32)], -1, (0, 255, 0), 3)

        side_length = self.board_side_length
        dst = np.array([[0,0], [side_length-1,0], [side_length-1,side_length-1], [0,side_length-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(sorted_rect, dst)
        self.inv_M = cv2.getPerspectiveTransform(dst, sorted_rect) # Store inverse matrix
        
        warped = cv2.warpPerspective(screenshot, M, (side_length, side_length))
        
        return warped

    def split_into_squares(self, board_img):
        height, width, _ = board_img.shape
        square_h, square_w = height // 8, width // 8
        squares = []
        for i in range(8):
            for j in range(8):
                square = board_img[i*square_h:(i+1)*square_h, j*square_w:(j+1)*square_w]
                squares.append(square)
        return squares

    def identify_piece_in_square(self, square_img, threshold=0.45):
        gray_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        best_match = (None, -1)

        h, w = gray_square.shape
        center_h_start, center_h_end = int(h * 0.25), int(h * 0.75)
        center_w_start, center_w_end = int(w * 0.25), int(w * 0.75)
        
        if center_h_start >= center_h_end: center_h_end = center_h_start + 1
        if center_w_start >= center_w_end: center_w_end = center_w_start + 1

        center_region = gray_square[center_h_start:center_h_end, center_w_start:center_w_end]
        
        std_dev = np.std(center_region)
        UNIFORMITY_THRESHOLD = 15.0 

        if std_dev < UNIFORMITY_THRESHOLD:
            return None

        avg_brightness = np.mean(center_region)
        BRIGHTNESS_THRESHOLD = 128 
        is_light_square = avg_brightness > BRIGHTNESS_THRESHOLD
        COLOR_BIAS = 0.25

        for piece_code, template_data in self.piece_templates.items():
            template = template_data['template']
            mask = template_data['mask']

            if template.shape[0] > gray_square.shape[0] or template.shape[1] > gray_square.shape[1]:
                scale_factor = min(gray_square.shape[0] / template.shape[0], gray_square.shape[1] / template.shape[1]) * 0.9
                new_h = int(template.shape[0] * scale_factor)
                new_w = int(template.shape[1] * scale_factor)
                if new_h == 0 or new_w == 0: continue
                template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if mask is not None:
                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(gray_square, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            _, max_val, _, _ = cv2.minMaxLoc(res)

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

        if best_match[1] > threshold:
            return best_match[0]
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
        print("正在擷取整個螢幕...")
        screenshot = self.capture_screen()
        
        print("正在自動偵測棋盤...")
        try:
            # Wrap potential cv2 failures in a try block
            board_img = self.find_board(screenshot)
        except Exception as e:
            print(f"棋盤偵測發生錯誤: {e}")
            return None
        
        if board_img is None:
            print("無法從螢幕上找到棋盤。")
            return None

        print("成功找到棋盤，正在進行辨識...")
        # debug write removed
        squares = self.split_into_squares(board_img)

        board_representation = []
        for i, square in enumerate(squares):
            piece = self.identify_piece_in_square(square)
            board_representation.append(piece)

        if player_color == 'b':
            board_representation = board_representation[::-1]

        piece_fen = self.board_to_fen(board_representation)
        
        full_fen = f"{piece_fen} {active_player} {castling} {en_passant} {halfmove} {fullmove}"
        return full_fen

    def get_move_screen_coords(self, uci_move, player_color='w'):
        """
        將 UCI 格式的步法 (例如 "e2e4") 轉換為螢幕上的起點與終點絕對座標。
        需要在呼叫過 get_fen_from_screen (保存了 inv_M) 之後調用。
        
        Returns:
            ((start_x, start_y), (end_x, end_y))
        """
        if self.inv_M is None:
            print("找不到棋盤座標映射矩陣。請確保已成功辨識過一次棋盤。")
            return None
            
        if len(uci_move) < 4:
            return None
            
        start_sq = uci_move[0:2]
        end_sq = uci_move[2:4]
        
        def sq_to_center_pt(sq_str):
            file_idx = ord(sq_str[0]) - ord('a') # 0-7
            rank_idx = int(sq_str[1]) - 1        # 0-7
            
            if player_color == 'b':
                # 如果以黑方視角，a1 在右上角
                display_file = 7 - file_idx
                display_rank = rank_idx
            else:
                # 預設白方視角，a1 在左下角，對應圖像矩陣左下即 y=大小，所以 rank 要顛倒
                display_file = file_idx
                display_rank = 7 - rank_idx
                
            square_size = self.board_side_length / 8
            # 取得格子中心的戰圖坐標 (相對於 800x800 的扭曲後圖像)
            x_warped = display_file * square_size + square_size / 2
            y_warped = display_rank * square_size + square_size / 2
            
            # 使用反向透視矩陣轉換回螢幕絕對坐標
            pt = np.array([[[x_warped, y_warped]]], dtype="float32")
            pt_screen = cv2.perspectiveTransform(pt, self.inv_M)
            return int(pt_screen[0][0][0]), int(pt_screen[0][0][1])
            
        try:
            start_coord = sq_to_center_pt(start_sq)
            end_coord = sq_to_center_pt(end_sq)
            return (start_coord, end_coord)
        except Exception as e:
            print(f"轉換座標失敗: {e}")
            return None

if __name__ == '__main__':
    try:
        recognizer = ScreenRecognizer(templates_path="templates/")
        
        print("準備在 2 秒後自動偵測棋盤並進行辨識...")
        import time
        time.sleep(2)

        fen_string = recognizer.get_fen_from_screen(
            player_color='w',
            active_player='w',
            castling='KQkq',
            en_passant='-',
            halfmove=0,
            fullmove=1
        )
        
        if fen_string:
            print("\n辨識完成！")
            print(f"FEN: {fen_string}")
        else:
            print("\n辨識失敗。請檢查終端機輸出以了解詳細資訊。")

    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print("請確保 'templates' 資料夾存在，且包含所有12個棋子的PNG圖片。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
