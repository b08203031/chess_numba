"""
Train & export the Universal 13-Class Chess Square Classifier (ONNX for cv2.dnn).

Classes (index 0..12):
    0: empty (None)
    1..6:  wP, wN, wB, wR, wQ, wK
    7..12: bP, bN, bB, bR, bQ, bK

Input:
    RGB float32 tensor in [0, 1] of shape (N, 3, 48, 48)
    (At runtime in OpenCV: cv2.dnn.blobFromImages(squares_bgr, 1.0/255.0, (48, 48), swapRB=True))

Output:
    Logits float32 tensor of shape (N, 13)

Zero extra runtime dependencies: exports standard ONNX binary directly without requiring
the `onnx` or `protobuf` packages, and verifies numerical equivalence with `cv2.dnn`.
"""

import os
import sys
import time
import math
import random
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F

PIECE_CLASSES = [
    None,
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]
PIECE_CODES = PIECE_CLASSES[1:]
IMG_SIZE = 48

LICHESS_STYLES = [
    "cburnett", "merida", "alpha", "california", "cardinal",
    "gioco", "governor", "horsey", "icpieces", "kosal",
    "leipzig", "maestro", "pirouetti", "pixel", "riohacha",
    "spatial", "tatiana", "fresca", "companion", "dubrovny",
    "fantasy", "caliente", "chessnut", "cooke", "mpchess",
]


# ============================================================================
# 1. Minimal Zero-Dependency ONNX Protobuf Exporter (IR v7, Opset 11)
# ============================================================================

def _varint(val: int) -> bytes:
    if val < 0:
        val &= (1 << 64) - 1
    out = bytearray()
    while (val >> 7) != 0:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _tag(field_num: int, wire_type: int) -> bytes:
    return _varint((field_num << 3) | wire_type)


def _bytes_field(field_num: int, data: bytes) -> bytes:
    return _tag(field_num, 2) + _varint(len(data)) + data


def _str_field(field_num: int, text: str) -> bytes:
    return _bytes_field(field_num, text.encode("utf-8"))


def _int_field(field_num: int, val: int) -> bytes:
    return _tag(field_num, 0) + _varint(val)


def _float_field(field_num: int, val: float) -> bytes:
    return _tag(field_num, 5) + struct.pack("<f", val)


def _attr_ints(name: str, vals) -> bytes:
    b = _str_field(1, name)
    for v in vals:
        b += _int_field(8, int(v))
    b += _int_field(20, 7)
    return b


def _attr_int(name: str, val: int) -> bytes:
    return _str_field(1, name) + _int_field(3, int(val)) + _int_field(20, 2)


def _attr_float(name: str, val: float) -> bytes:
    return _str_field(1, name) + _float_field(2, float(val)) + _int_field(20, 1)


def _make_tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    b = bytearray()
    for d in arr.shape:
        b += _int_field(1, int(d))
    b += _int_field(2, 1)  # FLOAT = 1
    b += _str_field(8, name)
    b += _bytes_field(9, arr.tobytes())
    return bytes(b)


def _make_value_info(name: str, dims) -> bytes:
    shape_bytes = bytearray()
    for d in dims:
        dim_msg = _str_field(2, d) if isinstance(d, str) else _int_field(1, int(d))
        shape_bytes += _bytes_field(1, dim_msg)
    tensor_type = _int_field(1, 1) + _bytes_field(2, bytes(shape_bytes))
    type_proto = _bytes_field(1, tensor_type)
    return _str_field(1, name) + _bytes_field(2, type_proto)


def _make_node(op_type: str, inputs, outputs, name: str = "", attrs=()) -> bytes:
    b = bytearray()
    for i in inputs:
        b += _str_field(1, i)
    for o in outputs:
        b += _str_field(2, o)
    if name:
        b += _str_field(3, name)
    b += _str_field(4, op_type)
    for a in attrs:
        b += _bytes_field(5, a)
    return bytes(b)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    w = conv.weight.detach().clone().float()
    b = conv.bias.detach().clone().float() if conv.bias is not None else torch.zeros(w.shape[0])
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    scale = gamma / torch.sqrt(var + bn.eps)
    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return w_fused.cpu().numpy(), b_fused.cpu().numpy()


# ============================================================================
# 2. Compact Spatial-Aware CNN Architecture
# ============================================================================

class PieceClassifierCNN(nn.Module):
    """
    48x48 RGB -> 13 classes.
    Preserves a 3x3 spatial feature grid before the linear head so crown/head vs
    pedestal geometry (King cross, Queen coronet, Bishop slit, Pawn head) stays crisp.
    """

    def __init__(self, num_classes: int = 13):
        super().__init__()
        # 48x48 -> 24x24
        self.c1 = nn.Conv2d(3, 24, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.c2 = nn.Conv2d(24, 24, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(24)

        # 24x24 -> 12x12
        self.c3 = nn.Conv2d(24, 48, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(48)
        self.c4 = nn.Conv2d(48, 48, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(48)

        # 12x12 -> 6x6
        self.c5 = nn.Conv2d(48, 80, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(80)
        self.c6 = nn.Conv2d(80, 96, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(96)

        # 6x6 -> AveragePool(2,2) -> 3x3 spatial grid (96 * 9 = 864)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(96 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.max_pool2d(x, 2, 2)  # 24x24

        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.bn4(self.c4(x)))
        x = F.max_pool2d(x, 2, 2)  # 12x12

        x = F.relu(self.bn5(self.c5(x)))
        x = F.relu(self.bn6(self.c6(x)))
        x = F.max_pool2d(x, 2, 2)  # 6x6

        x = F.avg_pool2d(x, 2, 2)  # 3x3
        x = x.flatten(1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def export_onnx_bytes(self) -> bytes:
        self.eval()
        nodes = []
        inits = []

        conv_blocks = [
            ("1", self.c1, self.bn1, False),
            ("2", self.c2, self.bn2, True),
            ("3", self.c3, self.bn3, False),
            ("4", self.c4, self.bn4, True),
            ("5", self.c5, self.bn5, False),
            ("6", self.c6, self.bn6, True),
        ]

        prev = "input"
        for tag, conv, bn, do_pool in conv_blocks:
            w, b = _fuse_conv_bn(conv, bn)
            wn, bn_name = f"c{tag}_w", f"c{tag}_b"
            co, ro = f"c{tag}_out", f"r{tag}_out"
            inits.append(_make_tensor(wn, w))
            inits.append(_make_tensor(bn_name, b))
            nodes.append(_make_node("Conv", [prev, wn, bn_name], [co], f"conv{tag}", [
                _attr_ints("kernel_shape", [3, 3]),
                _attr_ints("pads", [1, 1, 1, 1]),
                _attr_ints("strides", [1, 1]),
            ]))
            nodes.append(_make_node("Relu", [co], [ro], f"relu{tag}"))
            prev = ro
            if do_pool:
                po = f"mp{tag}_out"
                nodes.append(_make_node("MaxPool", [prev], [po], f"mp{tag}", [
                    _attr_ints("kernel_shape", [2, 2]),
                    _attr_ints("strides", [2, 2]),
                ]))
                prev = po

        nodes.append(_make_node("AveragePool", [prev], ["ap_out"], "ap", [
            _attr_ints("kernel_shape", [2, 2]),
            _attr_ints("strides", [2, 2]),
        ]))
        nodes.append(_make_node("Flatten", ["ap_out"], ["flat_out"], "flat", [
            _attr_int("axis", 1),
        ]))

        fc1_w = self.fc1.weight.detach().cpu().float().numpy()
        fc1_b = self.fc1.bias.detach().cpu().float().numpy()
        fc2_w = self.fc2.weight.detach().cpu().float().numpy()
        fc2_b = self.fc2.bias.detach().cpu().float().numpy()
        inits.extend([
            _make_tensor("fc1_w", fc1_w),
            _make_tensor("fc1_b", fc1_b),
            _make_tensor("fc2_w", fc2_w),
            _make_tensor("fc2_b", fc2_b),
        ])

        nodes.append(_make_node("Gemm", ["flat_out", "fc1_w", "fc1_b"], ["fc1_out"], "fc1", [
            _attr_int("transB", 1),
            _attr_float("alpha", 1.0),
            _attr_float("beta", 1.0),
        ]))
        nodes.append(_make_node("Relu", ["fc1_out"], ["fc1_relu"], "relu_fc1"))
        nodes.append(_make_node("Gemm", ["fc1_relu", "fc2_w", "fc2_b"], ["logits"], "fc2", [
            _attr_int("transB", 1),
            _attr_float("alpha", 1.0),
            _attr_float("beta", 1.0),
        ]))

        vin = [_make_value_info("input", ["batch", 3, IMG_SIZE, IMG_SIZE])]
        vout = [_make_value_info("logits", ["batch", len(PIECE_CLASSES)])]

        g = bytearray()
        for n in nodes:
            g += _bytes_field(1, n)
        g += _str_field(2, "piece_classifier_graph")
        for init in inits:
            g += _bytes_field(5, init)
        for vi in vin:
            g += _bytes_field(11, vi)
        for vo in vout:
            g += _bytes_field(12, vo)
        opset = _str_field(1, "") + _int_field(2, 11)
        return (
            _int_field(1, 7)
            + _str_field(2, "chess_numba_piece_cnn")
            + _bytes_field(7, bytes(g))
            + _bytes_field(8, opset)
        )


# ============================================================================
# 3. Multi-Style RGBA Piece Bank Loader (Local + SVG + Wikipedia + Font)
# ============================================================================

def _load_rgba_png(path: str, target_size: int = 128, clean_corners: bool = False):
    with open(path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    if clean_corners:
        # Remove tiny edge/corner coordinate artifacts (e.g. bottom-right 'g' in default wN.png)
        h, w, _ = im.shape
        alpha = im[:, :, 3].copy()
        num_l, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha > 20).astype(np.uint8))
        if num_l > 2:
            cx, cy = w * 0.5, h * 0.5
            keep = np.zeros_like(alpha)
            best_idx = max(
                range(1, num_l),
                key=lambda i: stats[i, cv2.CC_STAT_AREA] / (1.0 + math.hypot(centroids[i][0] - cx, centroids[i][1] - cy)),
            )
            keep[labels == best_idx] = 255
            # Also keep legitimate detached parts near top center (like Bishop/King cross or i-dot)
            for i in range(1, num_l):
                if i == best_idx:
                    continue
                px, py = centroids[i]
                if 0.25 * w <= px <= 0.75 * w and py <= 0.45 * h:
                    keep[labels == i] = 255
            im[:, :, 3] = np.where(keep > 0, alpha, 0)
    if im.shape[0] != target_size or im.shape[1] != target_size:
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return im


def _recolor_neo_variant(style_dict: dict, w_rgb, b_rgb) -> dict:
    """Create realistic Chess.com flat/tinted variants (Neo Classic, Glass, Slate, Ivory)."""
    out = {}
    for code, rgba in style_dict.items():
        img = rgba.astype(np.float32).copy()
        alpha = img[:, :, 3:4] / 255.0
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
        target = np.array(w_rgb if code.startswith("w") else b_rgb, dtype=np.float32)
        if code.startswith("w"):
            # Preserve dark outline where gray is low, tint bright interior with target
            t = np.clip((gray - 0.25) / 0.65, 0.0, 1.0)[..., None]
            rgb = t * target + (1.0 - t) * (img[:, :, :3] * 0.35)
        else:
            # Preserve dark outline & subtle specular highlight
            t = np.clip(gray / 0.55, 0.0, 1.25)[..., None]
            rgb = np.clip(target * (0.45 + 0.65 * t), 0, 255)
        res = np.concatenate([np.clip(rgb, 0, 255), alpha * 255.0], axis=2).astype(np.uint8)
        out[code] = res
    return out


def load_all_piece_styles(repo_root: str) -> dict:
    """Returns dict[style_name -> dict[piece_code -> RGBA uint8 (128,128,4)]]."""
    cache_dir = os.path.join(tempfile.gettempdir(), "chess_numba_svg_piece_cache")
    os.makedirs(cache_dir, exist_ok=True)
    styles = {}

    # 1. Local Default & Custom templates
    default_dir = os.path.join(repo_root, "data", "templates")
    if os.path.isdir(default_dir):
        d = {}
        for p in PIECE_CODES:
            fp = os.path.join(default_dir, f"{p}.png")
            if os.path.isfile(fp):
                d[p] = _load_rgba_png(fp, 128, clean_corners=True)
        if len(d) == 12 and all(v is not None for v in d.values()):
            styles["local_default_wood"] = d

    custom_dir = os.path.join(repo_root, "data", "templates", "custom")
    if os.path.isdir(custom_dir):
        d = {}
        for p in PIECE_CODES:
            fp = os.path.join(custom_dir, f"{p}.png")
            if os.path.isfile(fp):
                d[p] = _load_rgba_png(fp, 128, clean_corners=False)
        if len(d) == 12 and all(v is not None for v in d.values()):
            styles["local_custom_neo"] = d
            styles["neo_pure"] = _recolor_neo_variant(d, (249, 249, 249), (52, 50, 48))
            styles["neo_slate"] = _recolor_neo_variant(d, (238, 242, 246), (38, 46, 56))
            styles["neo_warm"] = _recolor_neo_variant(d, (252, 240, 215), (72, 48, 34))

    # 2. Built-in python-chess SVG pieces via fitz
    try:
        import chess
        import chess.svg
        import fitz

        d_pychess = {}
        for p in PIECE_CODES:
            sym = p[1].upper() if p[0] == "w" else p[1].lower()
            svg_bytes = chess.svg.piece(chess.Piece.from_symbol(sym), size=128).encode("utf-8")
            doc = fitz.open(stream=svg_bytes, filetype="svg")
            pix = doc[0].get_pixmap(alpha=True)
            d_pychess[p] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4).copy()
        styles["python_chess_svg"] = d_pychess
    except Exception as e:
        print(f"[WARN] python-chess svg load skipped: {e}")

    # 3. Lichess open-source SVG sets + chessboardjs wikipedia PNG set
    try:
        import requests
        import fitz

        tasks = []
        for st in LICHESS_STYLES:
            for p in PIECE_CODES:
                cache_png = os.path.join(cache_dir, f"lichess_{st}_{p}.png")
                url = f"https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/{st}/{p}.svg"
                tasks.append((f"lichess_{st}", p, url, cache_png, "svg"))

        for p in PIECE_CODES:
            cache_png = os.path.join(cache_dir, f"wikipedia_{p}.png")
            url = f"https://raw.githubusercontent.com/oakmac/chessboardjs/master/website/img/chesspieces/wikipedia/{p}.png"
            tasks.append(("wikipedia", p, url, cache_png, "png"))

        def fetch_one(item):
            st_name, piece_code, url, cache_png, fmt = item
            if os.path.isfile(cache_png):
                im = _load_rgba_png(cache_png, 128)
                if im is not None:
                    return st_name, piece_code, im
            try:
                r = requests.get(url, timeout=8)
                if r.status_code != 200:
                    return st_name, piece_code, None
                if fmt == "svg":
                    doc = fitz.open(stream=r.content, filetype="svg")
                    scale = 128.0 / max(1.0, max(doc[0].rect.width, doc[0].rect.height))
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4).copy()
                    # Pad/resize cleanly to 128x128
                    if arr.shape[0] != 128 or arr.shape[1] != 128:
                        arr = cv2.resize(arr, (128, 128), interpolation=cv2.INTER_AREA)
                else:
                    im_arr = np.frombuffer(r.content, dtype=np.uint8)
                    bgr = cv2.imdecode(im_arr, cv2.IMREAD_UNCHANGED)
                    arr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
                    arr = cv2.resize(arr, (128, 128), interpolation=cv2.INTER_AREA)
                Image.fromarray(arr).save(cache_png)
                return st_name, piece_code, arr
            except Exception:
                return st_name, piece_code, None

        grouped = {}
        with ThreadPoolExecutor(max_workers=12) as ex:
            for st_name, piece_code, arr in ex.map(fetch_one, tasks):
                if arr is not None:
                    grouped.setdefault(st_name, {})[piece_code] = arr

        for st_name, d in grouped.items():
            if len(d) == 12:
                # Filter out SVG sets that rely on external CSS <style> blocks unsupported by MuPDF,
                # which would otherwise render white and black pieces identically as solid black.
                wp, bp = d["wP"], d["bP"]
                wp_fg = wp[:, :, :3][wp[:, :, 3] > 128]
                bp_fg = bp[:, :, :3][bp[:, :, 3] > 128]
                if len(wp_fg) > 50 and len(bp_fg) > 50:
                    if abs(float(wp_fg.mean()) - float(bp_fg.mean())) < 3.5:
                        continue
                styles[st_name] = d
    except Exception as e:
        print(f"[WARN] Remote open-source style download skipped: {e}")

    # 4. Windows Unicode Chess Font Style (used by Tkinter / terminal / minimal GUIs)
    try:
        font_path = "C:/Windows/Fonts/seguisym.ttf"
        if os.path.isfile(font_path):
            font = ImageFont.truetype(font_path, 96)
            glyph_map = {
                "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
                "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
            }
            d_uni = {}
            for code, ch in glyph_map.items():
                # Render filled interior + crisp dark outline for white pieces, solid dark for black
                canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
                dr = ImageDraw.Draw(canvas)
                bbox = dr.textbbox((0, 0), ch, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = (128 - tw) // 2 - bbox[0]
                ty = (128 - th) // 2 - bbox[1]
                if code.startswith("w"):
                    # Solid equivalent glyph filled white + dark stroke
                    solid_ch = glyph_map["b" + code[1]]
                    dr.text((tx, ty), solid_ch, font=font, fill=(248, 248, 245, 255))
                    dr.text((tx, ty), ch, font=font, fill=(20, 20, 20, 255))
                else:
                    dr.text((tx, ty), ch, font=font, fill=(22, 22, 22, 255))
                d_uni[code] = np.array(canvas)
            styles["unicode_symbol"] = d_uni
    except Exception:
        pass

    return styles


# ============================================================================
# 4. Realistic Domain-Randomized Chess Square Synthesizer
# ============================================================================

# Common real-world board palettes: (light_rgb, dark_rgb)
BOARD_PALETTES = [
    ((240, 217, 181), (181, 136, 99)),   # Lichess / Classic Wood
    ((238, 238, 210), (118, 150, 86)),   # Chess.com Green
    ((222, 227, 230), (140, 162, 173)),  # Lichess Blue
    ((235, 236, 208), (119, 149, 86)),   # Tournament Green
    ((240, 240, 240), (180, 180, 180)),  # Light Gray / Newspaper
    ((255, 255, 255), (208, 214, 220)),  # Borderless Minimal White
    ((180, 180, 180), (105, 105, 105)),  # Mid Gray
    ((115, 115, 115), (62, 60, 58)),     # Dark Mode Slate
    ((95, 92, 88), (50, 48, 45)),        # Ultra Dark Mode
    ((232, 218, 239), (142, 108, 168)),  # Purple Theme
    ((212, 230, 241), (84, 153, 199)),   # Icy Sea / Sky Blue
    ((245, 222, 179), (139, 90, 43)),    # Deep Walnut Wood
    ((234, 224, 200), (170, 92, 75)),    # Coral / Red-Brown
    ((220, 228, 210), (92, 122, 108)),   # Sage / Teal
]

HIGHLIGHT_COLORS = [
    (247, 247, 105),  # Chess.com yellow highlight
    (187, 203, 43),   # Lichess green-yellow last move
    (245, 246, 130),  # Soft yellow
    (130, 190, 240),  # Blue selection/arrow tint
    (235, 97, 80),    # Premove / right-click red highlight
    (100, 210, 130),  # Mint highlight
]


def _load_coord_font():
    for fp in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, 15)
            except Exception:
                pass
    return ImageFont.load_default()


COORD_FONT = _load_coord_font()


def synthesize_square(
    rng: random.Random,
    class_idx: int,
    style_list: list,
) -> np.ndarray:
    """
    Synthesizes one realistic (IMG_SIZE, IMG_SIZE, 3) uint8 RGB chess square.
    Covers diverse board colors, wood/noise textures, last-move highlights, check glow,
    empty legal-move dots, capture corner/ring indicators, rank/file coordinates,
    multi-scale downsampling, sub-pixel jitter, and partial arrow overlays.
    """
    S = 96  # Render at 96x96 then downsample via realistic random resolution to 48x48

    # Pick square base color (80% curated realistic palettes + jitter, 20% fully random color)
    if rng.random() < 0.80:
        pal = BOARD_PALETTES[rng.randrange(len(BOARD_PALETTES))]
        is_light = rng.random() < 0.5
        base = np.array(pal[0 if is_light else 1], dtype=np.float32)
        opp_base = np.array(pal[1 if is_light else 0], dtype=np.float32)
        base += np.array([rng.uniform(-16, 16) for _ in range(3)], dtype=np.float32)
    else:
        base = np.array([rng.uniform(35, 250) for _ in range(3)], dtype=np.float32)
        opp_base = 255.0 - base
    base = np.clip(base, 18, 252)

    bg = np.broadcast_to(base, (S, S, 3)).copy()

    # Optional subtle wood grain / gradient / texture (35% chance)
    tex_r = rng.random()
    if tex_r < 0.20:
        # Directional wood-like streak
        freq = rng.uniform(0.08, 0.30)
        axis = rng.choice([0, 1])
        coords = np.arange(S, dtype=np.float32)
        wave = np.sin(coords * freq + rng.uniform(0, 6.28)) * rng.uniform(3.0, 10.0)
        if axis == 0:
            bg += wave[:, None, None]
        else:
            bg += wave[None, :, None]
    elif tex_r < 0.35:
        # Subtle gradient
        grad = np.linspace(-8.0, 8.0, S, dtype=np.float32)
        bg += grad[:, None, None]

    # Last-move / selected square color highlight (28% chance)
    if rng.random() < 0.28:
        hl = np.array(HIGHLIGHT_COLORS[rng.randrange(len(HIGHLIGHT_COLORS))], dtype=np.float32)
        alpha_hl = rng.uniform(0.30, 0.65)
        bg = bg * (1.0 - alpha_hl) + hl * alpha_hl

    # King in check red radial glow (for Kings, or rarely nearby spill)
    piece_code = PIECE_CLASSES[class_idx]
    if piece_code in ("wK", "bK") and rng.random() < 0.20:
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        dist = np.sqrt((xx - S * 0.5) ** 2 + (yy - S * 0.5) ** 2) / (S * 0.55)
        glow = np.clip(1.0 - dist, 0.0, 1.0)[..., None] * 0.75
        red = np.array([235.0, 40.0, 40.0], dtype=np.float32)
        bg = bg * (1.0 - glow) + red * glow

    # Adjacent square 1-3px border bleed from slight calibration/warp jitter (40% chance)
    if rng.random() < 0.40:
        bleed = rng.randint(1, 3)
        side = rng.randrange(4)
        adj_col = np.clip(opp_base + rng.uniform(-10, 10), 15, 250)
        if side == 0:
            bg[:bleed, :, :] = adj_col
        elif side == 1:
            bg[-bleed:, :, :] = adj_col
        elif side == 2:
            bg[:, :bleed, :] = adj_col
        else:
            bg[:, -bleed:, :] = adj_col

    img_pil = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img_pil, "RGBA")

    # Coordinate labels '1'-'8' or 'a'-'h' in corners (25% chance)
    if rng.random() < 0.25:
        label_col = tuple(int(c) for c in np.clip(opp_base, 0, 255)) + (215,)
        if rng.random() < 0.5:
            ch = rng.choice("12345678")
            draw.text((rng.randint(2, 5), rng.randint(1, 4)), ch, fill=label_col, font=COORD_FONT)
        if rng.random() < 0.5:
            ch = rng.choice("abcdefgh")
            draw.text((S - rng.randint(12, 16), S - rng.randint(17, 21)), ch, fill=label_col, font=COORD_FONT)

    if class_idx == 0:
        # EMPTY SQUARE: crucial negative cases so UI markers are NEVER confused with pieces!
        marker_r = rng.random()
        if marker_r < 0.22:
            # Chess.com / Lichess legal-move center dot on empty square!
            rad = int(S * rng.uniform(0.11, 0.18))
            cx = S // 2 + rng.randint(-2, 2)
            cy = S // 2 + rng.randint(-2, 2)
            dot_col = (20, 20, 20, rng.randint(35, 95)) if rng.random() < 0.7 else (30, 90, 40, rng.randint(50, 110))
            draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=dot_col)
        elif marker_r < 0.28:
            # Partial analysis arrow shaft passing across an empty square
            x1, y1 = rng.randint(0, S), rng.randint(0, S)
            x2, y2 = rng.randint(0, S), rng.randint(0, S)
            ar_col = rng.choice([(255, 170, 0, 145), (30, 144, 255, 145), (80, 190, 60, 145)])
            draw.line((x1, y1, x2, y2), fill=ar_col, width=rng.randint(6, 13))
    else:
        # OCCUPIED SQUARE
        # Legal capture ring / corner triangles (14% chance)
        cap_r = rng.random()
        if cap_r < 0.08:
            # Chess.com capture ring around outer edge of square
            pad = int(S * rng.uniform(0.03, 0.07))
            ring_col = (20, 20, 20, rng.randint(40, 95))
            draw.ellipse((pad, pad, S - pad, S - pad), outline=ring_col, width=rng.randint(5, 8))
        elif cap_r < 0.14:
            # Lichess corner triangles
            tri_s = int(S * 0.20)
            tri_col = (20, 85, 30, 95)
            draw.polygon([(0, 0), (tri_s, 0), (0, tri_s)], fill=tri_col)
            draw.polygon([(S, 0), (S - tri_s, 0), (S, tri_s)], fill=tri_col)
            draw.polygon([(0, S), (tri_s, S), (0, S - tri_s)], fill=tri_col)
            draw.polygon([(S, S), (S - tri_s, S), (S, S - tri_s)], fill=tri_col)

        style_dict = style_list[rng.randrange(len(style_list))]
        piece_rgba = style_dict[piece_code]

        # Scale & slight shift jitter
        scale = rng.uniform(0.76, 1.00)
        pw = max(16, int(S * scale))
        ph = max(16, int(S * scale))
        piece_resized = cv2.resize(piece_rgba, (pw, ph), interpolation=cv2.INTER_AREA if pw < 128 else cv2.INTER_LINEAR)

        # Mild brightness/contrast jitter on piece (10% variation)
        if rng.random() < 0.5:
            p_float = piece_resized.astype(np.float32)
            gain = rng.uniform(0.90, 1.10)
            bias = rng.uniform(-10.0, 10.0)
            p_float[:, :, :3] = np.clip(p_float[:, :, :3] * gain + bias, 0, 255)
            piece_resized = p_float.astype(np.uint8)

        max_shift = int(S * 0.055)
        ox = (S - pw) // 2 + rng.randint(-max_shift, max_shift)
        oy = (S - ph) // 2 + rng.randint(-max_shift, max_shift)

        p_pil = Image.fromarray(piece_resized, mode="RGBA")
        img_pil.paste(p_pil, (ox, oy), p_pil)

        # Occasional partial arrow crossing over piece (6% chance)
        if rng.random() < 0.06:
            x1, y1 = rng.randint(0, S), rng.randint(0, S)
            x2, y2 = rng.randint(0, S), rng.randint(0, S)
            ar_col = rng.choice([(255, 170, 0, 120), (30, 144, 255, 120)])
            draw.line((x1, y1, x2, y2), fill=ar_col, width=rng.randint(5, 10))

    arr_rgb = np.array(img_pil, dtype=np.uint8)

    # Simulate screen resolution variation (e.g. small 280px mini-board = 35px/sq up to 1000px board)
    if rng.random() < 0.45:
        sim_sq = rng.randint(28, 72)
        arr_rgb = cv2.resize(arr_rgb, (sim_sq, sim_sq), interpolation=cv2.INTER_AREA)

    out_rgb = cv2.resize(arr_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Slight sensor/compression noise (25% chance)
    if rng.random() < 0.25:
        noise = np.random.normal(0.0, rng.uniform(1.0, 3.5), out_rgb.shape).astype(np.float32)
        out_rgb = np.clip(out_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out_rgb


def generate_dataset(num_samples: int, style_list: list, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    np.random.seed(seed)
    X = np.empty((num_samples, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = np.empty((num_samples,), dtype=np.int64)

    for i in range(num_samples):
        # Give slightly higher weight to Empty ( ~14% ) and balanced ~7.1% across all 12 pieces
        if rng.random() < 0.14:
            cls = 0
        else:
            cls = rng.randint(1, 12)
        rgb = synthesize_square(rng, cls, style_list)
        X[i] = np.transpose(rgb, (2, 0, 1))
        y[i] = cls
    return X, y


# ============================================================================
# 5. Training Loop & Verification
# ============================================================================

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "data", "models")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "piece_classifier.onnx")

    t0 = time.time()
    print("[1/4] Loading multi-style piece bank...")
    styles_dict = load_all_piece_styles(repo_root)
    style_names = sorted(styles_dict.keys())
    print(f"      Loaded {len(style_names)} complete 12-piece styles: {style_names}")
    if len(style_names) < 2:
        raise RuntimeError("Insufficient piece styles loaded for universal training!")

    all_styles = [styles_dict[k] for k in style_names]

    print("[2/4] Synthesizing domain-randomized training & validation squares...")
    # 52,000 train samples + 5,200 val samples
    X_train, y_train = generate_dataset(52000, all_styles, seed=20261030)
    X_val, y_val = generate_dataset(5200, all_styles, seed=99991)
    print(f"      Generated X_train={X_train.shape}, X_val={X_val.shape} in {time.time() - t0:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3/4] Training PieceClassifierCNN on {device}...")

    model = PieceClassifierCNN(num_classes=len(PIECE_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    epochs = 22
    batch_size = 256
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    X_val_t = torch.from_numpy(X_val).to(device, dtype=torch.float32).div_(255.0)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_acc = 0.0
    best_state = None
    n_train = len(X_train)

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device, dtype=torch.float32).div_(255.0)
            yb = torch.from_numpy(y_train[idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.02)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            for s in range(0, len(X_val_t), 512):
                val_logits.append(model(X_val_t[s:s + 512]))
            val_logits = torch.cat(val_logits, dim=0)
            preds = val_logits.argmax(dim=1)
            val_acc = (preds == y_val_t).float().mean().item() * 100.0

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep in (1, 5, 10, 15, 20, epochs):
            print(f"      Epoch {ep:02d}/{epochs} | Loss: {total_loss / n_train:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_acc:.2f}%)")

    model.load_state_dict(best_state)
    model.cpu().eval()

    print(f"[4/4] Exporting fused ONNX model to {onnx_path}...")
    onnx_bytes = model.export_onnx_bytes()
    with open(onnx_path, "wb") as f:
        f.write(onnx_bytes)
    size_kb = len(onnx_bytes) / 1024.0
    print(f"      Saved {onnx_path} ({size_kb:.1f} KB)")

    # Verify with OpenCV cv2.dnn on batch=64 and batch=1 (using buffer for Unicode path safety)
    net = cv2.dnn.readNetFromONNX(np.frombuffer(onnx_bytes, dtype=np.uint8))
    test_batch = (X_val[:64].astype(np.float32) / 255.0)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_batch)).numpy()

    net.setInput(test_batch)
    cv_out = net.forward()
    max_diff = float(np.max(np.abs(pt_out - cv_out)))

    # Benchmark CPU 64-square board inference latency
    t_bench = time.perf_counter()
    iters = 50
    for _ in range(iters):
        net.setInput(test_batch)
        _ = net.forward()
    ms_per_board = (time.perf_counter() - t_bench) * 1000.0 / iters

    print(f"      cv2.dnn max abs diff vs PyTorch: {max_diff:.2e}")
    print(f"      cv2.dnn 64-square board CPU latency: {ms_per_board:.2f} ms/board")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX export mismatch too high: {max_diff}")
    if best_acc < 98.5:
        raise RuntimeError(f"Validation accuracy below target: {best_acc:.2f}%")
    print("[SUCCESS] Universal PieceClassifierCNN trained and verified!")


if __name__ == "__main__":
    main()
