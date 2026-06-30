# 重力公式分析報告與歷史表修整計畫書

本報告深入分析目前引擎在「歷史表（History Heuristics）更新公式」與「最大邊界值」上的限制，並對比 Stockfish 的成熟架構，提出一套系統性的修整與優化計畫。

---

## 1. 目前架構與參數分析 (Current Issues)

目前我們的引擎在 [constants.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine/classical/constants.py#L740-L750) 與 [search.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine/classical/search.py) 中的歷史表設計存在兩個主要缺陷：

### 缺陷 A：上限值（HISTORY_MAX）缺乏分流 (Lack of Tiered Ceilings)
* **現狀**：
  * 主歷史表 (`HISTORY_MAX_MAIN`) = `16384`
  * 蝴蝶歷史表 (`HISTORY_MAX_BUTTERFLY`) = `16384`
  * 吃子歷史表 (`HISTORY_MAX_CAPTURE`) = `16384`
  * 延續歷史表 (`HISTORY_MAX_CONTINUATION`) = `16384`
* **影響**：所有表格的漸近上限（重力公式中的 $D$）全部相同。這會導致次級表（如吃子歷史、蝴蝶歷史）在深層搜尋時，分數會長到與主歷史表一樣大。這嚴重干擾了 Move Ordering 的決策權重，使得次級特徵過度膨脹，掩蓋了主歷史表的排序指向。

### 缺陷 B：更新步長過早飽和 (Early Saturation of Bonus/Malus)
* **現狀**：
  * `bonus = min(depth * depth + 120 * depth - 100, 1600)`
  * `malus = min(depth * depth + 100 * depth - 50, 1400)`
* **影響**：
  * 使用了二次方（$depth^2$）的增長曲線。
  * 在 **depth = 4** 時：`bonus = min(16 + 480 - 100, 1600) = 396`（學習率極快）。
  * 在 **depth >= 12** 時：`bonus` 會直接觸及上限 `1600`。
  * **致命點**：這意味著從深度 12 到深度 25，引擎每一次 Cutoff 施加的更新步長**完全一模一樣**。高深度的戰術高信賴度完全沒有在歷史表中獲得應有的權重，導致長時限下歷史表無法精準收斂。

---

## 2. 借鑑 Stockfish 設計哲學 (Stockfish Philosophy)

1. **分級上限（Tiered Ceilings）**：
   * 主歷史表與延續歷史表上限最高（$D = 30000$）。
   * 次級歷史表（吃子、蝴蝶、兵歷史）上限遞減（$D$ 分別為 `10692`、`7183`、`8192`）。
   * 確保輔助歷史資訊永遠只作為「修正項」，不反客為主。
2. **線性非飽和更新（Linear Scaling）**：
   * 步長隨深度呈線性增長（例如 $134 \times depth$），上限較高，保證在 depth 1 到 depth 20 之間更新步長能持續平滑地隨深度放大。

---

## 3. 修整計畫書 (Refactoring Plan)

### 階段一：重新定義上限值 (Tiered Ceilings)
修改 [constants.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine/classical/constants.py#L740-L750)，建立權重分流：
* `HISTORY_MAX_MAIN = 16384` (不宜再大，以防止越過 10000/14000 的殺手步門檻)
* `HISTORY_MAX_CONTINUATION = 12288` (調降，作為主表的修正)
* `HISTORY_MAX_BUTTERFLY = 8192` (調降，輔助安靜步防守)
* `HISTORY_MAX_CAPTURE = 8192` (調降，輔助吃子排序)
* `HISTORY_MAX_PAWN = 4096` (調降，微調兵結構)

### 階段二：改進更新步長為「線性縮放」 (Linear Updates)
修改 [search.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine/classical/search.py) 中所有的 `bonus` / `malus` 計算公式：
* 廢除二次方公式，改用平滑的線性縮放，確保在深層（12~22層）時更新步長能持續增長：
  * `bonus = min(120 * depth, 1800)`
  * `malus = min(120 * depth, 1600)`
* 對應修改的地方包括：
  * Cutoff 區塊
  * TT Fail-High 區塊
  * Countermove 反饋區塊

### 階段三：實作節點寬度補償（Node Width Scaling - 進階）
在非 PV 節點中，根據已經搜過的安靜步與吃子步的數量，動態放大 Cutoff 的 Bonus：
* `bonus += bonus * (quiet_moves_tried_count + capture_moves_tried_count) // 256`

---

## 4. 驗證與調優

1. **JIT 正確性測試**：
   * 執行 `python -m tests.test_search` 確保修改後 JIT 編譯無誤。
2. **多時限對抗測試**：
   * 在 `cutechess-cli` 中運行兩個對局組合：
     * **組合一 (快速棋 1s/move)**：確認在 1 秒下不退步。
     * **組合二 (長時限 10s/move)**：實證在 10 秒下，修整後的版本（Tiered + Linear）能夠大幅超越舊版本。
