# 搜尋歷史啟發 (History Heuristics) 技術解析

本文件詳細說明引擎在搜尋與剪枝過程中使用的各種歷史記錄機制。這些技術是提升 move ordering（移動排序）品質、增強 LMR（晚期移動歸約）準確性以及修正靜態評估誤差的核心手段。

---

## 1. 歷史啟發體系總覽

歷史啟發的本質是：**利用搜尋樹中已知的「好步」與「壞步」統計數據，來預測當前節點中哪些移動最值得優先嘗試。**

引擎目前實現了四種細分維度的歷史記錄：

| 歷史類型 | 數據結構與維度 | 核心作用 | 更新時機 |
|---|---|---|---|
| **主歷史 (Main History)** | `history_table[12][64]` | 給予安靜移動 (Quiet moves) 基礎排序分值，並參與 LMR 歸約計算。 | 產生 beta 截斷 (Beta Cutoff) 時增加；在 PV 節點或 Fail-Low 節點減分。 |
| **蝴蝶歷史 (Butterfly History)** | `butterfly_history[64][64]` | 記錄移動的起點與終點格。作為主歷史的輔助分值。 | 同上。 |
| **吃子歷史 (Capture History)** | `capture_history[12][64][12]` | 專門排序吃子移動。針對「行棋棋子 + 目標格 + 被吃棋子」進行評級。 | 吃子移動產生 beta 截斷或 Fail-Low 時更新。 |
| **延續歷史 (Continuation History)** | `continuation_history[5][12][64][12][64]` | 關聯歷史。記錄當前移動與前幾步歷史移動（1步、2步、3步、4步與6步前）之間的協同效應。 | 當前安靜移動產生 beta 截斷或 Fail-Low 時更新。 |
| **低層歷史 (Low Ply History)** | `low_ply_history[5][65536]` | 專門為靠近根部 (ply < 5) 的安靜移動提供更精確的排序偏置，彌補全域歷史在根部的盲點。 | 在 ply < 5 產生 beta 截斷或 Fail-Low 時更新。 |

這些表格的管理與計算實作於 [search_heuristics.py](search_heuristics.py)。

---

## 2. 重力公式 (Gravity Formula) 更新機制

為了防止歷史數值隨搜尋進行無窮增長導致整型溢出，同時確保「近期搜尋的結果」擁有更高的權重，我們採用了 **重力公式**（Gravity Formula，源於 Stockfish）：

### 數學公式
$$\text{History} \gets \text{History} + \text{Bonus} - \frac{\text{History} \times |\text{Bonus}|}{\text{Max\_Limit}}$$

*   當移動獲得獎勵 (Bonus > 0) 時，數值朝 `Max_Limit` 漸進。
*   當移動獲得懲罰 (Bonus < 0，又稱 Malus) 時，數值朝 `-Max_Limit` 漸進。
*   此公式天然具備自我限幅特性，無需手動 clamp 臨界值。

### 程式實作範例 (主歷史更新)
```python
@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_history(history_table, piece_type, to_square, bonus):
    current_value = history_table[piece_type, to_square]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_MAIN), HISTORY_MAX_MAIN)
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // HISTORY_MAX_MAIN
    history_table[piece_type, to_square] = new_value
```

---

## 3. 延續歷史 (Continuation History)

延續歷史主要解決安靜移動的**戰術協同**問題。

### 原理
例如：己方馬跳到 f3 (`prev_move`)，敵方回應了 h6，現在輪到己方下棋。如果跳馬到 e5 (`curr_move`) 在過去的類似歷史中經常有效，那麼這兩步棋之間就存在強關聯。

### 追溯層次 (Plies Offset)
*   `Index 0` (1-ply ago): 前一步（對手剛下的子）。
*   `Index 1` (2-plies ago): 前二步（己方上一次下的子）。
*   `Index 2` (3-plies ago): 前三步（對手上次下的子）。
*   `Index 3` (4-plies ago): 前四步（己方上上次下的子）。
*   `Index 4` (6-plies ago): 前六步（己方上上上次下的子），跳過 5-ply。

在 [score_quiets](search_heuristics.py#L306) 中，這些關聯分值會被加權累加到安靜移動的排序分中。

---

## 4. 糾錯歷史 (Correction History)

糾錯歷史 (Correction History) 與排序無關，它用來**動態修正評估函數的誤差**。

### 原理
手工評估函數 (HCE) 存在靜態盲區（例如：無法精確評估某個特定的兵型）。糾錯歷史記錄了「特定局面特徵下的評估分值 (Static Eval)」與「經過深層搜尋後得到的真實分值 (Search Score)」之間的長期系統性偏差。

### 記錄特徵
*   `pawn_correction_history`: 基於當前兵鍵值 (Pawn Key)。修正長期兵型特徵引起的估值偏頗。
*   `minor_correction_history`: 基於輕子鍵值 (Minor Key)。
*   `non_pawn_correction_history`: 分白黑兩方，基於非兵棋子鍵值。
*   **連續糾錯歷史 (Continuation Correction History)**：基於動態走法關聯。記錄「我方前一手棋 (`ply - 2`) + 對手回應 (`ply - 1`)」與當前靜態評估誤差的關聯。

### 應用方式
在 `_search` 的開頭，獲取靜態評估分值後，會根據當前的各種 Key 查表並疊加修正：
```python
correction_sum = (global_pawn * CORRECTION_HISTORY_PAWN_WEIGHT +
                 global_minor * CORRECTION_HISTORY_MINOR_WEIGHT +
                 non_pawn_weight + cntcv)
static_score = raw_static_eval + (correction_sum // CORRECTION_HISTORY_DIVISOR)
```
其中 `cntcv` 即為連續糾錯歷史分量，當上一手棋有效時，會查詢與疊加 $2\text{-ply}$ 與 $4\text{-ply}$ 偏移處的歷史偏差值，乘上 HCE 專屬權重 `10000`。這使得引擎即使在靜態評估偏低時，也能通過搜尋歷史學會「這類著法序列後其實很好」，從而避免過早剪枝或錯誤選擇路徑。

---

## 5. 歷史更新優化與非對稱縮放 (Part B & E)

為了讓歷史數據更平滑地反映搜尋優先級，我們在更新時引入了以下優化機制：

### 兵歷史非對稱縮放 (Pawn History Asymmetric Scaling - Part E)
在更新兵歷史時，我們對獎勵 (Bonus) 與懲罰 (Malus) 實施非對稱縮放。當傳入的 `bonus` 為小於 `-7` 的懲罰值時，我們將其減半以保留有價值的歷史痕跡：
```python
scaled_bonus = bonus * 1038 // 1024 if bonus > -7 else bonus * 525 // 1024
```
這能防止兵歷史因單次 beta 截斷而對其他可能可行的步產生過度懲罰。

### 安靜移動懲罰衰減 (Quiet Malus Gradual Decay - Part B)
當某個安靜移動觸發 beta 截斷時，我們需要對先前所有嘗試過但失敗 (fail-low) 的安靜移動施加 `malus` 懲罰。
為了讓排序較前（原本預期較好）的移動受到較重的懲罰，而排序較後（原本就不看好）的移動懲罰遞減，我們對 `malus` 進行逐步衰減：
* 初始懲罰設為：`actual_malus = malus * 1136 // 1024`。
* 每走一步，懲罰值乘以 `956 / 1024`（衰減約 6.6%）。
* 這使得越早搜到的 quiet 步懲罰越大，越遲搜到的步懲罰越小。此遞減懲罰同步應用於主歷史、蝴蝶歷史、兵歷史與延續歷史中。
