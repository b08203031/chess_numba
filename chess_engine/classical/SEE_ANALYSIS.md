# Static Exchange Evaluation (SEE) 分析報告

## 1. 精神與意義 (Spirit and Meaning)

靜態交換評估 (SEE) 的核心精神是在不進行完整搜尋的情況下，評估一個特定方格上的「交換序列」是否有利。它主要用於：
- **移動排序 (Move Ordering)**：區分「好的吃子」和「壞的吃子」，確保好的移動優先被搜尋。
- **靜態搜尋 (Quiescence Search)**：剪掉那些明顯不利的吃子動作。
- **搜尋剪枝 (Pruning)**：在某些深度較淺的節點，剪掉 SEE 分數過低的移動。

SEE 的基本假設是：雙方都會以「最小價值」的棋子依次吃掉目標方格上的棋子（LVA - Least Valuable Attacker）。

---

## 2. 演算法方法 (Methodology)

目前的實現採用了類似 Stockfish 的 **Swap 演算法**：

1. **初始值**：計算初始吃子的價值（被吃掉的棋子價值）。
2. **攻擊者集合**：找出所有攻擊目標方格的棋子。
3. **疊代模擬**：
   - 雙方輪流從攻擊者集合中選出價值最低的棋子進行「吃子」。
   - 每一步計算該方的「累積增益」。
   - **X-Ray 處理**：當一個滑動棋子（車、象、后）或兵被移走時，檢查其後方是否露出了新的 X-Ray 攻擊者。
   - **國王安全**：國王不能移動到受攻擊的方格進行吃子（這在模擬中代表吃子後會被立即反殺）。
4. **Minimax 傳論**：從序列最後端向前傳導分數，模擬雙方在發現後續不利時會選擇停止吃子的情況。

---

## 3. 細節檢查與邏輯驗證 (Details and Verification)

### 3.1 棋子價值 (Piece Values)
- **現狀**：SEE 使用專用表 `SEE_PIECE_VALUES`（`see.py`），國王價值為 **20000**，其餘與中局材質尺度一致（P/N/B/R/Q = 100/320/330/500/900）。
- **說明**：高國王價值避免在 Swap 中把國王當成可犧牲的普通棋子。

### 3.2 攻擊者尋找 (Attacker Identification)
- **現狀**：Magic Bitboards + 預計算攻擊表；在 `get_attackers_for_see` 與 X-Ray 路徑上，先以 `BISHOP_RAYS` / `ROOK_RAYS` 預過濾，再必要時做 Magic 查表。

### 3.3 X-Ray 處理 (X-Ray Handling)
- **現狀**：滑動棋子被移開後，沿射線補上後方攻擊者；邏輯已以 0–5 棋子類型索引處理，不依賴側翼偏移的冗餘判斷。

### 3.4 國王捕捉模擬 (King Capturing)
- **現狀**：正確處理了國王不能捕獲受保護棋子的情況（避免送王）。

### 3.5 過路兵 (En Passant)
- **現狀**：正確處理初始捕獲價值，並在 `occupied` 位元棋盤中移除被過路兵吃掉的兵，以確保 X-Ray 掃描結果精確。

### 3.6 兵升變 (Pawn Promotion)
- **現狀**：當兵移動到最後一排（Rank 1 或 8）進行吃子時，該兵在後續的交換序列中被視為「后」（Queen）計算價值。
- **分析**：這避免了引擎誤以為犧牲升變後的后只是一個兵的代價，從而優化了移動排序和剪枝邏輯。

---

## 4. 相關檔案說明 (Related Files)

- [see.py](see.py) 與 [nnue/see.py](../nnue/see.py): SEE 的核心實現（兩版本完全獨立）。
- [constants.py](constants.py) 與 [nnue/constants.py](../nnue/constants.py): 定義了棋子價值和 SEE 閾值。
- [search.py](search.py) 與 [nnue/search.py](../nnue/search.py): 在移動排序、靜態搜尋和淺深度剪枝中調用 `see_ge`。
- [move_generator.py](move_generator.py) 與 [nnue/move_generator.py](../nnue/move_generator.py): 提供基礎的攻擊生成邏輯。
- [bitboard_utils.py](bitboard_utils.py) 與 [nnue/bitboard_utils.py](../nnue/bitboard_utils.py): 提供硬體加速的位元掃描和預計算的射線表。

---

## 5. 優化與修正紀錄 (Optimization and Fix Records)

本分析過程中，已針對以下項目進行優化與修正：
1. **修正國王價值**：設置專門的 `SEE_PIECE_VALUES` (King = 20000)，確保國王在交換評估中不會被視為普通棋子犧牲。
2. **性能優化**：在 `get_attackers_for_see` 與 X-Ray 邏輯中加入射線預過濾 (`BISHOP_RAYS`, `ROOK_RAYS`)，顯著減少昂貴的 Magic Bitboard 查表次數。
3. **優化 X-Ray 邏輯**：移除冗餘的側翼索引判斷，簡化代碼邏輯。
4. **修正過路兵細節**：在模擬過程中正確從 `occupied` 移除被吃掉的過路兵，確保後方 X-Ray 攻擊能被正確發現。
5. **處理兵升變**：在交換序列中，若兵進入底線進行捕獲，將其價值提升為「后」，使評估更符合實際博弈情況。

這些改進提高了搜尋效率，並使引擎在處理複雜吃子序列時更加精準。
