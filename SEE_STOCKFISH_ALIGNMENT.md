# SEE 與 Stockfish 對齊報告

## 1. 概述

本任務的目標是確保我的引擎 (`chess_engine.see`) 的靜態交換評估 (SEE) 函數在邏輯上與 Stockfish 的 SEE 實現完全一致。

為了達成此目標，我執行了以下步驟：
1.  **修改 Stockfish**：在 `stockfish/src` 中修改了 `position.cpp`, `uci.cpp` 等文件，增加了一個自定義的 `see <move>` 指令，使其能輸出精確的 SEE 分數（而非僅是 `see_ge` 的布林值）。
2.  **自動化對比**：編寫了 `compare_see_aligned.py` 腳本，針對一系列具代表性的盤面（包括 Pawn Capture, Bad Capture, X-Ray, Pin, En Passant, Promotion 等）進行交叉驗證。
3.  **邏輯修正**：根據對比結果，發現並修復了我的引擎在 **En Passant (EP)** 處理上的邏輯缺陷，並驗證了 **Pin** 檢測的正確性。

## 2. 驗證結果

在最終的測試中，我的引擎與修改後的 Stockfish 在所有測試案例上均達成 **100% 邏輯一致**（數值一致性取決於棋子價值常數，邏輯一致性體現在相對分數與正負號）。

| 測試場景 | 描述 | My SEE | SF SEE | 結果 | 備註 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Simple Capture** | Pawn captures Pawn | 0 | 0 | **MATCH** | 基礎功能正常 |
| **Bad Capture** | Queen captures defended Pawn | -2330 | -2330 | **MATCH** | 負分邏輯正確 |
| **Quiet Move** | Knight move to empty sq | 0 | 0 | **MATCH** | 寧靜步處理一致 |
| **Pawn Push** | Pawn push (quiet) | -208 | -208 | **MATCH** | 基礎價值減扣一致 |
| **PxP** | Pawn exchange chain | 0 | 0 | **MATCH** | 鏈式交換正確 |
| **X-Ray** | RxQ (defended by R behind) | 1262 | 1262 | **MATCH** | X-Ray 穿透檢測正確 |
| **Pin** | Pinned Pawn cannot recapture | 0 | 0 | **MATCH** | Pin 檢測正確（若未檢測到 Pin，此處應為負分） |
| **En Passant** | EP Capture | 0 | 0 | **MATCH** | **已修復**：正確識別 EP 為有效吃子 |
| **Promotion** | Capture + Promotion | 573 | 573 | **MATCH** | 升變價值計算正確 |

*註：測試中為了數值完全匹配，暫時使用了 Stockfish 的棋子價值 (P=208, N=781...)。實際引擎運行時將使用自定義價值。*

## 3. 關鍵修正與發現

1.  **En Passant (EP) 處理**：
    *   **問題**：最初我的 `see` 函數對於 EP 移動返回負分（例如 -100），因為它將 EP 視為「移入空方格（價值0）但被吃回」，忽略了吃掉敵方兵的收益。
    *   **Stockfish 邏輯**：Stockfish 在 `see` 計算中，若檢測到 EP，會將初始 `swap` 值設為 `PawnValue`。
    *   **解決方案**：更新了 `chess_engine/see.py`，增加了對 EP 的檢測邏輯（攻擊者為兵且斜向移動至空方格），並正確賦予其捕獲價值。

2.  **Pin (牽制) 處理**：
    *   Stockfish 的 SEE 能正確識別被 Pin 的棋子無法參與交換。
    *   我的 `see` 函數通過 "Ray-Tracing" 檢查（移動後是否送王）也成功模擬了這一行為。在測試案例 `e2d4`（吃子方被 Pin）中，雙方都給出了 `0`（表示無法吃回，或者吃回是不合法的/被忽略的），證明邏輯一致。

3.  **Promotion (升變)**：
    *   Stockfish 的 `see_ge` 對升變的處理是 `PieceValue[Promotion] - PieceValue[Pawn]` 的價值增量。我的實現也已對齊此邏輯。

## 4. 結論

本次分析與改進已完成。`chess_engine/see.py` 現在是一個高度精確、經過 Stockfish 對標驗證的 SEE 實現。它不僅處理了基礎交換，還正確涵蓋了 X-Ray, Pin, En Passant 和 Promotion 等複雜邊界情況。

這將顯著提升搜尋算法中 Move Ordering（尤其是 Good Captures 排序）和 Quiescence Search 剪枝的準確性。
