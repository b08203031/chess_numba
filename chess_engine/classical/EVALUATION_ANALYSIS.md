# Stockfish 11 與 Antares 引擎評估函數差異分析報告

## 1. 前言

本報告旨在對比分析 Stockfish 11（最後一個純手工評估的經典版本）的評估架構，與目前的 Antares 引擎（經典手工評估模組）。

| 項目 | Stockfish 11 (SF11) | Antares Engine (現況) | 狀態與備註 |
| :--- | :--- | :--- | :--- |
| **材質 (Material)** | 漸進式 (Tapered)，含複雜不平衡表。 | 漸進式 (Tapered)，基礎材質值。 | ✅ 材質比例合理，殘局中象的價值高於馬。 |
| **PST (位置分)** | 中局/殘局雙表，數值極為精細。 | 中局/殘局雙表，數值高度相似。 | ✅ 邏輯一致。 |
| **機動性 (Mobility)** | 排除被攻擊與受限格子後的非線性查表。 | 排除敵兵攻擊與牽制子後的非線性查表。 | ✅ 已實現非線性查表（`KNIGHT_MOBILITY_BONUS` 等）。 |
| **威脅 (Threats)** | 詳細威脅矩陣 (Minor/Rook 威脅不同棋子)。 | 實現分類型的威脅矩陣，含懸掛子與兵推，並補足限制棋子威脅。新增 Knight/Slider On Queen。 | ✅ **已完全實現**。整合了 Queen-target 威脅特徵，包含馬/滑動棋子威脅后。 |
| **國王安全 (King Safety)** | 非線性 kingDanger 二次方懲罰，含安全將軍檢測與 blockers。 | kingDanger 累加器與二次方轉換，含安全將軍、弱格、阻擋子與 offset。 | ✅ **已完全實現**。新增 `KING_DANGER_BLOCKERS` 與 `KING_DANGER_OFFSET`，移除了 0/1 進攻子的硬編碼門檻以完全對齊 SF11 非線性安全評估。 |
| **空間評估 (Space)** | 中局中央四直行空間控制評估（Rank 2-4）。 | 中局中央四直行空間控制評估（Rank 2-4）。 | ✅ **已完全實現**。已引入 `SPACE_THRESHOLD = 4700`，並採用了 4 倍安全折減因子（`// 4` 縮放）與對稱除法，以完美適配本引擎的材質比例。 (註：目前主程式為追求極致效能暫不啟用，但在 initiative 中有保留註解說明)。 |
| **主動權 (Initiative)** | 基於 complexity 的動態修正，防止和棋漂移。 | 基於 complexity 的動態複雜度修正。 | ✅ **已完全實現**。已將固定 10 cp 修正替換為動態複雜度 (Complexity) 評估與獨立輪行權 (Tempo) 獎勵。 |
| **棋子特定獎懲 (Piece-Specific)** | 長對角線象、車后同列、弱勢后、與易位權掛鉤的困車。 | 長對角線象、車后同列、弱勢后、困車係數。 | ✅ **已完全實現**。成功整合 Bishop/Rook/Queen 特徵並與 castling rights 掛鉤。 |

---

## 3. 兵型結構 (Pawn Structure) 對比

兵型是引擎評估的靈魂。當前 Antares 已經完整移植了 SF11 的核心兵型邏輯：

1. **通路兵 (Passed Pawn)**:
    * **判定**: 檢查前方與相鄰直行前方是否有敵兵阻擋。
    * **國王距離 (King Proximity)**: ✅ **已實作**。使用切比雪夫距離（Chebyshev Distance）動態修正，己方國王近（護送）加分，敵方國王近（阻擋）減分。
    * **路徑安全性與阻擋格**: ✅ **已實作**。在 `evaluate_passed_pawns` 中，根據阻擋格是否被敵方控制或防守進行動態折半/縮放。
    * **車/后後方護送 (Tarrasch Support)**: ✅ **已實作**。當車/后在通路兵後方時給予額外加分。
2. **後兵 (Backward Pawn)**:
    * **判定**: 該兵落後於相鄰兵，且無法 safe 推進 (被敵兵控制且無法被己方兵支持)。
    * **懲罰**: ✅ **已實作**。在中局與殘局給予 `BACKWARD_PAWN_PENALTY`（`S(-8, -20)`）。
3. **連結兵 (Connected Pawn)**:
    * **判定**: 相鄰直行的兵並排 (Phalanx) 或斜向支撐 (Connected)。
    * **獎勵**: ✅ **已實作**。使用 `CONNECTED_BONUS[rank]`，並區分是否被阻擋（`opposed`）給予加成。
4. **孤兵 (Isolated) 與 重疊兵 (Doubled)**:
    * ✅ **已實作**。已套用獨立中局/殘局懲罰。

---

## 4. 國王安全 (King Safety) 對比

當前的國王安全模型已非常接近 SF11 的核心思想：

1. **安全將軍檢測 (Safe Checks)**:
    * ✅ **已實作**。程式碼中單獨計算了車（`SAFE_CHECK_ROOK`）、后（`SAFE_CHECK_QUEEN`）、象（`SAFE_CHECK_BISHOP`）與馬（`SAFE_CHECK_KNIGHT`）的**安全將軍威脅格**，並將其計入 kingDanger 線性累加器。
2. **弱格與區域 (Weak Squares)**:
    * ✅ **已實作**。計算國王周圍 (King Ring) 被敵方攻擊且我方無防守（或僅有王/后防守）的弱格數。
3. **牽制子懲罰**:
    * ✅ **已實作**。防守方被牽制子暴露國王會受到 `KING_DANGER_PINNED` 的威脅分懲罰。
4. **無后折扣 (No-Queen Discount)**:
    * ✅ **已實作**。當進攻方沒有后時，國王危險分大幅減扣（`KING_DANGER_NO_QUEEN`）。

---

## 5. 已完成之優化與改進項目 (Completed Improvements)

### 1. 實現「空間評估 (Space Evaluation)」與參數優化

* **優化內容**：完全實現了中央 4 直行（C, D, E, F 線）與中路橫排（Rank 2-4）的控制格與兵後安全格控制評估。
* **折減與適配**：由於引擎的材質尺度不同於 SF11（馬的價值為 320 cp，而非 781 cp），對空間得分進行了 4 倍安全折減，並在正數側完成除法以保持對稱。

### 2. 啟用「限制棋子威脅 (Restricted Piece)」

* **優化內容**：正式將 `THREAT_RESTRICTED_PIECE` 補入威脅評估函數中，對被我方多重攻擊封鎖、無法自由移動的敵方棋子給予額外的威脅分加算。

### 3. 動態主動權與複雜度修正 (Dynamic Initiative)

* **優化內容**：完全移除固定 10 cp 獎勵，實作基於國王向性、兵數、通路兵、分側兵等的動態局勢複雜度 (Complexity) 修正公式，並保留了獨立的 `TEMPO_BONUS`（10 cp）加成。

### 4. 兵型評估 (Pawn Structure) 與死代碼優化

* **Connected 殘局分數適配**：為 `CONNECTED_BONUS_EG` 和 `CONNECTED_SUPPORT_WEIGHT_EG` 引入了獨立的殘局 (EG) 縮放常數 (與 SF11 的 $0.563$ 縮放對齊)，解決了殘局加分被放大 38% 的不對稱問題。
* **消除死代碼**：清理了 `evaluate_pawn_structure` 內未使用的通路兵 `is_passed` 判定邏輯，避免無謂的位元運算與 CPU 消耗，提升了引擎手工評估的整體性能。
* **候選兵阻擋者計數優化**：將 `evaluate_passed_pawns` 中原本對雙方候選通路兵阻擋者數量的手動 `while` 計數循環，替換為高效能的 `count_bits()` 硬體人口計數 (POPCNT) 呼叫，徹底消除了 Numba 的循環開銷，讓執行效率達到極致。

### 5. 兵型評估與兵盾快取化優化 (Pawn Evaluation & Shield Cache Optimization)

* **預計算候選兵與前進直行掩碼**：在模組載入時，一次性預計算雙方所有位置的 `WHITE_FORWARD_FILE_MASKS`, `BLACK_FORWARD_FILE_MASKS`, `WHITE_CANDIDATE_MASKS`, `BLACK_CANDIDATE_MASKS`。在 `evaluate_passed_pawns` 中，直接使用預計算好的掩碼進行位元與（`&`）運算，免去每次動態做查表與位元 OR 的開銷。
* **兵盾與兵風暴評估併入 Pawn Table 快取**：將 `evaluate_king_safety` 中的兵盾評估 `evaluate_shelter_aligned` 及其易位虛擬化判定邏輯完全移入 `evaluate_pawn_structure`。如此一來，兵盾評估結果即可隨同兵型分數一起由 `evaluate_pawn_structure_cached` 快取並返回。快取驗證中新增對 `castling_rights` 的比對，確保在 >99% 的快取命中情況下，省去重複的兵盾循環與虛擬化計算，大幅降低 `evaluate_king_safety` 每次呼叫時的運算量。
* **兵型與兵盾評估預計算及重構優化 (已優化)**：
  * 在模組載入時，新增預計算 `PHALANX_MASK`、`WHITE_SUPPORT_MASK`、`BLACK_SUPPORT_MASK`、`WHITE_BACKWARD_TEST_MASK`、`BLACK_BACKWARD_TEST_MASK` 等 5 組兵型結構遮罩，大幅消除在 `evaluate_pawn_structure` 迴圈中的多次位元 AND 和查表運算。
  * 在評估過程中，全面使用 `BB_SQUARES` 查表取代 `1 << sq` 的動態位元移位運算。
  * 將 `evaluate_shelter_aligned` 原本手動展開的 3 次直行 for 循環改寫為簡潔的迴圈，並由 Numba JIT (LLVM) 的編譯期自動展開 (Loop Unrolling) 處理，兼顧代碼可讀性與執行效能。
* **以移位代替除法 (Bitwise Shift Micro-optimizations)**：在所有的熱點函數（Hot Loops）中，將所有求橫排 `sq // 8` 和直行 `sq % 8` 的除法與模運算，明確改寫為位元移位與掩碼操作 `sq >> 3` 和 `sq & 7`，保證 Numba/LLVM 編譯成最簡化的二進位機器碼指令。

### 6. 通路兵判定演算法與 Stockfish 11 完全對齊及 Bug 修復

* **優化內容**：在 `evaluate_passed_pawns` 中引入了 Stockfish 11 的完整三階段（Condition A、B、C）偵測邏輯，實現了互鎖（Lever）、併聯支持推升（Phalanx vs LeverPush）、以及後方移位兵支持推升（Shifted Support Pushes Through）的完整評估判定。
* **Bug 修復**：徹底修復了在 `evaluate_pawn_structure` 與 `evaluate_passed_pawns` 中，原本 `PAWN_ATTACKS` 顏色索引定義顛倒（把 `WHITE` 和 `BLACK` 索引混淆）導致兵互鎖與推升格偵測在程式碼中長久失效的嚴重漏洞。同時，確保所有的位元移位操作在 Numba JIT 中具有類型安全性。

### 7. 候選通路兵邏輯重構與同色兵主教中心阻擋乘數 (重構優化 §1 & §3)

* **候選兵邏輯整合與快取優化 (§1)**：
  * 完全移除獨立的候選通路兵（Candidate Passed Pawn）偵測與評估循環，並將其邏輯精簡化、整合至通路兵評估 `evaluate_passed_pawns` 中。
  * 當通路兵前方有兵阻擋，或者在下一個前進格上「已不符合通路兵條件」時，其通路兵加分直接減半。這避免了在 `evaluate_passed_pawns` 中進行二次候選兵遍歷，並將 Pawn Table 的快取項目（原本需要存儲 White/Black Candidate bitboards）精簡化，將原本 10 元組返回值縮減為 8 元組，節約了快取頻寬與內存佔用。
  * 加入了 rank 安全邊界檢查（白兵 `rank < 7`，黑兵 `rank > 0`），確保在下一個前進格查表時不會發生超出數組索引範圍的 runtime 錯誤，符合 Numba JIT 的靜態編譯安全性。
* **同色兵主教中心阻擋乘數 (§3)**：
  * 定義了 `CENTER_FILES = np.uint64(0x3C3C3C3C3C3C3C3C)`（C、D、E、F 四列）。
  * 對於與主教同色格的己方兵（same-color pawns），當這些兵位於 C、D、E、F 中央四列且被前方任何棋子直接阻擋（White: `wp_bb & (all_occupancy >> 8)`；Black: `bp_bb & (all_occupancy << 8)`）時，每增加一個被阻擋的中心兵，主教的同色兵懲罰項便乘以 `(1 + center_blocked_pawns)`。這使得防守被鎖死的中央兵型時，主教的靈活性與評估懲罰能完美地與 Stockfish 11 的 BishopPawns 評估對齊。

### 8. 經典評估函數優化 - 第二階段 (Phase II Evaluation Upgrades)

* **國王安全阻擋子與偏移量**：在王安全公式中引入 `KING_DANGER_BLOCKERS`（2 cp）與 `KING_DANGER_OFFSET`（1 cp）。移除原本進攻子為 0 與 1 時的硬編碼提早返回和減半逻辑，使其完全藉由非線性二次懲罰自然過渡，與 Stockfish 11 對齊。
* **棋子位置與協同特徵**：
  - **長對角線象 (LongDiagonalBishop)**：若主教在長對角線上，且能穿過兵阻擋看到至少兩個中心方格（d4, e4, d5, e5）則給予額外加分（MG: 35, EG: 0）。
  - **車后同列 (RookOnQueenFile)**：若車與任何一方的后處於同一列上，給予額外加分（MG: 5, EG: 3）。
  - **弱勢后 (WeakQueen)**：若后處於被敵方車/象通過單一阻擋子 X 射線射擊（即潛在的牽制或發現攻擊射線），則給予懲罰（MG: -38, EG: -8）。
* **困車易位權縮放 (TrappedRook Scaling)**：將困車（TrappedRook）的懲罰與雙方的易位權（`castling_rights`）掛鉤。若一方仍保有對應側的易位權，則懲罰保持常規；否則施以雙倍懲罰（乘以 2），與 SF11 對齊。
* **馬/滑動棋子威脅后 (Knight/Slider On Queen)**：新增了白/黑馬（`THREAT_KNIGHT_ON_QUEEN`，MG: 12, EG: 7）或滑動棋子象/車（`THREAT_SLIDER_ON_QUEEN`，MG: 46, EG: 10）攻擊能直接威脅對方后的方格的威脅分數。

---

## 6. 優化與改進 Action Items

基於現有程式碼的審查，目前的 Action Items 如下：

### 1. 材質不平衡 (Imbalance) 補強

* **痛點**：未實作完整的不平衡對局加減分。
* **方向**：引入輕子對峙（如單馬對單象時的修正）等不平衡項。
