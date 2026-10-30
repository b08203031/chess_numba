path = 'chess_engine/classical/SEARCH_ANALYSIS.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = """### 4.5 晚期移動剪枝 (Late Move Pruning, LMP)
*   **理念**：如果在一個節點中已經搜尋了大量的安靜移動卻仍未找到更好的移動，那麼剩餘排序極後的安靜移動不太可能有所幫助。
*   **邏輯**：
    *   條件：已嘗試的安靜移動數超過特定深度的閾值。
    *   操作：直接忽略該節點剩餘的所有安靜移動。"""

replacement = """### 4.5 晚期移動剪枝 (Late Move Pruning, LMP)
*   **理念**：如果在一個節點中已經搜尋了大量的安靜移動卻仍未找到更好的移動，那麼剩餘排序極後的安靜移動不太可能有所幫助。
*   **邏輯與計算公式**：
    *   條件：非 PV 節點且盤面無將軍，已嘗試的安靜移動數超過特定深度與局面改善狀態計算的動態閾值：
        $$\\text{limit} = \\max\\left(\\text{LMP\\_MIN\\_LIMIT}, \\frac{\\text{LMP\\_MOVE\\_COUNT}[\\text{depth}]}{2 - \\text{improving}} \\times \\frac{\\text{LMP\\_SCALE\\_PERCENT}}{100}\\right)$$
    *   操作：超過此門檻的安靜著法直接跳過搜尋（`DIAG_LMP_SKIP`）。
*   **時控敏感性與深層盲區實證（E99，2026-10-28）**：
    *   在短時控（10k nodes）下過寬會浪費節點；但在長時控（400k nodes LTC）下，搜尋深入至 $d = 20 \\sim 28$，若門檻過緊（如 135% 或 150%），深層安靜走法過早被截斷，造成嚴重的深層殘局與戰術防守盲區（135% 錄得 -30.7 Elo 大敗）。
    *   放寬至 `LMP_SCALE_PERCENT = 165%` 於 400k 長時控下取得 53.1% 勝率（+21.6 Elo，淨勝 +23 局，LLR=+1.16），成功化解深層防守盲區。"""

assert target in text, "target not found in SEARCH_ANALYSIS.md"
text = text.replace(target, replacement, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated SEARCH_ANALYSIS.md successfully.")

