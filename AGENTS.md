# Chess Numba Engine — AI Agent Instructions & Guidelines

歡迎來到本專案！此檔案是 AI 助手在進入工作區時的首要導覽指南。旨在為各種 AI 助理（如 Google Antigravity、Gemini、Cursor、Claude Code 等）提供專案背景與開發導引。

---

## 專案簡介 (Project Overview)

本專案是一個採用 **Python** 開發的高性能西洋棋引擎，核心運算透過 **Numba JIT** 靜態編譯為機器碼，以發揮極致效能。引擎具備兩個主要評估架構：

1. **Classical 模組**：傳統手工評估與精心調優的 Alpha-Beta 搜尋。
2. **NNUE 模組**：半精緻神經網路評估，包含 PyTorch 訓練模型與 Numba 高速推理引擎。

### 核心技術棧 (Technology Stack)

* **語言環境**：Python 3.10
* **編譯加速**：Numba JIT Compiler (`@numba.njit`)
* **資料儲存**：NumPy Arrays（強烈類型約束，如 `np.uint64` 位元棋盤）
* **神經網路**：PyTorch (NNUE 訓練、驗證與導出)
* **通訊協定**：UCI (Universal Chess Interface)

---

## 模組化規則指引 (Workspace Rules Directory)

為避免上下文污染並提高任務的精準度，本專案的詳細開發限制已拆分為**模組化規則檔案**，並儲存於 `.agents/rules/` 目錄下。

AI 助手在執行任務時，應依據觸發條件自動載入並遵循以下特定檔案中的規範：

1. **工作流程與測試規範**：參見 [workflow.md](file:///.agents/rules/workflow.md) (Always-on)
    * *重要提醒*：
        * **工作前讀文檔**：修改任何檔案前，必須先閱讀 `README.md` 的「📂 專案文件導覽 (Documentation Map)」以檢索相關背景。
        * **工作後更新文檔**：新增功能或重構後，必須主動更新技術文檔並同步 `README.md` 中的導覽索引。
        * **嚴禁自動測試**：AI 助手嚴禁自動執行測試，必須引導使用者手動執行。
2. **Numba 編譯相容性限制**：參見 [numba-jit.md](file:///.agents/rules/numba-jit.md) (Always-on)
    * 標準 Numba `nopython` 限制（禁止 Class, dynamic types, dict/set comprehensions, try/except）與專案數組規範。
3. **棋盤完整性與可逆性**：參見 [board-integrity.md](file:///.agents/rules/board-integrity.md) (Glob: `board_operations.py`, `board.py`, `move_generator.py`, `zobrist.py`)
    * 位元棋盤佔用同步、Make/Unmake 原地修改約束與 Zobrist 增量維護。
4. **搜尋演算法正確性**：參見 [search-constraints.md](file:///.agents/rules/search-constraints.md) (Glob: `search.py`, `search_heuristics.py`)
    * Negamax 對稱性、將死分數 ply 調整、重複/50步和局判定與剪枝安全防護。
5. **NNUE 自定義架構開發與推理**：參見 [nnue-specification.md](file:///.agents/rules/nnue-specification.md) (Glob: `nnue/**/*.py`)
    * 包含 HalfKA 特徵編碼、8分桶架構、ClippedReLU與整數化量化推理的極致效能約束。
6. **NNUE 離線訓練與權重導出**：參見 [nnue-training.md](file:///.agents/rules/nnue-training.md) (Glob: `nnue/ml_eval/**/*.py`)
    * 包含 Factorizer與分桶殘差的重參數化（Structural Re-parameterization）、差異化學習率與Weight Decay、分桶平衡採樣與 physical 物理防爆極限約束。

