# Chess Numba Engine — AI Agent Instructions

專案入口：Python / Numba 西洋棋引擎，雙評估架構 **Classical HCE** 與 **NNUE**。

| 項目 | 內容 |
| :--- | :--- |
| 語言 | Python **3.10** |
| 加速 | Numba `@njit` |
| 資料 | NumPy（含 `np.uint64` 位元棋盤） |
| NNUE 訓練 | PyTorch |
| 協定 | UCI |

---

## 必讀索引

1. **文件地圖**：根目錄 [`README.md`](README.md) →「📂 專案文件導覽」
2. **模組規則**：下方 `.agents/rules/`（有 frontmatter 的工具會自動載入；否則**手動開啟**與當前檔案對應的規則）

### Always-on

| 規則 | 內容 |
| :--- | :--- |
| [`.agents/rules/workflow.md`](.agents/rules/workflow.md) | 改前讀文件、改後同步導覽、輕/重測試、JIT 耐心、Stockfish 參考路徑 |
| [`.agents/rules/numba-jit.md`](.agents/rules/numba-jit.md) | nopython 禁令、棋盤陣列型別、`cache` 預設與 NNUE 例外 |

### Glob（依編輯路徑）

| 規則 | 觸發（摘要） | 內容 |
| :--- | :--- | :--- |
| [board-integrity.md](.agents/rules/board-integrity.md) | `board_operations` / `move_generator` / `zobrist` / `bitboard_utils` / `move` / `fen_parser` | 佔用同步、make/unmake、Zobrist |
| [search-constraints.md](.agents/rules/search-constraints.md) | `search` / `search_heuristics` / `time_manager` / `transposition_table` | Negamax、將死距離、重複/50 步、剪枝安全 |
| [hce-evaluation.md](.agents/rules/hce-evaluation.md) | `evaluation` / `pawns` / `material` / `endgame` / `constants` | HCE↔SF11、殘局尺度 |
| [nnue-specification.md](.agents/rules/nnue-specification.md) | `chess_engine/nnue/**/*.py` | 拓撲、HalfKA、累加器、整數推理 |
| [nnue-training.md](.agents/rules/nnue-training.md) | `chess_engine/nnue/ml_eval/**/*.py` | Factorizer 導出、loss、LR、採樣 |

### 參考原始碼（倉庫相對路徑）

* 搜尋：`stockfish_repo/`（SF 18）
* HCE：`stockfish_11/`（SF 11）

---

## 文件與目錄政策

* 技術 Markdown **唯一索引** = `README.md` 文件導覽；增刪更名必須同步。
* **禁止**在根目錄堆積 session / 一次性計畫；考古放 `archive/`。
* `classical_old/` 僅 `tools/sync_classical_old.py` 同步 **`.py`**，不維護 md。
* 詳細約束以 `.agents/rules/*.md` 為準；本檔只做導覽，不重複貼全文。
