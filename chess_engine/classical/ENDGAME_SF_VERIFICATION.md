# 專用殘局 vs Stockfish 驗證說明

**範圍：** `chess_engine/classical` 專用殘局評估（`endgame.py`）與 Stockfish 搜尋對照  
**目的：** 用可重跑的 oracle 與單元測試，確認 specialized EG 勝負方向正確。

**Value 短路（SF11 子集）：** KXK · KPK · KRKP · KQKP · KBNK · **KNNK · KRKB · KRKN · KQKR · KNNKP**

本文件**不寫死**某一輪的 cp 數值表；以「怎麼驗證」為主。最新一輪表格可用：

```bash
python tools/verify_endgame_vs_sf.py --write-md
```

寫入的是**當次** oracle 結果表（可選）；日常回歸以單元測試為準。

---

## 1. 驗證工具

| 工具 | 路徑 | 用途 |
| :--- | :--- | :--- |
| SF 搜尋 oracle | `tools/verify_endgame_vs_sf.py` | 外部 Stockfish deep search + 本引擎 special/eval |
| **KPK golden** | `tests/test_kpk_golden.py`、`tools/verify_kpk_golden.py`、`tools/kpk_sf11_reference.py` | 獨立 SF11 參考 bitbase 全表對位 + 理論 FEN；可選 SF 抽樣 |
| 殘局單元測試 | `tests/test_endgame_conformance.py` | HCE 尺度勝負回歸（`VALUE_KNOWN_WIN ≈ 10000`） |
| HCE 階段 A 測試 | `tests/test_phase_a_eval.py` | A1–A4、KPK 邊角等 |

Stockfish 預設：`external/stockfish/stockfish-windows-x86-64-avx2.exe`（可用 `--sf` 覆寫）。

---

## 2. 執行方式

```bash
# 日常 / CI 友好
python -m unittest tests.test_endgame_conformance tests.test_phase_a_eval tests.test_kpk_golden -v

# KPK golden CLI（全表 vs 獨立參考 + 理論 FEN）
python tools/verify_kpk_golden.py
# 可選：再抽 N 個局面對外部 SF 搜尋（勿當 CI）
python tools/verify_kpk_golden.py --sf-sample 48 --depth 20

# 偶爾對照 SF（需本機 SF；勿在 CI 自動跑）
python tools/verify_endgame_vs_sf.py
python tools/verify_endgame_vs_sf.py --depth 22 --write-md
```

---

## 3. 判定注意

1. 現代 Stockfish **NNUE** 靜態 `eval` 與 SF11 HCE **尺度不同**，不可直接比 `10000`。
2. Oracle 以 **搜尋** 的 mate/cp **方向** 為準，不是 HCE 分數對齊。
3. 新增殘局：先寫 `test_endgame_conformance.py`，再跑 oracle 交叉確認。

## 4. 相關原始碼

| 路徑 | 角色 |
| :--- | :--- |
| `chess_engine/classical/endgame.py` | 專用殘局 + KPK bitbase |
| `chess_engine/classical/evaluation.py` | 先 special EG，再一般 HCE |
| `chess_engine/classical/HCE_SF11_GAP_AUDIT.md` | HCE 對齊主文件 |
| `stockfish_11/src/endgame.cpp` / `bitbase.cpp` | 參考 |
