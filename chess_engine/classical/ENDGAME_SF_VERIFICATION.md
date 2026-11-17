# 專用殘局驗證說明

**範圍：** `chess_engine/classical/endgame.py` 專用殘局評估與 Stockfish 搜尋對照  
**目的：** 以可重跑的單元測試確認 specialized EG 的勝負方向與 HCE 尺度（`VALUE_KNOWN_WIN ≈ 10000`）正確。

**Value 短路（SF11 子集）：** KXK · KPK · KRKP · KQKP · KBNK · KNNK · KRKB · KRKN · KQKR · KNNKP

---

## 1. 現行驗證工具

| 工具 | 路徑 | 用途 |
| :--- | :--- | :--- |
| 殘局單元測試 | `tests/test_endgame_conformance.py` | HCE 尺度勝負回歸與各專用殘局不變量 |
| SF11 對齊回歸 | `tests/test_phase4_sf11_alignment.py` | 與 SF11 結構對齊的評估回歸 |

```bash
python -m unittest tests.test_endgame_conformance tests.test_phase4_sf11_alignment -v
```

> **已移除的工具**：`tools/verify_endgame_vs_sf.py`（SF 搜尋 oracle）、`tools/verify_kpk_golden.py`、`tools/kpk_sf11_reference.py`、`tests/test_kpk_golden.py`、`tests/test_phase_a_eval.py` 已於 commit `ae30946`（2026-09-26）清理。需要重新做 KPK 全表或 SF oracle 對照時，用 `git show ae30946^:<path>` 取回。

---

## 2. 判定注意

1. 現代 Stockfish **NNUE** 靜態 `eval` 與 SF11 HCE **尺度不同**，不可直接比 `10000`。
2. SF oracle 以**搜尋**的 mate/cp **方向**為準，不是 HCE 分數對齊。
3. 新增殘局：先寫 `tests/test_endgame_conformance.py`，再視需要用外部 SF 交叉確認。

## 3. 相關原始碼

| 路徑 | 角色 |
| :--- | :--- |
| `chess_engine/classical/endgame.py` | 專用殘局 + KPK bitbase |
| `chess_engine/classical/evaluation.py` | 先 special EG，再一般 HCE |
| `chess_engine/classical/HCE_SF11_GAP_AUDIT.md` | HCE 對齊主文件 |
| `stockfish_11/src/endgame.cpp` / `bitbase.cpp` | 參考 |
