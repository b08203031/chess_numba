# 引擎對戰測試指南 (Tournament Tutorial)

本文件將引導您如何使用 [tournament.py](tournament.py) 腳本，讓兩個不同版本的西洋棋引擎進行對戰。

## 環境準備

在開始測試之前，請確保您的環境已經安裝了所需的依賴套件。

您可以透過以下指令安裝必要的 Python 套件：

```bash
pip install chess numpy numba
```

## 基本使用方法

在工作區根目錄下，您可以使用以下指令啟動對戰（該腳本預設會尋找根目錄的 `main.py`）：

```bash
python tools/tournament.py
```

這會以預設設定進行 20 場比賽，每步考慮時間為 100 毫秒。

## 自訂對戰參數

您可以透過命令列參數自訂對戰：

```bash
python tools/tournament.py --games 50 --time 200
```

- `--games`：總對戰場數（預設：20）
- `--time`：每步考慮時間，單位為毫秒（預設：100）

### 自訂引擎路徑與名稱

如果您想要測試其他版本的引擎，可以使用 `--engine1`, `--engine2` 參數指定對應的 `main.py` 路徑，並使用 `--name1`, `--name2` 參數為它們命名：

```bash
python tools/tournament.py --engine1 archive/main_old.py --name1 old_version --engine2 main.py --name2 new_version --games 50
```

## 查看結果

對戰進行時，終端機會即時顯示：
- 每場比賽的結果
- 目前的累計得分

對戰結束後，腳本會輸出最終的統計數據，包括：
- 總比賽場數
- 雙方的最終得分、勝平負場數
- 基於對戰結果估算出來的 **Elo 差異 (Estimated ELO Difference)** 及 **95% 信賴區間**。這有助於利用統計學方法（如 SPRT 測試）來判斷代碼修改是否真正帶來了棋力增長。

### 檢視棋譜 (PGN)

所有的對局記錄將會自動儲存到工作區根目錄下的 **`tournament_analysis/tournament_results.pgn`** 檔案中。您可以將此檔案匯入任何西洋棋 GUI 軟體（如 Arena, Cute Chess, Lichess 等）來回放並分析對局。
