# 引擎對戰測試指南 (Tournament Tutorial)

本文件將引導您如何使用 `tournament.py` 腳本，讓兩個不同版本的西洋棋引擎（例如當前版本的 `chess_engine` 和另一版本的 `chess_engine_v2`）進行對戰。

## 環境準備

在開始測試之前，請確保您的環境已經安裝了所需的依賴套件。

您可以透過以下指令安裝必要的 Python 套件：

```bash
pip install chess numpy numba
```

## 基本使用方法

腳本已經預設好了 `chess_engine` (當前目錄的 `main.py`) 以及 `chess_engine_v2` (`chess_engine_v2/main.py`) 作為對戰雙方。如果您只需要進行預設的對戰，只需執行以下指令：

```bash
python tournament.py
```

這會以預設設定進行 20 場比賽，每步考慮時間為 100 毫秒。

## 自訂對戰參數

您可以透過命令列參數自訂對戰：

```bash
python tournament.py --games 50 --time 200
```

- `--games`：總對戰場數（預設：20）
- `--time`：每步考慮時間，單位為毫秒（預設：100）

### 自訂引擎路徑與名稱

如果您想要測試其他版本的引擎，可以使用 `--engine1`, `--engine2` 參數指定對應的 `main.py` 路徑，並使用 `--name1`, `--name2` 參數為它們命名：

```bash
python tournament.py --engine1 custom_engine/main.py --name1 custom_engine --engine2 main.py --name2 default_engine --games 10
```

## 查看結果

對戰進行時，終端機會即時顯示：
- 每場比賽的結果
- 目前的累計得分

對戰結束後，腳本會輸出最終的統計數據，包括：
- 總比賽場數
- 雙方的最終得分、勝平負場數
- 基於對戰結果估算出來的 **Elo 差異** 及 **95% 信賴區間**

### 檢視棋譜 (PGN)

所有的對局記錄將會自動儲存到當前目錄下的 `tournament_results.pgn` 檔案中。您可以將此檔案匯入任何西洋棋 GUI 軟體（如 Arena, Cute Chess, Lichess 等）來回放並分析對局。