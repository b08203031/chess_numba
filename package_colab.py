import os
import shutil
import zipfile
import json

def package():
    root = os.path.dirname(os.path.abspath(__file__))
    staging = os.path.join(root, 'colab_nnue')
    
    # 1. Clean previous staging
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    
    # 2. Copy chess_engine/ excluding heavy or unnecessary artifacts
    print("Copying chess_engine...")
    shutil.copytree(
        os.path.join(root, 'chess_engine'),
        os.path.join(staging, 'chess_engine'),
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.vscode')
    )
    
    # 3. Copy tools/eval_comparison.py and tools/nnue_diagnostic.py
    print("Copying tools...")
    os.makedirs(os.path.join(staging, 'tools'), exist_ok=True)
    shutil.copy2(os.path.join(root, 'tools', 'eval_comparison.py'), os.path.join(staging, 'tools', 'eval_comparison.py'))
    shutil.copy2(os.path.join(root, 'tools', 'nnue_diagnostic.py'), os.path.join(staging, 'tools', 'nnue_diagnostic.py'))
    
    # 4. Create tuner/ directory and a readme inside it
    os.makedirs(os.path.join(staging, 'tuner'), exist_ok=True)
    with open(os.path.join(staging, 'tuner', 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# Tuner Folder\nPlace ultimate_halfka_farseerT75.npz in this directory.\n")
        
    # 5. Create the Jupyter Notebook
    print("Generating Google Colab Notebook...")
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Chess NNUE Colab Training & Evaluation Notebook\n",
                    "本 Notebook 包含在 Google Colab 上掛載雲端硬碟、解壓代碼、訓練 NNUE 模型、驗證推理與進行靜態估值對比的完整指令。\n",
                    "請確保您在執行前已開啟 GPU 加速（執行階段 -> 變更執行階段類型 -> 選擇 T4/L4 或 A100 GPU）。"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 1: 掛載 Google Drive\n",
                    "掛載雲端硬碟以載入大規模資料集與保存模型檢查點。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 2: 解壓縮專案程式碼至雲端硬碟\n",
                    "將 `colab_nnue.zip` 上傳至雲端硬碟根目錄或 `chess` 資料夾下，並在這裡解壓。\n",
                    "解壓在 Google Drive 的好處是：訓練過程中產生的 `best_model.pth` 及量化 `.npy` 權重會**即時同步回您的 Google Drive**，就算 Colab 連線中斷，進度也不會遺失！"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 設定工作目錄在雲端硬碟上的位置\n",
                    "import os\n",
                    "WORKDIR = '/content/drive/MyDrive/chess_nnue'\n",
                    "os.makedirs(WORKDIR, exist_ok=True)\n",
                    "\n",
                    "# 解壓上傳的程式碼壓縮包至雲端硬碟 (請確認 colab_nnue.zip 已放在您的 Drive 根目錄)\n",
                    "# 如果放在其他子資料夾，請調整 /content/drive/MyDrive/colab_nnue.zip 路徑\n",
                    "!unzip -o /content/drive/MyDrive/colab_nnue.zip -d /content/drive/MyDrive/\n",
                    "\n",
                    "# 切換工作目錄\n",
                    "%cd {WORKDIR}"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 3: 設定訓練資料集移動與配置\n",
                    "請先將您的資料集 `ultimate_halfka_farseerT75.npz` 上傳至 Google Drive 根目錄。\n",
                    "在此儲存格設定資料集來源，程式會自動使用 `shutil.move` 將資料集移到工作區的 `tuner/` 目錄下（在同個雲端硬碟內移動僅為元數據修改，瞬間即可完成，免去複製的等待時間）。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import shutil\n",
                    "\n",
                    "# 請修改為您在雲端硬碟上存放資料集的實際路徑：\n",
                    "dataset_src = '/content/drive/MyDrive/ultimate_halfka_farseerT75.npz'\n",
                    "dataset_dst = os.path.join(WORKDIR, 'tuner/ultimate_halfka_farseerT75.npz')\n",
                    "\n",
                    "os.makedirs(os.path.join(WORKDIR, 'tuner'), exist_ok=True)\n",
                    "if os.path.exists(dataset_src) and not os.path.exists(dataset_dst):\n",
                    "    print(\"正在將資料集從雲端硬碟根目錄移動至工作區 tuner 目錄下...\")\n",
                    "    shutil.move(dataset_src, dataset_dst)\n",
                    "    print(\"資料集配置成功！\")\n",
                    "elif os.path.exists(dataset_dst):\n",
                    "    print(\"資料集已存在於工作區 tuner 資料夾下。\")\n",
                    "else:\n",
                    "    print(\"【警告】找不到您的資料集檔案！請確認您已上傳 'ultimate_halfka_farseerT75.npz' 到 Google Drive 根目錄。\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 4: 檢查 GPU 環境\n",
                    "檢查是否成功使用 GPU 加速運行。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "print(\"GPU 是否可用:\", torch.cuda.is_available())\n",
                    "if torch.cuda.is_available():\n",
                    "    print(\"使用 GPU 裝置:\", torch.cuda.get_device_name(0))\n",
                    "else:\n",
                    "    print(\"【嚴重警告】未啟用 GPU 加速！請至選單「執行階段 -> 變更執行階段類型」，在硬體加速器選擇 T4 / L4 或 A100 GPU。\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 5: 開始 / 續訓 NNUE 模型\n",
                    "啟動訓練主循環。若先前已有訓練好的 `best_model.pth` 存放在 `chess_engine/nnue/ml_eval/weights/`，將會自動讀取並繼續增量訓練（Fine-tune）。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python chess_engine/nnue/ml_eval/train.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 6: 運行 NNUE 診斷（推理驗證與權重分析）\n",
                    "此指令會對剛訓練好的權重進行「PyTorch 浮點推論 vs Numba 量化整數推論」的靜態估值對等性校驗（Parity Check），並印出神經元活性與決策層參數統計。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!python tools/nnue_diagnostic.py"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 步驟 7: 進行多位置靜態估值對比（本機 NNUE vs 經典評估 vs 官方 Stockfish）\n",
                    "此指令會先在 Colab 的 Linux 環境下自動安裝 Stockfish，隨後計算並產出多個複雜局面（含中局戰術、殘局國王防護等）的對比報告。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 安裝 Linux Stockfish 引擎\n",
                    "!apt-get update && apt-get install -y stockfish\n",
                    "\n",
                    "# 執行評估對比報告生成\n",
                    "!python tools/eval_comparison.py"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(os.path.join(staging, 'NNUE_Training_Colab.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=4, ensure_ascii=False)
        
    # 6. Zip the folder
    zip_path = os.path.join(root, 'colab_nnue.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print("Zipping package...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for foldername, subfolders, filenames in os.walk(staging):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                arcname = os.path.relpath(filepath, staging)
                zip_file.write(filepath, arcname)
                
    # 7. Clean up staging folder to keep workspace clean
    shutil.rmtree(staging)
    print(f"Success! Packed package at: {zip_path}")

if __name__ == '__main__':
    package()
