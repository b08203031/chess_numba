import torch
import os
import sys

# 加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chess_engine.ml_eval.train import SimpleNNUE, export_weights

def convert():
    device = torch.device("cpu")
    model = SimpleNNUE()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'chess_engine', 'ml_eval', 'weights')
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    if not os.path.exists(best_model_path):
        print(f"找不到最佳模型：{best_model_path}")
        return
        
    print(f"正在載入模型：{best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 執行新版的 export_weights (會存成 int16)
    export_weights(model)
    print("量化轉檔完成！")

if __name__ == "__main__":
    convert()
