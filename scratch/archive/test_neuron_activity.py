import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Copy definitions from train.py
class ClippedReLU(nn.Module):
    def __init__(self, upper_bound=127.0):
        super(ClippedReLU, self).__init__()
        self.upper_bound = upper_bound
        
    def forward(self, x):
        return torch.clamp(x, 0.0, self.upper_bound)

class LayerStackNNUE(nn.Module):
    def __init__(self, num_buckets=8):
        super(LayerStackNNUE, self).__init__()
        self.num_buckets = num_buckets

        self.fc1 = nn.EmbeddingBag(22529, 512, mode="sum", padding_idx=22528)
        self.fc1_bias = nn.Parameter(torch.zeros(512))
        self.crelu = ClippedReLU(127.0)

        self.fact_fc2 = nn.Linear(1024, 32)
        self.fact_fc3 = nn.Linear(32, 32)
        self.fact_fc4 = nn.Linear(32, 1)

        self.delta_fc2 = nn.ModuleList([nn.Linear(1024, 32) for _ in range(num_buckets)])
        self.delta_fc3 = nn.ModuleList([nn.Linear(32, 32) for _ in range(num_buckets)])
        self.delta_fc4 = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_buckets)])

        for b in range(num_buckets):
            nn.init.zeros_(self.delta_fc2[b].weight)
            nn.init.zeros_(self.delta_fc2[b].bias)
            nn.init.zeros_(self.delta_fc3[b].weight)
            nn.init.zeros_(self.delta_fc3[b].bias)
            nn.init.zeros_(self.delta_fc4[b].weight)
            nn.init.zeros_(self.delta_fc4[b].bias)

    def forward(self, x_stm, x_nstm, bucket_ids):
        # We need model to refer to self here for this standalone test
        model = self
        acc_stm  = self.crelu(self.fc1(x_stm)  + self.fc1_bias)
        acc_nstm = self.crelu(self.fc1(x_nstm) + self.fc1_bias)
        x = torch.cat((acc_stm, acc_nstm), dim=1)

        output = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        for b in range(self.num_buckets):
            mask = (bucket_ids == b)
            if mask.any():
                xb = x[mask]
                w2 = self.fact_fc2.weight + self.delta_fc2[b].weight
                b2 = self.fact_fc2.bias   + self.delta_fc2[b].bias
                xb = self.crelu(F.linear(xb, w2, b2))

                w3 = model.fact_fc3.weight + model.delta_fc3[b].weight
                b3 = model.fact_fc3.bias   + model.delta_fc3[b].bias
                xb = self.crelu(F.linear(xb, w3, b3))

                w4 = model.fact_fc4.weight + model.delta_fc4[b].weight
                b4 = model.fact_fc4.bias   + model.delta_fc4[b].bias
                output[mask] = F.linear(xb, w4, b4).to(output.dtype)

        return output

def fen_to_halfka_indices(fen):
    parts = fen.split(' ')
    board_part = parts[0]
    stm_part = parts[1]
    
    pieces = {}
    ranks = board_part.split('/')
    for r_idx, rank in enumerate(ranks):
        rank_val = 7 - r_idx
        file_val = 0
        for char in rank:
            if char.isdigit():
                file_val += int(char)
            else:
                sq = rank_val * 8 + file_val
                pieces[sq] = char
                file_val += 1
                
    piece_count = len(pieces)
    stm = 0 if stm_part == 'w' else 1
    
    wk_sq = next(sq for sq, char in pieces.items() if char == 'K')
    bk_sq = next(sq for sq, char in pieces.items() if char == 'k')
    
    def get_indices(active_side):
        if active_side == 0:
            k_sq = wk_sq
            opp_k_sq = bk_sq
        else:
            k_sq = bk_sq
            opp_k_sq = wk_sq
            
        if active_side == 1:
            k_sq_sym = k_sq ^ 56
        else:
            k_sq_sym = k_sq
            
        flip_h = (k_sq_sym & 7) < 4
        if flip_h:
            king_mapped = k_sq_sym ^ 7
        else:
            king_mapped = k_sq_sym
            
        bucket = (7 - (king_mapped >> 3)) * 4 + (7 - (king_mapped & 7))
        
        indices = []
        # Own king
        sq_own_mapped = k_sq ^ (56 if active_side == 1 else 0)
        if flip_h: sq_own_mapped ^= 7
        indices.append(bucket * 704 + 10 * 64 + sq_own_mapped)
        
        # Opp king
        sq_opp_mapped = opp_k_sq ^ (56 if active_side == 1 else 0)
        if flip_h: sq_opp_mapped ^= 7
        indices.append(bucket * 704 + 10 * 64 + sq_opp_mapped)
        
        PIECE_CHARS = ['P', 'N', 'B', 'R', 'Q'] if active_side == 0 else ['p', 'n', 'b', 'r', 'q']
        OPP_CHARS = ['p', 'n', 'b', 'r', 'q'] if active_side == 0 else ['P', 'N', 'B', 'R', 'Q']
        
        for sq, char in pieces.items():
            if char in ['K', 'k']:
                continue
            if char in PIECE_CHARS:
                channel = PIECE_CHARS.index(char)
                mt = channel * 2
            elif char in OPP_CHARS:
                channel = OPP_CHARS.index(char)
                mt = channel * 2 + 1
            else:
                continue
                
            sq_mapped = sq ^ (56 if active_side == 1 else 0)
            if flip_h: sq_mapped ^= 7
            indices.append(bucket * 704 + mt * 64 + sq_mapped)
            
        while len(indices) < 32:
            indices.append(22528)
        return indices[:32]
        
    if stm == 0:
        stm_indices = get_indices(0)
        nstm_indices = get_indices(1)
    else:
        stm_indices = get_indices(1)
        nstm_indices = get_indices(0)
        
    return stm_indices, nstm_indices, piece_count

def test():
    device = torch.device("cpu")
    torch.manual_seed(42)
    model = LayerStackNNUE(num_buckets=8)
    
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
    
    inputs_stm = torch.tensor([stm_idx], dtype=torch.long, device=device)
    inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long, device=device)
    bucket_id = max(0, min(7, (pc - 1) // 4))
    
    acc_stm = model.crelu(model.fc1(inputs_stm) + model.fc1_bias)
    acc_nstm = model.crelu(model.fc1(inputs_nstm) + model.fc1_bias)
    x = torch.cat((acc_stm, acc_nstm), dim=1)
    
    w2 = model.fact_fc2.weight + model.delta_fc2[bucket_id].weight
    b2 = model.fact_fc2.bias + model.delta_fc2[bucket_id].bias
    h2_raw = F.linear(x, w2, b2)
    h2 = model.crelu(h2_raw)
    
    w3 = model.fact_fc3.weight + model.delta_fc3[bucket_id].weight
    b3 = model.fact_fc3.bias + model.delta_fc3[bucket_id].bias
    h3_raw = F.linear(h2, w3, b3)
    h3 = model.crelu(h3_raw)
    
    print(f"FC1 Active: {(acc_stm > 0).sum().item()}/512")
    print(f"FC2 raw output min/max/mean: {h2_raw.min().item():.4f} / {h2_raw.max().item():.4f} / {h2_raw.mean().item():.4f}")
    print(f"FC2 Active: {(h2 > 0).sum().item()}/32")
    print(f"FC3 raw output min/max/mean: {h3_raw.min().item():.4f} / {h3_raw.max().item():.4f} / {h3_raw.mean().item():.4f}")
    print(f"FC3 Active: {(h3 > 0).sum().item()}/32")

if __name__ == '__main__':
    test()
