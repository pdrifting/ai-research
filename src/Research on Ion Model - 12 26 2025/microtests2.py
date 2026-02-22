import torch
import torch.nn as nn
import numpy as np
import zlib
import csv

class MicroGenerator(nn.Module):
    def __init__(self, width=128, layers=3):
        super().__init__()
        chain = [nn.Linear(128, width), nn.LeakyReLU(0.2)]
        for _ in range(layers - 2):
            chain += [nn.Linear(width, width), nn.LeakyReLU(0.2)]
        chain += [nn.Linear(width, 1), nn.Tanh()]
        self.model = nn.Sequential(*chain)
    def forward(self, z): return self.model(z)

def get_entropy(model):
    model.eval()
    with torch.no_grad():
        z = torch.randn(2048, 128).cuda()
        data = model(z).flatten().cpu().numpy()
        m2 = len(np.where(np.diff(np.sign(data)) != 0)[0]) / len(data)
        m3 = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        # Magnitude check: Are the weights 'quiet' or 'loud'?
        weight_mag = torch.mean(torch.stack([p.abs().mean() for p in model.parameters()])).item()
        return (m2 + m3) / 2, weight_mag

def scan_bloodlines(num_seeds=100):
    print(f"Scanning {num_seeds} bloodlines for genetic potential...")
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = MicroGenerator(128, 3).cuda()
        
        # Initial potential
        base_ent, mag = get_entropy(model)
        
        # Quick Squeeze: 50 fast iterations to see if it responds to tuning
        best_ent = base_ent
        for _ in range(50):
            state = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn(p.size()).cuda() * 0.005)
            
            ent, _ = get_entropy(model)
            if ent > best_ent: best_ent = ent
            else: model.load_state_dict(state)
        
        improvement = best_ent - base_ent
        results.append([seed, base_ent, best_ent, improvement, mag])
        
        if seed % 10 == 0:
            print(f"Seed {seed} | Pot: {best_ent:.4f} | Gain: {improvement:.4f}")

    # Save to CSV for Analysis
    with open('bloodline_scan.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'base_entropy', 'final_entropy', 'improvement', 'weight_mag'])
        writer.writerows(results)

if __name__ == "__main__":
    scan_bloodlines(100)