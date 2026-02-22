import torch
import torch.nn as nn
import numpy as np
import zlib

# --- MICRO-GENERATOR FOR MAPPING ---
class MicroGenerator(nn.Module):
    def __init__(self, width=64, layers=3):
        super().__init__()
        chain = [nn.Linear(128, width), nn.LeakyReLU(0.2)]
        for _ in range(layers - 2):
            chain += [nn.Linear(width, width), nn.LeakyReLU(0.2)]
        chain += [nn.Linear(width, 1), nn.Tanh()]
        self.model = nn.Sequential(*chain)

    def forward(self, z):
        return self.model(z)

def get_fitness(model):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1024, 128).cuda()
        data = model(z).flatten().cpu().numpy()
        # Metric: Combination of Compression and Sign-Changes (Proxy for Entropy)
        m2 = len(np.where(np.diff(np.sign(data)) != 0)[0]) / len(data)
        m3 = len(zlib.compress(data.tobytes(), level=1)) / len(data.tobytes())
        return (m2 + m3) / 2

# --- CEILING TESTER ---
def map_ceiling(width, layers, iterations=5000):
    model = MicroGenerator(width, layers).cuda()
    best_score = get_fitness(model)
    
    for i in range(iterations):
        state = {k: v.clone() for k, v in model.state_dict().items()}
        # Apply mutation
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.size()).cuda() * 0.001)
        
        score = get_fitness(model)
        if score > best_score:
            best_score = score
        else:
            model.load_state_dict(state)
            
    return best_score

print("Mapping Entropy Capacity...")
results = {}
for w in [32, 64, 128]:
    for l in [2, 3, 4]:
        cap = map_ceiling(w, l)
        results[f"W{w}_L{l}"] = cap
        print(f"Config W:{w} L:{l} | Max Entropy: {cap:.6f}")