import torch
import torch.nn as nn
import numpy as np
import zlib
import csv
import os
import time

# --- BATCHED MICRO-SCANNER ---
class BatchedMicroGenerator(nn.Module):
    def __init__(self, batch_size, width=128, layers=3):
        super().__init__()
        self.batch_size = batch_size
        # We use grouped convolutions or specialized mapping to simulate 
        # multiple independent models in one GPU pass.
        # For simplicity in this scan, we utilize a weight-per-batch mapping.
        self.layers = nn.ModuleList([nn.Linear(128, width) for _ in range(layers-1)])
        self.out = nn.Linear(width, 1)
        self.act = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z: [batch, 128]
        x = z
        for layer in self.layers:
            x = self.act(layer(x))
        return self.tanh(self.out(x))

def get_entropy_batch(data_numpy):
    # data_numpy shape: [batch, samples]
    batch_size = data_numpy.shape[0]
    scores = []
    for i in range(batch_size):
        row = data_numpy[i]
        m2 = len(np.where(np.diff(np.sign(row)) != 0)[0]) / len(row)
        m3 = len(zlib.compress(row.tobytes(), level=1)) / len(row.tobytes())
        scores.append((m2 + m3) / 2)
    return np.array(scores)

def mega_scan(total_seeds=100000, squeeze_steps=300):
    device = torch.device("cuda")
    output_file = 'mega_bloodline_100k.csv'
    batch_size = 100  # Adjust based on VRAM
    
    print(f"[*] Launching 100k Seed Scan | Steps: {squeeze_steps} | Batch: {batch_size}")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'base_ent', 'final_ent', 'gain', 'mag', 'std'])

        for b_start in range(0, total_seeds, batch_size):
            b_end = min(b_start + batch_size, total_seeds)
            current_batch_size = b_end - b_start
            
            # Initialize models for this batch
            models = []
            for s in range(b_start, b_end):
                torch.manual_seed(s)
                m = BatchedMicroGenerator(1, 128, 3).to(device)
                models.append(m)
            
            # Get Base Fitness
            z = torch.randn(current_batch_size, 2048, 128).to(device)
            base_scores = []
            for i, m in enumerate(models):
                out = m(z[i]).flatten().cpu().detach().numpy()
                base_scores.append(get_entropy_batch(out.reshape(1, -1))[0])
            
            best_scores = np.array(base_scores)
            best_states = [ {k: v.cpu().clone() for k, v in m.state_dict().items()} for m in models ]

            # Deep Squeeze
            for step in range(squeeze_steps):
                radius = 0.015 * (0.985 ** step) # Sharper decay for accuracy
                
                for i, m in enumerate(models):
                    with torch.no_grad():
                        for p in m.parameters():
                            p.add_(torch.randn(p.size(), device=device) * radius)
                    
                    # Test new fitness
                    z_test = torch.randn(2048, 128).to(device)
                    new_out = m(z_test).flatten().cpu().detach().numpy()
                    new_score = get_entropy_batch(new_out.reshape(1, -1))[0]
                    
                    if new_score > best_scores[i]:
                        best_scores[i] = new_score
                        best_states[i] = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                    else:
                        m.load_state_dict(best_states[i])

            # Final Stats Logging
            for i in range(current_batch_size):
                final_weights = torch.cat([p.flatten() for p in models[i].parameters()])
                mag = final_weights.abs().mean().item()
                std = final_weights.std().item()
                seed = b_start + i
                writer.writerow([seed, base_scores[i], best_scores[i], best_scores[i]-base_scores[i], mag, std])

            if b_start % 500 == 0:
                print(f"Progress: {b_start}/{total_seeds} | Last Batch Max Gain: {np.max(best_scores - base_scores):.4f}")

if __name__ == "__main__":
    mega_scan()