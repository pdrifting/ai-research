import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import secrets
import os
from datetime import datetime
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reseed_model(model_class, input_dim=100, output_dim=1024, dropout_rate=0.1):
    torch.manual_seed(secrets.randbits(32))
    return model_class(input_dim, output_dim, dropout_rate).to(device)

class RandomByteDataset(Dataset):
    def __init__(self, size, seq_length=1024):
        self.size = size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = np.frombuffer(secrets.token_bytes(self.seq_length), dtype=np.uint8).astype(np.float32) / 255.0
        return torch.tensor(data, dtype=torch.float32)
    
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1024, dropout_rate=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
def save_byte_stream(generator, output_dir, cycle, epoch):
    generator.eval()
    with torch.no_grad():
        output = []
        while len(output) < 1048576:
            noise = torch.randn(1024, 100).to(device)
            gen = generator(noise).cpu().numpy()
            output.extend((gen * 255).astype(np.uint8).flatten())
        byte_stream = np.array(output[:1048576], dtype=np.uint8)
        filename = os.path.join(output_dir, f"byte_stream_cycle_{cycle}_epoch_{epoch}.bin")
        with open(filename, "wb") as f:
            f.write(byte_stream.tobytes())
    generator.train()
    return filename, byte_stream

def compute_run_variance(byte_data):
    bits = ''.join(f'{b:08b}' for b in byte_data)
    runs = [len(run) for run in bits.split('0') + bits.split('1')]
    return round(np.var(runs) if runs else 0.0, 4)

def blend_generators(state_a, state_b, ratio_a=0.5):
    blended_state = {}
    for key in state_a:
        if key in state_b:
            blended_state[key] = ratio_a * state_a[key] + (1 - ratio_a) * state_b[key]
    return blended_state

def train_cycle(cycle_num, generator, discriminator, dataloader, g_opt, d_opt, output_dir,
                epochs, config, log_path):
    criterion = nn.BCELoss()
    best_runvar = float("inf")
    best_state = None

    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(config) + "\n")

    for epoch in range(1, epochs + 1):
        g_loss_total, d_loss_total, batches = 0.0, 0.0, 0

        for real_data in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # --- Input Noise Strategy ---
            if epoch <= 3:
                noise = torch.rand(batch_size, 100).to(device)
            else:
                noise1 = torch.randn(batch_size, 100).to(device)
                noise2 = torch.rand(batch_size, 100).to(device)
                noise = (noise1 + noise2) / 2

            # --- Train Discriminator ---
            d_opt.zero_grad()
            gen_output = generator(noise).detach()
            fake_data = gen_output + 0.01 * torch.randn_like(gen_output).to(device)
            d_real = discriminator(real_data)
            d_real_loss = criterion(d_real, real_labels)
            d_fake = discriminator(fake_data)
            d_fake_loss = criterion(d_fake, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()

            # --- Train Generator ---
            for _ in range(2):
                g_opt.zero_grad()
                if epoch <= 3:
                    noise = torch.rand(batch_size, 100).to(device)
                else:
                    noise1 = torch.randn(batch_size, 100).to(device)
                    noise2 = torch.rand(batch_size, 100).to(device)
                    noise = (noise1 + noise2) / 2
                gen_output = generator(noise)
                fake_data = gen_output + 0.01 * torch.randn_like(gen_output).to(device)
                d_fake = discriminator(fake_data)
                g_loss = criterion(d_fake, real_labels)
                g_loss.backward()
                g_opt.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            batches += 1

        avg_d = d_loss_total / batches
        avg_g = g_loss_total / batches
        stream_file, byte_stream = save_byte_stream(generator, output_dir, cycle_num, epoch)
        runvar = compute_run_variance(byte_stream)

        print(f"Cycle {cycle_num}, Epoch {epoch}/{epochs}, D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}, RunVar: {runvar}")
        if runvar < best_runvar:
            best_runvar = runvar
            best_state = generator.state_dict()
            torch.save(best_state, os.path.join(output_dir, f"best_generator_cycle_{cycle_num}.pt"))

    return best_runvar, best_state

def main():
    # Hyperparameters
    num_epochs = 50
    batch_size = 64
    input_dim = 100
    output_dim = 1024
    ratios = [0.9, 0.7, 0.5, 0.3, 0.1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = f"entropy_lab_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)
    log_path = os.path.join(root_dir, "runvar_log.jsonl")

    dataset = RandomByteDataset(size=5000, seq_length=output_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # INITIAL TRAINING CYCLE
    G1 = Generator(input_dim, output_dim).to(device)
    D1 = Discriminator(output_dim).to(device)
    g_opt_1 = optim.RMSprop(G1.parameters(), lr=0.0001)
    d_opt_1 = optim.RMSprop(D1.parameters(), lr=0.00005)
    cycle_dir1 = os.path.join(root_dir, "cycle_1")
    os.makedirs(cycle_dir1, exist_ok=True)
    config_1 = {
        "cycle": "1",
        "blend": "None",
        "lr_g": 0.0001,
        "lr_d": 0.00005,
        "dropout": 0.1,
        "notes": "fresh start with chaos injection"
    }
    runvar_1, state_1 = train_cycle(1, G1, D1, dataloader, g_opt_1, d_opt_1, cycle_dir1, num_epochs, config_1, log_path)

    # SECOND TRAINING CYCLE ON BEST STATE
    G2 = Generator(input_dim, output_dim).to(device)
    G2.load_state_dict(state_1)
    D2 = Discriminator(output_dim).to(device)
    g_opt_2 = optim.RMSprop(G2.parameters(), lr=0.0001)
    d_opt_2 = optim.RMSprop(D2.parameters(), lr=0.00005)
    cycle_dir2 = os.path.join(root_dir, "cycle_2")
    os.makedirs(cycle_dir2, exist_ok=True)
    config_2 = {
        "cycle": "2",
        "blend": "retrain from cycle 1 best",
        "lr_g": 0.0001,
        "lr_d": 0.00005,
        "dropout": 0.1
    }
    runvar_2, state_2 = train_cycle(2, G2, D2, dataloader, g_opt_2, d_opt_2, cycle_dir2, num_epochs, config_2, log_path)

    # BLEND TRIALS BETWEEN BEST STATES
    for i, r in enumerate(ratios, start=1):
        blend_state = blend_generators(state_1, state_2, ratio_a=r)
        G_blend = Generator(input_dim, output_dim).to(device)
        G_blend.load_state_dict(blend_state)
        D_blend = Discriminator(output_dim).to(device)
        g_opt_b = optim.RMSprop(G_blend.parameters(), lr=0.0001)
        d_opt_b = optim.RMSprop(D_blend.parameters(), lr=0.00005)

        trial_dir = os.path.join(root_dir, f"blend_trial_{i}")
        os.makedirs(trial_dir, exist_ok=True)

        config_blend = {
            "cycle": f"blend_trial_{i}",
            "blend": f"{r:.2f} A + {(1 - r):.2f} B",
            "lr_g": 0.0001,
            "lr_d": 0.00005,
            "dropout": 0.1,
            "notes": "chaos injected training on blended model"
        }

        runvar_b, state_b = train_cycle(f"blend_trial_{i}", G_blend, D_blend, dataloader,
                                        g_opt_b, d_opt_b, trial_dir, num_epochs, config_blend, log_path)

        # If RunVar improves, save best blend snapshot
        if runvar_b < min(runvar_1, runvar_2):
            torch.save(state_b, os.path.join(trial_dir, "final_best_blend.pt"))
            print(f"Blend trial {i} improved RunVar to {runvar_b}")
        else:
            print(f"Blend trial {i} did not improve. RunVar: {runvar_b}")

    print("\nAll cycles and blend trials complete. Configs and metrics logged.")

if __name__ == "__main__":
    main()