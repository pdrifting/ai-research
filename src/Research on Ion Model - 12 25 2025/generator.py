import os
import secrets
import numpy as np
import time
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

# --- Remove all problematic networking variables ---
for var in ['GLOO_SOCKET_IFNAME', 'USE_LIBUV', 'MASTER_ADDR', 'MASTER_PORT']:
    if var in os.environ: del os.environ[var]

os.environ['PYTHONUNBUFFERED'] = '1'

GPU_IDS = [0, 1, 2]
WORLD_SIZE = len(GPU_IDS)

def force_print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class Generator(nn.Module):
    def __init__(self, latent_dim=128, data_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048), nn.LeakyReLU(0.2),
            nn.Linear(2048, data_dim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_shard(rank):
    """
    Independent Training Shard: Runs on a specific GPU without 
    network synchronization to avoid Windows socket/libuv hangs.
    """
    try:
        gpu_id = GPU_IDS[rank]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        force_print(f"Shard {rank} (GPU {gpu_id}): Engine Ignited.")

        # Model Setup (No DDP wrapper needed)
        netG = Generator().to(device)
        netD = Discriminator().to(device)
        
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4)
        criterion = nn.BCELoss()

        for gen in range(1, 1000):
            # Training generation
            for epoch in range(50):
                # Generate high-entropy "NIST-style" data
                real_raw = [[secrets.randbelow(10**6)/10**6 for _ in range(1024)] for _ in range(64)]
                real_data = (torch.tensor(real_raw, dtype=torch.float32).to(device) * 2 - 1)
                z = torch.randn(64, 128, device=device)
                
                # Update Discriminator
                netD.zero_grad()
                d_real = netD(real_data)
                fake = netG(z)
                d_fake = netD(fake.detach())
                lossD = criterion(d_real, torch.ones(64, 1, device=device)) + \
                        criterion(d_fake, torch.zeros(64, 1, device=device))
                lossD.backward()
                optimizerD.step()

                # Update Generator
                netG.zero_grad()
                lossG = criterion(netD(fake), torch.ones(64, 1, device=device))
                lossG.backward()
                optimizerG.step()

                if rank == 0 and epoch % 10 == 0:
                    vram = torch.cuda.memory_allocated(device) / 1024**2
                    force_print(f"LIVE | Gen {gen} | Ep {epoch}/50 | LossG: {lossG.item():.4f} | VRAM: {vram:.0f}MB")

            # Coordination: Every shard saves its own progress
            ts = int(time.time())
            save_path = f"shard_{rank}_gen_{gen}_{ts}.pt"
            torch.save({'g_state': netG.state_dict(), 'rank': rank}, save_path)
            
            # Periodically cleanup old shards to save space
            if rank == 0:
                old_files = glob.glob(f"shard_{rank}_gen_{gen-2}_*.pt")
                for f in old_files:
                    try: os.remove(f)
                    except: pass

    except Exception as e:
        force_print(f"Shard {rank} CRITICAL ERROR: {str(e)}")
    finally:
        os._exit(0)

if __name__ == "__main__":
    force_print(f"MASTER: Launching Independent 3-GPU Cluster (No Handshake Mode)")
    
    mp.set_start_method('spawn', force=True)
    processes = []
    
    try:
        for i in range(WORLD_SIZE):
            p = mp.Process(target=train_shard, args=(i,))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        force_print("\n[!] Force-stopping all shards...")
        os._exit(0)