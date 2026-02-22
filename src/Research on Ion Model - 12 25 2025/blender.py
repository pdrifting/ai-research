import os
import glob
import time
import torch
import torch.nn as nn
import blake3
import numpy as np
from scipy.stats import chisquare
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

# --- Configuration ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
DATA_DIM = 1024

class Generator(nn.Module):
    def __init__(self, latent_dim=128, data_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, data_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class AI_DRBG_Node:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        self.blenders = []
        self.keymaster = None
        self._load_latest_champions()

    def _load_latest_champions(self):
        """Finds most recent models based on 1970 Epoch timestamps in filenames."""
        blender_files = sorted(glob.glob("blender_*.pt"), reverse=True)[:3]
        keymaster_files = sorted(glob.glob("keymaster_*.pt"), reverse=True)[:1]

        if len(blender_files) < 3 or not keymaster_files:
            print(f"Current Blenders: {len(blender_files)} | Keymasters: {len(keymaster_files)}")
            raise FileNotFoundError("Required: 3 'blender_*.pt' and 1 'keymaster_*.pt' files.")

        print(f"[*] Loading Blenders: {blender_files}")
        for pf in blender_files:
            m = Generator(LATENT_DIM, DATA_DIM).to(self.device)
            m.load_state_dict(torch.load(pf, map_location=self.device)['g_state'])
            m.eval()
            self.blenders.append(m)

        print(f"[*] Loading Keymaster: {keymaster_files[0]}")
        self.keymaster = Generator(LATENT_DIM, DATA_DIM).to(self.device)
        self.keymaster.load_state_dict(torch.load(keymaster_files[0], map_location=self.device)['g_state'])
        self.keymaster.eval()

    def _generate_master_seed(self):
        """Blends 3 blenders + 1 keymaster into a 32-byte BLAKE3 hash."""
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, device=self.device)
            hasher = blake3.blake3()
            
            # Blend the first 3 champions
            for i, m in enumerate(self.blenders):
                output = m(z).cpu().numpy().tobytes()
                hasher.update(output)
                hasher.update(f"blender_{i}".encode())
            
            # Seal with the Keymaster
            km_output = self.keymaster(z).cpu().numpy().tobytes()
            hasher.update(km_output)
            hasher.update(b"keymaster_seal")
            
        return hasher.digest()

    def get_nist_ascii(self, total_bits=1000000):
        """Outputs as ASCII '0'/'1' characters. 1,000,000 bits = 1,000,000 bytes."""
        binary_data = self.get_binary_stream((total_bits + 7) // 8)
        bit_string = "".join(f"{byte:08b}" for byte in binary_data)
        return bit_string[:total_bits].encode('ascii')

    def get_binary_stream(self, num_bytes):
        """Outputs packed binary using ChaCha20 seeded by AI-Master-Seed."""
        seed = self._generate_master_seed()
        nonce = b'\x00' * 16 
        
        algorithm = algorithms.ChaCha20(seed, nonce)
        cipher = Cipher(algorithm, mode=None)
        encryptor = cipher.encryptor()
        
        return encryptor.update(b'\x00' * num_bytes)

def run_pre_test(data, is_ascii=True):
    """Chi-Squared test for bit distribution uniformity."""
    if is_ascii:
        # Convert ASCII bytes ('0' or '1') to integers
        bits = np.array([int(chr(b)) for b in data])
    else:
        # Unpack binary bytes to bits
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    ones = np.sum(bits)
    zeros = len(bits) - ones
    expected = len(bits) / 2
    
    stat, p_value = chisquare([zeros, ones], f_exp=[expected, expected])
    
    print("\n--- Entropy Pre-Test (Chi-Squared) ---")
    print(f"Total Bits: {len(bits)}")
    print(f"Zero count: {zeros} | One count: {ones}")
    print(f"P-Value:     {p_value:.6f} (Target > 0.01)")
    
    if p_value > 0.01:
        print("[RESULT] PASS: Distribution is statistically uniform.")
    else:
        print("[RESULT] FAIL: Distribution is biased.")

if __name__ == "__main__":
    try:
        node = AI_DRBG_Node()
        ts = int(time.time())

        # 1. NIST ASCII Test File (1M bits = 1MB)
        nist_file = f"nist_test_ascii_{ts}.txt"
        print(f"[*] Generating {nist_file}...")
        nist_data = node.get_nist_ascii(1000000)
        with open(nist_file, "wb") as f:
            f.write(nist_data)
        run_pre_test(nist_data, is_ascii=True)

        # 2. Production Binary Stream (e.g., 5MB)
        prod_file = f"prod_stream_bin_{ts}.bin"
        print(f"\n[*] Generating {prod_file}...")
        prod_data = node.get_binary_stream(5 * 1024 * 1024)
        with open(prod_file, "wb") as f:
            f.write(prod_data)
        
        print("\n[COMPLETE] All streams generated using 1970 Epoch timestamps.")

    except FileNotFoundError as e:
        print(f"\n[!] Error: {e}")
        print("Please ensure your training script saves models as 'blender_174000000.pt', etc.")