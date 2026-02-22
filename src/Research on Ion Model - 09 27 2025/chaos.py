import os
import torch
import torch.nn as nn
import random
import math
from collections import Counter
from copy import deepcopy
import secrets

def ensure_dir(path):
    """Creates directory if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

ensure_dir("byte_chaos")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cs_stream(length):
    """Generates a list of cryptographically secure byte values."""
    return [secrets.randbelow(256) for _ in range(length)]

# Example:
cs_ref = generate_cs_stream(1024)  # 1024 random bytes

# ----------------------------
# 8-Bit Vocabulary Setup
# ----------------------------
vocab_size = 256
token_to_bits = {i: format(i, "08b") for i in range(vocab_size)}  # 0 → "00000000", ..., 255 → "11111111"
bits_to_token = {v: k for k, v in token_to_bits.items()}

# ----------------------------
# Bytewise Entropy GRU Generator
# ----------------------------
class ByteGRUModel(nn.Module):
    def __init__(self, vocab_size=256, emb_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden
		
# ----------------------------
# Entropy Leaderboard & Checkpoint Saver
# ----------------------------
def deviation_score(run_dev, monobit_dev, threshold=0.005):
    def scale(dev):
        if dev <= threshold:
            # Reward: closer to 0 gets higher score
            return 100.0 * (1 - (dev / threshold))
        else:
            # Penalize: scale down based on how far above threshold
            penalty = (dev - threshold) / (1.0 - threshold)
            return max(0.0, 100.0 * (1 - penalty))

    run_score = scale(run_dev)
    mono_score = scale(monobit_dev)

    return run_score, mono_score

def chi_square_score(chi_value, ideal=256, max_deviation=100):
    deviation = abs(chi_value - ideal)
    scaled = max(0.0, 100.0 * (1 - deviation / max_deviation))
    return round(scaled, 2)

class EntropyLeaderboard:
    def __init__(self, save_dir="byte_chaos"):
        self.best_score = float("-inf")
        self.best_epoch = -1
        self.best_stream = []
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def _score(self, m):
        # Scaled entropy score        
        entropy_score = (m["entropy"] / 8.0) * 100
        chi_penalty = chi_square_score(m["chi2"])
        run_penalty, mono_penalty = deviation_score(m.get("run_dev",0), m.get("monobits",0))
        return (entropy_score + chi_penalty + run_penalty + mono_penalty) / 4

    def update(self, model, epoch, byte_stream, metrics):
        score = self._score(metrics)

        print(f"📊 Epoch {epoch:03d} Ent: {metrics['entropy']:.5f} Chi: {metrics['chi2']:.5f} Run: {metrics.get('run_dev', 0):.5f} Mono: {metrics.get('monobits', 0):.5f} Score: {score:6.2f}")

        if score > self.best_score:
            print("   -> New Best")
            self.best_score = score
            self.best_epoch = epoch
            self.best_stream = byte_stream

            tag = f"epoch{epoch:03d}_score{score:.2f}"

            # Save raw byte stream (.bin)
            with open(f"{self.save_dir}/stream_{tag}.bin", "wb") as f:
                f.write(bytes(byte_stream))

            # Save hex-dump for inspection
            hex_string = " ".join(f"{b:02x}" for b in byte_stream)
            with open(f"{self.save_dir}/stream_{tag}.hex.txt", "w", encoding="utf-8") as f:
                f.write(hex_string)

            # Save model weights
            torch.save(model.state_dict(), f"{self.save_dir}/best_model_{tag}.pt")

# ----------------------------
# Entropy Metric Analyzer
# ----------------------------
def analyze_byte_stream(byte_stream):
    counts = Counter(byte_stream)
    total = len(byte_stream)

    # Frequency histogram
    probs = [counts[b] / total for b in range(256)]

    # Shannon Entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Chi-Square Uniformity Test
    expected = total / 256
    chi2 = sum(((counts[b] - expected) ** 2) / expected for b in range(256))

    # Bit run deviation
    bits = "".join(format(b, "08b") for b in byte_stream)
    transitions = sum(bits[i] != bits[i + 1] for i in range(len(bits) - 1))
    ideal_transitions = len(bits) - 1
    run_dev = abs(transitions - ideal_transitions / 2) / (ideal_transitions / 2)

    # Monobit deviation
    ones = bits.count("1")
    zeros = bits.count("0")
    monobits = abs(ones - zeros) / len(bits)

    return {
        "entropy": entropy,
        "chi2": chi2,
        "run_dev": run_dev,
        "monobits": monobits
    }
	
# ----------------------------
# Full Training Loop
# ----------------------------
def train_byte_entropy_model(model, epochs=100, stream_len=1024, save_dir="byte_chaos"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    leaderboard = EntropyLeaderboard(save_dir)

    for epoch in range(1, epochs + 1):
        model.train()

        # Generate CSPRNG training stream
        cs_stream = generate_cs_stream(stream_len + 1)
        inputs = torch.tensor(cs_stream[:-1], dtype=torch.long).unsqueeze(0).to(device)
        targets = torch.tensor(cs_stream[1:], dtype=torch.long).unsqueeze(0).to(device)

        # Forward & backward pass
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, 256), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        generated = generate_model_stream(model, length=stream_len)
        metrics = analyze_byte_stream(generated)
        leaderboard.update(model, epoch, generated, metrics)
        #leaderboard.update(model, epoch, "".join(map(chr, generated)), metrics)
		
# ----------------------------
# Model Byte Stream Generator
# ----------------------------
def generate_model_stream(model, length=1024, temperature=1.0):
    model.eval()
    generated = []
    input_id = torch.randint(0, 256, (1, 1)).to(device)
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_id, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            input_id = torch.multinomial(probs, 1)
            generated.append(input_id.item())

    return generated
	
if __name__ == "__main__":
    print("\nLaunching Byte Entropy Engine...")
    model = ByteGRUModel(vocab_size=256)
    train_byte_entropy_model(model, epochs=100, stream_len=2048)