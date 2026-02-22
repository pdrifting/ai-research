import string
import random
import math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================
# Config
# =====================
VOCAB_CHARS = '_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' \
              '!@#$%^&*()_+-={}[]|:;\"\'<>,.?/\\~'
VOCAB_SIZE = len(VOCAB_CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 5
DROPOUT = 0.2
BATCH_SIZE = 64
LR = 0.003
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MAX_EPOCHS_PER_TRY = 25
EVAL_EVERY = 5
FIRST_CYCLE_MAX_EPOCHS = 10
FIRST_CYCLE_LOSS_TARGET = 1.65
MAX_RETRIES_PER_CYCLE = 3
IMPROVE_EPS = 1e-4

# =====================
# Utilities
# =====================
def safe_write_text(path, txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)

# =====================
# Data Generation
# =====================
ALPHA_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
SPECIAL_CHARS = '!@#$%^&*()_+-={}[]|:;\"\'<>,.?/\\~'

def generate_char_string(length):
    result = []
    last_char_type = None
    last_char = ''
    while len(result) < length:
        stream = random.choice([ALPHA_CHARS, SPECIAL_CHARS])
        c = random.choice(stream)
        # enforce no consecutive type repeats
        char_type = 'alpha' if c in ALPHA_CHARS else 'special'
        if last_char_type == char_type and len(result) > 0:
            continue
        if c.lower() == last_char.lower():
            continue
        result.append(c)
        last_char_type = char_type
        last_char = c
    return ''.join(result)

def calculate_entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    ent = -sum((v/total)*math.log2(v/total) for v in counts.values())
    return ent

def bitstring_entropy_tests(samples):
    bitstring = ''.join(f"{ord(c):08b}" for s in samples for c in s)
    if not bitstring:
        return 0.5, 0.5, 0.5
    ones = bitstring.count('1')
    zeros = len(bitstring) - ones
    total = len(bitstring)
    p1 = ones / total
    monobit_dev = abs(p1 - 0.5)
    runs = 1
    for i in range(1, total):
        if bitstring[i] != bitstring[i-1]:
            runs += 1
    expected_runs = 2 * ones * zeros / total + 1 if total > 0 else 1
    runs_dev = abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 0.0
    counts = Counter(''.join(samples))
    avg_len = np.mean([len(s) for s in samples]) if samples else 1
    expected = len(samples) * avg_len / (VOCAB_SIZE - 1)
    chi2 = sum((counts.get(c,0)-expected)**2/expected for c in VOCAB_CHARS) if expected>0 else 0.0
    p_val = math.exp(-0.5*chi2/max(VOCAB_SIZE,1))
    return monobit_dev, runs_dev, p_val

def compute_score(entropy, monobit_dev, runs_dev, chi2_p, loss):
    max_entropy = math.log2(VOCAB_SIZE)
    entropy_score = entropy / max_entropy
    monobit_score = 1.0 - min(abs(monobit_dev), 1.0)
    runs_score = 1.0 - min(abs(runs_dev), 1.0)
    chi2_score = 1.0 - min(abs(chi2_p - 0.5) * 2, 1.0)
    loss_score = 1.0 - min(max((loss - 2.05), 0.0)/1.0, 1.0)
    total = (0.4*entropy_score + 0.15*monobit_score + 0.15*runs_score + 0.2*chi2_score + 0.1*loss_score)*10
    return float(round(total,4))

def generate_data(n_samples, min_len=40, max_len=60):
    samples = []
    attempts = 0
    max_attempts = n_samples * 2000

    types_stream1 = string.ascii_letters + string.digits
    types_stream2 = "!@#$%^&*()_+-={}[]|:;\"'<>,.?/~\\"

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        length = random.randint(min_len, max_len)
        s = []
        last_char = ''
        last_type_stream1 = ''
        last_type_stream2 = ''

        for _ in range(length):
            if random.random() < 0.7:
                # pick from stream1
                c = random.choice(types_stream1)
                if c == last_type_stream1:
                    continue
                last_type_stream1 = c
            else:
                # pick from stream2
                c = random.choice(types_stream2)
                if c == last_type_stream2:
                    continue
                last_type_stream2 = c

            if c == last_char:
                continue
            last_char = c
            s.append(c)

        candidate = ''.join(s)
        if len(candidate) < min_len:
            continue

        # compute entropy and simple bit-level tests
        entropy_val = calculate_entropy(candidate)
        monobit_dev, runs_dev, chi2_p = bitstring_entropy_tests([candidate])

        # enforce >95% entropy + reasonable monobit/runs
        max_ent = math.log2(len(types_stream1 + types_stream2))
        score = entropy_val / max_ent
        if score >= 0.86 and monobit_dev < 0.15 and runs_dev < 0.15:
            samples.append(candidate)
            print(f"{entropy_val:.3f} {score:.3f} {candidate}")

    if len(samples) < n_samples:
        print(f"Warning: only generated {len(samples)} samples after {attempts} attempts")
    return samples

def preprocess_data(samples):
    max_len = max(len(s) for s in samples)
    tensor_data = torch.zeros(len(samples), max_len, dtype=torch.long)
    for i, s in enumerate(samples):
        for j, c in enumerate(s):
            tensor_data[i, j] = CHAR_TO_IDX[c]
    return tensor_data

# =====================
# Model
# =====================
class CharRNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.fc(out)

# =====================
# Text Generation
# =====================
def generate_text(model, start_char='A', max_len=30, temperature=0.8, repetition_penalty=1.1):
    device = next(model.parameters()).device
    model.eval()
    target_len = random.randint(int(max_len*0.8), int(max_len*1.2))
    cur = torch.tensor([[CHAR_TO_IDX[start_char]]], device=device)
    generated = start_char
    last4 = []
    with torch.no_grad():
        for _ in range(target_len-1):
            logits = model(cur)[:, -1, :] / max(temperature,1e-6)
            for c in last4:
                logits[0, CHAR_TO_IDX[c]] /= repetition_penalty
            prob = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, 1)
            ch = IDX_TO_CHAR[next_idx.item()]
            generated += ch
            last4.append(ch)
            if len(last4) > 4:
                last4.pop(0)
            cur = torch.cat([cur, next_idx.unsqueeze(0)], dim=1)
    return generated

# =====================
# Training & Evaluation
# =====================
def run_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = criterion(logits.reshape(-1,VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss/max(len(loader),1)

def evaluate_model_path(model_path, samples_for_eval=300):
    device = torch.device('cpu')
    model = CharRNN()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    samples = [generate_text(model, random.choice(VOCAB_CHARS)) for _ in range(samples_for_eval)]
    entropy = float(np.mean([calculate_entropy(s) for s in samples]))
    monobit_dev, runs_dev, chi2_p = bitstring_entropy_tests(samples)
    loss_est = 2.05
    score = compute_score(entropy, monobit_dev, runs_dev, chi2_p, loss_est)
    scored = [(s, calculate_entropy(s)) for s in samples]
    scored.sort(key=lambda x: -x[1])
    top_entropy = [s for s,_ in scored[:100]]
    top_middle = [s for s,_ in scored[100:200]]
    top_low = [s for s,_ in scored[200:]]
    diverse_top = top_entropy[:50]+top_middle[:30]+top_low[:20]
    return {'score':score,'model_path':model_path,'diverse_top':diverse_top,'entropy':entropy,
            'monobit_dev':monobit_dev,'runs_dev':runs_dev,'chi2_p':chi2_p}

def train_until_better(model, train_data, device, target_score, cycle_num, ensure_loss_leq=None):
    dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_IDX['_'])
    best_local = -float('inf')
    last_eval_score = -float('inf')
    if ensure_loss_leq is not None:
        for epoch in range(1,FIRST_CYCLE_MAX_EPOCHS+1):
            avg_loss = run_epoch(model, loader, device, optimizer, criterion)
            if avg_loss <= ensure_loss_leq+1e-8:
                break
    for epoch in range(1,MAX_EPOCHS_PER_TRY+1):
        avg_loss = run_epoch(model, loader, device, optimizer, criterion)
        if epoch % EVAL_EVERY==0:
            model.eval()
            with torch.no_grad():
                samples = [generate_text(model, random.choice(VOCAB_CHARS)) for _ in range(120)]
                ent = float(np.mean([calculate_entropy(s) for s in samples]))
                mdev,rdev,pval = bitstring_entropy_tests(samples)
                score_now = compute_score(ent,mdev,rdev,pval,avg_loss)
            if score_now > best_local + IMPROVE_EPS:
                best_local = score_now
            if score_now > target_score + IMPROVE_EPS and score_now > last_eval_score + IMPROVE_EPS:
                last_eval_score = score_now
                checkpoint_path = f'checkpoints/model_cycle{cycle_num}_score{score_now:.4f}.pt'
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
    return model, best_local, avg_loss

# =====================
# Evolution Loop
# =====================
def evolve(cycles=20, n_samples=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_score = -float('inf')
    best_model_path = None
    for cycle in range(1, cycles+1):
        print(f"\n=== Cycle {cycle} ===")
        data_samples = generate_data(n_samples)
        tensor_data = preprocess_data(data_samples)
        model = CharRNN().to(device)
        model, cycle_score, last_loss = train_until_better(model, tensor_data, device, best_score, cycle, ensure_loss_leq=FIRST_CYCLE_LOSS_TARGET)
        # Evaluate final model
        model_path = f'checkpoints/final_cycle{cycle}.pt'
        torch.save(model.state_dict(), model_path)
        eval_info = evaluate_model_path(model_path)
        print(f"Cycle {cycle} score: {eval_info['score']:.4f}, entropy: {eval_info['entropy']:.4f}, monobit_dev: {eval_info['monobit_dev']:.4f}")
        if eval_info['score'] > best_score:
            best_score = eval_info['score']
            best_model_path = model_path
            safe_write_text(f'checkpoints/best_diverse_top_cycle{cycle}.txt', '\n'.join(eval_info['diverse_top']))
    print(f"\nBest model: {best_model_path} with score {best_score:.4f}")
    return best_model_path, best_score

# =====================
# Main
# =====================
if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    best_path, best_score = evolve(cycles=20, n_samples=500)
    print(f"Finished evolution. Best model saved at {best_path} with score {best_score:.4f}")
    # Sample generation from best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharRNN().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    print("\nSample generated strings:")
    for _ in range(10):
        print(generate_text(model, random.choice(VOCAB_CHARS), max_len=40))