import os
import math
import random
import shutil
import secrets
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === Constants ===
CHARS = '_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
SYMBOLS = '!@#$%^&*()_+-={}|:;"\'<>,.?/\\~'
MIN_LEN = 10
MAX_LEN = 52
MAX_ATTEMPTS = 120000
TARGET_ENTROPY_RATIO = 0.95  # normalized entropy >95%

STREAM_1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
STREAM_2 = "!@#$%^&*()_+-={}|\:;\"'<>,.?/\\~"
VOCAB = STREAM_1 + STREAM_2
VOCAB_SIZE = len(VOCAB)

CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}

# Training hyperparams
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
MAX_EPOCHS_PER_TRY = 250
FIRST_CYCLE_MAX_EPOCHS = 250
FIRST_CYCLE_LOSS_TARGET = 2.0
EVAL_EVERY = 2
IMPROVE_EPS = 1e-3
MAX_RETRIES_PER_CYCLE = 3

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 5
DROPOUT = 0.3
BATCH_SIZE = 64
MAX_EPOCHS_PER_TRY = 200
EVAL_EVERY = 1  # epochs
LR = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

# === Utilities ===

def safe_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

# === Entropy & Randomness Tests ===

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
    avg_len = sum(len(s) for s in samples) / len(samples) if samples else 1
    expected = len(samples) * avg_len / VOCAB_SIZE
    chi2 = sum((counts.get(c, 0) - expected)**2 / expected for c in VOCAB) if expected > 0 else 0.0
    p_val = math.exp(-0.5 * chi2 / max(VOCAB_SIZE, 1))

    return monobit_dev, runs_dev, p_val

def is_case_variant(c1, c2):
    return c1.lower() == c2.lower() and c1 != c2

def calculate_entropy(s: str) -> float:
    counts = Counter(s)
    total = len(s)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return entropy

def generate_data(n_samples: int) -> list[str]:
    samples = []
    attempts = 0

    while len(samples) < n_samples and attempts < MAX_ATTEMPTS:
        attempts += 1
        length = random.randint(MIN_LEN, MAX_LEN)
        result = []
        last_letter = last_digit = last_symbol = ''

        while len(result) < length:
            # Decide stream type
            if not result:
                stream_type = random.choice(['letter', 'digit', 'symbol'])
            else:
                prev_char = result[-1]
                if prev_char.isalpha():
                    stream_type = random.choice(['digit', 'symbol'])
                elif prev_char.isdigit():
                    stream_type = random.choice(['letter', 'symbol'])
                else:
                    stream_type = random.choice(['letter', 'digit'])

            # Pick character from chosen stream
            if stream_type == 'letter':
                candidates = [c for c in CHARS[1:] if c.isalpha() and c != last_letter and c.lower() != last_letter.lower()]
                if not candidates:
                    continue
                ch = random.choice(candidates)
                last_letter = ch
            elif stream_type == 'digit':
                candidates = [c for c in CHARS[1:] if c.isdigit() and c != last_digit]
                if not candidates:
                    continue
                ch = random.choice(candidates)
                last_digit = ch
            else:  # symbol
                candidates = [c for c in SYMBOLS if c != last_symbol]
                if not candidates:
                    continue
                ch = random.choice(candidates)
                last_symbol = ch

            result.append(ch)

        candidate = ''.join(result)
        entropy = calculate_entropy(candidate)
        max_entropy = math.log2(len(set(candidate)))
        if max_entropy > 0 and (entropy / max_entropy) >= TARGET_ENTROPY_RATIO:
            samples.append(candidate)

    if len(samples) < n_samples:
        print(f"Warning: Stopping after {attempts} attempts, generated {len(samples)} samples")
    return samples

def compute_score(entropy, monobit_dev, runs_dev, chi2_p, loss):
    max_entropy = math.log2(VOCAB_SIZE)
    entropy_score = entropy / max_entropy
    monobit_score = 1.0 - min(abs(monobit_dev), 1.0)
    runs_score = 1.0 - min(abs(runs_dev), 1.0)
    chi2_score = 1.0 - min(abs(chi2_p - 0.5) * 2, 1.0)
    loss_score = 1.0 - min(max((loss - FIRST_CYCLE_LOSS_TARGET), 0.0) / 1.0, 1.0)
    total = (0.4 * entropy_score + 0.15 * monobit_score + 0.15 * runs_score +
             0.2 * chi2_score + 0.1 * loss_score) * 10
    return float(round(total, 4))

# === Model ===

class CharRNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.fc(out)

# === Helpers ===

def char_tensor(string):
    return torch.tensor([CHAR_TO_IDX[c] for c in string], dtype=torch.long)

def generate_text(model, start_char='A', max_len=52, temperature=0.8, repetition_penalty=1.1):
    device = next(model.parameters()).device
    model.eval()
    target_len = random.randint(int(max_len * 0.8), int(max_len * 1.2))
    cur = torch.tensor([[CHAR_TO_IDX[start_char]]], device=device)
    generated = start_char
    with torch.no_grad():
        for _ in range(target_len - 1):
            output = model(cur)
            if isinstance(output, tuple):
                logits = output[0][:, -1]
            else:
                logits = output[:, -1]

            logits = logits / max(temperature, 1e-6)
            for c in generated[-4:]:
                logits[0, CHAR_TO_IDX[c]] /= repetition_penalty

            prob = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, 1)
            ch = IDX_TO_CHAR[next_idx.item()]
            if ch == '_':
                break
            generated += ch
            cur = torch.cat([cur, next_idx], dim=1)
    return generated

# === Training loop ===

def run_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inp)
        loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def evaluate_model_path(model_path, samples_for_eval=300):
    device = torch.device('cpu')
    model = CharRNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    samples = [generate_text(model, random.choice(CHARS[1:])) for _ in range(samples_for_eval)]
    entropy = float(np.mean([calculate_entropy(s) for s in samples]))
    monobit_dev, runs_dev, chi2_p = bitstring_entropy_tests(samples)
    loss_est = FIRST_CYCLE_LOSS_TARGET  # placeholder
    score = compute_score(entropy, monobit_dev, runs_dev, chi2_p, loss_est)

    score_txt = (
        f"Entropy: {entropy:.4f} "
        f"Monobit Dev: {monobit_dev:.4f} "
        f"Runs Dev: {runs_dev:.4f} "
        f"Chi2 p: {chi2_p:.4f} "
        f"Score: {score:.4f} "
        f"Model Path: {model_path}"
    )

    print(f"\n[Sample Outputs] from model {model_path}:")
    for i in range(10):
        sample = samples[random.randint(0, len(samples)-1)]
        print(f"  {i+1}: {sample}")
    print()

    # Create diverse sample pools for next cycle
    scored = [(s, calculate_entropy(s)) for s in samples]
    scored.sort(key=lambda x: -x[1])
    top_entropy = [s for s, _ in scored[:100]]
    top_middle = [s for s, _ in scored[100:200]]
    top_low = [s for s, _ in scored[200:]]
    diverse_top = top_entropy[:50] + top_middle[:30] + top_low[:20]

    return {
        'score': score,
        'model_path': model_path,
        'diverse_top': diverse_top,
        'score_txt': score_txt,
        'entropy': entropy,
        'monobit_dev': monobit_dev,
        'runs_dev': runs_dev,
        'chi2_p': chi2_p,
    }

def train_until_better(model, train_data, device, target_score, cycle_num, ensure_loss_leq=None):
    dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_IDX['_'])

    best_local = -float('inf')
    last_eval_score = -float('inf')

    if ensure_loss_leq is not None:
        for epoch in range(1, FIRST_CYCLE_MAX_EPOCHS + 1):
            avg_loss = run_epoch(model, loader, device, optimizer, criterion)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            if avg_loss <= ensure_loss_leq + 1e-8:
                break

    for epoch in range(1, MAX_EPOCHS_PER_TRY + 1):
        avg_loss = run_epoch(model, loader, device, optimizer, criterion)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                samples = [generate_text(model, random.choice(CHARS[1:])) for _ in range(120)]
                ent = float(np.mean([calculate_entropy(s) for s in samples]))
                mdev, rdev, pval = bitstring_entropy_tests(samples)
                score_now = compute_score(ent, mdev, rdev, pval, avg_loss)

            if score_now > best_local + IMPROVE_EPS:
                print(f"[local] Improved score: {score_now:.4f} > {best_local:.4f} (prev)")
                best_local = score_now

            if score_now > target_score + IMPROVE_EPS and score_now > last_eval_score + IMPROVE_EPS:
                print(f"[✓] Improved Score: {score_now:.4f} > {target_score:.4f} (prev target)")
                last_eval_score = score_now
                return avg_loss, score_now

    return avg_loss, max(best_local, last_eval_score)

def preprocess_data(data):
    # Pad or truncate strings to max length, encode to tensor
    max_len = max(len(s) for s in data)
    tensor_data = torch.full((len(data), max_len), fill_value=CHAR_TO_IDX['_'], dtype=torch.long)
    for i, s in enumerate(data):
        tensor_data[i, :len(s)] = char_tensor(s)
    return tensor_data

def train_and_save_on_gpu(gpu_id, cycle_num, top_samples, best_model_path=None, previous_score=-1.0):
    assert torch.cuda.is_available(), "CUDA not available for GPU worker"
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Using {torch.cuda.get_device_name(gpu_id)}")

    # Generate dataset for this GPU cycle
    data = generate_data(2500) + top_samples + random.sample(top_samples, max(1, len(top_samples)//2))
    padded = preprocess_data(data)

    device = torch.device(f"cuda:{gpu_id}")
    model = CharRNN().to(device)

    # Seed randomness per process
    torch.manual_seed(int.from_bytes(os.urandom(4), 'little'))
    random.seed()
    np.random.seed()

    # Load previous best model if any
    if best_model_path and os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

    ensure_loss = FIRST_CYCLE_LOSS_TARGET if cycle_num == 1 else None
    final_loss, final_score = train_until_better(model, padded, device, previous_score, cycle_num, ensure_loss_leq=ensure_loss)

    model_path = f"models/model_cycle_{cycle_num:02d}_gpu{gpu_id}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    return model_path, cycle_num, final_loss, final_score

# === Main evolve orchestrator ===

def evolve(cycles=20):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs found. Cannot run parallel GPU training.")

    best_overall_score = -float('inf')
    best_overall_path = ''

    best_entropy = -float('inf')
    best_monobit = float('inf')
    best_runs = float('inf')
    best_chi2_gap = float('inf')

    top_samples = generate_data(2400)

    for cycle in range(1, cycles + 1):
        print(f"=== Cycle {cycle}/{cycles} ===")
        retries = 0
        cycle_improved = False

        while retries == 0 or (not cycle_improved and retries < MAX_RETRIES_PER_CYCLE):
            model_paths = []
            with ProcessPoolExecutor(max_workers=num_gpus) as ex:
                futures = [ex.submit(train_and_save_on_gpu, gpu_id, cycle, top_samples,
                                     best_overall_path if best_overall_path else None, best_overall_score)
                           for gpu_id in range(num_gpus)]
                for fut in as_completed(futures):
                    mpath, cyc, _, _ = fut.result()
                    model_paths.append(mpath)

            results = []
            with ProcessPoolExecutor(max_workers=min(8, len(model_paths))) as ex:
                futs = [ex.submit(evaluate_model_path, p) for p in model_paths]
                for fut in as_completed(futs):
                    results.append(fut.result())

            for res in sorted(results, key=lambda r: -r['score']):
                print("[Evaluate Model] Model Path:", res['model_path'])
                print("[Evaluate Model] Score:", f"{res['score']:.4f}")
                print("[Evaluate Model] Score Data:" + res['score_txt'])

            results.sort(key=lambda r: -r['score'])
            best_res = results[0]
            best_score_round = best_res['score']
            best_path_round = best_res['model_path']

            print(f"[Evolve] Best score from results: {best_score_round:.4f}")
            print(f"[Evolve] Best model path: {best_path_round}")
            print("[Evolve] Score data to write:" + best_res['score_txt'])

            safe_write_text(f"scores/model_cycle_{cycle:02d}_score.txt", best_res['score_txt'])
            scored_copy_path = f"models/best_score_cycle_{cycle:02d}_{best_score_round:.4f}.pth"
            shutil.copyfile(best_path_round, scored_copy_path)

            print(f"Top Score: {best_score_round:.4f}, From: {os.path.basename(best_path_round)}")

            if best_score_round > best_overall_score + IMPROVE_EPS:
                print(f"[✓] New global best: {best_score_round:.4f} > {best_overall_score:.4f} (prev)")
                best_overall_score = best_score_round
                best_overall_path = best_path_round
                shutil.copyfile(best_path_round, f"models/best_overall_{best_overall_score:.4f}.pth")
                cycle_improved = True
            else:
                print(f"[Info] No global improvement this round (best: {best_overall_score:.4f}).")

            # Metric-specific tracking
            if best_res['entropy'] > best_entropy + IMPROVE_EPS:
                best_entropy = best_res['entropy']
                shutil.copyfile(best_path_round, f"models/best_entropy_{best_entropy:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_entropy_{best_entropy:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            if best_res['monobit_dev'] < best_monobit - IMPROVE_EPS:
                best_monobit = best_res['monobit_dev']
                shutil.copyfile(best_path_round, f"models/best_monobit_{best_monobit:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_monobit_{best_monobit:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            if best_res['runs_dev'] < best_runs - IMPROVE_EPS:
                best_runs = best_res['runs_dev']
                shutil.copyfile(best_path_round, f"models/best_runs_{best_runs:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_runs_{best_runs:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            chi_gap = abs(best_res['chi2_p'] - 0.5)
            if chi_gap < best_chi2_gap - IMPROVE_EPS:
                best_chi2_gap = chi_gap
                shutil.copyfile(best_path_round, f"models/best_chi2_{1.0 - 2*best_chi2_gap:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_chi2_{1.0 - 2*best_chi2_gap:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            print(f"[Cycle {cycle}] Best Score so far: {best_overall_score:.4f} from {os.path.basename(best_overall_path) if best_overall_path else 'n/a'}")

            # Prepare next seed pool (diversity + noise)
            pool = []
            for res in results[:2]:
                pool.extend(res['diverse_top'])
            pool.extend(generate_data(600))
            seen = set()
            uniq = []
            for s in pool:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
                if len(uniq) >= 2400:
                    break
            top_samples = uniq

            if cycle_improved:
                break

            retries += 1
            print(f"[Retry] No improvement; retrying training for cycle {cycle} (attempt {retries}/{MAX_RETRIES_PER_CYCLE})")

    print("=== Finished ===")
    print(f"Best overall score: {best_overall_score:.4f}")
    print(f"Best overall model: {best_overall_path}")

if __name__ == '__main__':
    evolve(cycles=20)
