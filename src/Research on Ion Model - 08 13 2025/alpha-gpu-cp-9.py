# entropy_rnn_evolver_parallel.py

import os
import math
import random
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import secrets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

# =====================
# Config / Constants
# =====================
CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
VOCAB_SIZE = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}

# Paths
os.makedirs("models", exist_ok=True)
os.makedirs("scores", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Training hyperparams
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

# Gates
FIRST_CYCLE_LOSS_TARGET = 1.25   # ensure first cycle gets loss <= this
FIRST_CYCLE_MAX_EPOCHS = 400
IMPROVE_EPS = 1e-6               # min delta to consider "improved"
MAX_RETRIES_PER_CYCLE = 2        # if no model beats previous best, retry training more times

# =====================
# Utilities
# =====================

def debug_startup():
    print("==== Debug: Environment ====")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"[GPU {i}] {name} | capability {cap}")
    print("============================")

def safe_write_text(path: str, text: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

# =====================
# Model
# =====================
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

# =====================
# Data
# =====================

def save_training_samples(samples, filename="training_samples.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")

def calculate_entropy(s):
    # Shannon entropy in bits
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

# Max entropy for a given length (full unique distribution = log2(len(CHARS)) * length)
# For normalization, use this as the "ideal" cap.
MAX_ENTROPY_BY_LENGTH = {
    length: math.log2(len(CHARS)) * length
    for length in range(10, 53)
}

def generate_data(num_samples=10000, min_len=10, max_len=52):
    accepted = []
    attempts = 0

    while len(accepted) < num_samples and attempts < num_samples * 5000:
        # Fresh pools
        digits = list("0123456789")
        lowers = list("abcdefghijklmnopqrstuvwxyz")
        uppers = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        pools = {
            "digit": digits,
            "lower": lowers,
            "upper": uppers
        }

        candidate = []
        last_type = None
        length = secrets.randbelow(max_len - min_len + 1) + min_len

        while len(candidate) < length:
            # Allowed categories (avoid repeats)
            possible_types = [t for t in pools if pools[t] and t != last_type]
            if not possible_types:
                break

            char_type = secrets.choice(possible_types)
            pool = pools[char_type]
            ch = pool.pop(secrets.randbelow(len(pool)))

            # Block adjacent upper/lower variant
            if candidate:
                prev = candidate[-1]
                if prev.lower() == ch.lower() and prev != ch:
                    pool.append(ch)
                    continue

            candidate.append(ch)
            last_type = char_type

        if len(candidate) < min_len:
            attempts += 1
            continue

        s = "".join(candidate)
        entropy = calculate_entropy(s)
        max_entropy = MAX_ENTROPY_BY_LENGTH[len(s)]
        normalized_entropy = entropy / max_entropy

        if normalized_entropy >= 0.01:  # keep only high-entropy sequences
            accepted.append(s)
            print(f"{len(s)} chars | Entropy: {entropy:.4f} / {max_entropy:.4f} ({normalized_entropy:.2%}) | {s}")

        attempts += 1

    return accepted

def generate_data_BAD(num_samples=10000, max_len=30, min_entropy=4.5, max_monobit_dev=0.1, max_runs_dev=0.2):
    accepted = []
    attempts = 0
    while len(accepted) < num_samples and attempts < num_samples * 10:
        s = ''.join(random.choices(CHARS[1:], k=random.randint(5, max_len)))
        entropy = calculate_entropy(s)
        monobit_dev, runs_dev, _ = bitstring_entropy_tests([s])
        
        # Check if it meets your randomness thresholds
        if entropy >= min_entropy and monobit_dev <= max_monobit_dev and runs_dev <= max_runs_dev:
            accepted.append(s)
        else:
            print(f"rejected {s}")
        attempts += 1

    if len(accepted) < num_samples:
        print(f"Warning: Only generated {len(accepted)} samples meeting criteria out of requested {num_samples}")

    save_training_samples(accepted)
    return accepted
    

def preprocess_data(data):
    seqs = [torch.tensor([CHAR_TO_IDX[c] for c in s], dtype=torch.long) for s in data]
    return pad_sequence(seqs, batch_first=True, padding_value=CHAR_TO_IDX['_'])

# =====================
# Metrics / Scoring
# =====================

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
    # Expected per symbol (rough heuristic)
    avg_len = np.mean([len(s) for s in samples]) if samples else 1
    expected = len(samples) * avg_len / (VOCAB_SIZE - 1)
    chi2 = sum((counts.get(c, 0) - expected)**2 / expected for c in CHARS[1:]) if expected > 0 else 0.0
    # crude p-value proxy in [0,1]
    p_val = math.exp(-0.5 * chi2 / max(VOCAB_SIZE, 1))

    return monobit_dev, runs_dev, p_val

def compute_score(entropy, monobit_dev, runs_dev, chi2_p, loss):
    max_entropy = math.log2(VOCAB_SIZE)
    entropy_score = entropy / max_entropy
    monobit_score = 1.0 - min(abs(monobit_dev), 1.0)
    runs_score = 1.0 - min(abs(runs_dev), 1.0)
    chi2_score = 1.0 - min(abs(chi2_p - 0.5) * 2, 1.0)
    loss_score = 1.0 - min(max((loss - 2.05), 0.0) / 1.0, 1.0)
    total = (0.4 * entropy_score + 0.15 * monobit_score + 0.15 * runs_score + 0.2 * chi2_score + 0.1 * loss_score) * 10
    return float(round(total, 4))

# =====================
# Text generation
# =====================

def generate_text(model, start_char='A', max_len=30, temperature=0.8, repetition_penalty=1.1):
    device = next(model.parameters()).device
    model.eval()
    target_len = random.randint(int(max_len * 0.8), int(max_len * 1.2))
    cur = torch.tensor([[CHAR_TO_IDX[start_char]]], device=device)
    generated = start_char
    with torch.no_grad():
        for _ in range(target_len - 1):
            logits = model(cur)[:, -1] / max(temperature, 1e-6)
            for c in generated[-4:]:
                logits[0, CHAR_TO_IDX[c]] /= repetition_penalty
            prob = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, 1)
            ch = IDX_TO_CHAR[next_idx.item()]
            if ch == '_':
                break
            generated += ch
            cur = torch.cat([cur, next_idx.view(1, 1)], dim=1)
    return generated

# =====================
# Training
# =====================

def run_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)

def evaluate_model_path(model_path, samples_for_eval=300):
    device = torch.device('cpu')
    model = CharRNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    samples = [generate_text(model, random.choice(CHARS[1:])) for _ in range(samples_for_eval)]
    entropy = float(np.mean([calculate_entropy(s) for s in samples]))
    monobit_dev, runs_dev, chi2_p = bitstring_entropy_tests(samples)
    loss_est = 2.05  # placeholder in scoring mix
    score = compute_score(entropy, monobit_dev, runs_dev, chi2_p, loss_est)

    score_txt = (
        f"Entropy: {entropy:.4f}"
        f"Monobit Dev: {monobit_dev:.4f}"
        f"Runs Dev: {runs_dev:.4f}"
        f"Chi2 p: {chi2_p:.4f}"
        f"Score: {score:.4f}"
        f"Model Path: {model_path}"
    )

    print(f"\n[Sample Outputs] from model {model_path}:")
    for i in range(10):
        sample = samples[random.randint(0, len(samples)-1)]
        print(f"  {i+1}: {sample}")
    print()

    # Build diverse set for next cycle
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
    # Dataset / loader
    dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_IDX['_'])

    best_local = -float('inf')
    last_eval_score = -float('inf')

    # Optional bootstrap for first cycle to reach target loss
    if ensure_loss_leq is not None:
        for epoch in range(1, FIRST_CYCLE_MAX_EPOCHS + 1):
            avg_loss = run_epoch(model, loader, device, optimizer, criterion)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            if avg_loss <= ensure_loss_leq + 1e-8:
                break

    # Main training until score improves or max epochs elapsed
    for epoch in range(1, MAX_EPOCHS_PER_TRY + 1):
        avg_loss = run_epoch(model, loader, device, optimizer, criterion)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if epoch % EVAL_EVERY == 0:
            # quick eval using text samples
            model.eval()
            with torch.no_grad():
                samples = [generate_text(model, random.choice(CHARS[1:])) for _ in range(120)]
                ent = float(np.mean([calculate_entropy(s) for s in samples]))
                mdev, rdev, pval = bitstring_entropy_tests(samples)
                score_now = compute_score(ent, mdev, rdev, pval, avg_loss)

            if score_now > best_local + IMPROVE_EPS:
                print(f"[local] Improved score: {score_now:.4f} > {best_local:.4f} (prev)")
                best_local = score_now

            # only announce improvement vs target_score when truly better
            if score_now > target_score + IMPROVE_EPS and score_now > last_eval_score + IMPROVE_EPS:
                print(f"[✓] Improved Score: {score_now:.4f} > {target_score:.4f} (prev target)")
                last_eval_score = score_now
                return avg_loss, score_now

    # return best seen (may be <= target)
    return avg_loss, max(best_local, last_eval_score)

# =====================
# Per-GPU worker
# =====================

def train_and_save_on_gpu(gpu_id, cycle_num, top_samples, best_model_path=None, previous_score=-1.0):
    # isolate CUDA per process
    assert torch.cuda.is_available(), "CUDA not available for GPU worker"
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Using {torch.cuda.get_device_name(gpu_id)}")

    # Build dataset for this GPU
    data = generate_data(2500) + top_samples + random.sample(top_samples, max(1, len(top_samples)//2))
    padded = preprocess_data(data)

    device = torch.device(f"cuda:{gpu_id}")
    model = CharRNN().to(device)

    # seed randomness per process
    torch.manual_seed(int.from_bytes(os.urandom(4), 'little'))
    random.seed()
    np.random.seed()

    # Load best so far if provided
    if best_model_path and os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # First cycle bootstrap to loss threshold
    ensure_loss = FIRST_CYCLE_LOSS_TARGET if cycle_num == 1 else None
    final_loss, final_score = train_until_better(model, padded, device, previous_score, cycle_num, ensure_loss_leq=ensure_loss)

    # Save checkpoint
    model_path = f"models/model_cycle_{cycle_num:02d}_gpu{gpu_id}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path, cycle_num, final_loss, final_score

# =====================
# Evolution Orchestrator
# =====================

def evolve(cycles=20):
    debug_startup()

    num_gpus = min(torch.cuda.device_count(), 3) if torch.cuda.is_available() else 0
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs found. Cannot run parallel GPU training.")

    # metric-best trackers across all cycles
    best_overall_score = -float('inf')
    best_overall_path = ''

    best_entropy = -float('inf')
    best_entropy_path = ''

    best_monobit = float('inf')  # lower is better
    best_monobit_path = ''

    best_runs = float('inf')     # lower is better
    best_runs_path = ''

    best_chi2_gap = float('inf') # |p-0.5| lower is better
    best_chi2_path = ''

    # seed pool
    top_samples = generate_data(1200)

    for cycle in range(1, cycles + 1):
        print(f"=== Cycle {cycle}/{cycles} ===")

        # We require improvement over previous global best to advance; but will allow a few retries.
        retries = 0
        cycle_improved = False

        while retries == 0 or (not cycle_improved and retries < MAX_RETRIES_PER_CYCLE):
            # 1) Train in parallel on all GPUs
            model_paths = []
            with ProcessPoolExecutor(max_workers=num_gpus) as ex:
                futs = [
                    ex.submit(train_and_save_on_gpu, gid, cycle, top_samples, best_overall_path if best_overall_path else None, best_overall_score)
                    for gid in range(num_gpus)
                ]
                for fut in as_completed(futs):
                    mpath, cyc, _, _ = fut.result()
                    model_paths.append(mpath)

            # 2) Evaluate all models (CPU)
            results = []
            with ProcessPoolExecutor(max_workers=min(8, len(model_paths))) as ex:
                futs = [ex.submit(evaluate_model_path, p) for p in model_paths]
                for fut in as_completed(futs):
                    res = fut.result()
                    results.append(res)

            # Pretty print each result
            for res in sorted(results, key=lambda r: -r['score']):
                print("[Evaluate Model] Model Path:", res['model_path'])
                print("[Evaluate Model] Score:", f"{res['score']:.4f}")
                print("[Evaluate Model] Score Data:" + res['score_txt'])

            # 3) Pick best of this round
            results.sort(key=lambda r: -r['score'])
            best_res = results[0]
            best_score_round = best_res['score']
            best_path_round = best_res['model_path']

            print(f"[Evolve] Best score from results: {best_score_round:.4f}")
            print(f"[Evolve] Best model path: {best_path_round}")
            print("[Evolve] Score data to write:" + best_res['score_txt'])

            # 4) Save per-cycle best score file + duplicate model with score in name
            score_file = f"scores/model_cycle_{cycle:02d}_score.txt"
            safe_write_text(score_file, best_res['score_txt'])

            scored_copy_path = f"models/best_score_cycle_{cycle:02d}_{best_score_round:.4f}.pth"
            shutil.copyfile(best_path_round, scored_copy_path)

            print(f"Top Score: {best_score_round:.4f}, From: {os.path.basename(best_path_round)}")

            # 5) Update global overall best
            if best_score_round > best_overall_score + IMPROVE_EPS:
                print(f"[✓] New global best: {best_score_round:.4f} > {best_overall_score:.4f} (prev)")
                best_overall_score = best_score_round
                best_overall_path = best_path_round
                # also store a canonical copy
                shutil.copyfile(best_path_round, f"models/best_overall_{best_overall_score:.4f}.pth")
                cycle_improved = True
            else:
                print(f"[Info] No global improvement this round (best: {best_overall_score:.4f}).")

            # 6) Update metric-specific bests (across ALL cycles)
            # Entropy (higher is better)
            if best_res['entropy'] > best_entropy + IMPROVE_EPS:
                best_entropy = best_res['entropy']
                best_entropy_path = best_path_round
                shutil.copyfile(best_path_round, f"models/best_entropy_{best_entropy:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_entropy_{best_entropy:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            # Monobit dev (lower is better)
            if best_res['monobit_dev'] < best_monobit - IMPROVE_EPS:
                best_monobit = best_res['monobit_dev']
                best_monobit_path = best_path_round
                shutil.copyfile(best_path_round, f"models/best_monobit_{best_monobit:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_monobit_{best_monobit:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            # Runs dev (lower is better)
            if best_res['runs_dev'] < best_runs - IMPROVE_EPS:
                best_runs = best_res['runs_dev']
                best_runs_path = best_path_round
                shutil.copyfile(best_path_round, f"models/best_runs_{best_runs:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_runs_{best_runs:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            # Chi2 closeness to 0.5 (minimize |p-0.5|)
            chi_gap = abs(best_res['chi2_p'] - 0.5)
            if chi_gap < best_chi2_gap - IMPROVE_EPS:
                best_chi2_gap = chi_gap
                best_chi2_path = best_path_round
                shutil.copyfile(best_path_round, f"models/best_chi2_{1.0 - 2*best_chi2_gap:.4f}_cycle{cycle:02d}.pth")
                safe_write_text(f"scores/best_chi2_{1.0 - 2*best_chi2_gap:.4f}_cycle{cycle:02d}.txt", best_res['score_txt'])

            print(f"[Cycle {cycle}] Best Score so far: {best_overall_score:.4f} from {os.path.basename(best_overall_path) if best_overall_path else 'n/a'}")

            # 7) Prepare next-cycle seed pool (mix diversity + fresh noise)
            pool = []
            # collect top-2 diverse sets if available
            for res in results[:2]:
                pool.extend(res['diverse_top'])
            # add fresh noise
            pool.extend(generate_data(600))
            # dedupe and cap
            seen = set()
            uniq = []
            for s in pool:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
                if len(uniq) >= 1200:
                    break
            top_samples = uniq

            # If improved, break retry loop
            if cycle_improved:
                break

            retries += 1
            print(f"[Retry] No improvement; retrying training for cycle {cycle} (attempt {retries}/{MAX_RETRIES_PER_CYCLE})")

    print("=== Finished ===")
    print(f"Best overall score: {best_overall_score:.4f}")
    print(f"Best overall model: {best_overall_path}")


if __name__ == '__main__':
    evolve(cycles=20)