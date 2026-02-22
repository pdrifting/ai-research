import torch
from torch import nn
import secrets
import string
import os
import random
import hashlib
import bisect
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from scipy.stats import chisquare, ks_2samp, entropy
import io
import gzip
from nistrng import SP800_22R1A_BATTERY

fixed_lengths = [8,16,32,64,128,256,512,1024,2028,4096,8192]
random_ranges = [(50,1000),(250,5000),(1000,10000),(1000,250000)]

# Track entropy scores across training epochs
entropy_progression = []
global_entropy_curve = []

def track_entropy_progress(score):
    global_entropy_curve.append(score)

# Helper to generate fresh salt/IV/nonce
def random_salt(length=32):
    chars = string.ascii_letters + string.digits + "!@#$%^&*()-_=+<>?:"
    return ''.join(secrets.choice(chars) for _ in range(length))

def random_bytes(length):
    return secrets.token_bytes(length)

def bytes_to_bits(byte_stream):
    return [int(bit) for b in byte_stream for bit in f"{b:08b}"]

# STREAM GENERATORS

# ATMOSPHERIC FIXED LENGTH STREAM
ATMOSPHERIC_DIR = "atmospheric"
ATMOSPHERIC_CACHE = []
ATMOSPHERIC_INDEX = 0

def initialize_atmospheric_pool():
    global ATMOSPHERIC_CACHE, ATMOSPHERIC_INDEX
    files = [f for f in os.listdir(ATMOSPHERIC_DIR) if f.startswith("RandomNumbers")]
    random.shuffle(files)
    stream = []
    
    for file in files:
        path = os.path.join(ATMOSPHERIC_DIR, file)
        with open(path, "rb") as f:
            stream.extend(list(f.read()))
    
    ATMOSPHERIC_CACHE = stream
    ATMOSPHERIC_INDEX = 0
    print(f"[✔] Atmospheric entropy pool initialized with {len(ATMOSPHERIC_CACHE)} bytes.")

def generate_atmospheric_stream(byte_count):
    global ATMOSPHERIC_CACHE, ATMOSPHERIC_INDEX
    
    total = len(ATMOSPHERIC_CACHE)
    result = []
    
    while byte_count > 0:
        remaining = total - ATMOSPHERIC_INDEX
        if remaining >= byte_count:
            result.extend(ATMOSPHERIC_CACHE[ATMOSPHERIC_INDEX:ATMOSPHERIC_INDEX + byte_count])
            ATMOSPHERIC_INDEX += byte_count
            byte_count = 0
        else:
            result.extend(ATMOSPHERIC_CACHE[ATMOSPHERIC_INDEX:])
            ATMOSPHERIC_INDEX = 0
            byte_count -= remaining
    
    return result

# SHA DIGEST STREAM (DRGB)
def drbg_sha(seed="KelvinSha2512", hash_type="sha512", length=16384):
    hash_fn = lambda x: hashlib.new(hash_type, x).digest()
    block_size = 64  # For both SHA2-512 and SHA3-512 outputs (512 bits = 64 bytes)
    stream = bytearray()
    
    counter = 0
    salt = secrets.token_bytes(32)
    base = f"{seed}".encode() + salt

    while len(stream) < length:
        input_block = base + counter.to_bytes(4, 'big')
        stream.extend(hash_fn(input_block))
        counter += 1

    return list(stream[:length])

# AES STREAM (DRGB)
def drbg_aes(mode="cbc", length=16384):
    key = secrets.token_bytes(32)      # AES-256 key
    iv = secrets.token_bytes(16)       # Initialization vector
    backend = default_backend()

    # Choose block cipher mode
    if mode == "cbc":
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    elif mode == "gcm":
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=backend)
    else:
        raise ValueError("Unsupported AES mode")

    encryptor = cipher.encryptor()

    # Generate and encrypt zeroed blocks until target length
    block = b"\x00" * 4096  # AES block multiples
    stream = bytearray()

    while len(stream) < length:
        stream.extend(encryptor.update(block))

    stream.extend(encryptor.finalize())
    return list(stream[:length])

# CHACHA20 STREAM (DRGB)
def drbg_chacha(length=16384):
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(16)  # ChaCha20 uses 96-bit nonce
    backend = default_backend()

    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=backend)
    encryptor = cipher.encryptor()

    block = b"\x00" * 4096
    stream = bytearray()

    while len(stream) < length:
        stream.extend(encryptor.update(block))

    return list(stream[:length])

# RDS 1024 BIT DIGEST STREAM (DRGB)
def drbg_rsa(length=16384, key_size=1024):
    key = RSA.generate(key_size)
    cipher = PKCS1_OAEP.new(key)
    block_size = key_size // 8 - 42  # Max OAEP input per encrypt

    stream = bytearray()
    counter = 0

    while len(stream) < length:
        msg = secrets.token_bytes(block_size - 4) + counter.to_bytes(4, 'big')
        encrypted = cipher.encrypt(msg)
        stream.extend(encrypted)
        counter += 1

    return list(stream[:length])

# PGP DIGEST STREAM (DRGB)
def drbg_pgp(length=16384):
    key = secrets.token_bytes(32)
    iv = secrets.token_bytes(16)
    base = secrets.token_bytes(64)

    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    enc = cipher.encryptor()

    stream = bytearray()
    counter = 0

    while len(stream) < length:
        m = hashlib.sha512(base + counter.to_bytes(4, 'big')).digest()
        encrypted = enc.update(m)
        stream.extend(encrypted)
        counter += 1

    return list(stream[:length])

# BLAKE2B DIGEST OF STRONGEST ENTROPY STREAMS (DRGB)
def drgb_blake2b_pool(sources=None, length=16384):
    if sources is None:
        sources = ["sha512", "aes256", "chacha", "sha3_512", "rsa", "pgp", "atmospheric"]

    pool = bytearray()
    salt = secrets.token_bytes(64)

    # Collect entropy from multiple secure sources
    for src in sources:
        chunk = generate_secure_stream(src, length=4096)
        pool.extend(chunk)

    # Mix with Blake2b hash
    blake_hash = hashlib.blake2b(pool + salt, digest_size=64).digest()

    # Expand output by chaining digest rounds
    output = bytearray()
    counter = 0
    while len(output) < length:
        h = hashlib.blake2b(blake_hash + counter.to_bytes(4, "big"), digest_size=64)
        output.extend(h.digest())
        counter += 1

    return list(output[:length])

# GENERATOR FOR SPECIFIC SECURE STREAMS
def generate_secure_stream(source="sha512", length=16384):
    salt = random_salt()

    if source == "sha512":
        return drbg_sha(f"FooBar2512-{salt}", "sha512", length)
    if source == "sha3_512":
        return drbg_sha(f"FooBarSha3512-{salt}", "sha3_512", length)
    if source == "aes256":
        return drbg_aes("cbc", length)
    if source == "aes_gcm":
        return drbg_aes("gcm", length)
    if source == "chacha":
        return drbg_chacha(length)
    if source == "rsa":
        return drbg_rsa(length)
    if source == "pgp":
        return drbg_pgp(length)
    if source == "atmospheric":
        return generate_atmospheric_stream(length)
    if source == "blake2b_fusion":
        return drgb_blake2b_pool(length=length)
    raise ValueError("Unknown secure source")

#------- END OF SECURE STREAMS GENERATION ------

# Insecure Steam Generators
def generate_constant_stream(value=127, length=16384):
    return [value] * length

def generate_pattern_stream(pattern=[1, 2, 3, 4], length=16384):
    return (pattern * (length // len(pattern)))[:length]

def generate_biased_stream(length=16384):
    weights = [1]*240 + [10]*16
    return random.choices(range(256), weights=weights, k=length)

def generate_sorted_stream(length=16384):
    return sorted(random.choices(range(256), k=length))

def generate_low_bit_entropy_stream(length=16384):
    return [random.randint(0, 15) << 4 for _ in range(length)]

def generate_insecure_stream(source="constant"):
    if source == "constant": return generate_constant_stream()
    if source == "pattern": return generate_pattern_stream()
    if source == "biased": return generate_biased_stream()
    if source == "sorted": return generate_sorted_stream()
    if source == "lowbit": return generate_low_bit_entropy_stream()
    raise ValueError("Unknown bad source")

#------- END OF INSECURE STREAMS GENERATION ------

# Graded Entropy Stream Models
class GradientDRBGTracker:
    def __init__(self):
        self.buckets = {round(score, 1): [] for score in [x * 0.1 for x in range(30, 81)]}

    def update(self, model_name, stream, metrics):
        score = metrics["scipy"]["entropy"]
        key = round(score, 1)
        if key not in self.buckets:
            return
        self.buckets[key].append({
            "model": model_name,
            "score": score,
            "metrics": metrics,
            "stream": stream
        })
        # Keep only top 3 scoring models per entropy bucket
        self.buckets[key] = sorted(self.buckets[key], key=lambda x: x["score"], reverse=True)[:3]

    def get_top_models(self, entropy_level):
        key = round(entropy_level, 1)
        return self.buckets.get(key, [])

def drbg_from_model(model, device, length=16384):
    model.eval()
    stream = generate_model_stream(model, device=device, length=length)
    return stream

# Dispatcher for training data streams...

def generate_stream_by_hardness(hardness="secure", method="sha512", length=16384):
    if hardness == "secure":
        return generate_secure_stream(source=method)[:length]
    elif hardness == "insecure":
        return generate_insecure_stream(source=method)[:length]
    raise ValueError("Invalid hardness label")

# Testing Metric Algorithms and Intepretation

def interpret_score(metrics):
    e = metrics["entropy"]
    m = metrics["monobit"]
    c = metrics["compression"]
    
    if e < 5.0: return "Terrible"
    if e < 6.0: return "Bad"
    if e < 7.0: return "Moderate"
    if e < 7.6 and m < 0.1 and c < 0.8: return "Good"
    if e >= 7.6 and m < 0.05 and c < 0.6: return "Excellent"
    if e >= 7.9 and m < 0.03 and c < 0.5 and metrics["nist_freq"] > 0.99:
        return "Secure"
    return "Unclassified"

def run_nistrng_tests(byte_stream):
    bitstream = bytes_to_bits(byte_stream)
    results = {}

    for name, test_obj in SP800_22R1A_BATTERY.items():
        try:
            result = test_obj.execute(bitstream)
            results[name] = round(result.p_value, 5)
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    
    return results

def run_scipy_tests(stream):
    hist = [stream.count(i) for i in range(256)]
    ks_ref = list(range(len(stream)))
    return {
        "chi": chisquare(hist),
        "ks": ks_2samp(stream, ks_ref),
        "shannon": entropy(hist)
    }

def compression_ratio(stream):
    raw = bytes(stream)
    compressed = gzip.compress(raw)
    return round(len(compressed) / len(raw), 4)

def monobit_frequency(stream):
    bits = "".join(f"{b:08b}" for b in stream)
    ones = bits.count("1")
    zeros = bits.count("0")
    return round(abs(ones - zeros) / len(bits), 4)

def grade_entropy(stream):
    # Scipy metrics
    hist = [stream.count(i) for i in range(256)]
    ks_ref = list(range(len(stream)))
    scipy_results = {
        "entropy": round(entropy(hist), 5),
        "monobit": monobit_frequency(stream),
        "compression": compression_ratio(stream),
        "chi2_stat": round(chisquare(hist).statistic, 2),
        "ks_test": round(ks_2samp(stream, ks_ref).pvalue, 5)
    }

    # NISTRNG full suite
    nistrng_results = run_nistrng_tests(stream)

    # Interpretation
    label = interpret_score(scipy_results)
    
    return {
        "grade": label,
        "scipy": scipy_results,
        "nist_extended": nistrng_results
    }

# Choose best available GPU configuration
def get_devices():
    available = torch.cuda.device_count()
    if available >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    elif available == 1:
        return torch.device("cuda:0"), torch.device("cpu")
    else:
        return torch.device("cpu"), torch.device("cpu")

# Placeholder generator (replace with DCGAN/Transformer/etc.)
class SecureEntropyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 16384),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Entropy classifier (binary secure/insecure)
class EntropyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 2)  # Output: secure/insecure
        )

    def forward(self, x):
        return self.net(x)

# Load models onto devices
device_g, device_d = get_devices()
generator = SecureEntropyGenerator().to(device_g)
classifier = EntropyClassifier().to(device_d)

def train_entropy_generator(generator, classifier, device_g, device_d, tracker, epoch=0):
    generator.train()
    optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Generate noise
    z = torch.randn(1, 100).to(device_g)
    output = generator(z).view(-1)

    # Convert to uint8 entropy stream
    stream = ((output + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).tolist()
    metrics = grade_entropy(stream)
    label = metrics["grade"]
    entropy_score = metrics["scipy"]["entropy"]

    # Track entropy curve
    entropy_progression.append(entropy_score)

    entropy_target = 7.0
    scaling = min(max((entropy_target - entropy_score) / entropy_target, 0), 1)
    penalty_weight = 0.5 + scaling  # escalates penalty when below target

    # Classifier prediction
    classifier.eval()
    stream_tensor = torch.tensor(stream, dtype=torch.float32).view(1, 1, -1).to(device_d)
    target = torch.tensor([1]).to(device_d)

    prediction = classifier(stream_tensor)
    classification_loss = loss_fn(prediction, target)

    # Add entropy-based penalty
    entropy_penalty = max(7.0 - entropy_score, 0)
    total_loss = classification_loss + entropy_penalty * penalty_weight

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Checkpoint if generator hits a tier
    if entropy_score >= 7.0:
        torch.save(generator.state_dict(), f"drbg_gen_checkpoint_epoch_{epoch}.pt")

    tracker.update(f"Generator_Epoch_{epoch}", stream, metrics)
    print(f"🎯 Epoch {epoch}: Gen Entropy {entropy_score:.3f} | Grade: {label} | Loss: {total_loss.item():.4f}")
    track_entropy_progress(entropy_score)

def evolve_generator_if_needed():
    recent_scores = global_entropy_curve[-10:]
    avg = sum(recent_scores) / len(recent_scores)

    if avg >= 7.2:
        print("🧬 Generator shows sustained entropy growth. Time to evolve model.")
        # Replace model or deepen layers here

def train_entropy_classifier(classifier, device, epoch=0):
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Streams
    secure_stream = generate_secure_stream("blake2b_fusion", length=16384)
    insecure_stream = generate_insecure_stream("pattern")

    # Prepare training batch
    inputs = torch.tensor([secure_stream, insecure_stream], dtype=torch.float32).view(2, 1, -1).to(device)
    labels = torch.tensor([1, 0]).to(device)

    output = classifier(inputs)
    loss = loss_fn(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"🛡️ Epoch {epoch}: Classifier Loss {loss.item():.4f}")

tracker = GradientDRBGTracker()

def start_entropy_models(total_epochs, generator, classifier, tracker):
    for epoch in range(total_epochs):
        train_entropy_generator(generator, classifier, device_g, device_d, tracker, epoch)
        train_entropy_classifier(classifier, device_d, epoch)
        print_epoch_summary(epoch)

        if epoch % 3 == 0:
            print("🔍 Top DRBGs @ entropy tier 7.0:")
            for model in tracker.get_top_models(7.0):
                print(f"🧬 {model['model']} | Score: {model['score']:.3f}")        

def adjust_generator_for_next_generation(gen):
    if gen % 2 == 0:
        # Increase depth or learning rate
        for param_group in generator_optimizer.param_groups:
            param_group["lr"] *= 1.1
        print("⚙️ Boosted generator learning rate for next generation")

def run_entropy_generations(generations=5, epochs_per_generation=50):
    for gen in range(1, generations + 1):
        print(f"\n🚀 Starting Entropy Generation {gen}")

        # Fresh tracker per generation
        tracker = GradientDRBGTracker()

        # Load fresh or evolved models
        generator = SecureEntropyGenerator().to(device_g)
        classifier = EntropyClassifier().to(device_d)

        start_entropy_models(
            total_epochs=epochs_per_generation,
            generator=generator,
            classifier=classifier,
            tracker=tracker
        )

        # Save checkpoints
        torch.save(generator.state_dict(), f"gen_{gen}_generator.pt")
        torch.save(classifier.state_dict(), f"gen_{gen}_classifier.pt")
        print(f"💾 Saved Generation {gen} models")

        #adjust_generator_for_next_generation(gen)
        evolve_generator_if_needed()  # Optionally reset optimizer or increase learning pressure
        
def print_epoch_summary(epoch):
    print(f"\n📊 Epoch {epoch} Summary:")
    print(f"   Generator Entropy Progression: {entropy_progression[-1]:.3f}")
    if entropy_progression[-1] >= 7.0:
        print("   ✅ DRBG milestone reached! Model checkpoint saved.")
    else:
        print("   ⚠️ Entropy below target. Reinforcement required.")

def main():
    print("🧬 Initializing Entropy Engine...")
    initialize_atmospheric_pool()

    print("⚙️ Bootstrapping entropy model ecosystem...\n")
    run_entropy_generations(generations=5, epochs_per_generation=50)

    print("\n🏁 All generations complete.")
    print("📊 Final top-performing DRBG models:")
    for tier in [7.0, 7.5, 7.9]:
        models = tracker.get_top_models(tier)
        for m in models:
            print(f"🧬 Entropy {tier:.1f} → {m['model']} → Score: {m['score']:.3f}")

if __name__ == "__main__":
    main()