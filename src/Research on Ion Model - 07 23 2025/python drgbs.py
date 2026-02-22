import secrets
import hashlib
import hmac
import threading
import pgpy
import blake3
import time
import random
import string
from typing import Literal
from bitarray import bitarray
from pgpy.constants import PubKeyAlgorithm, KeyFlags, HashAlgorithm, SymmetricKeyAlgorithm, CompressionAlgorithm
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, constant_time
from cryptography.hazmat.backends import default_backend
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP


# Helper to generate fresh salt/IV/nonce
def random_salt(length=32, charset=None):
    if charset is None:
        charset = string.ascii_letters + string.digits + "!@#$%^&*()-_=+<>?:"
    return ''.join(secrets.choice(charset) for _ in range(length))

def random_bytes(length):
    return secrets.token_bytes(length)

def bytes_to_bits(byte_stream):
    return [(b >> i) & 1 for b in byte_stream for i in reversed(range(8))]

def bits_to_bytes(bit_list):
    return bytes(
        sum((bit << (7 - i % 8)) for i, bit in enumerate(bit_list[j:j+8]))
        for j in range(0, len(bit_list), 8)
    )

class _BaseDRBG:
    def __init__(self, hash_type="sha512"):
        self._lock = threading.Lock()
        self.hash_type = hash_type.lower()
        self.buffer = bytearray()
        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []
        self.reseed_interval = 1024 * 1024  # 1MB
        self.reseed_callback = None
        self.prediction_resistance = False
        self._reinitialize_seeds()

    def set_reseed_callback(self, func: callable):
        self.reseed_callback = func

    def enable_prediction_resistance(self, enabled: bool = True):
        self.prediction_resistance = enabled

    def _reinitialize_seeds(self):
        with self._lock:
            self.counter = 0
            self.hmac_counter = 0
            self._total_bytes_generated = 0

            if self.reseed_callback:
                seed_material = self.reseed_callback()
            else:
                seed_material = secrets.token_bytes(64)

            salt = secrets.token_bytes(32)
            ikm = seed_material + salt
            self.key = HKDF(
                algorithm=hashes.SHA512(),
                length=64,
                salt=None,
                info=b"drbg-stream"
            ).derive(ikm)

    def _maybe_reseed(self):
        if self.prediction_resistance or self._total_bytes_generated >= self.reseed_interval:
            self._reinitialize_seeds()

    def _generate_block_hash(self):
        block = self.key + self.counter.to_bytes(8, "big")
        self.counter += 1
        return hashlib.new(self.hash_type, block).digest()

    def _generate_block_hmac(self):
        msg = self.hmac_counter.to_bytes(8, "big")
        self.hmac_counter += 1
        return hmac.new(self.key, msg, digestmod=self.hash_type).digest()

    def next_bytes(self, n):
        with self._lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                self.buffer.extend(self._generate_block_hash())
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return result

    def next_bytes_hmac(self, n):
        with self._lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                result.extend(self._generate_block_hmac())
            self._total_bytes_generated += n
            return result[:n]

    def next_bits_from_bytes(self, byte_fn, n):
        with self._lock:
            bits = bitarray()
            while len(bits) < n:
                if self._bit_pos == 8:
                    self._bit_stream = list(byte_fn(32))
                    self._bit_byte_index = 0
                    self._bit_pos = 0

                if self._bit_byte_index >= len(self._bit_stream):
                    self._bit_pos = 8
                    continue

                byte = self._bit_stream[self._bit_byte_index]
                bit = (byte >> (7 - self._bit_pos)) & 1
                bits.append(bit)

                self._bit_pos += 1
                if self._bit_pos == 8:
                    self._bit_pos = 0
                    self._bit_byte_index += 1
            return bits[:n]

    def next_bits(self, n):
        return self.next_bits_from_bytes(self.next_bytes, n)

    def next_bits_hmac(self, n):
        return self.next_bits_from_bytes(self.next_bytes_hmac, n)

    def export_bytes_to_bin_file(self, filename, n, use_hmac=False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "wb") as f:
            f.write(bytearray(int(b) for byte in data for b in f"{byte:08b}"))

    def export_bytes_to_hex_file(self, filename, n, use_hmac=False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

    def export_bits_to_file(self, filename, n, use_hmac=False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

class DRBG_HashStream(_BaseDRBG):
    def __init__(self, algo="sha512"):
        super().__init__(hash_type=algo)

class DRBG_AES(_BaseDRBG):
    def __init__(self, mode="ctr"):
        super().__init__()
        self.mode = mode.lower()
        self._init_cipher()

    def _init_cipher(self):
        if self.reseed_callback:
            key_material = self.reseed_callback()
        else:
            key_material = secrets.token_bytes(48)
        self.key = key_material[:32]
        self.iv = key_material[32:48]

        if self.mode == "gcm":
            mode_obj = modes.GCM(self.iv)
        elif self.mode == "ctr":
            mode_obj = modes.CTR(self.iv)
        else:
            mode_obj = modes.CBC(self.iv)

        cipher = Cipher(algorithms.AES(self.key), mode_obj)
        self.encryptor = cipher.encryptor()

    def next_bytes(self, n):
        with self._lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                block = secrets.token_bytes(4096)
                encrypted = self.encryptor.update(block)
                self.buffer.extend(encrypted)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)
        
class DRBG_ChaCha20(_BaseDRBG):
    def __init__(self, hash_type="sha512"):
        super().__init__(hash_type)
        self._init_cipher()

    def _init_cipher(self):
        key_material = self.reseed_callback() if self.reseed_callback else secrets.token_bytes(44)
        self.key = key_material[:32]
        self.nonce = key_material[32:44]
        self.encryptor = Cipher(
            algorithms.ChaCha20(self.key, self.nonce),
            mode=None
        ).encryptor()

    def next_bytes(self, n):
        with self._lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                block = secrets.token_bytes(64)
                self.buffer.extend(self.encryptor.update(block))
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)

class DRBG_BLAKE(_BaseDRBG):
    def __init__(self,
                 hash_variant="blake2b",
                 entropy_sources=None,
                 hmac_sources=None,
                 use_hmac=True,
                 pgp_key=None,
                 pgp_pass=None):
        super().__init__()
        self.hash_variant = hash_variant.lower()
        self.use_hmac = use_hmac
        self.entropy_sources = entropy_sources
        self.hmac_sources = hmac_sources
        self.pgp_key = pgp_key
        self.pgp_pass = pgp_pass
        self._initialize_seeds()

    def _digest_entropy_pool(self, sources, block_size=64):
        raw = bytearray()
        for src in sources[:4]:
            data = src.next_bytes_hmac(block_size) if self.use_hmac else src.next_bytes(block_size)
            raw.extend(data)

        if self.hash_variant == "blake2b":
            return hashlib.blake2b(raw, digest_size=64).digest()
        elif self.hash_variant == "blake3":
            hasher = blake3.blake3()
            hasher.update(raw)
            return hasher.digest(length=64)
        else:
            raise ValueError("Unsupported hash variant")

    def _generate_secure_email(self):
        user = secrets.token_hex(16)
        host = secrets.token_hex(16)
        tld = secrets.token_hex(4)
        return f"{user}@{host}.{tld}"

    def _generate_temp_pgp_key(self):
        key = pgpy.PGPKey.new(PubKeyAlgorithm.RSAEncryptOrSign, 2048)
        email = self._generate_secure_email()
        uid = pgpy.PGPUID.new("Entropy Synthesizer", email=email)
        key.add_uid(
            uid,
            usage={KeyFlags.Sign},
            hashes=[HashAlgorithm.SHA512],
            ciphers=[SymmetricKeyAlgorithm.AES256],
            compression=[CompressionAlgorithm.ZLIB]
        )
        passphrase = secrets.token_hex(16)
        key.unlock(passphrase)
        return key, passphrase

    def _initialize_seeds(self):
        with self._lock:
            seed = secrets.token_bytes(64)

            if not self.pgp_key:
                self.pgp_key, self.pgp_pass = self._generate_temp_pgp_key()

            if not self.entropy_sources:
                self.entropy_sources = [
                    DRBG_HashStream("sha3_512"),
                    DRBG_ChaCha20("sha512"),
                    DRBG_AES("gcm"),
                    DRBG_PGP(pgp_privkey=self.pgp_key, passphrase=self.pgp_pass)
                ]

            if not self.hmac_sources:
                self.hmac_sources = self.entropy_sources

            entropy_pool = self._digest_entropy_pool(self.entropy_sources)
            combined = seed + entropy_pool

            if self.hash_variant == "blake2b":
                self.key = hashlib.blake2b(combined, digest_size=64).digest()
            elif self.hash_variant == "blake3":
                hasher = blake3.blake3()
                hasher.update(combined)
                self.key = hasher.digest(length=64)

            self.buffer.clear()
            self.hmac_counter = 0
            self._total_bytes_generated = 0

    def _maybe_reseed(self):
        if self.prediction_resistance or self._total_bytes_generated >= self.reseed_interval:
            self._initialize_seeds()

    def next_bytes(self, n):
        with self._lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                block = self.key + self.hmac_counter.to_bytes(8, "big")
                if self.hash_variant == "blake2b":
                    self.buffer.extend(hashlib.blake2b(block, digest_size=64).digest())
                elif self.hash_variant == "blake3":
                    hasher = blake3.blake3(key=self.key)
                    hasher.update(block)
                    self.buffer.extend(hasher.digest(length=64))
                self.hmac_counter += 1
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += len(result)
            return bytes(result)

    def next_bytes_hmac(self, n):
        with self._lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                entropy_mix = self._digest_entropy_pool(self.hmac_sources)
                msg = self.hmac_counter.to_bytes(8, "big") + entropy_mix
                if self.hash_variant == "blake2b":
                    digest = hmac.new(self.key, msg, digestmod=hashlib.blake2b).digest()
                elif self.hash_variant == "blake3":
                    digest = hmac.new(self.key, msg, digestmod="sha512").digest()  # fallback
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += len(result)
            return result[:n]
    
class DRBG_PGP(_BaseDRBG):
    def __init__(self, pgp_privkey: pgpy.PGPKey, passphrase: str = None, hash_type="sha512", seed=None):
        super().__init__(hash_type)
        
        if pgp_privkey is None:
            raise ValueError("PGP private key is required")
        self.pgp_privkey = pgp_privkey
        if passphrase:
            self.pgp_privkey.unlock(passphrase)

        self._initialize(seed)

    def _sign_counter(self):
        msg = str(self.hmac_counter).encode()
        self.hmac_counter += 1
        sig = self.pgp_privkey.sign(msg, detached=True, hash=self.hash_type.upper())
        return bytes(sig.__bytes__())

    def _initialize(self, seed):
        self.counter = 0
        self.buffer.clear()
        self.hmac_counter = 0

    def next_bytes(self, n):
        with self._lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                sig_bytes = self._sign_counter()
                digest = hashlib.new(self.hash_type, sig_bytes).digest()
                self.buffer.extend(digest)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return result

    def next_bytes_hmac(self, n):
        with self._lock:
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, "big")
                sig = self._sign_counter()
                digest = hmac.new(sig, msg, digestmod=self.hash_type).digest()
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += n
            return result[:n]

class DRBG_RSA(_BaseDRBG):
    def __init__(self, private_key=None, hash_type="sha512", key_size=2048, public_exponent=373587883):
        super().__init__(hash_type)
        self.hash_type = hash_type.lower()
        self.key_size = key_size
        self.public_exponent = public_exponent

        if private_key is None:
            self.private_key = self._generate_key()
        else:
            self.private_key = private_key

        self.public_key = self.private_key.public_key()
        self._reinitialize_seeds()

    def _generate_key(self):
        print(f"[RSA DRBG] Generating {self.key_size}-bit RSA key with exponent {self.public_exponent}...")
        return rsa.generate_private_key(
            public_exponent=self.public_exponent,
            key_size=self.key_size
        )

    def _initialize_seeds(self):
        with self._lock:
            self.private_key = self._generate_key()
            self.public_key = self.private_key.public_key()
            self.buffer.clear()
            self.hmac_counter = 0
            self.counter = 0
            self._total_bytes_generated = 0

    def _maybe_reseed(self):
        if self.prediction_resistance or self._total_bytes_generated >= self.reseed_interval:
            self._initialize_seeds()

    def _rsa_encrypt_counter(self):
        msg = self.hmac_counter.to_bytes(128, "big")
        self.hmac_counter += 1
        return self.public_key.encrypt(
            msg,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def next_bytes(self, n):
        with self._lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                ct = self._rsa_encrypt_counter()
                digest = hashlib.new(self.hash_type, ct).digest()
                self.buffer.extend(digest)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += len(result)
            return result

    def next_bytes_hmac(self, n):
        with self._lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, "big")
                ct = self._rsa_encrypt_counter()
                digest = hmac.new(ct, msg, digestmod=self.hash_type).digest()
                result.extend(digest)
            self._total_bytes_generated += len(result)
            return result[:n]
