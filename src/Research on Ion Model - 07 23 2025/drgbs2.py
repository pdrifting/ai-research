import secrets
import hashlib
import hmac
import threading
import pgpy
import blake3
import time
import random
from typing import Literal
from bitarray import bitarray
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, constant_time
from cryptography.hazmat.backends import default_backend

class DRBG_AES:
    def __init__(
        self,
        mode: Literal["cbc", "gcm"] = "cbc",
        seed: str | bytes = None,
    ):
        self.mode = mode.lower()
        self.backend = default_backend()
        self._thread_lock = threading.Lock()

        self.buffer = bytearray()
        self.hmac_buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_buffer = 0
        self._bit_stream = []
        self._bit_byte_index = 0

        self._hmac_bit_pos = 8
        self._hmac_bit_buffer = 0
        self._hmac_bit_stream = []
        self._hmac_bit_byte_index = 0

        self._total_bytes_generated = 0
        self.reseed_interval = 1024 * 1024  # 1 MB
        self.reseed_callback = None

        self._initialize(seed)

    def _initialize(self, seed: str | bytes):
        if seed is None:
            seed = secrets.token_bytes(64)
        elif isinstance(seed, str):
            seed = seed.encode("utf-8")

        salt = secrets.token_bytes(32)

        key_material = HKDF(
            algorithm=hashes.SHA512(),
            length=112,
            salt=salt,
            info=b"drbg key derivation",
            backend=self.backend
        ).derive(seed)

        self.key = key_material[:32]
        self.iv = key_material[32:48]
        self.hmac_key = key_material[48:]

        self.encryptor = self._init_cipher()
        self.buffer = bytearray()
        self.hmac_buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0

    def _init_cipher(self):
        if self.mode == "cbc":
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=self.backend)
        elif self.mode == "gcm":
            cipher = Cipher(algorithms.AES(self.key), modes.GCM(self.iv), backend=self.backend)
        else:
            raise ValueError("Unsupported AES mode")
        return cipher.encryptor()

    def _maybe_reseed(self):
        if self._total_bytes_generated >= self.reseed_interval:
            if self.reseed_callback:
                seed = self.reseed_callback()
            else:
                seed = secrets.token_bytes(64)
            self._initialize(seed)

    def set_reseed_callback(self, func: callable):
        self.reseed_callback = func

    def reseed(self, seed: str | bytes):
        with self._thread_lock:
            self._initialize(seed)

    def next_bytes(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                block = secrets.token_bytes(4096)  # Use real entropy as input
                encrypted = self.encryptor.update(block)
                self.buffer.extend(encrypted)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)

    def next_bytes_hmac(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, 'big')
                digest = hmac.new(self.hmac_key, msg, hashlib.sha512).digest()
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += n
            return bytes(result[:n])

    def next_bits(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._bit_pos == 8:
                    self._bit_stream = list(self.next_bytes(32))
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

    def next_bits_hmac(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_stream = list(self.next_bytes_hmac(32))
                    self._hmac_bit_byte_index = 0
                    self._hmac_bit_pos = 0

                if self._hmac_bit_byte_index >= len(self._hmac_bit_stream):
                    self._hmac_bit_pos = 8
                    continue

                byte = self._hmac_bit_stream[self._hmac_bit_byte_index]
                bit = (byte >> (7 - self._hmac_bit_pos)) & 1
                bits.append(bit)

                self._hmac_bit_pos += 1
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_pos = 0
                    self._hmac_bit_byte_index += 1
            return bits[:n]

    def export_bits_to_file(self, filename: str, n: int, use_hmac: bool = False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

    def export_bytes_to_hex_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

    def export_bytes_to_bin_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(''.join(f"{byte:08b}" for byte in data))

class DRBG_SHA:
    def __init__(
        self,
        hash_type: Literal["sha512", "sha3_512"] = "sha512",
        seed: str | bytes = None,
    ):
        self.backend = default_backend()
        self.hash_type = hash_type.lower()
        if self.hash_type not in {"sha512", "sha3_512"}:
            raise ValueError("Unsupported hash_type. Use 'sha512' or 'sha3_512'.")

        self._thread_lock = threading.Lock()
        self.buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0
        self.reseed_interval = 1024 * 1024  # 1 MB
        self.reseed_callback = None

        self._initialize(seed)

    def _initialize(self, seed: str | bytes):
        if seed is None:
            seed = secrets.token_bytes(64)
        elif isinstance(seed, str):
            seed = seed.encode("utf-8")

        salt = secrets.token_bytes(32)

        alg = hashes.SHA512() if self.hash_type == "sha512" else hashes.SHA3_512()
        self.hash_len = alg.digest_size

        # Derive a base key from seed and salt
        key_material = HKDF(
            algorithm=alg,
            length=self.hash_len,
            salt=salt,
            info=b"drbg sha key derivation",
            backend=self.backend,
        ).derive(seed)

        self.base = key_material  # Base secret for counter hashing
        self.counter = 0

        self.buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0

    def _hash(self, data: bytes) -> bytes:
        return hashlib.new(self.hash_type, data).digest()

    def _maybe_reseed(self):
        if self._total_bytes_generated >= self.reseed_interval:
            if self.reseed_callback:
                seed = self.reseed_callback()
            else:
                seed = secrets.token_bytes(64)
            self._initialize(seed)

    def set_reseed_callback(self, func: callable):
        self.reseed_callback = func

    def reseed(self, seed: str | bytes):
        with self._thread_lock:
            self._initialize(seed)

    def next_bytes(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                input_block = self.base + self.counter.to_bytes(8, "big")
                self.buffer.extend(self._hash(input_block))
                self.counter += 1
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)

    def next_bytes_hmac(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, "big")
                digest = hmac.new(
                    key=self.base,
                    msg=msg,
                    digestmod=self.hash_type,
                ).digest()
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += n
            return bytes(result[:n])

    def next_bits(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._bit_pos == 8:
                    self._bit_stream = list(self.next_bytes(32))
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

    def next_bits_hmac(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_stream = list(self.next_bytes_hmac(32))
                    self._hmac_bit_byte_index = 0
                    self._hmac_bit_pos = 0

                if self._hmac_bit_byte_index >= len(self._hmac_bit_stream):
                    self._hmac_bit_pos = 8
                    continue

                byte = self._hmac_bit_stream[self._hmac_bit_byte_index]
                bit = (byte >> (7 - self._hmac_bit_pos)) & 1
                bits.append(bit)

                self._hmac_bit_pos += 1
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_pos = 0
                    self._hmac_bit_byte_index += 1
            return bits[:n]

    def export_bits_to_file(self, filename: str, n: int, use_hmac: bool = False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

    def export_bytes_to_hex_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

    def export_bytes_to_bin_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write("".join(f"{byte:08b}" for byte in data))

class DRBG_ChaCha20:
    def __init__(
        self,
        hash_type: Literal["sha512", "sha3_512"] = "sha512",
        seed: str | bytes = None,
    ):
        self.backend = default_backend()
        self.hash_type = hash_type.lower()
        if self.hash_type not in {"sha512", "sha3_512"}:
            raise ValueError("Unsupported hash_type. Use 'sha512' or 'sha3_512'.")

        self._thread_lock = threading.Lock()
        self.buffer = bytearray()
        self.hmac_buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0
        self.reseed_interval = 1024 * 1024  # 1 MB
        self.reseed_callback = None

        self._initialize(seed)

    def _initialize(self, seed: str | bytes):
        if seed is None:
            seed = secrets.token_bytes(64)
        elif isinstance(seed, str):
            seed = seed.encode("utf-8")

        salt = secrets.token_bytes(32)

        # Use HKDF with chosen hash
        alg = hashes.SHA512() if self.hash_type == "sha512" else hashes.SHA3_512()
        key_material = HKDF(
            algorithm=alg,
            length=44,  # 32 bytes key + 12 bytes nonce
            salt=salt,
            info=b"drbg chacha20 key derivation",
            backend=self.backend,
        ).derive(seed)

        self.key = key_material[:32]
        self.nonce = key_material[32:44]  # ChaCha20 nonce is 12 bytes

        self.encryptor = self._init_cipher()
        self.buffer = bytearray()
        self.hmac_buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0

    def _init_cipher(self):
        cipher = Cipher(
            algorithms.ChaCha20(self.key, self.nonce),
            mode=None,
            backend=self.backend
        )
        return cipher.encryptor()

    def _maybe_reseed(self):
        if self._total_bytes_generated >= self.reseed_interval:
            if self.reseed_callback:
                seed = self.reseed_callback()
            else:
                seed = secrets.token_bytes(64)
            self._initialize(seed)

    def set_reseed_callback(self, func: callable):
        self.reseed_callback = func

    def reseed(self, seed: str | bytes):
        with self._thread_lock:
            self._initialize(seed)

    def next_bytes(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                block = secrets.token_bytes(64)  # Input entropy for cipher
                encrypted = self.encryptor.update(block)
                self.buffer.extend(encrypted)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)

    def next_bytes_hmac(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, "big")
                digest = hmac.new(
                    key=self.key,
                    msg=msg,
                    digestmod=self.hash_type
                ).digest()
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += n
            return bytes(result[:n])

    def next_bits(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._bit_pos == 8:
                    self._bit_stream = list(self.next_bytes(32))
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

    def next_bits_hmac(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_stream = list(self.next_bytes_hmac(32))
                    self._hmac_bit_byte_index = 0
                    self._hmac_bit_pos = 0

                if self._hmac_bit_byte_index >= len(self._hmac_bit_stream):
                    self._hmac_bit_pos = 8
                    continue

                byte = self._hmac_bit_stream[self._hmac_bit_byte_index]
                bit = (byte >> (7 - self._hmac_bit_pos)) & 1
                bits.append(bit)

                self._hmac_bit_pos += 1
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_pos = 0
                    self._hmac_bit_byte_index += 1
            return bits[:n]

    def export_bits_to_file(self, filename: str, n: int, use_hmac: bool = False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

    def export_bytes_to_hex_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

    def export_bytes_to_bin_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write("".join(f"{byte:08b}" for byte in data))

class DRBG_PGP:
    def __init__(self, pgp_privkey: pgpy.PGPKey = None, passphrase: str = None, hash_type="sha512", seed=None):
        self.hash_type = hash_type.lower()
        if self.hash_type not in {"sha512", "sha3_512"}:
            raise ValueError("Unsupported hash_type")
        self._thread_lock = threading.Lock()

        self.buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0
        self.reseed_interval = 1024 * 1024  # 1MB
        self.reseed_callback = None

        if pgp_privkey is None:
            raise ValueError("A loaded PGP private key (pgpy.PGPKey) is required")

        self.pgp_privkey = pgp_privkey
        if passphrase:
            self.pgp_privkey.unlock(passphrase)

        self._initialize(seed)

    def _initialize(self, seed):
        if seed is None:
            seed = secrets.token_bytes(64)
        elif isinstance(seed, str):
            seed = seed.encode("utf-8")

        self.counter = 0
        self.buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0

    def _maybe_reseed(self):
        if self._total_bytes_generated >= self.reseed_interval:
            if self.reseed_callback:
                seed = self.reseed_callback()
            else:
                seed = secrets.token_bytes(64)
            self._initialize(seed)

    def set_reseed_callback(self, func: callable):
        self.reseed_callback = func

    def reseed(self, seed):
        with self._thread_lock:
            self._initialize(seed)

    def _sign_counter(self):
        """
        Sign the counter using the PGP private key.
        Returns raw bytes of the signature packet.
        """
        msg = str(self.counter).encode()
        self.counter += 1

        sig = self.pgp_privkey.sign(msg, detached=True, hash=self.hash_type.upper())
        # signature blob is ASCII armored, extract bytes:
        sig_bytes = bytes(sig.__bytes__())
        return sig_bytes

    def next_bytes(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                sig_bytes = self._sign_counter()
                digest = hashlib.new(self.hash_type, sig_bytes).digest()
                self.buffer.extend(digest)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)

    def next_bytes_hmac(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, "big")
                sig_bytes = self._sign_counter()
                digest = hmac.new(
                    key=sig_bytes,
                    msg=msg,
                    digestmod=self.hash_type,
                ).digest()
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += n
            return bytes(result[:n])

    def next_bits(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._bit_pos == 8:
                    self._bit_stream = list(self.next_bytes(32))
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

    def next_bits_hmac(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_stream = list(self.next_bytes_hmac(32))
                    self._hmac_bit_byte_index = 0
                    self._hmac_bit_pos = 0

                if self._hmac_bit_byte_index >= len(self._hmac_bit_stream):
                    self._hmac_bit_pos = 8
                    continue

                byte = self._hmac_bit_stream[self._hmac_bit_byte_index]
                bit = (byte >> (7 - self._hmac_bit_pos)) & 1
                bits.append(bit)

                self._hmac_bit_pos += 1
                if self._hmac_bit_pos == 0:
                    self._hmac_bit_byte_index += 1
            return bits[:n]

    def export_bits_to_file(self, filename: str, n: int, use_hmac: bool = False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

    def export_bytes_to_hex_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

class DRBG_RSA:
    def __init__(self, private_key=None, hash_type="sha512", seed=None):
        """
        private_key: cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey instance
        If None, generates a new 8192-bit RSA private key (slow!)
        """
        self.backend = default_backend()
        self.hash_type = hash_type.lower()
        if self.hash_type not in {"sha512", "sha3_512"}:
            raise ValueError("Unsupported hash_type")

        self._thread_lock = threading.Lock()
        self.buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0
        self.reseed_interval = 1024 * 1024  # 1 MB
        self.reseed_callback = None

        if private_key is None:
            print("Generating 8192-bit RSA private key, please wait...")
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=8192,
                backend=self.backend
            )
        else:
            self.private_key = private_key

        self.public_key = self.private_key.public_key()

        # Seed is used to initialize counter salt if needed
        self._initialize(seed)

    def _initialize(self, seed):
        if seed is None:
            seed = secrets.token_bytes(64)
        elif isinstance(seed, str):
            seed = seed.encode("utf-8")

        self.counter = 0
        self.buffer = bytearray()
        self.hmac_counter = 0

        self._bit_pos = 8
        self._bit_byte_index = 0
        self._bit_stream = []

        self._hmac_bit_pos = 8
        self._hmac_bit_byte_index = 0
        self._hmac_bit_stream = []

        self._total_bytes_generated = 0

    def _maybe_reseed(self):
        if self._total_bytes_generated >= self.reseed_interval:
            if self.reseed_callback:
                seed = self.reseed_callback()
            else:
                seed = secrets.token_bytes(64)
            self._initialize(seed)

    def set_reseed_callback(self, func: callable):
        self.reseed_callback = func

    def reseed(self, seed):
        with self._thread_lock:
            self._initialize(seed)

    def _rsa_encrypt_counter(self):
        """
        Encrypt counter using RSA with OAEP padding.
        Returns ciphertext bytes.
        """
        counter_bytes = self.counter.to_bytes(128, "big")  # 8192 bits = 128 bytes
        ciphertext = self.public_key.encrypt(
            counter_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        self.counter += 1
        return ciphertext

    def next_bytes(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            while len(self.buffer) < n:
                ct = self._rsa_encrypt_counter()
                # Hash ciphertext to fixed-length digest
                digest = hashlib.new(self.hash_type, ct).digest()
                self.buffer.extend(digest)
            result = self.buffer[:n]
            del self.buffer[:n]
            self._total_bytes_generated += n
            return bytes(result)

    def next_bytes_hmac(self, n: int) -> bytes:
        with self._thread_lock:
            self._maybe_reseed()
            result = bytearray()
            while len(result) < n:
                msg = self.hmac_counter.to_bytes(8, "big")
                ct = self._rsa_encrypt_counter()
                digest = hmac.new(
                    key=ct,
                    msg=msg,
                    digestmod=self.hash_type,
                ).digest()
                result.extend(digest)
                self.hmac_counter += 1
            self._total_bytes_generated += n
            return bytes(result[:n])

    def next_bits(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._bit_pos == 8:
                    self._bit_stream = list(self.next_bytes(32))
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

    def next_bits_hmac(self, n: int) -> bitarray:
        with self._thread_lock:
            bits = bitarray()
            while len(bits) < n:
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_stream = list(self.next_bytes_hmac(32))
                    self._hmac_bit_byte_index = 0
                    self._hmac_bit_pos = 0

                if self._hmac_bit_byte_index >= len(self._hmac_bit_stream):
                    self._hmac_bit_pos = 8
                    continue

                byte = self._hmac_bit_stream[self._hmac_bit_byte_index]
                bit = (byte >> (7 - self._hmac_bit_pos)) & 1
                bits.append(bit)

                self._hmac_bit_pos += 1
                if self._hmac_bit_pos == 8:
                    self._hmac_bit_pos = 0
                    self._hmac_bit_byte_index += 1
            return bits[:n]

    def export_bits_to_file(self, filename: str, n: int, use_hmac: bool = False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

    def export_bytes_to_hex_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

    def export_bytes_to_bin_file(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write("".join(f"{byte:08b}" for byte in data))

class DRBG_BLAKE:
    def __init__(
        self,
        hash_variant: Literal["blake2b", "blake3"] = "blake2b",
        seed: str | bytes = None,
        entropy_sources: list[Callable[[int], bytes]] = None,
        hmac_sources: list[Callable[[int], bytes]] = None
    ):
        self.hash_variant = hash_variant
        self._thread_lock = threading.Lock()
        self.buffer = bytearray()
        self.hmac_counter = 0
        self.entropy_sources = entropy_sources or []
        self.hmac_sources = hmac_sources or []
        self.entropy_pool = self._gather_entropy(self.entropy_sources)
        self.hkdf = None
        self._initialize(seed)

    def _gather_entropy(self, sources: list[Callable[[int], bytes]]) -> bytes:
        return b"".join(src(64) for src in sources[:4])

    def _get_hkdf_algorithm(self):
        if self.hash_variant == "blake2b":
            return hashes.SHA512()
        elif self.hash_variant == "blake3":
            return hashes.SHA3_512()
        else:
            raise ValueError("Unsupported hash variant")

    def _initialize(self, seed: str | bytes):
        if seed is None:
            seed = secrets.token_bytes(64)
        elif isinstance(seed, str):
            seed = seed.encode("utf-8")

        combined = seed + self.entropy_pool

        if self.hash_variant == "blake2b":
            self.key = hashlib.blake2b(combined, digest_size=64).digest()
        elif self.hash_variant == "blake3":
            hasher = blake3.blake3()
            hasher.update(combined)
            self.key = hasher.digest(length=64)
        else:
            raise ValueError("Unsupported hash variant")

        self.hkdf = HKDF(
            algorithm=self._get_hkdf_algorithm(),
            length=64,
            salt=b"",
            info=b"drbg-blake",
            backend=default_backend()
        )

        self.buffer.clear()
        self.hmac_counter = 0

    def reseed(self, seed: str | bytes, entropy_sources: list[Callable[[int], bytes]] = None):
        with self._thread_lock:
            if entropy_sources:
                self.entropy_sources = entropy_sources
                self.entropy_pool = self._gather_entropy(self.entropy_sources)
            self._initialize(seed)

    def next_bytes(self, n: int) -> bytes:
        with self._thread_lock:
            while len(self.buffer) < n:
                data = self.key + self.hmac_counter.to_bytes(8, 'big')
                if self.hash_variant == "blake2b":
                    self.buffer.extend(hashlib.blake2b(data, digest_size=64).digest())
                elif self.hash_variant == "blake3":
                    hasher = blake3.blake3(key=self.key)
                    hasher.update(self.hmac_counter.to_bytes(8, 'big'))
                    self.buffer.extend(hasher.digest(length=64))
                self.hmac_counter += 1
            result = self.buffer[:n]
            del self.buffer[:n]
            return bytes(result)

    def next_bytes_hmac(self, n: int) -> bytes:
        with self._thread_lock:
            result = bytearray()
            while len(result) < n:
                entropy = self._gather_entropy(self.hmac_sources)
                msg = self.hmac_counter.to_bytes(8, 'big') + entropy
                digest = hmac.new(self.key, msg, digestmod=self.hash_variant).digest()
                result.extend(digest)
                self.hmac_counter += 1
            return bytes(result[:n])

    def next_bits(self, n: int) -> bitarray:
        bits = bitarray()
        data = self.next_bytes((n + 7) // 8)
        for byte in data:
            bits.extend(f"{byte:08b}")
        return bitarray(bits.tolist()[:n])

    def next_bits_hmac(self, n: int) -> bitarray:
        bits = bitarray()
        data = self.next_bytes_hmac((n + 7) // 8)
        for byte in data:
            bits.extend(f"{byte:08b}")
        return bitarray(bits.tolist()[:n])

    def export_bin(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "wb") as f:
            f.write(data)

    def export_hex(self, filename: str, n: int, use_hmac: bool = False):
        data = self.next_bytes_hmac(n) if use_hmac else self.next_bytes(n)
        with open(filename, "w") as f:
            f.write(data.hex())

    def export_bits(self, filename: str, n: int, use_hmac: bool = False):
        bits = self.next_bits_hmac(n) if use_hmac else self.next_bits(n)
        with open(filename, "wb") as f:
            bits.tofile(f)

    def derive_key_hkdf(self, length: int, salt: bytes = b"", info: bytes = b"drbg-blake") -> bytes:
        hkdf = HKDF(
            algorithm=self._get_hkdf_algorithm(),
            length=length,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(self.key)

class BaseDRBG:
    def __init__(self):
        self._lock = threading.Lock()
        self.key = secrets.token_bytes(64)

    def derive_key_hkdf(self, length: int, salt: bytes = b"", info: bytes = b"drbg") -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA512(),
            length=length,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(self.key)

    def next_bytes_hmac(self, n: int, algo: str) -> bytes:
        result = bytearray()
        counter = 0
        while len(result) < n:
            msg = counter.to_bytes(8, 'big')
            result.extend(hmac.new(self.key, msg, digestmod=algo).digest())
            counter += 1
        return bytes(result[:n])

class DRBG_OSRandom(BaseDRBG):
    def reseed(self):
        with self._lock:
            self.key = secrets.token_bytes(64)

    def next_bytes(self, n: int) -> bytes:
        with self._lock:
            return os.urandom(n)

    def next_bits(self, n: int) -> bitarray:
        data = self.next_bytes((n + 7) // 8)
        bits = bitarray()
        for byte in data:
            bits.extend(f"{byte:08b}")
        return bitarray(bits.tolist()[:n])

    def export_bin(self, filename: str, n: int):
        with open(filename, "wb") as f:
            f.write(self.next_bytes(n))

    def export_hex(self, filename: str, n: int):
        with open(filename, "w") as f:
            f.write(self.next_bytes(n).hex())

    def export_bits(self, filename: str, n: int):
        with open(filename, "wb") as f:
            self.next_bits(n).tofile(f)


class DRBG_TimeXORRandom(BaseDRBG):
    def reseed(self):
        with self._lock:
            self.key = secrets.token_bytes(64)

    def next_bytes(self, n: int) -> bytes:
        with self._lock:
            output = bytearray()
            for _ in range(n // 8 + 1):
                t = time.time_ns()
                r = secrets.randbits(64)
                x = (t ^ r).to_bytes(8, 'big')
                output.extend(x)
            return bytes(output[:n])

    def next_bits(self, n: int) -> bitarray:
        data = self.next_bytes((n + 7) // 8)
        bits = bitarray()
        for byte in data:
            bits.extend(f"{byte:08b}")
        return bitarray(bits.tolist()[:n])

    def export_bin(self, filename: str, n: int):
        with open(filename, "wb") as f:
            f.write(self.next_bytes(n))

    def export_hex(self, filename: str, n: int):
        with open(filename, "w") as f:
            f.write(self.next_bytes(n).hex())

    def export_bits(self, filename: str, n: int):
        with open(filename, "wb") as f:
            self.next_bits(n).tofile(f)


class DRBG_RandomBits(BaseDRBG):
    def reseed(self):
        with self._lock:
            self.key = secrets.token_bytes(64)

    def next_bytes(self, n: int) -> bytes:
        with self._lock:
            output = bytearray()
            for _ in range(n):
                output.append(random.getrandbits(8))
            return bytes(output)

    def next_bits(self, n: int) -> bitarray:
        data = self.next_bytes((n + 7) // 8)
        bits = bitarray()
        for byte in data:
            bits.extend(f"{byte:08b}")
        return bitarray(bits.tolist()[:n])

    def export_bin(self, filename: str, n: int):
        with open(filename, "wb") as f:
            f.write(self.next_bytes(n))

    def export_hex(self, filename: str, n: int):
        with open(filename, "w") as f:
            f.write(self.next_bytes(n).hex())

    def export_bits(self, filename: str, n: int):
        with open(filename, "wb") as f:
            self.next_bits(n).tofile(f)


class DRBG_Hash(BaseDRBG):
    def __init__(self, algo: Literal["whirlpool", "ripemd160", "md5"]):
        super().__init__()
        self.algo = algo
        self.counter = 0

    def reseed(self):
        with self._lock:
            self.key = secrets.token_bytes(64)
            self.counter = 0

    def next_bytes(self, n: int) -> bytes:
        with self._lock:
            stream = bytearray()
            while len(stream) < n:
                data = self.counter.to_bytes(8, 'big')
                h = hashlib.new(self.algo, data).digest()
                stream.extend(h)
                self.counter += 1
            return bytes(stream[:n])

    def next_bits(self, n: int) -> bitarray:
        data = self.next_bytes((n + 7) // 8)
        bits = bitarray()
        for byte in data:
            bits.extend(f"{byte:08b}")
        return bitarray(bits.tolist()[:n])

    def export_bin(self, filename: str, n: int):
        with open(filename, "wb") as f:
            f.write(self.next_bytes(n))

    def export_hex(self, filename: str, n: int):
        with open(filename, "w") as f:
            f.write(self.next_bytes(n).hex())

    def export_bits(self, filename: str, n: int):
        with open(filename, "wb") as f:
            self.next_bits(n).tofile(f)
