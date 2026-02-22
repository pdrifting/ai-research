import os
import sys
import argparse
from pathlib import Path
from argon2.low_level import hash_secret_raw, Type
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

# Argon2id Key Derivation
def derive_key(password: bytes, salt: bytes, length=32):
    return hash_secret_raw(
        secret=password,
        salt=salt,
        time_cost=3,
        memory_cost=2**16,
        parallelism=2,
        hash_len=length,
        type=Type.ID
    )

# ChaCha20 DRBG
class ChaCha20DRBG:
    def __init__(self, seed: bytes, nonce: bytes):
        self.seed = seed
        self.nonce = nonce

    def generate(self, length: int):
        cipher = Cipher(algorithms.ChaCha20(self.seed, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(b'\x00' * length)

# AES-CTR Encryption
def aes_ctr_encrypt(key: bytes, plaintext: bytes, iv: bytes):
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(plaintext) + encryptor.finalize()

# RSA Signing
def rsa_sign(data: bytes, output_dir: Path):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    signature = private_key.sign(
        data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA512()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA512()
    )

    (output_dir / "rsa.private.key").write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
    )

    (output_dir / "rsa.public").write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )

    return signature

# Hashing
def hash_data(data: bytes):
    digest = hashes.Hash(hashes.SHA512(), backend=default_backend())
    digest.update(data)
    return digest.finalize()

# Generation Mode
def generate_pipeline(password: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    password_bytes = password.encode()
    salt = os.random(16)
    nonce = os.random(16)
    iv = os.random(16)

    derived_key = derive_key(password_bytes, salt)
    drbg = ChaCha20DRBG(seed=derived_key, nonce=nonce)
    random_data = drbg.generate(64)
    encrypted = aes_ctr_encrypt(derived_key, random_data, iv)
    hashed = hash_data(encrypted)
    signature = rsa_sign(hashed, output_dir)

    # Save all artifacts
    (output_dir / "password.raw").write_bytes(password_bytes)
    (output_dir / "password.hash").write_bytes(hashed)
    (output_dir / "argon2.salt").write_bytes(salt)
    (output_dir / "chacha20.nonce").write_bytes(nonce)
    (output_dir / "aes.iv").write_bytes(iv)

    print("Generation complete. Files written to:", output_dir)

# Verification Mode
def verify_pipeline(input_dir: Path):
    password_bytes = (input_dir / "password.raw").read_bytes()
    stored_hash = (input_dir / "password.hash").read_bytes()
    salt = (input_dir / "argon2.salt").read_bytes()
    nonce = (input_dir / "chacha20.nonce").read_bytes()
    iv = (input_dir / "aes.iv").read_bytes()
    public_key_bytes = (input_dir / "rsa.public").read_bytes()
    private_key_bytes = (input_dir / "rsa.private.key").read_bytes()

    derived_key = derive_key(password_bytes, salt)
    drbg = ChaCha20DRBG(seed=derived_key, nonce=nonce)
    random_data = drbg.generate(64)
    encrypted = aes_ctr_encrypt(derived_key, random_data, iv)
    new_hash = hash_data(encrypted)

    public_key = serialization.load_pem_public_key(public_key_bytes, backend=default_backend())
    private_key = serialization.load_pem_private_key(private_key_bytes, password=None, backend=default_backend())

    try:
        public_key.verify(
            private_key.sign(
                new_hash,
                padding.PSS(mgf=padding.MGF1(hashes.SHA512()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA512()
            ),
            new_hash,
            padding.PSS(mgf=padding.MGF1(hashes.SHA512()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA512()
        )
        if new_hash == stored_hash:
            print("Verification successful: hash matches.")
        else:
            print("Verification failed: hash mismatch.")
    except Exception as e:
        print("Verification error:", str(e))

# CLI Entry Point
def main():
    parser = argparse.ArgumentParser(description="Cryptographic pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gen_parser = subparsers.add_parser("gen", help="Generate cryptographic artifacts")
    gen_parser.add_argument("password", type=str, help="Password to derive key")
    gen_parser.add_argument("output_dir", type=str, help="Directory to store output files")

    verify_parser = subparsers.add_parser("verify", help="Verify password and hash")
    verify_parser.add_argument("input_dir", type=str, help="Directory containing stored files")

    args = parser.parse_args()

    if args.mode == "gen":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_pipeline(args.password, output_dir)

    elif args.mode == "verify":
        input_dir = Path(args.input_dir)
        verify_pipeline(input_dir)

if __name__ == "__main__":
    main()
