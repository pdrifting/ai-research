import numpy as np
import secrets
from runbatterytests import NISTFull, run_all_nist # Assuming the previous class is saved

def perform_sanity_check(bit_count=10000000):
    print(f"[*] Generating {bit_count} bits from OS Entropy (secrets)...")
    
    # Generate random bytes and convert to bit array
    raw_bytes = secrets.token_bytes(bit_count // 8)
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
    
    print("[*] Running NIST Battery...")
    results = run_all_nist(bits)
    
    print("\n[SANITY CHECK RESULTS: secrets vs NIST]")
    print(f"{'Test':<25} | {'P-Value':<10} | {'Status'}")
    print("-" * 45)
    
    passed_tests = 0
    for test_name, p_val in results.items():
        status = "PASS" if p_val >= 0.01 else "FAIL"
        if status == "PASS": passed_tests += 1
        print(f"{test_name:<25} | {p_val:<10.6f} | {status}")
        
    print("-" * 45)
    print(f"[RESULT] {passed_tests}/15 Tests Passed.")
    
    if passed_tests >= 14:
        print("\n[VERDICT] NIST Implementation is VALID. Secrets performed as expected.")
    else:
        print("\n[VERDICT] NIST Implementation may have logic errors or sample size issues.")

if __name__ == "__main__":
    perform_sanity_check()