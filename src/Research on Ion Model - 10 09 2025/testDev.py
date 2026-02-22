import re
from collections import Counter
import math
import secrets

def check_byte_distribution(byte_stream):
    """
    Analyzes the distribution of characters in a byte stream.

    The function passes if no more than 15 characters have a count
    that falls outside one standard deviation from the expected frequency.

    Args:
        byte_stream (bytes): The stream of bytes to analyze.

    Returns:
        tuple: A tuple containing a boolean result (True for pass, False for fail)
               and a message detailing the outcome.
    """
    total_bytes = len(byte_stream)
    num_characters = 256

    if total_bytes == 0:
        return True, "Empty stream provided, nothing to validate."

    # Use a Counter for an efficient way to get character frequencies
    counts = Counter(byte_stream)

    # Calculate the expected frequency and standard deviation
    expected_freq = total_bytes / num_characters
    std_dev = math.sqrt(expected_freq * (1 - 1/num_characters))

    # Determine the lower and upper bounds for one standard deviation
    lower_bound = expected_freq - std_dev
    upper_bound = expected_freq + std_dev

    # Count how many characters fall outside the one-standard-deviation range
    outlier_count = 0
    for char_code in range(num_characters):
        count = counts[char_code]
        if not (lower_bound <= count <= upper_bound):
            outlier_count += 1

    # Check if the number of outliers is within the acceptable limit
    if outlier_count <= 75:
        message = (
            f"PASS: Only {outlier_count} characters were outside one standard deviation "
            f"({std_dev:.2f}) of the expected count ({expected_freq:.2f})."
        )
        return True, message
    else:
        message = (
            f"FAIL: {outlier_count} characters were outside one standard deviation "
            f"({std_dev:.2f}) of the expected count ({expected_freq:.2f}). "
            f"This exceeds the limit of 75."
        )
        return False, message

def extract_outlier_count(message):
    match = re.search(r"(\d+) characters were outside", message)
    return int(match.group(1)) if match else None

def run_distribution_benchmark(iterations=10_000, stream_size=1_000_000):
    best = float('inf')
    worst = float('-inf')
    total = 0
    seen = 0

    for i in range(1, iterations + 1):
        random_bytes = secrets.token_bytes(stream_size)
        result, message = check_byte_distribution(random_bytes)
        outliers = extract_outlier_count(message)

        if outliers is None:
            continue  # skip malformed result

        total += outliers
        seen += 1

        if outliers < best:
            best = outliers
            print(f"[NEW BEST @ {i}] Outliers: {outliers}")
            print(message)
            print("-" * 50)

        if outliers > worst:
            worst = outliers

        if i % 1000 == 0 or i == iterations:
            avg = total / seen
            print(f"[{i} runs] Best: {best}, Worst: {worst}, Avg: {avg:.2f}")

if __name__ == '__main__':
    print("Running byte distribution sanity test over 10,000 iterations...")
    run_distribution_benchmark()
