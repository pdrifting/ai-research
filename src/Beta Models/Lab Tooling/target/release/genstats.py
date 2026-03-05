import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kstest, uniform
import matplotlib.pyplot as plt

RESULTS_DIR = ".\\"   # adjust if needed
PLOT = True               # set False to disable plots

def analyze_test_file(path):
    name = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)

    if "p_value" not in df.columns:
        print(f"[WARN] {name}: no p_value column found")
        return None

    p = df["p_value"].astype(float).values
    n = len(p)

    if n == 0:
        print(f"[WARN] {name}: empty file")
        return None

    # Basic stats
    stats = {
        "test": name,
        "samples": n,
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "mean": float(np.mean(p)),
        "median": float(np.median(p)),
        "std": float(np.std(p)),
    }

    # Histogram bins
    hist_counts, _ = np.histogram(p, bins=10, range=(0,1))
    hist_probs = hist_counts / n
    stats["histogram"] = hist_probs.tolist()

    # KS test for uniformity
    ks_stat, ks_p = kstest(p, uniform.cdf)
    stats["ks_stat"] = float(ks_stat)
    stats["ks_p"] = float(ks_p)

    # Optional plots
    if PLOT:
        plt.figure(figsize=(10,4))
        plt.hist(p, bins=20, range=(0,1), alpha=0.7, color="steelblue")
        plt.title(f"{name} — p-value histogram (n={n})")
        plt.xlabel("p-value")
        plt.ylabel("count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name}_hist.png")
        plt.close()

        # QQ plot
        plt.figure(figsize=(6,6))
        sorted_p = np.sort(p)
        uniform_q = np.linspace(0,1,len(p))
        plt.plot(uniform_q, sorted_p, "o", markersize=2)
        plt.plot([0,1],[0,1], "r--")
        plt.title(f"{name} — QQ plot vs Uniform(0,1)")
        plt.xlabel("Theoretical quantile")
        plt.ylabel("Empirical quantile")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name}_qq.png")
        plt.close()

    return stats


def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "scalar_*.csv"))
    if not files:
        print("No scalar_*.csv files found.")
        return

    all_stats = []

    for path in files:
        stats = analyze_test_file(path)
        if stats:
            all_stats.append(stats)

    # Print summary table
    print("\n=== SUMMARY ===")
    for s in all_stats:
        print(f"\nTest: {s['test']}")
        print(f"  Samples: {s['samples']}")
        print(f"  Min:     {s['min']:.6f}")
        print(f"  Max:     {s['max']:.6f}")
        print(f"  Mean:    {s['mean']:.6f}")
        print(f"  Median:  {s['median']:.6f}")
        print(f"  Std:     {s['std']:.6f}")
        print(f"  KS stat: {s['ks_stat']:.6f}")
        print(f"  KS p:    {s['ks_p']:.6f}")
        print(f"  Histogram (10 bins): {s['histogram']}")

    # Save full report
    report_df = pd.DataFrame(all_stats)
    report_df.to_csv("test_analysis_summary.csv", index=False)
    print("\nSaved summary to test_analysis_summary.csv")


if __name__ == "__main__":
    main()
